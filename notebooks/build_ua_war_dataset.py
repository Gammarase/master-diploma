"""Build the UA-war news dataset used by ``demo.ipynb``.

Items are news claims about the 2022 Russian invasion of Ukraine, labelled
``true`` or ``false``, built from the EUvsDisinfo project (EEAS East StratCom
Task Force):

* false - titles of EUvsDisinfo disinformation cases (Mendeley dump, CC BY 4.0,
  doi:10.17632/yhdtkszvgp.3), debunked on or after 2022-02-24.
* true  - headlines of the "trustworthy" articles cited by EUvsDisinfo debunks
  (Leite et al., CIKM 2024, Zenodo 10514307, CC BY-SA 4.0), scraped with
  trafilatura and published on or after 2022-02-24.

Headlines are kept only if a local Ollama model judges them to be a
self-contained, checkable factual statement. A small uk/ru subset is added from
native-language articles and Ollama translations (flagged ``translated``).

Usage (from the project root, Ollama running):
    .venv/Scripts/python notebooks/build_ua_war_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import trafilatura
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "ua_war_news"
RAW_DIR = OUT_DIR / "raw"

CASES_URL = (
    "https://data.mendeley.com/public-files/datasets/yhdtkszvgp/files/"
    "481ecd12-8a6e-4c98-8bf5-3d8cc58703fa/file_downloaded"
)
BASE_URL = "https://zenodo.org/records/10514307/files/euvsdisinfo_base.csv?download=1"

INVASION_DATE = pd.Timestamp("2022-02-24")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"

WAR_PATTERN = re.compile(
    r"ukrain|kyiv|kiev|zelensk|donbas|donetsk|luhansk|kherson|mariupol|crimea|"
    r"zaporizh|kharkiv|bucha|azov|nato|sanction|special military operation|"
    r"russian (?:army|troops|forces|military|invasion)",
    re.IGNORECASE,
)
# URLs whose headlines describe claims rather than state facts, or are not news.
SKIP_URL_PATTERN = re.compile(
    r"fact-?check|disinfo|stopfake|/live/|live-updates|/video|/opinion|"
    r"wikipedia\.org|euneighbourseast\.eu|understandingwar\.org",
    re.IGNORECASE,
)
SKIP_TITLE_PATTERN = re.compile(
    r"^(analysis|opinion|explainer|live|video|watch|timeline|q&a|in pictures)\b|"
    r"fact[- ]?check|\?|^factbox|"
    # Debunk headlines ("Fake: ...") give the answer away; they are not plain news.
    r"fake|фейк|disinform|дезинформ|дезінформ|\bmyth|\bміф|\bмиф",
    re.IGNORECASE,
)
# Trailing " | Site name" / " - Site name" appended to scraped <title>s.
TITLE_SUFFIX = re.compile(r"\s+[|\-–—]\s+([^|\-–—]{2,40})$")
LANG_CODES = {"English": "en", "Ukrainian": "uk", "Russian": "ru"}
LANG_NAMES = {"uk": "Ukrainian", "ru": "Russian"}


def download(url: str, path: Path) -> Path:
    if not path.exists():
        print(f"Downloading {path.name} ...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            path.write_bytes(r.read())
    return path


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title)).strip().strip('"').strip()


def strip_site_suffix(title: str, *site_names: str) -> str:
    """Drop a trailing " | Reuters"-style suffix, but only when it names the site."""
    match = TITLE_SUFFIX.search(title)
    if not match:
        return title
    suffix = re.sub(r"[^a-z0-9а-яіїєґ]", "", match.group(1).lower())
    for name in site_names:
        token = re.sub(r"[^a-z0-9а-яіїєґ]", "", str(name or "").lower().split(".")[0])
        if len(token) >= 2 and (token in suffix or suffix in token):
            return title[: match.start()].strip()
    return title


def word_count_ok(title: str, lo: int = 5, hi: int = 30) -> bool:
    return lo <= len(title.split()) <= hi


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def ollama(prompt: str, client: httpx.Client) -> str:
    resp = client.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


CHECKWORTHY_PROMPT = """You are building a fact-checking benchmark about the Russian invasion of Ukraine.
Decide whether the news headline below is a self-contained factual statement that can be checked as true or false.
Answer NO if it is a question, a topic label, a report or section title, a quote without a claim, an opinion piece title, or too vague to verify.
Answer NO if it is not related to Russia, Ukraine or the war.

Headline: {title}

Answer with a single word: YES or NO."""

TRANSLATE_PROMPT = """Translate the following news headline into {language}.
Keep names, numbers and meaning exactly. Output only the translation, nothing else.

Headline: {title}"""


def load_json_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_json_cache(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def filter_checkworthy(titles: list[str], client: httpx.Client) -> dict[str, bool]:
    cache_path = RAW_DIR / "checkworthy_cache.json"
    cache = load_json_cache(cache_path)
    todo = [t for t in dict.fromkeys(titles) if t not in cache]
    for i, title in enumerate(tqdm(todo, desc="checkworthy filter")):
        answer = ollama(CHECKWORTHY_PROMPT.format(title=title), client)
        cache[title] = answer.upper().startswith("YES")
        if i % 25 == 0:
            save_json_cache(cache_path, cache)
    save_json_cache(cache_path, cache)
    return cache


def translate(title: str, lang: str, client: httpx.Client, cache: dict) -> str:
    key = f"{lang}::{title}"
    if key not in cache:
        text = ollama(TRANSLATE_PROMPT.format(language=LANG_NAMES[lang], title=title), client)
        cache[key] = clean_title(text.splitlines()[0])
    return cache[key]


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def load_false_items(cases_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cases_path, encoding="utf-8", encoding_errors="replace")
    df["dt"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df = df[df["dt"] >= INVASION_DATE].copy()
    text = df["Title"].fillna("") + " " + df["Country"].fillna("") + " " + df["Disinformation"].fillna("")
    df = df[text.str.contains(WAR_PATTERN)]
    df["claim"] = df["Title"].map(clean_title)
    df = df[df["claim"].map(word_count_ok) & ~df["claim"].str.contains("�")]
    return pd.DataFrame(
        {
            "claim": df["claim"],
            "label": "false",
            "lang_code": "en",
            "date": df["dt"].dt.strftime("%Y-%m-%d"),
            "source_url": "https://euvsdisinfo.eu" + df["Links"].astype(str),
            "publisher": df["Outlets"].fillna(""),
            "origin": "euvsdisinfo_case",
            "context": df["Disinformation"].fillna("").str.strip().str.slice(0, 600),
            "rationale": df["Information"].fillna("").str.strip().str.slice(0, 600),
        }
    ).drop_duplicates("claim")


def fetch_headline(url: str) -> dict:
    try:
        html = trafilatura.fetch_url(url)
        meta = trafilatura.extract_metadata(html) if html else None
    except Exception as exc:  # noqa: BLE001 - network errors are just skipped
        return {"url": url, "error": str(exc)[:200]}
    if meta is None:
        return {"url": url, "error": "unavailable"}
    return {"url": url, "title": meta.title, "date": meta.date, "sitename": meta.sitename}


def load_true_items(base_path: Path, workers: int) -> pd.DataFrame:
    df = pd.read_csv(base_path)
    df["dt"] = pd.to_datetime(df["debunk_date"], format="%d-%m-%Y", errors="coerce")
    df = df[(df["class"] == "trustworthy") & (df["dt"] >= INVASION_DATE)]
    df = df[df["article_language"].isin(LANG_CODES)]
    df = df[~df["article_url"].str.contains(SKIP_URL_PATTERN)].drop_duplicates("article_url")

    cache_path = RAW_DIR / "headlines_cache.json"
    cache = load_json_cache(cache_path)
    todo = [u for u in df["article_url"] if u not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(tqdm(pool.map(fetch_headline, todo), total=len(todo), desc="fetch headlines")):
                cache[rec["url"]] = rec
                if i % 50 == 0:
                    save_json_cache(cache_path, cache)
        save_json_cache(cache_path, cache)

    rows = []
    for _, r in df.iterrows():
        rec = cache.get(r["article_url"], {})
        title = strip_site_suffix(
            clean_title(rec.get("title") or ""),
            rec.get("sitename"), r["article_domain"], r["article_publisher"], "bbc", "news",
        )
        published = pd.to_datetime(rec.get("date"), errors="coerce")
        if not title or pd.isna(published) or published < INVASION_DATE:
            continue
        if SKIP_TITLE_PATTERN.search(title) or not word_count_ok(title):
            continue
        rows.append(
            {
                "claim": title,
                "label": "true",
                "lang_code": LANG_CODES[r["article_language"]],
                "date": published.strftime("%Y-%m-%d"),
                "source_url": r["article_url"],
                "publisher": rec.get("sitename") or r["article_domain"],
                "origin": "euvsdisinfo_trustworthy",
                "context": "",
                "rationale": "",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("claim")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = download(CASES_URL, RAW_DIR / "euvsdisinfo_cases.csv")
    base_path = download(BASE_URL, RAW_DIR / "euvsdisinfo_base.csv")

    false_df = load_false_items(cases_path)
    true_df = load_true_items(base_path, args.workers)
    print(f"Candidates: false={len(false_df)} true={len(true_df)} "
          f"(true by lang: {true_df['lang_code'].value_counts().to_dict()})")

    with httpx.Client() as client:
        # Only screen as many candidates as we can use, sampled with a fixed seed.
        false_pool = false_df.sample(frac=1, random_state=args.seed).head(args.per_class * 2)
        true_en = true_df[true_df["lang_code"] == "en"].sample(frac=1, random_state=args.seed).head(args.per_class * 2)
        true_native = true_df[true_df["lang_code"] != "en"]
        verdicts = filter_checkworthy(
            list(false_pool["claim"]) + list(true_en["claim"]) + list(true_native["claim"]), client
        )
        keep = lambda d: d[d["claim"].map(verdicts).fillna(False).astype(bool)]  # noqa: E731
        false_pool, true_en, true_native = keep(false_pool), keep(true_en), keep(true_native)
        print(f"Check-worthy: false={len(false_pool)} true_en={len(true_en)} true_native={len(true_native)}")

        # Reserve items for translation first so English and translated rows never overlap.
        n_spare = 2 * args.per_lang
        false_spare, true_spare = false_pool.head(n_spare), true_en.head(n_spare)
        n_en = min(args.per_class, len(false_pool) - n_spare, len(true_en) - n_spare)
        false_en = false_pool.iloc[n_spare:n_spare + n_en]
        true_en_sel = true_en.iloc[n_spare:n_spare + n_en]

        tcache_path = RAW_DIR / "translation_cache.json"
        tcache = load_json_cache(tcache_path)
        parts = [false_en, true_en_sel]
        for i, lang in enumerate(("uk", "ru")):
            native = true_native[true_native["lang_code"] == lang].head(args.per_lang)
            need_true = args.per_lang - len(native)
            src_false = false_spare.iloc[i * args.per_lang:(i + 1) * args.per_lang]
            src_true = true_spare.iloc[i * need_true:(i + 1) * need_true] if need_true > 0 else true_spare.head(0)
            for src in (src_false, src_true):
                tr = src.copy()
                tr["orig_claim"] = tr["claim"]
                tr["claim"] = [translate(t, lang, client, tcache) for t in tqdm(tr["claim"], desc=f"translate {lang}")]
                tr["lang_code"] = lang
                tr["translated"] = True
                parts.append(tr)
            parts.append(native.assign(translated=False))
            save_json_cache(tcache_path, tcache)

    out = pd.concat(parts, ignore_index=True)
    out["translated"] = out["translated"].fillna(False).astype(bool)
    out["orig_claim"] = out.get("orig_claim", pd.Series(dtype=str)).fillna("")
    out = out.drop_duplicates("claim").sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.insert(0, "id", [f"uaw-{i:04d}" for i in range(len(out))])
    columns = ["id", "claim", "label", "lang_code", "date", "source_url", "publisher",
               "origin", "translated", "orig_claim", "context", "rationale"]
    return out[columns]


def write_readme(df: pd.DataFrame) -> None:
    counts = "```\n" + pd.crosstab([df["lang_code"], df["translated"]], df["label"], margins=True).to_string() + "\n```"
    (OUT_DIR / "README.md").write_text(
        f"""# UA-war news (EUvsDisinfo) - demo dataset

Built {date.today().isoformat()} by `notebooks/build_ua_war_dataset.py`. {len(df)} items.

| column | meaning |
|---|---|
| claim | headline / claim text to verify |
| label | `true` or `false` |
| lang_code | en / uk / ru |
| date | publication date of the article (true) or of the debunk (false) |
| source_url | trustworthy article URL (true) or EUvsDisinfo debunk page (false) |
| publisher | article publisher (true) or pro-Kremlin outlets that spread it (false) |
| origin | `euvsdisinfo_trustworthy` or `euvsdisinfo_case` |
| translated | True if machine-translated from English with Ollama ({OLLAMA_MODEL}) |
| orig_claim | English original of translated rows |
| context / rationale | disinformation text and EUvsDisinfo disproof (false rows only) |

## Counts

{counts}

## Sources and licences

* False: EUvsDisinfo disinformation cases, Mendeley Data doi:10.17632/yhdtkszvgp.3 (CC BY 4.0).
* True: trustworthy article URLs from J. A. Leite et al., "EUvsDisinfo: A Dataset for Multilingual
  Detection of Pro-Kremlin Disinformation in News Articles", CIKM 2024, Zenodo 10514307 (CC BY-SA 4.0);
  headlines scraped with trafilatura.
* All items dated on or after 2022-02-24 and screened by {OLLAMA_MODEL} for being a checkable statement.

## Caveats

* Label semantics differ by origin: a false item is a debunked claim; a true item is a headline from a
  credible outlet (true by source reputation, not individually fact-checked).
* The EUvsDisinfo debunk page and the original true article are both reachable by web search, so a
  retriever without a temporal filter can find the "answer key".
* Translated rows inherit the English label; translation errors are possible.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=170, help="English items per label")
    parser.add_argument("--per-lang", type=int, default=25, help="uk/ru items per label and language")
    parser.add_argument("--workers", type=int, default=8, help="parallel headline fetchers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "ua_war_news.csv", index=False, encoding="utf-8")
    write_readme(df)
    print(pd.crosstab([df["lang_code"], df["translated"]], df["label"], margins=True))
    print(f"Wrote {OUT_DIR / 'ua_war_news.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
