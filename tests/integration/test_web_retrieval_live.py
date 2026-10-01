"""
Live check of WebEvidenceRetriever against local SearXNG and Ollama.

Not part of the default run. Execute explicitly:
    .venv/Scripts/python -m pytest -m live -s

Uses the real configuration (source policy, reranker, Ollama query
generation) with a temporary cache. Besides the assertions it prints, for
every grouped ``site:`` request, the hit count and the share of hits on the
group's domains — to check whether the engines honour OR-joined ``site:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from configs import Settings, get_settings
from retrieval.cache import WebCache
from retrieval.page_fetcher import PageFetcher
from retrieval.query_builder import QueryBuilder
from retrieval.reranker import Reranker
from retrieval.source_policy import SourcePolicy
from retrieval.web_retriever import WebEvidenceRetriever
from retrieval.web_search import (
    OllamaSearchBackend,
    SearchHit,
    SearchResponse,
    SearxngBackend,
)
from verification.rag_verifier import OllamaClient

pytestmark = pytest.mark.live

CLAIMS = [
    (
        "en",
        "The IAEA said the Zaporizhzhia nuclear power plant was shelled in "
        "August 2022.",
    ),
    ("uk", "Росія розпочала повномасштабне вторгнення в Україну 24 лютого 2022 року."),
]


@dataclass
class _Request:
    query: str
    sites: list[str] | None
    hits: list[str] = field(default_factory=list)
    engine_errors: int = 0
    error: str = ""


class RecordingBackend:
    """Wraps a backend and records every request with its hits."""

    def __init__(self, inner: SearxngBackend) -> None:
        self._inner = inner
        self.name = inner.name
        self.requests: list[_Request] = []

    def search(
        self, query: str, language: str, max_results: int, sites: list[str] | None = None
    ) -> SearchResponse:
        record = _Request(query, sites)
        self.requests.append(record)
        try:
            response = self._inner.search(query, language, max_results, sites=sites)
        except Exception as exc:
            record.error = str(exc)
            raise
        record.hits = [h.url for h in response.hits]
        record.engine_errors = len(response.engine_errors)
        return response

    def health_check(self) -> bool:
        return self._inner.health_check()


def _reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=5).status_code < 500
    except httpx.HTTPError:
        return False


def _on_domain(url: str, sites: list[str]) -> bool:
    host = (httpx.URL(url).host or "").lower()
    return any(host == d or host.endswith("." + d) for d in sites)


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = get_settings()
    if not _reachable(f"{s.search.base_url.rstrip('/')}/"):
        pytest.skip(f"SearXNG not reachable at {s.search.base_url}")
    if not _reachable(f"{s.ollama.base_url.rstrip('/')}/api/tags"):
        pytest.skip(f"Ollama not reachable at {s.ollama.base_url}")
    return s


@pytest.fixture(scope="module")
def reranker(settings: Settings) -> Reranker:
    return Reranker(settings)


@pytest.mark.parametrize(("language", "claim"), CLAIMS, ids=[c[0] for c in CLAIMS])
def test_live_retrieval(
    settings: Settings, reranker: Reranker, tmp_path: Path, language: str, claim: str
) -> None:
    policy = SourcePolicy.from_settings(settings)
    cache = WebCache(tmp_path, mode="read_write")
    backend = RecordingBackend(SearxngBackend.from_settings(settings))
    fetcher = PageFetcher.from_settings(settings, policy, cache)
    ollama = OllamaClient(settings)
    retriever = WebEvidenceRetriever(
        settings,
        backend,  # type: ignore[arg-type]
        fetcher,
        policy,
        QueryBuilder(ollama.generate, enabled=settings.retrieval.query_generation),
        reranker,
        cache=cache,
    )

    result = retriever.retrieve(claim, language)

    print(f"\n=== [{language}] {claim}")
    print(f"status={result.status} queries={result.queries}")
    for i, req in enumerate(backend.requests):
        sites = req.sites or []
        on = sum(_on_domain(u, sites) for u in req.hits) if sites else len(req.hits)
        ratio = f"{on}/{len(req.hits)}" if req.hits else "0/0"
        group = f"{len(sites)} sites ({sites[0]}…)" if sites else "open"
        extra = f" engine_errors={req.engine_errors}" if req.engine_errors else ""
        extra += f" ERROR={req.error}" if req.error else ""
        print(f"  req {i}: q={req.query[:40]!r} group={group} hits={len(req.hits)} on-domain={ratio}{extra}")
    for ev in result.evidences:
        print(
            f"  [{ev.score:.2f}] {ev.publisher} ({ev.source_tier}, trust {ev.trust}) "
            f"{'snippet ' if ev.is_snippet else ''}{ev.source_url}"
        )

    assert result.status in ("ok", "unavailable")
    if result.status == "ok":
        assert len(result.evidences) >= 1
        assert all(policy.allows(ev.source_url) for ev in result.evidences)
        assert all(ev.source_url and ev.publisher for ev in result.evidences)


# ── Ollama hosted web search / web fetch ────────────────────────────────────

_OLLAMA_SITES = ["reuters.com", "bbc.com"]


@pytest.fixture(scope="module")
def ollama_key() -> str:
    key = get_settings().search.ollama_api_key
    value = key.get_secret_value().strip() if key else ""
    if not value:
        pytest.skip("SEARCH_OLLAMA_API_KEY (or OLLAMA_API_KEY) is not set")
    return value


@pytest.fixture(scope="module")
def ollama_hits(ollama_key: str) -> list[str]:
    web_url = get_settings().search.ollama_web_url
    backend = OllamaSearchBackend(ollama_key, web_url)
    response = backend.search(
        "Zaporizhzhia nuclear power plant IAEA", "en", 5, sites=_OLLAMA_SITES
    )
    print(f"\nOllama query={response.query!r}")
    for hit in response.hits:
        print(f"  {hit.rank}: {hit.url}")
    return [h.url for h in response.hits]


def test_ollama_site_restricted_search(ollama_hits: list[str]) -> None:
    assert ollama_hits, "Ollama web search returned no hits"
    assert all(_on_domain(u, _OLLAMA_SITES) for u in ollama_hits)


def test_ollama_web_fetch(ollama_key: str, ollama_hits: list[str]) -> None:
    if not ollama_hits:
        pytest.skip("no search hit to fetch")
    s = get_settings().search
    fetcher = PageFetcher(
        SourcePolicy({"tiers": {"t": {"trust": 1.0, "domains": _OLLAMA_SITES}}}),
        user_agent=s.user_agent,
        respect_robots=False,  # the hit's site may disallow generic agents
        mode="ollama",
        ollama_api_key=ollama_key,
        ollama_web_url=s.ollama_web_url,
    )
    page = fetcher.fetch(SearchHit(url=ollama_hits[0]))
    assert page is not None and not page.is_snippet
    assert page.text.strip()
    print(f"\nFetched {page.url}: {page.title!r}, {len(page.text)} chars")
