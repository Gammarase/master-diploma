"""
Page fetching and main-text extraction for web evidence.

Fetches a search hit's page with an identifying user agent, honours
robots.txt, caps the response size, re-checks the source policy on the
final URL after redirects, and extracts the main article text with
``trafilatura``. When a page cannot be used, the search snippet becomes a
single fallback passage.

In ``ollama`` mode the page text comes from Ollama's hosted web fetch API
instead of a local download; robots.txt is still checked first, and the
cache entries are kept apart from the direct fetcher's.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from logging_config import get_logger
from retrieval.cache import utc_now_iso
from retrieval.web_search import OLLAMA_KEY_ENV, ollama_auth_headers

if TYPE_CHECKING:
    from configs import Settings
    from retrieval.cache import WebCache
    from retrieval.source_policy import SourcePolicy
    from retrieval.web_search import SearchHit

__all__ = ["PageFetcher", "FetchedPage"]

logger = get_logger(__name__)

_HTML_TYPES = ("text/html", "application/xhtml+xml")
STATUS_OK = "ok"
# Transient failures are not cached, so a later run can retry them.
_TRANSIENT = "error"
# Cache-key prefix of pages fetched through Ollama's web fetch API.
_OLLAMA_CACHE_PREFIX = "ollama:"

FetcherMode = Literal["direct", "ollama"]


@dataclass
class FetchedPage:
    """Extracted content of one page (or its snippet fallback).

    Attributes:
        url: Requested URL.
        final_url: URL after redirects.
        text: Main text (or ``title. snippet`` for the fallback).
        title: Page title.
        sitename: Site name reported by the page ("" if absent).
        published_at: Publication date ``YYYY-MM-DD`` ("" if unknown).
        language: Detected language code.
        retrieved_at: ISO-8601 UTC fetch time.
        is_snippet: True for the snippet fallback.
    """

    url: str
    final_url: str
    text: str
    title: str = ""
    sitename: str = ""
    published_at: str = ""
    language: str = ""
    retrieved_at: str = ""
    is_snippet: bool = False


def _detect_language(text: str) -> str:
    """Deterministic language detection; "" when detection fails."""
    try:
        from langdetect import DetectorFactory, detect  # type: ignore[import-untyped]

        DetectorFactory.seed = 0
        return str(detect(text[:2000]))
    except Exception:
        return ""


def _normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return ""


def _extract(html: str, url: str) -> dict[str, str] | None:
    """Run trafilatura; return text/title/sitename/date or None if no text."""
    import trafilatura

    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if doc is None:
        return None
    data = doc.as_dict() if hasattr(doc, "as_dict") else dict(doc)
    text = (data.get("text") or "").strip()
    if not text:
        return None
    return {
        "text": text,
        "title": str(data.get("title") or ""),
        "sitename": str(data.get("sitename") or ""),
        "date": _normalize_date(data.get("date")),
    }


class PageFetcher:
    """Fetch and extract pages for search hits.

    Args:
        policy: Source policy, re-checked on the final URL.
        user_agent: User-Agent header (identifies the project honestly).
        timeout: Per-request timeout in seconds.
        max_bytes: Maximum response size; larger pages are skipped.
        respect_robots: Whether to honour robots.txt.
        cache: Optional page cache; in read-only mode no requests are sent.
        client: Optional preconfigured ``httpx.Client`` (used by tests).
        mode: ``direct`` downloads pages; ``ollama`` asks Ollama's web fetch
            API for their text.
        ollama_api_key: API key for ``ollama`` mode.
        ollama_web_url: Ollama service URL for ``ollama`` mode.
    """

    def __init__(
        self,
        policy: "SourcePolicy",
        user_agent: str = "DisinfoDetection-Research/1.0",
        timeout: float = 10.0,
        max_bytes: int = 2_000_000,
        respect_robots: bool = True,
        cache: "WebCache | None" = None,
        client: httpx.Client | None = None,
        mode: FetcherMode = "direct",
        ollama_api_key: str | None = None,
        ollama_web_url: str = "https://ollama.com",
    ) -> None:
        if mode not in ("direct", "ollama"):
            raise ValueError(f"Unknown page fetcher mode: {mode!r}")
        self._mode: FetcherMode = mode
        self._ollama_key = ollama_api_key or None
        self._ollama_url = ollama_web_url.rstrip("/")
        self._auth_warned = False
        self._policy = policy
        self._user_agent = user_agent
        self._max_bytes = max_bytes
        self._respect_robots = respect_robots
        self._cache = cache
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )
        self._robots: dict[str, RobotFileParser | bool] = {}
        self._robots_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        policy: "SourcePolicy",
        cache: "WebCache | None" = None,
    ) -> "PageFetcher":
        s = settings.search
        ollama = s.page_fetcher == "ollama"
        key = s.ollama_api_key.get_secret_value() if ollama and s.ollama_api_key else None
        return cls(
            policy,
            user_agent=s.user_agent,
            timeout=s.fetch_timeout_seconds,
            max_bytes=s.max_page_bytes,
            respect_robots=s.respect_robots,
            cache=cache,
            mode="ollama" if ollama else "direct",
            ollama_api_key=key,
            ollama_web_url=s.ollama_web_url if ollama else "https://ollama.com",
        )

    @property
    def mode(self) -> FetcherMode:
        return self._mode

    # ── public ──────────────────────────────────────────────────────────────

    def fetch(self, hit: "SearchHit", fallback_language: str = "") -> FetchedPage | None:
        """Fetch *hit*'s page and extract its main text.

        Args:
            hit: The search hit (its URL should already pass the policy).
            fallback_language: Language used when detection fails.

        Returns:
            The extracted page; the snippet fallback when the page is
            unusable and the hit has a snippet; None when the final URL is
            not allowed or nothing usable remains.
        """
        status, final_url, content, retrieved_at = self._load(hit.url)

        if not self._policy.allows(final_url):
            logger.info(
                "Discarding %s: final URL %s is not allowed by the source policy.",
                hit.url,
                final_url,
            )
            return None

        if status == STATUS_OK and content:
            text = content["text"]
            return FetchedPage(
                url=hit.url,
                final_url=final_url,
                text=text,
                title=content.get("title") or hit.title,
                sitename=content.get("sitename", ""),
                published_at=content.get("date") or hit.published_at,
                language=_detect_language(text) or fallback_language,
                retrieved_at=retrieved_at,
            )

        logger.debug("Page %s unusable (%s); trying snippet fallback.", hit.url, status)
        snippet = hit.snippet.strip()
        if not snippet:
            return None
        title = hit.title.strip()
        text = f"{title}. {snippet}" if title else snippet
        return FetchedPage(
            url=hit.url,
            final_url=hit.url,
            text=text,
            title=title,
            published_at=hit.published_at,
            language=_detect_language(text) or fallback_language,
            retrieved_at=retrieved_at,
            is_snippet=True,
        )

    # ── loading (cache → network) ───────────────────────────────────────────

    def _load(self, url: str) -> tuple[str, str, dict[str, str] | None, str]:
        """Return ``(status, final_url, content, retrieved_at)`` for *url*."""
        # Ollama-fetched pages live under their own key, so a failure of one
        # fetcher (e.g. a direct 403) never blocks the other.
        key = _OLLAMA_CACHE_PREFIX + url if self._mode == "ollama" else url
        if self._cache is not None:
            cached = self._cache.get_page(key)
            if cached is not None:
                return (
                    cached["status"],
                    cached["final_url"],
                    cached["content"],
                    cached["retrieved_at"],
                )
            if self._cache.offline:
                return "cache_miss", url, None, ""

        retrieved_at = utc_now_iso()
        if self._mode == "ollama":
            status, final_url, content = self._download_ollama(url)
        else:
            status, final_url, content = self._download(url)
        if self._cache is not None and status != _TRANSIENT:
            self._cache.put_page(key, final_url, status, content, retrieved_at)
        return status, final_url, content, retrieved_at

    def _download(self, url: str) -> tuple[str, str, dict[str, str] | None]:
        if self._respect_robots and not self._robots_allows(url):
            return "robots_disallowed", url, None
        try:
            with self._client.stream("GET", url) as response:
                final_url = str(response.url)
                if response.status_code >= 400:
                    return str(response.status_code), final_url, None
                ctype = response.headers.get("content-type", "").split(";")[0]
                if ctype.strip().lower() not in _HTML_TYPES:
                    return "not_html", final_url, None
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self._max_bytes:
                    return "too_large", final_url, None
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self._max_bytes:
                        return "too_large", final_url, None
                encoding = response.charset_encoding or "utf-8"
        except httpx.HTTPError as exc:
            logger.info("Fetching %s failed: %s", url, exc)
            return _TRANSIENT, url, None

        try:
            html = bytes(body).decode(encoding, errors="replace")
        except LookupError:
            html = bytes(body).decode("utf-8", errors="replace")
        try:
            content = _extract(html, final_url)
        except Exception as exc:  # trafilatura/lxml edge cases
            logger.info("Extraction failed for %s: %s", final_url, exc)
            content = None
        if content is None:
            return "no_text", final_url, None
        return STATUS_OK, final_url, content

    def _download_ollama(self, url: str) -> tuple[str, str, dict[str, str] | None]:
        """Get *url*'s text from Ollama's web fetch API.

        The API reports no redirects, so the final URL is the requested one.
        """
        if self._respect_robots and not self._robots_allows(url):
            return "robots_disallowed", url, None
        if not self._ollama_key:
            self._warn_auth(f"no Ollama API key is set; set {OLLAMA_KEY_ENV}")
            return _TRANSIENT, url, None
        try:
            response = self._client.post(
                f"{self._ollama_url}/api/web_fetch",
                json={"url": url},
                headers=ollama_auth_headers(self._ollama_key),
            )
        except httpx.HTTPError as exc:
            logger.info("Ollama web fetch of %s failed: %s", url, exc)
            return _TRANSIENT, url, None
        if response.status_code in (401, 403):
            self._warn_auth(
                f"Ollama rejected the API key (HTTP {response.status_code}); "
                f"set {OLLAMA_KEY_ENV}"
            )
            return _TRANSIENT, url, None
        if response.status_code >= 400:
            logger.info(
                "Ollama web fetch of %s returned HTTP %d.", url, response.status_code
            )
            return _TRANSIENT, url, None
        try:
            data = response.json()
        except ValueError:
            logger.info("Ollama web fetch of %s returned malformed JSON.", url)
            return _TRANSIENT, url, None
        if not isinstance(data, dict):
            logger.info("Ollama web fetch of %s returned a malformed response.", url)
            return _TRANSIENT, url, None
        text = str(data.get("content") or "").strip()[: self._max_bytes]
        if not text:
            return "no_text", url, None
        return (
            STATUS_OK,
            url,
            {"text": text, "title": str(data.get("title") or ""), "sitename": "", "date": ""},
        )

    def _warn_auth(self, message: str) -> None:
        """Log an API key problem once per fetcher (never the key itself)."""
        if not self._auth_warned:
            self._auth_warned = True
            logger.warning("Ollama web fetch failed: %s.", message)

    # ── robots.txt ──────────────────────────────────────────────────────────

    def _robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            rules = self._robots.get(origin)
        if rules is None:
            rules = self._load_robots(origin)
            with self._robots_lock:
                self._robots[origin] = rules
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(self._user_agent, url)

    def _load_robots(self, origin: str) -> RobotFileParser | bool:
        """Parsed robots.txt; True (allow all) on errors/4xx, False on 5xx."""
        try:
            response = self._client.get(f"{origin}/robots.txt")
        except httpx.HTTPError:
            return True
        if response.status_code >= 500:
            return False
        if response.status_code >= 400:
            return True
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser

