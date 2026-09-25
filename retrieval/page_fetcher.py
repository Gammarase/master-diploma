"""
Page fetching and main-text extraction for web evidence.

Fetches a search hit's page with an identifying user agent, honours
robots.txt, caps the response size, re-checks the source policy on the
final URL after redirects, and extracts the main article text with
``trafilatura``. When a page cannot be used, the search snippet becomes a
single fallback passage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from logging_config import get_logger
from retrieval.cache import utc_now_iso

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
    ) -> None:
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
        return cls(
            policy,
            user_agent=s.user_agent,
            timeout=s.fetch_timeout_seconds,
            max_bytes=s.max_page_bytes,
            respect_robots=s.respect_robots,
            cache=cache,
        )

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
        if self._cache is not None:
            cached = self._cache.get_page(url)
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
        status, final_url, content = self._download(url)
        if self._cache is not None and status != _TRANSIENT:
            self._cache.put_page(url, final_url, status, content, retrieved_at)
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

