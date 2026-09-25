"""
Web search backends for evidence retrieval.

``SearxngBackend`` queries a self-hosted SearXNG instance through its JSON
API. Queries can be restricted to trusted domains with ``site:`` operators;
the backend reports engines that failed (rate limits, CAPTCHAs) so the
caller can tell an empty result from a blocked search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx

from exceptions import SearchBackendError
from logging_config import get_logger

if TYPE_CHECKING:
    from configs import Settings

__all__ = [
    "SearchHit",
    "SearchResponse",
    "SearchBackend",
    "SearxngBackend",
    "build_query",
]

logger = get_logger(__name__)


@dataclass
class SearchHit:
    """One search result.

    Attributes:
        url: Result URL.
        title: Result title.
        snippet: Result snippet (may be empty).
        published_at: Publication date reported by the engine ("" if absent).
        engine: Upstream engine(s) that returned the result.
        rank: Position within its response (0-based).
    """

    url: str
    title: str = ""
    snippet: str = ""
    published_at: str = ""
    engine: str = ""
    rank: int = 0


@dataclass
class SearchResponse:
    """Hits for one search request plus the engines that reported errors.

    Attributes:
        query: The full query string sent (including ``site:`` operators).
        hits: Results in rank order.
        engine_errors: ``[engine, reason]`` pairs for unresponsive engines.
    """

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    engine_errors: list[list[str]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """True when no hits came back because upstream engines failed."""
        return not self.hits and bool(self.engine_errors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResponse":
        return cls(
            query=str(data.get("query", "")),
            hits=[SearchHit(**h) for h in data.get("hits", [])],
            engine_errors=[list(e) for e in data.get("engine_errors", [])],
        )


class SearchBackend(Protocol):
    """Interface of a web search backend."""

    name: str

    def search(
        self,
        query: str,
        language: str,
        max_results: int,
        sites: list[str] | None = None,
    ) -> SearchResponse: ...

    def health_check(self) -> bool: ...


def build_query(query: str, sites: list[str] | None) -> str:
    """Append ``site:`` operators to *query*.

    No sites → *query*; one site → ``query site:a``; several →
    ``query (site:a OR site:b …)``.
    """
    if not sites:
        return query
    if len(sites) == 1:
        return f"{query} site:{sites[0]}"
    return f"{query} ({' OR '.join('site:' + s for s in sites)})"


def _normalize_date(value: Any) -> str:
    """Keep the ``YYYY-MM-DD`` part of an engine-reported date."""
    if not value:
        return ""
    text = str(value).strip()
    return text[:10] if len(text) >= 10 and text[4] == "-" and text[7] == "-" else ""


class SearxngBackend:
    """SearXNG JSON API client.

    Args:
        base_url: Instance URL, e.g. ``http://localhost:8080``.
        timeout: Request timeout in seconds.
        user_agent: User-Agent header sent to the instance.
        client: Optional preconfigured ``httpx.Client`` (used by tests).
    """

    name = "searxng"

    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
        user_agent: str = "DisinfoDetection-Research/1.0",
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=timeout, headers={"User-Agent": user_agent}
        )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "SearxngBackend":
        cfg = settings.search
        return cls(cfg.base_url, cfg.timeout_seconds, cfg.user_agent)

    @property
    def base_url(self) -> str:
        return self._base_url

    def search(
        self,
        query: str,
        language: str,
        max_results: int,
        sites: list[str] | None = None,
    ) -> SearchResponse:
        """Run one search request.

        Args:
            query: Search query text.
            language: Language code for the engines ("" means all languages).
            max_results: Maximum number of hits returned.
            sites: Domains for ``site:`` operators; None for an open search.

        Returns:
            SearchResponse with hits and engine errors.

        Raises:
            SearchBackendError: On connection errors, timeouts or non-2xx
                responses.
        """
        q = build_query(query, sites)
        params: dict[str, str | int] = {
            "q": q,
            "format": "json",
            "language": language or "all",
            "safesearch": 0,
        }
        try:
            response = self._client.get(f"{self._base_url}/search", params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise SearchBackendError(
                f"SearXNG at '{self._base_url}' returned HTTP "
                f"{exc.response.status_code}.",
                original_error=exc,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchBackendError(
                f"SearXNG request to '{self._base_url}' failed.", original_error=exc
            ) from exc

        hits: list[SearchHit] = []
        for item in data.get("results", []) or []:
            url = item.get("url") or ""
            if not url:
                continue
            engines = item.get("engines") or [item.get("engine", "")]
            hits.append(
                SearchHit(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or ""),
                    published_at=_normalize_date(item.get("publishedDate")),
                    engine=",".join(str(e) for e in engines if e),
                    rank=len(hits),
                )
            )
            if len(hits) >= max_results:
                break
        errors = [
            [str(x) for x in e] if isinstance(e, (list, tuple)) else [str(e), ""]
            for e in data.get("unresponsive_engines", []) or []
        ]
        if not hits and errors:
            logger.warning("SearXNG returned no results; engine errors: %s", errors)
        return SearchResponse(query=q, hits=hits, engine_errors=errors)

    def health_check(self) -> bool:
        """Check that SearXNG is reachable and serves the JSON format.

        Returns:
            True when the instance answers a JSON search request.

        Raises:
            SearchBackendError: With an "enable the json format" hint on
                HTTP 403, or naming the URL when the service is unreachable.
        """
        url = f"{self._base_url}/search"
        try:
            response = self._client.get(url, params={"q": "test", "format": "json"})
        except httpx.HTTPError as exc:
            raise SearchBackendError(
                f"SearXNG is not reachable at '{self._base_url}'. "
                "Ensure the service is running.",
                original_error=exc,
            ) from exc
        if response.status_code == 403:
            raise SearchBackendError(
                f"SearXNG at '{self._base_url}' refused the JSON format (HTTP 403). "
                "Enable the json format: add 'json' to search.formats in "
                "SearXNG settings.yml."
            )
        if response.status_code >= 400:
            raise SearchBackendError(
                f"SearXNG at '{self._base_url}' returned HTTP "
                f"{response.status_code} to a health-check search."
            )
        logger.debug("SearXNG health check passed at %s.", self._base_url)
        return True
