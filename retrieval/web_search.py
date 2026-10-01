"""
Web search backends for evidence retrieval.

``SearxngBackend`` queries a self-hosted SearXNG instance through its JSON
API. Queries can be restricted to trusted domains with ``site:`` operators;
the backend reports engines that failed (rate limits, CAPTCHAs) so the
caller can tell an empty result from a blocked search.

``OllamaSearchBackend`` sends the same query strings to Ollama's hosted web
search API (needs an API key); ``create_search_backend`` picks one of the
two from ``search.backend``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

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
    "OllamaSearchBackend",
    "OLLAMA_KEY_ENV",
    "build_query",
    "create_search_backend",
    "ollama_auth_headers",
    "ollama_health_check",
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

    def off_target(self, sites: list[str] | None) -> bool:
        """True when a ``site:``-filtered request came back degraded.

        When the engines that honour ``site:`` are suspended, SearXNG may
        still answer from engines that ignore the operators. Such a response
        has engine errors and no hit on any requested site; it must be
        treated as failed, not cached as a real result.

        Args:
            sites: Domains the request was restricted to; None for an open
                search (never off-target).
        """
        if not sites or not self.engine_errors:
            return False
        domains = [s.split("/", 1)[0].lower() for s in sites if s]
        for hit in self.hits:
            host = (urlsplit(hit.url).hostname or "").lower()
            if any(host == d or host.endswith("." + d) for d in domains):
                return False
        return True

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


OLLAMA_KEY_ENV = "SEARCH_OLLAMA_API_KEY"
_OLLAMA_MAX_RESULTS = 10  # web_search API maximum


def ollama_auth_headers(api_key: str | None) -> dict[str, str]:
    """``Authorization`` header for Ollama's hosted APIs ({} without a key)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def ollama_health_check(
    client: httpx.Client, web_url: str, api_key: str | None
) -> bool:
    """Check that a key is set and the Ollama web service is reachable.

    Sends only ``GET {web_url}``, never a search or fetch, so it uses no
    quota. The key itself is not validated.

    Raises:
        SearchBackendError: When no key is set, or naming *web_url* when the
            service is unreachable or answers with a 5xx status.
    """
    if not api_key:
        raise SearchBackendError(
            f"No Ollama API key is set; set {OLLAMA_KEY_ENV} "
            "(get one at https://ollama.com/settings/keys)."
        )
    try:
        response = client.get(web_url)
    except httpx.HTTPError as exc:
        raise SearchBackendError(
            f"Ollama web service is not reachable at '{web_url}'.",
            original_error=exc,
        ) from exc
    if response.status_code >= 500:
        raise SearchBackendError(
            f"Ollama web service at '{web_url}' returned HTTP "
            f"{response.status_code} to a health check."
        )
    logger.debug("Ollama web health check passed at %s.", web_url)
    return True


class OllamaSearchBackend:
    """Ollama hosted web search API client (``POST /api/web_search``).

    The query string is sent as is, so ``site:`` and ``OR`` operators from
    ``build_query`` work. The API has no language parameter.

    Args:
        api_key: Ollama account API key (sent only as a bearer token).
        web_url: Service URL, default ``https://ollama.com``.
        timeout: Request timeout in seconds.
        user_agent: User-Agent header.
        client: Optional preconfigured ``httpx.Client`` (used by tests).
    """

    name = "ollama"

    def __init__(
        self,
        api_key: str | None,
        web_url: str = "https://ollama.com",
        timeout: float = 20.0,
        user_agent: str = "DisinfoDetection-Research/1.0",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or None
        self._web_url = web_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=timeout, headers={"User-Agent": user_agent}
        )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "OllamaSearchBackend":
        cfg = settings.search
        key = cfg.ollama_api_key.get_secret_value() if cfg.ollama_api_key else None
        return cls(key, cfg.ollama_web_url, cfg.timeout_seconds, cfg.user_agent)

    @property
    def base_url(self) -> str:
        return self._web_url

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
            language: Ignored; the API has no language parameter.
            max_results: Maximum number of hits (capped at 10).
            sites: Domains for ``site:`` operators; None for an open search.

        Returns:
            SearchResponse with hits and no engine errors.

        Raises:
            SearchBackendError: On connection errors, timeouts, non-2xx or
                malformed responses; ``fatal`` on HTTP 401/403 or a missing key.
        """
        if not self._api_key:
            raise SearchBackendError(
                f"No Ollama API key is set; set {OLLAMA_KEY_ENV}.", fatal=True
            )
        q = build_query(query, sites)
        limit = max(1, min(max_results, _OLLAMA_MAX_RESULTS))
        url = f"{self._web_url}/api/web_search"
        try:
            response = self._client.post(
                url,
                json={"query": q, "max_results": limit},
                headers=ollama_auth_headers(self._api_key),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise SearchBackendError(
                    f"Ollama rejected the API key (HTTP {status}); "
                    f"set {OLLAMA_KEY_ENV}.",
                    fatal=True,
                ) from exc
            raise SearchBackendError(
                f"Ollama web search at '{self._web_url}' returned HTTP {status}.",
                original_error=exc,
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchBackendError(
                f"Ollama web search request to '{self._web_url}' failed.",
                original_error=exc,
            ) from exc

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise SearchBackendError(
                f"Ollama web search at '{self._web_url}' returned a malformed "
                "response (no 'results' list)."
            )
        hits: list[SearchHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or ""),
                    engine=self.name,
                    rank=len(hits),
                )
            )
            if len(hits) >= limit:
                break
        return SearchResponse(query=q, hits=hits)

    def health_check(self) -> bool:
        """Check the key is set and the service is reachable (no quota used).

        Raises:
            SearchBackendError: See ``ollama_health_check``.
        """
        return ollama_health_check(self._client, self._web_url, self._api_key)


def create_search_backend(settings: "Settings") -> SearchBackend:
    """Build the search backend named by ``search.backend``."""
    if settings.search.backend == "ollama":
        return OllamaSearchBackend.from_settings(settings)
    return SearxngBackend.from_settings(settings)
