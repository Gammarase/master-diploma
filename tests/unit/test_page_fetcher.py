"""Unit tests for retrieval.page_fetcher.PageFetcher."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from retrieval.cache import WebCache
from retrieval.page_fetcher import PageFetcher
from retrieval.source_policy import SourcePolicy
from retrieval.web_search import SearchHit

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "article.html"
_HTML = _FIXTURE.read_bytes()

_POLICY_DATA = {
    "tiers": {
        "wire": {"trust": 0.95, "domains": ["apnews.com", "reuters.com"]},
    },
    "blocklist": ["rt.com"],
}

Handler = Callable[[httpx.Request], httpx.Response]


def _html_response(body: bytes = _HTML, **headers: str) -> httpx.Response:
    return httpx.Response(
        200, content=body, headers={"content-type": "text/html; charset=utf-8", **headers}
    )


def _router(pages: dict[str, Handler], robots: str | int = 404) -> tuple[Handler, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if request.url.path == "/robots.txt":
            if isinstance(robots, int):
                return httpx.Response(robots)
            return httpx.Response(200, text=robots)
        for prefix, h in pages.items():
            if url.startswith(prefix):
                return h(request)
        return httpx.Response(404)

    return handler, calls


def _fetcher(
    handler: Handler,
    mode: str = "strict",
    cache: WebCache | None = None,
    max_bytes: int = 2_000_000,
) -> PageFetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    policy = SourcePolicy(_POLICY_DATA, mode=mode)  # type: ignore[arg-type]
    return PageFetcher(policy, cache=cache, client=client, max_bytes=max_bytes)


def _hit(url: str = "https://apnews.com/article/xyz", snippet: str = "IAEA arrives.") -> SearchHit:
    return SearchHit(
        url=url, title="IAEA at plant", snippet=snippet, published_at="2022-09-02"
    )


class TestExtraction:
    def test_article_body_extracted(self) -> None:
        handler, _ = _router({"https://apnews.com/": lambda r: _html_response()})
        page = _fetcher(handler).fetch(_hit())
        assert page is not None
        assert page.is_snippet is False
        assert "Rafael Grossi" in page.text
        assert "largest in Europe" in page.text
        assert "NAVMENU" not in page.text
        assert "SCRIPT_MARKER" not in page.text
        assert page.title == "IAEA experts reach Zaporizhzhia nuclear plant"
        assert page.published_at == "2022-09-01"
        assert page.sitename == "Example News"
        assert page.language == "en"
        assert page.final_url == "https://apnews.com/article/xyz"
        assert page.retrieved_at.endswith("Z")

    def test_hit_date_used_when_page_has_none(self) -> None:
        html = _HTML.replace(
            b'<meta property="article:published_time" content="2022-09-01T12:30:00Z">', b""
        )
        handler, _ = _router({"https://apnews.com/": lambda r: _html_response(html)})
        page = _fetcher(handler).fetch(_hit())
        assert page is not None and not page.is_snippet
        assert page.published_at == "2022-09-02"


class TestFallbacksAndSkips:
    def test_403_falls_back_to_snippet(self) -> None:
        handler, _ = _router({"https://apnews.com/": lambda r: httpx.Response(403)})
        page = _fetcher(handler).fetch(_hit())
        assert page is not None
        assert page.is_snippet is True
        assert page.text == "IAEA at plant. IAEA arrives."
        assert page.published_at == "2022-09-02"

    def test_403_without_snippet_skipped(self) -> None:
        handler, _ = _router({"https://apnews.com/": lambda r: httpx.Response(403)})
        assert _fetcher(handler).fetch(_hit(snippet="")) is None

    def test_oversize_page_skipped(self) -> None:
        handler, _ = _router({"https://apnews.com/": lambda r: _html_response()})
        page = _fetcher(handler, max_bytes=500).fetch(_hit(snippet=""))
        assert page is None

    def test_oversize_by_content_length_skipped(self) -> None:
        handler, _ = _router(
            {"https://apnews.com/": lambda r: _html_response(b"<html></html>", **{"content-length": "99999999"})}
        )
        page = _fetcher(handler, max_bytes=1000).fetch(_hit())
        assert page is not None and page.is_snippet

    def test_non_html_skipped(self) -> None:
        handler, _ = _router(
            {"https://apnews.com/": lambda r: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})}
        )
        assert _fetcher(handler).fetch(_hit(snippet="")) is None

    def test_robots_disallowed_skipped(self) -> None:
        handler, calls = _router(
            {"https://apnews.com/": lambda r: _html_response()},
            robots="User-agent: *\nDisallow: /article/\n",
        )
        assert _fetcher(handler).fetch(_hit(snippet="")) is None
        assert "https://apnews.com/article/xyz" not in calls

    def test_robots_5xx_disallows(self) -> None:
        handler, calls = _router({"https://apnews.com/": lambda r: _html_response()}, robots=503)
        assert _fetcher(handler).fetch(_hit(snippet="")) is None
        assert "https://apnews.com/article/xyz" not in calls

    def test_robots_cached_per_host(self) -> None:
        handler, calls = _router({"https://apnews.com/": lambda r: _html_response()})
        fetcher = _fetcher(handler)
        fetcher.fetch(_hit("https://apnews.com/article/1"))
        fetcher.fetch(_hit("https://apnews.com/article/2"))
        assert calls.count("https://apnews.com/robots.txt") == 1

    def test_no_text_falls_back(self) -> None:
        handler, _ = _router({"https://apnews.com/": lambda r: _html_response(b"<html><body></body></html>")})
        page = _fetcher(handler).fetch(_hit())
        assert page is not None and page.is_snippet

    def test_connection_error_falls_back(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        handler, _ = _router({"https://apnews.com/": boom})
        page = _fetcher(handler).fetch(_hit())
        assert page is not None and page.is_snippet


class TestRedirects:
    def test_redirect_to_blocked_domain_discarded(self) -> None:
        handler, _ = _router(
            {
                "https://apnews.com/": lambda r: httpx.Response(
                    302, headers={"location": "https://www.rt.com/landing"}
                ),
                "https://www.rt.com/": lambda r: _html_response(),
            }
        )
        assert _fetcher(handler).fetch(_hit()) is None

    def test_redirect_to_unknown_domain_discarded_in_strict(self) -> None:
        handler, _ = _router(
            {
                "https://apnews.com/": lambda r: httpx.Response(
                    302, headers={"location": "https://tracker.example/landing"}
                ),
                "https://tracker.example/": lambda r: _html_response(),
            }
        )
        assert _fetcher(handler).fetch(_hit()) is None

    def test_redirect_within_allowed_domain(self) -> None:
        handler, _ = _router(
            {
                "https://apnews.com/": lambda r: httpx.Response(
                    301, headers={"location": "https://www.reuters.com/world/x"}
                ),
                "https://www.reuters.com/": lambda r: _html_response(),
            }
        )
        page = _fetcher(handler).fetch(_hit())
        assert page is not None
        assert page.final_url == "https://www.reuters.com/world/x"


class TestCache:
    def test_page_cached_and_replayed_offline(self, tmp_path: Path) -> None:
        handler, calls = _router({"https://apnews.com/": lambda r: _html_response()})
        first = _fetcher(handler, cache=WebCache(tmp_path)).fetch(_hit())

        def no_network(request: httpx.Request) -> httpx.Response:
            raise AssertionError("network used in read_only mode")

        ro = WebCache(tmp_path, mode="read_only")
        second = _fetcher(no_network, cache=ro).fetch(_hit())
        assert first == second

    def test_negative_result_cached(self, tmp_path: Path) -> None:
        handler, calls = _router({"https://apnews.com/": lambda r: httpx.Response(403)})
        cache = WebCache(tmp_path)
        _fetcher(handler, cache=cache).fetch(_hit())
        entry = cache.get_page("https://apnews.com/article/xyz")
        assert entry is not None and entry["status"] == "403"

    def test_transient_error_not_cached(self, tmp_path: Path) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        handler, _ = _router({"https://apnews.com/": boom})
        cache = WebCache(tmp_path)
        _fetcher(handler, cache=cache).fetch(_hit())
        assert cache.get_page("https://apnews.com/article/xyz") is None

    def test_offline_miss_uses_snippet_without_network(self, tmp_path: Path) -> None:
        def no_network(request: httpx.Request) -> httpx.Response:
            raise AssertionError("network used in read_only mode")

        ro = WebCache(tmp_path, mode="read_only")
        page = _fetcher(no_network, cache=ro).fetch(_hit())
        assert page is not None and page.is_snippet


def test_from_settings() -> None:
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.search.user_agent = "UA/1"
    settings.search.fetch_timeout_seconds = 5
    settings.search.max_page_bytes = 1000
    settings.search.respect_robots = False
    fetcher = PageFetcher.from_settings(settings, SourcePolicy(_POLICY_DATA))
    assert fetcher._max_bytes == 1000

