"""Unit tests for retrieval.web_search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from exceptions import SearchBackendError
from retrieval.web_search import (
    SearchHit,
    SearchResponse,
    SearxngBackend,
    build_query,
)

_OK_JSON: dict[str, Any] = {
    "query": "zaporizhzhia iaea",
    "results": [
        {
            "url": "https://www.iaea.org/newscenter/zaporizhzhia",
            "title": "IAEA mission to Zaporizhzhia",
            "content": "The IAEA team arrived at the plant.",
            "publishedDate": "2022-09-01T10:00:00",
            "engine": "brave",
            "engines": ["brave", "duckduckgo"],
        },
        {
            "url": "https://news.un.org/en/story/2022/09/1125",
            "title": "UN News",
            "content": "",
            "publishedDate": None,
            "engine": "google",
        },
        {"url": "", "title": "no url"},
    ],
    "unresponsive_engines": [],
}

_BLOCKED_JSON: dict[str, Any] = {
    "query": "x",
    "results": [],
    "unresponsive_engines": [
        ["brave", "too many requests"],
        ["duckduckgo", "CAPTCHA"],
        ["google cse", "too many requests"],
    ],
}


def _backend(handler: Callable[[httpx.Request], httpx.Response]) -> SearxngBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SearxngBackend("http://searx.test:8080/", client=client)


def _json_handler(
    payload: dict[str, Any], seen: list[httpx.Request] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return handler


class TestBuildQuery:
    def test_no_sites(self) -> None:
        assert build_query("iaea zaporizhzhia", None) == "iaea zaporizhzhia"
        assert build_query("iaea zaporizhzhia", []) == "iaea zaporizhzhia"

    def test_one_site(self) -> None:
        assert build_query("q", ["apnews.com"]) == "q site:apnews.com"

    def test_three_sites(self) -> None:
        assert (
            build_query("q", ["a.com", "b.org", "c.net"])
            == "q (site:a.com OR site:b.org OR site:c.net)"
        )


class TestSearch:
    @pytest.mark.parametrize(
        ("sites", "expected_q"),
        [
            (None, "iaea plant"),
            (["iaea.org"], "iaea plant site:iaea.org"),
            (
                ["iaea.org", "un.org", "reuters.com"],
                "iaea plant (site:iaea.org OR site:un.org OR site:reuters.com)",
            ),
        ],
    )
    def test_request_params(
        self, sites: list[str] | None, expected_q: str
    ) -> None:
        seen: list[httpx.Request] = []
        resp = _backend(_json_handler(_OK_JSON, seen)).search(
            "iaea plant", "en", 10, sites=sites
        )
        req = seen[0]
        assert req.url.path == "/search"
        assert req.url.params["q"] == expected_q
        assert req.url.params["format"] == "json"
        assert req.url.params["language"] == "en"
        assert req.url.params["safesearch"] == "0"
        assert resp.query == expected_q

    def test_empty_language_means_all(self) -> None:
        seen: list[httpx.Request] = []
        _backend(_json_handler(_OK_JSON, seen)).search("q", "", 10)
        assert seen[0].url.params["language"] == "all"

    def test_normal_response(self) -> None:
        resp = _backend(_json_handler(_OK_JSON)).search("q", "en", 10)
        assert [h.url for h in resp.hits] == [
            "https://www.iaea.org/newscenter/zaporizhzhia",
            "https://news.un.org/en/story/2022/09/1125",
        ]
        first = resp.hits[0]
        assert first.title == "IAEA mission to Zaporizhzhia"
        assert first.snippet == "The IAEA team arrived at the plant."
        assert first.published_at == "2022-09-01"
        assert first.engine == "brave,duckduckgo"
        assert [h.rank for h in resp.hits] == [0, 1]
        assert resp.hits[1].published_at == ""
        assert resp.engine_errors == []
        assert resp.failed is False

    def test_max_results(self) -> None:
        resp = _backend(_json_handler(_OK_JSON)).search("q", "en", 1)
        assert len(resp.hits) == 1

    def test_unresponsive_engines_no_results(self) -> None:
        resp = _backend(_json_handler(_BLOCKED_JSON)).search("q", "en", 10)
        assert resp.hits == []
        assert resp.engine_errors[0] == ["brave", "too many requests"]
        assert len(resp.engine_errors) == 3
        assert resp.failed is True

    def test_empty_but_healthy_is_not_failed(self) -> None:
        payload = {"results": [], "unresponsive_engines": []}
        resp = _backend(_json_handler(payload)).search("q", "en", 10)
        assert resp.failed is False

    def test_http_error_raises(self) -> None:
        backend = _backend(lambda r: httpx.Response(500))
        with pytest.raises(SearchBackendError, match="HTTP 500"):
            backend.search("q", "en", 10)

    def test_connection_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(SearchBackendError, match="searx.test"):
            _backend(handler).search("q", "en", 10)

    def test_invalid_json_raises(self) -> None:
        backend = _backend(lambda r: httpx.Response(200, text="<html>"))
        with pytest.raises(SearchBackendError):
            backend.search("q", "en", 10)


class TestSerialization:
    def test_round_trip(self) -> None:
        resp = SearchResponse(
            query="q",
            hits=[SearchHit(url="https://a.com", title="t", rank=0)],
            engine_errors=[["brave", "x"]],
        )
        assert SearchResponse.from_dict(resp.to_dict()) == resp


class TestOffTarget:
    _ERR = [["brave", "Suspended: too many requests"]]

    def _resp(self, *urls: str, errors: list[list[str]] | None = None) -> SearchResponse:
        return SearchResponse(
            query="q",
            hits=[SearchHit(url=u) for u in urls],
            engine_errors=self._ERR if errors is None else errors,
        )

    def test_junk_hits_with_errors(self) -> None:
        r = self._resp("https://dictionary.cambridge.org/huge", "https://www.huge.com/")
        assert r.off_target(["reuters.com", "bbc.com"]) is True

    def test_hit_on_requested_site(self) -> None:
        r = self._resp("https://dictionary.cambridge.org/huge", "https://www.reuters.com/world/x")
        assert r.off_target(["reuters.com", "bbc.com"]) is False

    def test_subdomain_counts(self) -> None:
        assert self._resp("https://news.bbc.co.uk/a").off_target(["bbc.co.uk"]) is False

    def test_path_site_matches_domain(self) -> None:
        r = self._resp("https://www.reuters.com/fact-check/x")
        assert r.off_target(["reuters.com/fact-check"]) is False

    def test_suffix_is_not_a_subdomain(self) -> None:
        assert self._resp("https://notreuters.com/a").off_target(["reuters.com"]) is True

    def test_no_engine_errors_is_not_off_target(self) -> None:
        r = self._resp("https://www.huge.com/", errors=[])
        assert r.off_target(["reuters.com"]) is False

    def test_open_search_is_never_off_target(self) -> None:
        assert self._resp("https://www.huge.com/").off_target(None) is False

    def test_no_hits_with_errors(self) -> None:
        # Also ``failed``; off_target agrees.
        assert self._resp().off_target(["reuters.com"]) is True


class TestHealthCheck:
    def test_ok(self) -> None:
        seen: list[httpx.Request] = []
        assert _backend(_json_handler(_OK_JSON, seen)).health_check() is True
        assert seen[0].url.params["format"] == "json"

    def test_403_hints_json_format(self) -> None:
        backend = _backend(lambda r: httpx.Response(403))
        with pytest.raises(SearchBackendError, match="(?i)enable the json format"):
            backend.health_check()

    def test_other_error_status(self) -> None:
        backend = _backend(lambda r: httpx.Response(502))
        with pytest.raises(SearchBackendError, match="502"):
            backend.health_check()

    def test_unreachable_names_url(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(SearchBackendError, match="http://searx.test:8080"):
            _backend(handler).health_check()


def test_from_settings() -> None:
    settings = MagicMock()
    settings.search.base_url = "http://localhost:8080"
    settings.search.timeout_seconds = 5
    settings.search.user_agent = "UA/1.0"
    backend = SearxngBackend.from_settings(settings)
    assert backend.base_url == "http://localhost:8080"
    assert backend.name == "searxng"
