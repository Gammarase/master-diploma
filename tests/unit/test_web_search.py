"""Unit tests for retrieval.web_search."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import SecretStr

from exceptions import SearchBackendError
from retrieval.web_search import (
    OllamaSearchBackend,
    SearchHit,
    SearchResponse,
    SearxngBackend,
    build_query,
    create_search_backend,
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


# ── Ollama web search ───────────────────────────────────────────────────────

_KEY = "sk-test-ollama-0123456789"

_OLLAMA_JSON: dict[str, Any] = {
    "results": [
        {
            "title": "Drone attack on Kyiv",
            "url": "https://www.reuters.com/world/europe/kyiv-drones",
            "content": "Russia launched drones at Kyiv overnight.",
        },
        {"title": "no url", "url": "", "content": "skipped"},
        {
            "title": "BBC report",
            "url": "https://www.bbc.com/news/world-europe-1",
            "content": "Air defences were active.",
        },
        {
            "title": "Third",
            "url": "https://www.bbc.com/news/world-europe-2",
            "content": "More detail.",
        },
    ]
}


def _ollama(
    handler: Callable[[httpx.Request], httpx.Response], key: str | None = _KEY
) -> OllamaSearchBackend:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaSearchBackend(key, "https://ollama.test/", client=client)


class TestOllamaSearch:
    def test_request_carries_query_and_bearer(self) -> None:
        seen: list[httpx.Request] = []
        resp = _ollama(_json_handler(_OLLAMA_JSON, seen)).search(
            "Kyiv drone attack", "uk", 5, sites=["reuters.com", "bbc.com"]
        )
        req = seen[0]
        assert req.method == "POST"
        assert str(req.url) == "https://ollama.test/api/web_search"
        assert req.headers["Authorization"] == f"Bearer {_KEY}"
        body = json.loads(req.content)
        assert body == {
            "query": "Kyiv drone attack (site:reuters.com OR site:bbc.com)",
            "max_results": 5,
        }
        assert resp.query == body["query"]

    def test_max_results_capped_at_10(self) -> None:
        seen: list[httpx.Request] = []
        _ollama(_json_handler(_OLLAMA_JSON, seen)).search("q", "en", 20)
        assert json.loads(seen[0].content)["max_results"] == 10

    def test_results_mapped_and_urlless_skipped(self) -> None:
        resp = _ollama(_json_handler(_OLLAMA_JSON)).search("q", "en", 10)
        assert [h.url for h in resp.hits] == [
            "https://www.reuters.com/world/europe/kyiv-drones",
            "https://www.bbc.com/news/world-europe-1",
            "https://www.bbc.com/news/world-europe-2",
        ]
        assert resp.hits[0].snippet == "Russia launched drones at Kyiv overnight."
        assert resp.hits[0].title == "Drone attack on Kyiv"
        assert [h.rank for h in resp.hits] == [0, 1, 2]
        assert resp.engine_errors == []
        assert resp.failed is False
        assert resp.off_target(["example.org"]) is False

    def test_empty_results_are_not_failed(self) -> None:
        resp = _ollama(_json_handler({"results": []})).search("q", "en", 10)
        assert resp.hits == [] and resp.failed is False

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_is_fatal_without_key(self, status: int) -> None:
        backend = _ollama(lambda r: httpx.Response(status, text=f"bad {_KEY}"))
        with pytest.raises(SearchBackendError) as info:
            backend.search("q", "en", 10)
        assert info.value.fatal is True
        assert "SEARCH_OLLAMA_API_KEY" in str(info.value)
        assert "API key" in str(info.value)
        assert _KEY not in str(info.value)
        assert _KEY not in repr(info.value)

    @pytest.mark.parametrize("status", [429, 500])
    def test_http_errors_not_fatal(self, status: int) -> None:
        backend = _ollama(lambda r: httpx.Response(status))
        with pytest.raises(SearchBackendError, match=f"HTTP {status}") as info:
            backend.search("q", "en", 10)
        assert info.value.fatal is False
        assert _KEY not in str(info.value)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="<html>"),
            httpx.Response(200, json={"results": "nope"}),
            httpx.Response(200, json=[1, 2]),
        ],
    )
    def test_malformed_response_raises(self, response: httpx.Response) -> None:
        with pytest.raises(SearchBackendError):
            _ollama(lambda r: response).search("q", "en", 10)

    def test_connection_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(SearchBackendError, match="ollama.test") as info:
            _ollama(handler).search("q", "en", 10)
        assert isinstance(info.value.original_error, httpx.TransportError)

    def test_missing_key_is_fatal_and_sends_nothing(self) -> None:
        seen: list[httpx.Request] = []
        with pytest.raises(SearchBackendError, match="SEARCH_OLLAMA_API_KEY") as info:
            _ollama(_json_handler(_OLLAMA_JSON, seen), key=None).search("q", "en", 10)
        assert info.value.fatal is True
        assert seen == []


class TestOllamaHealthCheck:
    def test_ok_without_search_call(self) -> None:
        seen: list[httpx.Request] = []
        assert _ollama(_json_handler({}, seen)).health_check() is True
        assert [(r.method, r.url.path) for r in seen] == [("GET", "/")]
        assert all("web_search" not in str(r.url) for r in seen)

    def test_4xx_counts_as_reachable(self) -> None:
        assert _ollama(lambda r: httpx.Response(404)).health_check() is True

    def test_5xx_raises(self) -> None:
        with pytest.raises(SearchBackendError, match="503"):
            _ollama(lambda r: httpx.Response(503)).health_check()

    def test_unreachable_names_url(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(SearchBackendError, match="https://ollama.test"):
            _ollama(handler).health_check()

    def test_missing_key_raises(self) -> None:
        with pytest.raises(SearchBackendError, match="SEARCH_OLLAMA_API_KEY"):
            _ollama(_json_handler({}), key=None).health_check()


def _search_settings(backend: str, key: str | None = _KEY) -> MagicMock:
    settings = MagicMock()
    settings.search.backend = backend
    settings.search.base_url = "http://localhost:8080"
    settings.search.ollama_web_url = "https://ollama.com"
    settings.search.ollama_api_key = SecretStr(key) if key else None
    settings.search.timeout_seconds = 5
    settings.search.user_agent = "UA/1.0"
    return settings


class TestCreateSearchBackend:
    def test_default_is_searxng(self) -> None:
        backend = create_search_backend(_search_settings("searxng"))
        assert isinstance(backend, SearxngBackend)

    def test_ollama(self) -> None:
        backend = create_search_backend(_search_settings("ollama"))
        assert isinstance(backend, OllamaSearchBackend)
        assert backend.name == "ollama"
        assert backend.base_url == "https://ollama.com"
