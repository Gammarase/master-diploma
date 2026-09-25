"""Unit tests for retrieval.web_retriever.WebEvidenceRetriever."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from exceptions import SearchBackendError
from retrieval.cache import WebCache
from retrieval.evidence import RetrievedEvidence
from retrieval.page_fetcher import FetchedPage
from retrieval.source_policy import SourcePolicy
from retrieval.web_retriever import WebEvidenceRetriever, normalize_url
from retrieval.web_search import SearchHit, SearchResponse

_CLAIM = "The IAEA confirmed shelling at the Zaporizhzhia plant."


# ─── Fakes ────────────────────────────────────────────────────────────────────


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        responder: Callable[[str, list[str] | None], SearchResponse] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, list[str] | None]] = []
        self._responder = responder or (lambda q, sites: SearchResponse(query=q))

    def search(
        self, query: str, language: str, max_results: int, sites: list[str] | None = None
    ) -> SearchResponse:
        self.calls.append((query, language, sites))
        return self._responder(query, sites)

    def health_check(self) -> bool:
        return True


class FakeFetcher:
    """Returns a page whose text is given per URL (default: 2 sentences)."""

    def __init__(self, texts: dict[str, str] | None = None) -> None:
        self.texts = texts or {}
        self.fetched: list[str] = []

    def fetch(self, hit: SearchHit, fallback_language: str = "") -> FetchedPage | None:
        self.fetched.append(hit.url)
        text = self.texts.get(hit.url, f"Passage about {hit.url}.")
        if text is None:
            return None
        return FetchedPage(
            url=hit.url,
            final_url=hit.url,
            text=text,
            title=hit.title or "Title",
            published_at="2022-08-12",
            language="en",
            retrieved_at="2026-09-25T10:00:00Z",
        )


class FakeReranker:
    def __init__(self, scores: dict[str, float] | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self._scores = scores or {}
        self.calls = 0

    def rerank(self, query: str, evs: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        self.calls += 1
        out = [
            dataclasses.replace(ev, score=self._scores.get(ev.evidence_text, 0.5))
            for ev in evs
        ]
        return sorted(out, key=lambda e: e.score, reverse=True)


class FakeQueries:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def build(self, claim_text: str, language: str) -> list[str]:
        return list(self.queries)


def _hits(*urls: str) -> list[SearchHit]:
    return [SearchHit(url=u, title=f"T{i}", snippet="s", rank=i) for i, u in enumerate(urls)]


def _static(*urls: str) -> Callable[[str, list[str] | None], SearchResponse]:
    return lambda q, sites: SearchResponse(query=q, hits=_hits(*urls))


# ─── Setup ────────────────────────────────────────────────────────────────────

_TIERS = {
    "fact_checkers": {"trust": 1.0, "domains": ["politifact.com", "reuters.com/fact-check"]},
    "wire_agencies": {"trust": 0.95, "domains": ["reuters.com", "apnews.com"]},
    "primary_sources": {"trust": 0.9, "domains": ["iaea.org", "un.org"]},
    "major_outlets": {"trust": 0.8, "domains": ["bbc.com", "bbc.co.uk"]},
}


def _settings(**overrides: Any) -> MagicMock:
    s = MagicMock()
    r = s.retrieval
    r.top_k = 5
    r.candidate_k = 40
    r.min_relevance = 0.2
    r.max_passages_per_source = 2
    r.max_passage_chars = 800
    r.max_search_requests = 8
    r.max_pages_per_claim = 8
    s.search.results_per_query = 10
    s.search.min_interval_seconds = 1.0
    s.search.site_filter = "grouped"
    s.search.site_group_size = 10
    for key, value in overrides.items():
        section, name = key.split("__")
        setattr(getattr(s, section), name, value)
    return s


def _policy(mode: str = "strict", tiers: dict | None = None) -> SourcePolicy:
    return SourcePolicy(
        {"tiers": tiers or _TIERS, "blocklist": ["rt.com"]}, mode=mode  # type: ignore[arg-type]
    )


def _retriever(
    backend: FakeBackend,
    fetcher: FakeFetcher | None = None,
    reranker: FakeReranker | None = None,
    policy: SourcePolicy | None = None,
    queries: list[str] | None = None,
    cache: WebCache | None = None,
    sleeps: list[float] | None = None,
    **settings: Any,
) -> WebEvidenceRetriever:
    clock = iter(range(0, 10_000))
    return WebEvidenceRetriever(
        _settings(**settings),
        backend,
        fetcher or FakeFetcher(),  # type: ignore[arg-type]
        policy or _policy(),
        FakeQueries(queries or [_CLAIM]),  # type: ignore[arg-type]
        reranker or FakeReranker(),  # type: ignore[arg-type]
        cache=cache,
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
        clock=lambda: 0.0 if sleeps is not None else float(next(clock)),
    )


# ─── Two-stage retrieval ──────────────────────────────────────────────────────


class TestRanking:
    def test_reranked_order_returned(self) -> None:
        # 20 candidate passages: 10 pages × 2 chunks, each page on its own host.
        urls = [f"https://www.bbc.com/news/{i}" for i in range(10)]
        texts = {
            # Letters, not digits: "7. " would be read as a list-item marker.
            u: f"First part of page {chr(65 + i)}. Second part of page {chr(65 + i)} here."
            for i, u in enumerate(urls)
        }
        # Candidate #14 (0-based) is page 7, passage 0.
        reranker = FakeReranker({"First part of page H.": 0.99})
        seen: list[int] = []
        orig = reranker.rerank

        def spy(q: str, evs: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
            seen.append(len(evs))
            assert evs[14].evidence_text == "First part of page H."
            return orig(q, evs)

        reranker.rerank = spy  # type: ignore[method-assign]
        retriever = _retriever(
            FakeBackend(_static(*urls)),
            FakeFetcher(texts),
            reranker,
            retrieval__max_passage_chars=30,
            retrieval__max_passages_per_source=10,
            retrieval__max_pages_per_claim=10,
        )
        result = retriever.retrieve(_CLAIM, "en")
        assert seen == [20]
        assert result.status == "ok"
        assert result.evidences[0].evidence_text == "First part of page H."
        assert len(result.evidences) == 5

    def test_retrieval_score_keeps_pre_rerank_score(self) -> None:
        backend = FakeBackend(_static("https://apnews.com/a"))
        result = _retriever(backend, reranker=FakeReranker({})).retrieve(_CLAIM, "en")
        ev = result.evidences[0]
        assert ev.score == 0.5
        assert ev.retrieval_score == 1.0

    def test_reranker_disabled_uses_search_order(self) -> None:
        urls = ["https://apnews.com/1", "https://www.bbc.com/2", "https://iaea.org/3"]
        texts = {u: "One sentence here. Another sentence here." for u in urls}
        reranker = FakeReranker(enabled=False)
        result = _retriever(
            FakeBackend(_static(*urls)),
            FakeFetcher(texts),
            reranker,
            retrieval__max_passage_chars=25,
            retrieval__min_relevance=0.99,
            retrieval__top_k=4,
        ).retrieve(_CLAIM, "en")
        assert reranker.calls == 0
        assert [(e.search_rank, e.passage_index) for e in result.evidences] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        assert [e.score for e in result.evidences] == [1.0, 0.5, pytest.approx(1 / 3), 0.25]

    def test_no_relevant_evidence(self) -> None:
        backend = FakeBackend(_static("https://apnews.com/a", "https://www.bbc.com/b"))
        reranker = FakeReranker({})
        result = _retriever(
            backend, reranker=reranker, retrieval__min_relevance=0.6
        ).retrieve(_CLAIM, "en")
        assert result.evidences == []
        assert result.status == "ok"

    def test_candidate_k_limits_passages(self) -> None:
        urls = [f"https://apnews.com/{i}" for i in range(5)]
        texts = {u: ("Sentence number one. " * 10) for u in urls}
        reranker = FakeReranker()
        seen: list[int] = []
        orig = reranker.rerank

        def spy(q: str, evs: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
            seen.append(len(evs))
            return orig(q, evs)

        reranker.rerank = spy  # type: ignore[method-assign]
        _retriever(
            FakeBackend(_static(*urls)),
            FakeFetcher(texts),
            reranker,
            retrieval__max_passage_chars=50,
            retrieval__candidate_k=7,
        ).retrieve(_CLAIM, "en")
        assert seen == [7]


class TestDiversityCap:
    def test_cap_by_publisher(self) -> None:
        urls = [
            "https://www.reuters.com/world/1",
            "https://reuters.com/world/2",
            "https://www.reuters.com/world/3",
            "https://apnews.com/a",
            "https://www.bbc.com/b",
        ]
        texts = {u: f"Text {i}." for i, u in enumerate(urls)}
        scores = {"Text 0.": 0.99, "Text 1.": 0.98, "Text 2.": 0.97, "Text 3.": 0.6, "Text 4.": 0.5}
        result = _retriever(
            FakeBackend(_static(*urls)), FakeFetcher(texts), FakeReranker(scores)
        ).retrieve(_CLAIM, "en")
        publishers = [e.publisher for e in result.evidences]
        assert publishers.count("reuters.com") == 2
        assert publishers == ["reuters.com", "reuters.com", "apnews.com", "bbc.com"]


# ─── Provenance ───────────────────────────────────────────────────────────────


class TestProvenance:
    def test_fields_filled(self) -> None:
        url = "https://apnews.com/article/xyz"
        r1 = _retriever(FakeBackend(_static(url))).retrieve(_CLAIM, "en")
        r2 = _retriever(FakeBackend(_static(url))).retrieve(_CLAIM, "en")
        ev = r1.evidences[0]
        assert ev.source_url == url
        assert ev.publisher == "apnews.com"
        assert ev.trust == 0.95
        assert ev.source_tier == "wire_agencies"
        assert ev.retrieved_at
        assert ev.published_at == "2022-08-12"
        assert ev.evidence_id.endswith("-p0") and len(ev.evidence_id) == 19
        assert ev.evidence_id == r2.evidences[0].evidence_id

    def test_path_tier_on_evidence(self) -> None:
        url = "https://www.reuters.com/fact-check/abc"
        ev = _retriever(FakeBackend(_static(url))).retrieve(_CLAIM, "en").evidences[0]
        assert (ev.source_tier, ev.trust, ev.publisher) == ("fact_checkers", 1.0, "reuters.com")

    def test_result_metadata(self) -> None:
        result = _retriever(
            FakeBackend(_static("https://apnews.com/a")), queries=["q1", "q2"]
        ).retrieve(_CLAIM, "en")
        assert result.queries == ["q1", "q2"]
        assert result.backend == "fake"


# ─── Search restriction and the request plan ─────────────────────────────────


class TestRequestPlan:
    def _many_domains(self) -> SourcePolicy:
        return SourcePolicy(
            {
                "tiers": {
                    "major": {"trust": 0.8, "domains": [f"major{i}.com" for i in range(10)]},
                    "fact": {"trust": 1.0, "domains": [f"fact{i}.org" for i in range(8)]},
                    "wire": {"trust": 0.95, "domains": [f"wire{i}.com" for i in range(7)]},
                }
            }
        )

    def test_queries_restricted_to_trusted_domains(self) -> None:
        backend = FakeBackend()
        _retriever(backend, policy=self._many_domains()).retrieve(_CLAIM, "en")
        assert len(backend.calls) == 3
        groups = [sites for _, _, sites in backend.calls]
        assert all(g is not None and len(g) <= 10 for g in groups)
        assert groups[0][:8] == [f"fact{i}.org" for i in range(8)]  # type: ignore[index]
        assert len({d for g in groups for d in g}) == 25  # type: ignore[union-attr]

    def test_path_entry_searched_by_domain(self) -> None:
        backend = FakeBackend()
        _retriever(backend).retrieve(_CLAIM, "en")
        flat = [d for _, _, g in backend.calls for d in (g or [])]
        assert flat.count("reuters.com") == 1
        assert not any("/" in d for d in flat)

    @pytest.mark.parametrize("mode", ["lenient", "off"])
    def test_lenient_and_off_search_openly(self, mode: str) -> None:
        backend = FakeBackend()
        _retriever(backend, policy=_policy(mode), queries=["q1", "q2"]).retrieve(_CLAIM, "en")
        assert [sites for _, _, sites in backend.calls] == [None, None]

    def test_site_filter_none(self) -> None:
        backend = FakeBackend()
        _retriever(backend, search__site_filter="none").retrieve(_CLAIM, "en")
        assert [sites for _, _, sites in backend.calls] == [None]

    def test_site_filter_per_domain(self) -> None:
        backend = FakeBackend()
        _retriever(
            backend, search__site_filter="per_domain", retrieval__max_search_requests=100
        ).retrieve(_CLAIM, "en")
        assert all(len(s) == 1 for _, _, s in backend.calls)  # type: ignore[arg-type]
        # 8 entries; the reuters.com/fact-check path entry dedupes into reuters.com.
        assert len(backend.calls) == 7

    def test_budget_respected(self) -> None:
        # 2 queries × 4 groups, budget 6 → q1 with all groups, q2 with the first 2.
        backend = FakeBackend()
        retriever = _retriever(
            backend,
            queries=["q1", "q2"],
            search__site_group_size=2,
            retrieval__max_search_requests=6,
        )
        groups = retriever.site_groups
        assert len(groups) == 4
        retriever.retrieve(_CLAIM, "en")
        assert [(q, s) for q, _, s in backend.calls] == [
            ("q1", groups[0]),
            ("q1", groups[1]),
            ("q1", groups[2]),
            ("q1", groups[3]),
            ("q2", groups[0]),
            ("q2", groups[1]),
        ]

    def test_language_passed_to_backend(self) -> None:
        backend = FakeBackend()
        _retriever(backend, search__site_filter="none").retrieve(_CLAIM, "uk")
        assert backend.calls[0][1] == "uk"
        backend2 = FakeBackend()
        _retriever(backend2, search__site_filter="none").retrieve(_CLAIM, "")
        assert backend2.calls[0][1] == "all"

    def test_throttling_between_requests(self) -> None:
        sleeps: list[float] = []
        backend = FakeBackend()
        _retriever(backend, queries=["q1", "q2"], sleeps=sleeps, search__site_filter="none").retrieve(
            _CLAIM, "en"
        )
        # Clock is frozen at 0: the second request waits the full interval.
        assert sleeps == [1.0]


class TestFiltering:
    def test_untrusted_results_never_fetched(self) -> None:
        urls = [
            "https://apnews.com/a",
            "https://random.example/1",
            "https://www.bbc.co.uk/news/b",
            "https://notreuters.com/x",
            "https://reuters.com.evil.io/x",
            "https://www.rt.com/news/1",
            "https://iaea.org/c",
            "https://blog.example/2",
            "https://another.example/3",
            "https://third.example/4",
        ]
        fetcher = FakeFetcher()
        result = _retriever(FakeBackend(_static(*urls)), fetcher).retrieve(_CLAIM, "en")
        assert len(fetcher.fetched) == 3
        policy = _policy()
        assert all(policy.allows(e.source_url) for e in result.evidences)

    def test_duplicate_urls_fetched_once(self) -> None:
        responder = _static(
            "https://apnews.com/a?utm_source=x#frag",
            "https://APNEWS.com/a",
            "https://apnews.com/a?fbclid=1",
        )
        fetcher = FakeFetcher()
        _retriever(FakeBackend(responder), fetcher, queries=["q1", "q2"]).retrieve(_CLAIM, "en")
        assert fetcher.fetched == ["https://apnews.com/a"]

    def test_page_cap(self) -> None:
        urls = [f"https://apnews.com/{i}" for i in range(20)]
        fetcher = FakeFetcher()
        _retriever(
            FakeBackend(_static(*urls)), fetcher, retrieval__max_pages_per_claim=8
        ).retrieve(_CLAIM, "en")
        assert len(fetcher.fetched) == 8

    def test_unfetchable_pages_skipped(self) -> None:
        fetcher = FakeFetcher({"https://apnews.com/a": None})  # type: ignore[dict-item]
        result = _retriever(
            FakeBackend(_static("https://apnews.com/a", "https://www.bbc.com/b")), fetcher
        ).retrieve(_CLAIM, "en")
        assert [e.publisher for e in result.evidences] == ["bbc.com"]
        assert result.evidences[0].search_rank == 1

    def test_fetch_exception_isolated(self) -> None:
        fetcher = FakeFetcher()
        orig = fetcher.fetch

        def flaky(hit: SearchHit, fallback_language: str = "") -> FetchedPage | None:
            if "bad" in hit.url:
                raise RuntimeError("boom")
            return orig(hit, fallback_language)

        fetcher.fetch = flaky  # type: ignore[method-assign]
        result = _retriever(
            FakeBackend(_static("https://apnews.com/bad", "https://apnews.com/good")), fetcher
        ).retrieve(_CLAIM, "en")
        assert [e.source_url for e in result.evidences] == ["https://apnews.com/good"]


# ─── Status ───────────────────────────────────────────────────────────────────


class TestStatus:
    def test_all_engines_rate_limited(self) -> None:
        blocked = SearchResponse(
            query="q", hits=[], engine_errors=[["brave", "too many requests"], ["duckduckgo", "CAPTCHA"]]
        )
        result = _retriever(FakeBackend(lambda q, s: blocked)).retrieve(_CLAIM, "en")
        assert result.status == "unavailable"
        assert result.evidences == []

    def test_backend_unreachable(self) -> None:
        def down(q: str, s: list[str] | None) -> SearchResponse:
            raise SearchBackendError(
                "down", original_error=httpx.ConnectError("refused")
            )

        backend = FakeBackend(down)
        result = _retriever(backend, queries=["q1", "q2"]).retrieve(_CLAIM, "en")
        assert result.status == "unavailable"
        assert len(backend.calls) == 1  # no further requests after a connection error

    def test_http_error_keeps_trying(self) -> None:
        calls = {"n": 0}

        def flaky(q: str, s: list[str] | None) -> SearchResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                raise SearchBackendError("HTTP 502")
            return SearchResponse(query=q, hits=_hits("https://apnews.com/a"))

        result = _retriever(FakeBackend(flaky), search__site_filter="none", queries=["q1", "q2"]).retrieve(
            _CLAIM, "en"
        )
        assert result.status == "ok"
        assert len(result.evidences) == 1

    def test_nothing_trusted_found(self) -> None:
        backend = FakeBackend(_static("https://random.example/1", "https://blog.example/2"))
        result = _retriever(backend).retrieve(_CLAIM, "en")
        assert result.status == "no_results"
        assert result.evidences == []

    def test_empty_results(self) -> None:
        result = _retriever(FakeBackend()).retrieve(_CLAIM, "en")
        assert result.status == "no_results"


# ─── Cache ────────────────────────────────────────────────────────────────────


class TestCache:
    def test_offline_replay(self, tmp_path: Path) -> None:
        urls = ["https://apnews.com/a", "https://www.bbc.com/b"]
        first_backend = FakeBackend(_static(*urls))
        first = _retriever(
            first_backend, cache=WebCache(tmp_path), queries=["q1", "q2"]
        ).retrieve(_CLAIM, "en")
        assert first_backend.calls

        def offline(q: str, s: list[str] | None) -> SearchResponse:
            raise AssertionError("network used in read_only mode")

        offline_backend = FakeBackend(offline)
        second = _retriever(
            offline_backend,
            cache=WebCache(tmp_path, mode="read_only"),
            queries=["q1", "q2"],
        ).retrieve(_CLAIM, "en")
        assert offline_backend.calls == []
        assert second.evidences == first.evidences
        assert second.status == "ok"

    def test_read_only_miss_is_no_results(self, tmp_path: Path) -> None:
        backend = FakeBackend(lambda q, s: pytest.fail("network used"))
        result = _retriever(
            backend, cache=WebCache(tmp_path, mode="read_only")
        ).retrieve(_CLAIM, "en")
        assert result.status == "no_results"
        assert backend.calls == []

    def test_cached_requests_do_not_count_toward_budget(self, tmp_path: Path) -> None:
        cache = WebCache(tmp_path)
        backend = FakeBackend()
        _retriever(
            backend, cache=cache, queries=["q1"], search__site_filter="none"
        ).retrieve(_CLAIM, "en")
        assert len(backend.calls) == 1
        backend2 = FakeBackend()
        _retriever(
            backend2,
            cache=cache,
            queries=["q1", "q2"],
            search__site_filter="none",
            retrieval__max_search_requests=1,
        ).retrieve(_CLAIM, "en")
        assert [q for q, _, _ in backend2.calls] == ["q2"]

    def test_failed_search_not_cached(self, tmp_path: Path) -> None:
        cache = WebCache(tmp_path)
        blocked = SearchResponse(query="q", engine_errors=[["brave", "too many requests"]])
        _retriever(FakeBackend(lambda q, s: blocked), cache=cache, search__site_filter="none").retrieve(
            _CLAIM, "en"
        )
        backend = FakeBackend()
        _retriever(backend, cache=cache, search__site_filter="none").retrieve(_CLAIM, "en")
        assert len(backend.calls) == 1


def test_normalize_url() -> None:
    assert (
        normalize_url("HTTPS://WWW.Example.com/Path?a=1&utm_source=x&gclid=2#frag")
        == "https://www.example.com/Path?a=1"
    )
