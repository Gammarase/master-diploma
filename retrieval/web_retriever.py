"""
Web evidence retriever: search → filter → fetch → chunk → rerank.

For each claim, ``WebEvidenceRetriever`` runs a fixed sequence of steps:

1. Build two search queries (``QueryBuilder``).
2. Pair each query with each ``site:`` group of trusted domains
   (query-major) and send the requests through the cache, throttled and
   within a per-claim budget.
3. Deduplicate hits by normalised URL and drop hits the source policy does
   not allow; keep the first ``max_pages_per_claim``.
4. Fetch the pages concurrently and extract their main text
   (``PageFetcher``), keeping search-rank order.
5. Chunk the text into passages and keep the first ``candidate_k``.
6. Rerank, drop passages below ``min_relevance``, cap passages per
   publisher, and cut to ``top_k``.

The search status (``ok`` / ``no_results`` / ``unavailable``) is reported
with the passages, so a blocked search is never mistaken for "no evidence".

SearXNG setup
-------------
* ``search.formats`` in SearXNG's ``settings.yml`` must include ``json``;
  otherwise the startup health check fails with an "enable the json format"
  error.
* Upstream engines rate-limit a single IP quickly (brave, duckduckgo and
  google cse were suspended after about 10 queries during planning). When
  every engine is suspended, SearXNG answers with no results and lists the
  engines in ``unresponsive_engines``; the claim's status is then
  ``unavailable`` and the failed request is not cached, so a later run
  retries it. Enabling more engines (for example bing, mojeek, qwant,
  startpage, wikipedia) spreads the load.
* A claim needs up to ``retrieval.max_search_requests`` (default 8)
  requests on a cold cache, spaced ``search.min_interval_seconds`` apart.
  If the engines do not honour OR-joined ``site:`` operators, set
  ``search.site_filter`` to ``per_domain`` (more requests) or ``none``
  (open search plus the policy post-filter).

Batch runs and reproducibility
------------------------------
Warm the cache first with ``retrieval.cache_mode: read_write``, over
several sessions if the engines throttle. Then run the evaluation with
``cache_mode: read_only``: no network requests are sent, a cache miss
counts as no results, and the same passages come back in the same order.
Archive ``cache/web_cache.sqlite`` with the results.

Known limitation: RU22Fact leakage
----------------------------------
There is no temporal filter. When evaluating on RU22Fact claims, search
may return the fact-check articles that produced the dataset labels, which
inflates accuracy. Report this in the evaluation; a publication-date filter
or excluding fact-check domains is a follow-up.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from exceptions import SearchBackendError
from logging_config import get_logger
from retrieval.chunking import chunk_evidence
from retrieval.evidence import (
    STATUS_NO_RESULTS,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    RetrievalResult,
    RetrievedEvidence,
    SearchStatus,
)
from retrieval.source_policy import publisher_for
from retrieval.web_search import SearchHit, SearchResponse, build_query

if TYPE_CHECKING:
    from configs import Settings
    from retrieval.cache import WebCache
    from retrieval.page_fetcher import FetchedPage, PageFetcher
    from retrieval.query_builder import QueryBuilder
    from retrieval.reranker import Reranker
    from retrieval.source_policy import SourcePolicy
    from retrieval.web_search import SearchBackend

__all__ = ["WebEvidenceRetriever", "normalize_url"]

logger = get_logger(__name__)

_FETCH_WORKERS = 4
_TRACKING_PARAMS = {"fbclid", "gclid"}


def normalize_url(url: str) -> str:
    """Lower-case the host, drop the fragment and tracking parameters."""
    parts = urlsplit(url.strip())
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
        ]
    )
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, query, "")
    )


def _evidence_id(url: str, passage_index: int) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-p{passage_index}"


@dataclass
class _Candidate:
    hit: SearchHit
    language: str


class WebEvidenceRetriever:
    """Retrieve evidence passages for a claim from trusted web sources.

    Args:
        settings: Application settings (``retrieval`` and ``search``).
        backend: Search backend.
        fetcher: Page fetcher (shares the cache and the policy).
        policy: Source policy.
        query_builder: Query builder.
        reranker: Cross-encoder reranker (may be disabled).
        cache: Web cache; None disables caching of searches.
        sleep: Sleep function (injected by tests).
        clock: Monotonic clock (injected by tests).
    """

    def __init__(
        self,
        settings: "Settings",
        backend: "SearchBackend",
        fetcher: "PageFetcher",
        policy: "SourcePolicy",
        query_builder: "QueryBuilder",
        reranker: "Reranker",
        cache: "WebCache | None" = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        r = settings.retrieval
        s = settings.search
        self._backend = backend
        self._fetcher = fetcher
        self._policy = policy
        self._queries = query_builder
        self._reranker = reranker
        self._cache = cache
        self._sleep = sleep
        self._clock = clock

        self._top_k: int = r.top_k
        self._candidate_k: int = r.candidate_k
        self._min_relevance: float = r.min_relevance
        self._per_source: int = r.max_passages_per_source
        self._max_chars: int = r.max_passage_chars
        self._budget: int = r.max_search_requests
        self._max_pages: int = r.max_pages_per_claim
        self._results_per_query: int = s.results_per_query
        self._min_interval: float = s.min_interval_seconds
        self._site_filter: str = s.site_filter
        self._group_size: int = s.site_group_size

        self._last_request: float | None = None
        self._throttle_lock = threading.Lock()
        self._site_groups = self._build_site_groups()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def site_groups(self) -> list[list[str] | None]:
        """``site:`` groups each query is paired with (None = no operators)."""
        return list(self._site_groups)

    def _build_site_groups(self) -> list[list[str] | None]:
        if self._policy.mode != "strict" or self._site_filter == "none":
            return [None]
        size = 1 if self._site_filter == "per_domain" else self._group_size
        groups: list[list[str] | None] = list(self._policy.site_groups(size))
        return groups or [None]

    # ── public ──────────────────────────────────────────────────────────────

    def retrieve(
        self, claim_text: str, language: str = "", top_k: int | None = None
    ) -> RetrievalResult:
        """Retrieve evidence passages for *claim_text*.

        Never raises for search failures: an unreachable or blocked search
        yields status ``unavailable`` with no passages.

        Args:
            claim_text: The claim.
            language: The claim's language code.
            top_k: Override for ``retrieval.top_k``.

        Returns:
            RetrievalResult with passages (best first), status and queries.
        """
        k = top_k if top_k is not None else self._top_k
        queries = self._queries.build(claim_text, language)
        responses, any_failed = self._search(queries, language)

        if not responses and any_failed:
            logger.warning(
                "Evidence search unavailable for claim '%s...': every search "
                "request failed.",
                claim_text[:50],
            )
            return self._result([], STATUS_UNAVAILABLE, queries)

        candidates = self._collect(responses, queries, language)
        if not candidates:
            return self._result([], STATUS_NO_RESULTS, queries)

        pages = self._fetch(candidates)
        passages = self._chunk(pages)
        evidences = self._rank(claim_text, passages, k)
        logger.debug(
            "Retrieved %d passages from %d pages (%d candidates) for claim '%s...'.",
            len(evidences),
            len(pages),
            len(passages),
            claim_text[:50],
        )
        return self._result(evidences, STATUS_OK, queries)

    def _result(
        self, evidences: list[RetrievedEvidence], status: SearchStatus, queries: list[str]
    ) -> RetrievalResult:
        return RetrievalResult(
            evidences=evidences,
            status=status,
            queries=list(queries),
            backend=self._backend.name,
        )

    # ── search ──────────────────────────────────────────────────────────────

    def _search(
        self, queries: list[str], language: str
    ) -> tuple[list[tuple[int, SearchResponse]], bool]:
        """Run the request plan.

        Returns:
            ``(responses, any_failed)`` where *responses* holds the
            successful responses with the index of their query.
        """
        lang = language or "all"
        offline = self._cache is not None and self._cache.offline
        sent = 0
        any_failed = False
        backend_down = False
        responses: list[tuple[int, SearchResponse]] = []

        for qi, query in enumerate(queries):
            for sites in self._site_groups:
                full_query = build_query(query, sites)
                cached = (
                    self._cache.get_search(self._backend.name, full_query, lang)
                    if self._cache is not None
                    else None
                )
                if cached is not None:
                    responses.append((qi, SearchResponse.from_dict(cached)))
                    continue
                if offline:
                    responses.append((qi, SearchResponse(query=full_query)))
                    continue
                if sent >= self._budget or backend_down:
                    continue

                self._throttle()
                sent += 1
                try:
                    response = self._backend.search(
                        query, lang, self._results_per_query, sites=sites
                    )
                except SearchBackendError as exc:
                    logger.warning("Search request failed: %s", exc)
                    any_failed = True
                    # Connection errors and timeouts would repeat for every
                    # remaining request, so stop sending them for this claim.
                    backend_down = isinstance(exc.original_error, httpx.TransportError)
                    continue
                if response.failed:
                    any_failed = True
                    continue
                if self._cache is not None:
                    self._cache.put_search(
                        self._backend.name, full_query, lang, response.to_dict()
                    )
                responses.append((qi, response))
        return responses, any_failed

    def _throttle(self) -> None:
        with self._throttle_lock:
            now = self._clock()
            if self._last_request is not None:
                wait = self._min_interval - (now - self._last_request)
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last_request = now

    # ── candidates ──────────────────────────────────────────────────────────

    def _collect(
        self,
        responses: list[tuple[int, SearchResponse]],
        queries: list[str],
        language: str,
    ) -> list[_Candidate]:
        """Dedupe hits, apply the policy and the page cap."""
        english = (language or "en").lower().startswith("en")
        seen: set[str] = set()
        out: list[_Candidate] = []
        for qi, response in responses:
            q_lang = language if (not english and qi == 0) else "en"
            for hit in response.hits:
                url = normalize_url(hit.url)
                if not url or url in seen:
                    continue
                seen.add(url)
                if not self._policy.allows(url):
                    continue
                out.append(
                    _Candidate(
                        SearchHit(
                            url=url,
                            title=hit.title,
                            snippet=hit.snippet,
                            published_at=hit.published_at,
                            engine=hit.engine,
                            rank=hit.rank,
                        ),
                        q_lang or "",
                    )
                )
                if len(out) >= self._max_pages:
                    return out
        return out

    def _fetch(self, candidates: list[_Candidate]) -> list[tuple[int, "FetchedPage"]]:
        """Fetch pages concurrently; keep search-rank order."""

        def fetch(c: _Candidate) -> "FetchedPage | None":
            try:
                return self._fetcher.fetch(c.hit, fallback_language=c.language)
            except Exception as exc:  # a single page must not fail the claim
                logger.warning("Fetching %s failed: %s", c.hit.url, exc)
                return None

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            pages = list(pool.map(fetch, candidates))
        return [(rank, page) for rank, page in enumerate(pages) if page is not None]

    def _chunk(self, pages: list[tuple[int, "FetchedPage"]]) -> list[RetrievedEvidence]:
        passages: list[RetrievedEvidence] = []
        for rank, page in pages:
            url = page.final_url
            publisher = publisher_for(url)
            tier = self._policy.tier_for(url)
            trust = self._policy.trust_for(url)
            for idx, text in enumerate(chunk_evidence(page.text, self._max_chars)):
                position = len(passages)
                passages.append(
                    RetrievedEvidence(
                        evidence_id=_evidence_id(url, idx),
                        score=1.0 / (1.0 + position),
                        evidence_text=text,
                        source_url=url,
                        publisher=publisher,
                        title=page.title,
                        published_at=page.published_at,
                        retrieved_at=page.retrieved_at,
                        source_tier=tier,
                        trust=trust,
                        language=page.language,
                        passage_index=idx,
                        search_rank=rank,
                        is_snippet=page.is_snippet,
                    )
                )
                if len(passages) >= self._candidate_k:
                    return passages
        return passages

    # ── ranking ─────────────────────────────────────────────────────────────

    def _rank(
        self, claim_text: str, passages: list[RetrievedEvidence], k: int
    ) -> list[RetrievedEvidence]:
        if not passages:
            return []
        if self._reranker is not None and self._reranker.enabled:
            ranked = self._reranker.rerank(claim_text, passages)
            ranked = [ev for ev in ranked if ev.score >= self._min_relevance]
        else:
            ranked = passages  # already in (rank, passage) order

        results: list[RetrievedEvidence] = []
        per_source: dict[str, int] = {}
        for ev in ranked:
            key = ev.source_key
            if per_source.get(key, 0) >= self._per_source:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            results.append(ev)
            if len(results) >= k:
                break
        return results

