"""
Evidence data model for the Disinformation Detection System.

``RetrievedEvidence`` is one passage of a web page, carrying its provenance
(URL, publisher, dates, source tier and trust). ``RetrievalResult`` bundles
the passages for one claim with the search status and the queries used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "RetrievedEvidence",
    "RetrievalResult",
    "SearchStatus",
    "STATUS_OK",
    "STATUS_NO_RESULTS",
    "STATUS_UNAVAILABLE",
]

SearchStatus = Literal["ok", "no_results", "unavailable"]

STATUS_OK: SearchStatus = "ok"
STATUS_NO_RESULTS: SearchStatus = "no_results"
STATUS_UNAVAILABLE: SearchStatus = "unavailable"


@dataclass
class RetrievedEvidence:
    """A single evidence passage taken from a web page.

    Attributes:
        evidence_id: Stable id, ``sha1(final_url)[:16]`` plus ``-p{passage_index}``.
        score: Relevance score: the reranker score when reranking is enabled,
            otherwise a search-rank-derived score.
        evidence_text: Passage text used by the verifiers.
        source_url: Final URL of the page (after redirects).
        publisher: Registrable domain of ``source_url``.
        title: Page title.
        published_at: Publication date as ``YYYY-MM-DD`` ("" when unknown).
        retrieved_at: ISO-8601 UTC timestamp of the fetch.
        source_tier: Source-policy tier name ("" when unmatched).
        trust: Source trust in [0, 1].
        language: Language code of the passage.
        passage_index: Position of the passage within its page.
        search_rank: Position of the page in the collected search results.
        is_snippet: True when the passage is the search snippet fallback.
        retrieval_score: Score before reranking (defaults to ``score``).
    """

    evidence_id: str
    score: float
    evidence_text: str
    source_url: str = ""
    publisher: str = ""
    title: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    source_tier: str = ""
    trust: float = 1.0
    language: str = ""
    passage_index: int = 0
    search_rank: int = 0
    is_snippet: bool = False
    retrieval_score: float | None = None

    def __post_init__(self) -> None:
        if self.retrieval_score is None:
            self.retrieval_score = self.score

    @property
    def passage_text(self) -> str:
        """Alias for ``evidence_text``."""
        return self.evidence_text

    @property
    def source_key(self) -> str:
        """Publisher domain (or URL when unknown), used for the diversity cap."""
        return self.publisher or self.source_url


@dataclass
class RetrievalResult:
    """Evidence retrieved for one claim, with the search status.

    Attributes:
        evidences: Passages, best first.
        status: ``ok``, ``no_results`` or ``unavailable``.
        queries: Search queries used, in order.
        backend: Name of the search backend.
    """

    evidences: list[RetrievedEvidence] = field(default_factory=list)
    status: SearchStatus = STATUS_OK
    queries: list[str] = field(default_factory=list)
    backend: str = ""
