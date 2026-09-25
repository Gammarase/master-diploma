"""Unit tests for retrieval.evidence."""

from __future__ import annotations

from retrieval.evidence import (
    STATUS_NO_RESULTS,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    RetrievalResult,
    RetrievedEvidence,
)


class TestRetrievedEvidence:
    def test_defaults(self) -> None:
        ev = RetrievedEvidence(evidence_id="abc-p0", score=0.7, evidence_text="t")
        assert ev.trust == 1.0
        assert ev.retrieval_score == 0.7
        assert ev.is_snippet is False
        assert ev.published_at == ""

    def test_explicit_retrieval_score_kept(self) -> None:
        ev = RetrievedEvidence(
            evidence_id="a", score=0.9, evidence_text="t", retrieval_score=0.1
        )
        assert ev.retrieval_score == 0.1

    def test_passage_text_alias(self) -> None:
        ev = RetrievedEvidence(evidence_id="a", score=0.5, evidence_text="body")
        assert ev.passage_text == "body"

    def test_source_key_prefers_publisher(self) -> None:
        ev = RetrievedEvidence(
            evidence_id="a",
            score=0.5,
            evidence_text="t",
            source_url="https://www.reuters.com/x",
            publisher="reuters.com",
        )
        assert ev.source_key == "reuters.com"

    def test_source_key_falls_back_to_url(self) -> None:
        ev = RetrievedEvidence(
            evidence_id="a",
            score=0.5,
            evidence_text="t",
            source_url="https://example.org/x",
        )
        assert ev.source_key == "https://example.org/x"


class TestRetrievalResult:
    def test_status_constants(self) -> None:
        assert (STATUS_OK, STATUS_NO_RESULTS, STATUS_UNAVAILABLE) == (
            "ok",
            "no_results",
            "unavailable",
        )

    def test_defaults(self) -> None:
        r = RetrievalResult()
        assert r.evidences == []
        assert r.status == "ok"
        assert r.queries == []
