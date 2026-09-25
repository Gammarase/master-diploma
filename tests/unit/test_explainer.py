"""Unit tests for explainability.explainer.Explainer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from claim_extraction.extractor import Claim
from explainability.explainer import ExplanationOutput, Explainer
from retrieval.evidence import RetrievalResult, RetrievedEvidence
from verification.aggregator import (
    REASON_CONFLICT,
    REASON_LOW_SCORE_MARGIN,
    REASON_NO_EVIDENCE,
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    VerificationResult,
)
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict


@pytest.fixture
def explainer() -> Explainer:
    return Explainer()


def _make_claim(text: str = "Russia launched missiles.", lang: str = "en") -> Claim:
    return Claim(
        text=text,
        original_sentence=text,
        sentence_index=0,
        language=lang,
    )


def _make_evidence(idx: int = 0) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=f"abcdef0123456789-p{idx}",
        score=0.9 - idx * 0.1,
        evidence_text=f"Evidence passage {idx} with important facts.",
        source_url="https://apnews.com/article/xyz" if idx == 0 else f"https://www.bbc.com/news/{idx}",
        publisher="apnews.com" if idx == 0 else "bbc.com",
        title=f"Title {idx}",
        published_at="2022-08-12" if idx == 0 else "",
        retrieved_at="2026-09-25T10:00:00Z",
        source_tier="wire_agencies" if idx == 0 else "major_outlets",
        trust=0.95 if idx == 0 else 0.8,
        language="en",
        passage_index=idx,
    )


def _make_nli_result(
    evidence_text: str,
    label: str = "entailment",
    confidence: float = 0.85,
) -> NLIResult:
    return NLIResult(
        claim="Russia launched missiles.",
        evidence_text=evidence_text,
        entailment_score=confidence if label == "entailment" else 0.1,
        contradiction_score=confidence if label == "contradiction" else 0.1,
        neutral_score=confidence if label == "neutral" else 0.1,
        predicted_label=label,
        confidence=confidence,
    )


def _make_rag_verdict(
    verdict: str = "SUPPORTED", confidence: float = 0.85
) -> RAGVerdict:
    return RAGVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning="Strong evidence supports the claim.",
        supporting_evidence_ids=[0],
        contradicting_evidence_ids=[],
    )


def _make_verification_result(
    verdict: str = VERDICT_CONFIRMED,
    nli_score: float = 0.85,
    rag_score: float = 0.85,
    final_score: float = 0.85,
    reason: str | None = None,
    n_evidence: int = 2,
) -> VerificationResult:
    evidences = [_make_evidence(i) for i in range(n_evidence)]
    nli_results = [
        _make_nli_result(ev.evidence_text, label=lbl)
        for ev, lbl in zip(evidences, ["entailment", "contradiction"])
    ]
    rag = _make_rag_verdict()
    rag.raw_response = '{"verdict": "SUPPORTED"}'
    return VerificationResult(
        claim=_make_claim(),
        nli_score=nli_score,
        rag_score=rag_score,
        final_score=final_score,
        verdict=verdict,
        nli_results=nli_results,
        rag_verdict=rag,
        retrieved_evidences=evidences,
        component_weights={"nli": 0.3, "rag": 0.7},
        uncertainty_reason=reason,
        effective_weights={"nli": 0.0, "rag": 1.0},
    )


@pytest.fixture
def settings_mock() -> MagicMock:
    s = MagicMock()
    s.ollama.evidence_mode = "web"
    s.ollama.model = "qwen3:14b"
    s.source_policy.mode = "strict"
    s.retrieval.reranker_model = "BAAI/bge-reranker-v2-m3"
    s.verification.nli_model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    return s


class TestExplain:
    def test_returns_explanation_output(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert isinstance(output, ExplanationOutput)

    def test_verdict_matches(self, explainer: Explainer) -> None:
        result = _make_verification_result(verdict=VERDICT_CONFIRMED)
        output = explainer.explain(result)
        assert output.verdict == VERDICT_CONFIRMED

    def test_final_score_matches(self, explainer: Explainer) -> None:
        result = _make_verification_result(final_score=0.75)
        output = explainer.explain(result)
        assert output.final_score == pytest.approx(0.75, abs=0.001)

    def test_claim_text_matches(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert output.claim_text == "Russia launched missiles."

    def test_language_matches(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert output.language == "en"

    def test_component_scores_present(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert "nli" in output.component_scores
        assert "rag" in output.component_scores

    def test_evidence_excerpts_count(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert len(output.evidence_excerpts) == 2

    def test_citations_count(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        assert len(output.citations) == 2

    def test_citation_for_wire_story(self, explainer: Explainer) -> None:
        output = explainer.explain(_make_verification_result())
        first = output.citations[0]
        assert first == {
            "passage_index": 0,
            "evidence_id": "abcdef0123456789-p0",
            "url": "https://apnews.com/article/xyz",
            "publisher": "apnews.com",
            "title": "Title 0",
            "published_at": "2022-08-12",
            "retrieved_at": "2026-09-25T10:00:00Z",
            "tier": "wire_agencies",
            "trust": 0.95,
            "language": "en",
        }
        assert output.citations[1]["published_at"] == ""
        assert [c["passage_index"] for c in output.citations] == [0, 1]

    def test_no_dataset_fields(self, explainer: Explainer) -> None:
        output = explainer.explain(_make_verification_result())
        parsed = json.loads(explainer.to_json(output))
        for citation in parsed["citations"]:
            assert not {"dataset", "split", "vector_id", "claim_id"} & set(citation)

    def test_excerpts_have_no_dataset_label(self, explainer: Explainer) -> None:
        output = explainer.explain(_make_verification_result())
        parsed = json.loads(explainer.to_json(output))
        for excerpt in parsed["evidence_excerpts"]:
            assert "label_from_dataset" not in excerpt
            assert set(excerpt) == {"passage_index", "text", "relevance_score", "stance"}

    def test_stance_aligned_by_position(self, explainer: Explainer) -> None:
        output = explainer.explain(_make_verification_result())
        assert [e["stance"] for e in output.evidence_excerpts] == [
            "supports", "contradicts"
        ]

    def test_stance_falls_back_to_text_match(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        result.nli_results = result.nli_results[1:]  # misaligned lengths
        output = explainer.explain(result)
        assert [e["stance"] for e in output.evidence_excerpts] == [
            "unknown", "contradicts"
        ]


class TestProcessingMetadata:
    _KEYS = {
        "uncertainty_reason",
        "effective_weights",
        "rag_model_verdict",
        "rag_confidence",
        "rag_raw_response",
        "evidence_mode",
        "search_backend",
        "search_status",
        "search_queries",
        "source_policy_mode",
        "models",
    }

    def test_metadata_keys_present(
        self, settings_mock: MagicMock
    ) -> None:
        retrieval = RetrievalResult(
            evidences=[], status="ok", queries=["q1", "q2"], backend="searxng"
        )
        output = Explainer(settings_mock).explain(
            _make_verification_result(verdict=VERDICT_UNCERTAIN, reason=REASON_CONFLICT),
            retrieval,
        )
        meta = output.processing_metadata
        assert self._KEYS <= set(meta)
        assert meta["uncertainty_reason"] == REASON_CONFLICT
        assert meta["effective_weights"] == {"nli": 0.0, "rag": 1.0}
        assert meta["rag_model_verdict"] == "SUPPORTED"
        assert meta["rag_confidence"] == pytest.approx(0.85)
        assert meta["rag_raw_response"] == '{"verdict": "SUPPORTED"}'
        assert meta["evidence_mode"] == "web"
        assert meta["search_backend"] == "searxng"
        assert meta["search_status"] == "ok"
        assert meta["search_queries"] == ["q1", "q2"]
        assert meta["source_policy_mode"] == "strict"
        assert "embeddings" not in meta["models"]
        assert meta["models"] == {
            "reranker": "BAAI/bge-reranker-v2-m3",
            "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "llm": "qwen3:14b",
        }

    def test_metadata_without_settings(self, explainer: Explainer) -> None:
        meta = explainer.explain(_make_verification_result()).processing_metadata
        assert self._KEYS <= set(meta)
        assert meta["evidence_mode"] is None
        assert meta["models"]["llm"] is None
        assert "embeddings" not in meta["models"]
        assert meta["uncertainty_reason"] is None
        assert meta["search_status"] is None
        assert meta["search_queries"] == []
        assert meta["source_policy_mode"] is None


class TestGenerateExplanationText:
    def test_confirmed_mentions_supported(self, explainer: Explainer) -> None:
        result = _make_verification_result(verdict=VERDICT_CONFIRMED)
        text = explainer._generate_explanation_text(result)
        assert "supported" in text.lower() or "accurate" in text.lower()

    def test_disinformation_mentions_contradiction(
        self, explainer: Explainer
    ) -> None:
        result = _make_verification_result(verdict=VERDICT_DISINFORMATION)
        text = explainer._generate_explanation_text(result)
        assert "contradicted" in text.lower() or "disinformation" in text.lower()

    def test_uncertain_mentions_insufficient(self, explainer: Explainer) -> None:
        result = _make_verification_result(verdict=VERDICT_UNCERTAIN)
        text = explainer._generate_explanation_text(result)
        assert "insufficient" in text.lower() or "inconsistent" in text.lower()

    def test_conflict_explanation(self, explainer: Explainer) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, nli_score=0.1, rag_score=0.9, reason=REASON_CONFLICT
        )
        text = explainer._generate_explanation_text(result)
        assert "opposite directions" in text
        assert "LLM assessment: Strong evidence supports the claim." in text
        assert text.index("opposite directions") < text.index("LLM assessment:")

    def test_low_score_margin_explanation(self, explainer: Explainer) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, final_score=0.5, reason=REASON_LOW_SCORE_MARGIN
        )
        text = explainer._generate_explanation_text(result)
        assert "too weak or mixed" in text

    def test_no_evidence_explanation(self, explainer: Explainer) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, reason=REASON_NO_EVIDENCE, n_evidence=0
        )
        text = explainer._generate_explanation_text(result)
        assert "No sufficiently relevant evidence from trusted sources" in text

    def test_no_evidence_with_ok_search(self, explainer: Explainer) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, reason=REASON_NO_EVIDENCE, n_evidence=0
        )
        output = explainer.explain(result, RetrievalResult(status="no_results"))
        assert "No sufficiently relevant evidence from trusted sources" in output.explanation

    def test_search_unavailable_explanation(self, explainer: Explainer) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, reason=REASON_NO_EVIDENCE, n_evidence=0
        )
        output = explainer.explain(
            result, RetrievalResult(status="unavailable", backend="searxng")
        )
        assert "evidence search could not be performed" in output.explanation
        assert "Retry later" in output.explanation
        assert "manual" in output.explanation.lower()
        assert "No sufficiently relevant evidence" not in output.explanation
        assert output.processing_metadata["search_status"] == "unavailable"

    def test_unavailable_status_only_affects_no_evidence(
        self, explainer: Explainer
    ) -> None:
        result = _make_verification_result(
            verdict=VERDICT_UNCERTAIN, reason=REASON_CONFLICT
        )
        output = explainer.explain(result, RetrievalResult(status="unavailable"))
        assert "opposite directions" in output.explanation

    def test_rag_reasoning_appended(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        text = explainer._generate_explanation_text(result)
        assert "LLM assessment: Strong evidence supports" in text

    def test_trivial_reasoning_not_appended(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        result.rag_verdict.reasoning = "ok"
        text = explainer._generate_explanation_text(result)
        assert "LLM assessment" not in text


class TestNliLabelToStance:
    def test_entailment_to_supports(self, explainer: Explainer) -> None:
        assert explainer._nli_label_to_stance("entailment") == "supports"

    def test_contradiction_to_contradicts(self, explainer: Explainer) -> None:
        assert explainer._nli_label_to_stance("contradiction") == "contradicts"

    def test_neutral_to_neutral(self, explainer: Explainer) -> None:
        assert explainer._nli_label_to_stance("neutral") == "neutral"

    def test_unknown_to_unknown(self, explainer: Explainer) -> None:
        assert explainer._nli_label_to_stance("garbage") == "unknown"


class TestToJson:
    def test_produces_valid_json(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        json_str = explainer.to_json(output)
        parsed = json.loads(json_str)
        assert parsed["verdict"] == VERDICT_CONFIRMED
        assert "evidence_excerpts" in parsed
        assert "citations" in parsed
