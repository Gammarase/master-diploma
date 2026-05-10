"""Unit tests for explainability.explainer.Explainer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from claim_extraction.extractor import Claim
from explainability.explainer import ExplanationOutput, Explainer
from retrieval.vector_store import RetrievedEvidence
from verification.aggregator import (
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


def _make_evidence(idx: int = 0, label: str = "Supported") -> RetrievedEvidence:
    return RetrievedEvidence(
        vector_id=f"ru22fact-{idx}-en",
        score=0.9 - idx * 0.1,
        claim_text="Claim",
        evidence_text=f"Evidence passage {idx} with important facts.",
        label=label,
        language="EN",
        explanation="Explanation here.",
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
) -> VerificationResult:
    evidences = [_make_evidence(0), _make_evidence(1)]
    nli_results = [
        _make_nli_result(evidences[0].evidence_text),
        _make_nli_result(evidences[1].evidence_text),
    ]
    return VerificationResult(
        claim=_make_claim(),
        nli_score=nli_score,
        rag_score=rag_score,
        final_score=final_score,
        verdict=verdict,
        nli_results=nli_results,
        rag_verdict=_make_rag_verdict(),
        retrieved_evidences=evidences,
        component_weights={"nli": 0.4, "rag": 0.6},
    )


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

    def test_citations_have_dataset_field(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        output = explainer.explain(result)
        for citation in output.citations:
            assert citation["dataset"] == "RU22Fact"


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

    def test_rag_reasoning_appended(self, explainer: Explainer) -> None:
        result = _make_verification_result()
        text = explainer._generate_explanation_text(result)
        assert "Strong evidence supports" in text


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
