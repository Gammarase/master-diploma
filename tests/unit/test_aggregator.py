"""Unit tests for verification.aggregator.ResultAggregator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verification.aggregator import (
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    ResultAggregator,
    VerificationResult,
)
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.verification.nli_weight = 0.4
    settings.verification.rag_weight = 0.6
    settings.verification.thresholds.disinformation = 0.3
    settings.verification.thresholds.confirmed = 0.7
    settings.verification.disagreement_delta = 0.3
    return settings


@pytest.fixture
def aggregator(mock_settings: MagicMock) -> ResultAggregator:
    return ResultAggregator(mock_settings)


@pytest.fixture
def mock_claim() -> MagicMock:
    claim = MagicMock()
    claim.text = "Russia launched missiles."
    return claim


def _make_rag(verdict: str = "SUPPORTED", confidence: float = 0.8) -> RAGVerdict:
    return RAGVerdict(verdict=verdict, confidence=confidence, reasoning="Test.")


class TestComputeFinalScore:
    def test_weighted_combination(self, aggregator: ResultAggregator) -> None:
        score = aggregator._compute_final_score(0.5, 0.5)
        # 0.4 * 0.5 + 0.6 * 0.5 = 0.5
        assert score == pytest.approx(0.5)

    def test_clamped_at_one(self, aggregator: ResultAggregator) -> None:
        assert aggregator._compute_final_score(1.0, 1.0) == pytest.approx(1.0)

    def test_clamped_at_zero(self, aggregator: ResultAggregator) -> None:
        assert aggregator._compute_final_score(0.0, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "nli,rag,expected_verdict",
    [
        (0.1, 0.1, VERDICT_DISINFORMATION),   # final ≈ 0.1 < 0.3
        (0.5, 0.5, VERDICT_UNCERTAIN),         # final = 0.5 in [0.3, 0.7)
        (0.9, 0.9, VERDICT_CONFIRMED),         # final ≈ 0.9 ≥ 0.7
        (0.2, 0.2, VERDICT_DISINFORMATION),   # final ≈ 0.2 < 0.3
        (0.7, 0.7, VERDICT_CONFIRMED),         # final = 0.7 ≥ 0.7
    ],
)
def test_apply_threshold(
    aggregator: ResultAggregator, nli: float, rag: float, expected_verdict: str
) -> None:
    score = aggregator._compute_final_score(nli, rag)
    verdict = aggregator._apply_threshold(score)
    assert verdict == expected_verdict


class TestDetectDisagreement:
    def test_no_disagreement(self, aggregator: ResultAggregator) -> None:
        assert aggregator._detect_disagreement(0.5, 0.5) is False

    def test_disagreement_detected(self, aggregator: ResultAggregator) -> None:
        assert aggregator._detect_disagreement(0.1, 0.9) is True

    def test_boundary_not_disagreement(self, aggregator: ResultAggregator) -> None:
        # |0.4 - 0.7| = 0.3, not strictly greater
        assert aggregator._detect_disagreement(0.4, 0.7) is False

    def test_just_over_boundary(self, aggregator: ResultAggregator) -> None:
        assert aggregator._detect_disagreement(0.39, 0.71) is True


class TestAggregate:
    def test_returns_verification_result(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        result = aggregator.aggregate(
            claim=mock_claim,
            nli_results=[],
            rag_verdict=_make_rag("SUPPORTED", 0.8),
            retrieved_evidences=[],
        )
        assert isinstance(result, VerificationResult)

    def test_disagreement_forces_uncertain(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        # NLI says 0.05 (disinformation), RAG says 0.9 (confirmed) → disagree
        nli_results = [
            NLIResult("c", "e", 0.0, 0.95, 0.05, "contradiction", 0.95)
        ]
        rag = _make_rag("SUPPORTED", 0.9)
        result = aggregator.aggregate(mock_claim, nli_results, rag, [])
        assert result.verdict == VERDICT_UNCERTAIN

    def test_component_weights_stored(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        result = aggregator.aggregate(mock_claim, [], _make_rag(), [])
        assert result.component_weights == {"nli": 0.4, "rag": 0.6}

    def test_confirmed_verdict_with_high_scores(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        nli_results = [
            NLIResult("c", "e", 0.9, 0.05, 0.05, "entailment", 0.9)
        ]
        rag = _make_rag("SUPPORTED", 0.9)
        result = aggregator.aggregate(mock_claim, nli_results, rag, [])
        assert result.verdict == VERDICT_CONFIRMED
