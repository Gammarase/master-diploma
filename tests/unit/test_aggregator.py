"""Unit tests for verification.aggregator (decide + ResultAggregator)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrieval.evidence import RetrievedEvidence
from verification.aggregator import (
    REASON_CONFLICT,
    REASON_LOW_SCORE_MARGIN,
    REASON_NO_EVIDENCE,
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    DecisionConfig,
    ResultAggregator,
    VerificationResult,
    decide,
    is_decisive,
)
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict

CFG = DecisionConfig(
    nli_weight=0.3,
    rag_weight=0.7,
    threshold_disinformation=0.35,
    threshold_confirmed=0.65,
    decisiveness_margin=0.15,
)


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.verification.nli_weight = 0.3
    settings.verification.rag_weight = 0.7
    settings.verification.thresholds.disinformation = 0.35
    settings.verification.thresholds.confirmed = 0.65
    settings.verification.decisiveness_margin = 0.15
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


def _evidences(n: int = 2) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence(evidence_id=f"v{i}", score=0.9, evidence_text=f"e{i}")
        for i in range(n)
    ]


def _nli(entail: float, contra: float) -> NLIResult:
    label = "entailment" if entail >= contra else "contradiction"
    return NLIResult("c", "e", entail, contra, 1 - entail - contra, label, max(entail, contra))


# ─── Spec scenarios: specs/verdict-aggregation/spec.md ───────────────────────

class TestDecisiveness:
    def test_neutral_nli_is_not_decisive(self) -> None:
        assert is_decisive(0.55, 0.15) is False

    def test_far_score_is_decisive(self) -> None:
        assert is_decisive(0.10, 0.15) is True
        assert is_decisive(0.90, 0.15) is True

    def test_boundary_is_not_decisive(self) -> None:
        assert is_decisive(0.6, 0.1 + 1e-9) is False


class TestAdaptiveWeighting:
    def test_nli_undecided_rag_confident(self) -> None:
        d = decide(0.50, 0.10, 3, CFG)
        assert d.final_score == pytest.approx(0.10)
        assert d.effective_weights == {"nli": 0.0, "rag": 1.0}

    def test_rag_undecided_nli_confident(self) -> None:
        d = decide(0.90, 0.50, 3, CFG)
        assert d.final_score == pytest.approx(0.90)
        assert d.effective_weights == {"nli": 1.0, "rag": 0.0}

    def test_both_decisive_and_agreeing(self) -> None:
        d = decide(0.80, 0.90, 3, CFG)
        assert d.final_score == pytest.approx(0.87, abs=0.001)
        assert d.verdict == VERDICT_CONFIRMED
        assert d.reason is None

    def test_neither_decisive_uses_configured_weights(self) -> None:
        d = decide(0.40, 0.60, 3, CFG)
        assert d.effective_weights == pytest.approx({"nli": 0.3, "rag": 0.7})
        assert d.final_score == pytest.approx(0.3 * 0.4 + 0.7 * 0.6)

    def test_weights_are_normalised(self) -> None:
        cfg = DecisionConfig(3.0, 7.0, 0.35, 0.65, 0.15)
        d = decide(0.80, 0.90, 3, cfg)
        assert d.final_score == pytest.approx(0.87, abs=0.001)
        assert sum(d.effective_weights.values()) == pytest.approx(1.0)

    def test_zero_weights_fall_back_to_equal(self) -> None:
        cfg = DecisionConfig(0.0, 0.0, 0.35, 0.65, 0.15)
        d = decide(0.40, 0.60, 3, cfg)
        assert d.effective_weights == {"nli": 0.5, "rag": 0.5}

    def test_score_clamped(self) -> None:
        assert 0.0 <= decide(0.0, 0.0, 1, CFG).final_score <= 1.0
        assert decide(1.0, 1.0, 1, CFG).final_score == pytest.approx(1.0)


class TestConflictRule:
    def test_opposite_decisive_components(self) -> None:
        d = decide(0.10, 0.90, 3, CFG)
        assert d.verdict == VERDICT_UNCERTAIN
        assert d.reason == REASON_CONFLICT

    def test_confident_rag_with_neutral_nli_is_not_conflict(self) -> None:
        d = decide(0.50, 0.05, 3, CFG)
        assert d.verdict == VERDICT_DISINFORMATION
        assert d.final_score == pytest.approx(0.05)
        assert d.reason is None

    def test_large_gap_same_side_is_not_conflict(self) -> None:
        # |0.66 - 0.99| is large, but both point towards "true".
        d = decide(0.66, 0.99, 3, CFG)
        assert d.verdict == VERDICT_CONFIRMED


class TestThresholds:
    def test_middle_score_low_margin(self) -> None:
        d = decide(0.50, 0.50, 3, CFG)
        assert d.final_score == pytest.approx(0.50)
        assert d.verdict == VERDICT_UNCERTAIN
        assert d.reason == REASON_LOW_SCORE_MARGIN

    def test_empty_evidence(self) -> None:
        d = decide(0.95, 0.95, 0, CFG)
        assert d.verdict == VERDICT_UNCERTAIN
        assert d.reason == REASON_NO_EVIDENCE

    @pytest.mark.parametrize(
        "nli,rag,expected",
        [
            (0.10, 0.10, VERDICT_DISINFORMATION),
            (0.90, 0.90, VERDICT_CONFIRMED),
            (0.50, 0.65, VERDICT_CONFIRMED),   # decisive RAG exactly at threshold
            (0.50, 0.349, VERDICT_DISINFORMATION),
            (0.50, 0.36, VERDICT_UNCERTAIN),   # not decisive, mid band
        ],
    )
    def test_threshold_mapping(self, nli: float, rag: float, expected: str) -> None:
        assert decide(nli, rag, 2, CFG).verdict == expected

    def test_definite_verdicts_have_no_reason(self) -> None:
        assert decide(0.1, 0.1, 2, CFG).reason is None
        assert decide(0.9, 0.9, 2, CFG).reason is None


# ─── ResultAggregator wrapper ────────────────────────────────────────────────

class TestResultAggregator:
    def test_config_from_settings(self, aggregator: ResultAggregator) -> None:
        assert aggregator.config == CFG

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
        assert result.uncertainty_reason == REASON_NO_EVIDENCE

    def test_conflict_forces_uncertain(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        nli_results = [_nli(0.0, 0.95)]  # nli score 0.025
        rag = _make_rag("SUPPORTED", 0.9)
        result = aggregator.aggregate(mock_claim, nli_results, rag, _evidences())
        assert result.verdict == VERDICT_UNCERTAIN
        assert result.uncertainty_reason == REASON_CONFLICT

    def test_neutral_nli_does_not_block_confident_rag(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        nli_results = [_nli(0.1, 0.1)]  # nli score 0.5
        rag = _make_rag("REFUTED", 0.9)  # rag score 0.1
        result = aggregator.aggregate(mock_claim, nli_results, rag, _evidences())
        assert result.verdict == VERDICT_DISINFORMATION
        assert result.final_score == pytest.approx(0.1)
        assert result.effective_weights == {"nli": 0.0, "rag": 1.0}
        assert result.uncertainty_reason is None

    def test_uses_max_abs_nli_aggregation(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        nli_results = [_nli(0.1, 0.05), _nli(0.05, 0.85), _nli(0.15, 0.05)]
        result = aggregator.aggregate(
            mock_claim, nli_results, _make_rag("INSUFFICIENT_EVIDENCE", 0.5), _evidences(3)
        )
        assert result.nli_score == pytest.approx(0.10)

    def test_component_weights_stored(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        result = aggregator.aggregate(mock_claim, [], _make_rag(), [])
        assert result.component_weights == {"nli": 0.3, "rag": 0.7}

    def test_confirmed_verdict_with_high_scores(
        self, aggregator: ResultAggregator, mock_claim: MagicMock
    ) -> None:
        nli_results = [_nli(0.9, 0.05)]
        rag = _make_rag("SUPPORTED", 0.9)
        result = aggregator.aggregate(mock_claim, nli_results, rag, _evidences())
        assert result.verdict == VERDICT_CONFIRMED
        assert result.uncertainty_reason is None
