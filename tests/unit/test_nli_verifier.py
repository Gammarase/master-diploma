"""Unit tests for verification.nli_verifier.NLIVerifier."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from exceptions import NLIError
from retrieval.evidence import RetrievedEvidence
from verification.nli_verifier import NLIResult, NLIVerifier

# mDeBERTa-xnli order: entailment, neutral, contradiction
_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.verification.nli_model = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    return settings


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
    model.config = SimpleNamespace(label2id=dict(_LABEL2ID))
    return model


@pytest.fixture
def verifier(mock_settings: MagicMock, mock_cross_encoder: MagicMock) -> NLIVerifier:
    v = NLIVerifier(mock_settings)
    v._model = mock_cross_encoder
    return v


def _make_evidence(
    text: str = "Evidence text.", score: float = 0.9, trust: float = 1.0
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="test-id",
        score=score,
        evidence_text=text,
        language="en",
        trust=trust,
    )


def _result(support: float, trust: float = 1.0) -> NLIResult:
    entail = max(0.0, support)
    contra = max(0.0, -support)
    return NLIResult(
        "c", "e", entail, contra, 1 - entail - contra, "neutral", 0.5, trust=trust
    )


class TestVerifySingle:
    def test_returns_nli_result(self, verifier: NLIVerifier) -> None:
        result = verifier.verify_single("Russia attacked Ukraine.", "Evidence text.")
        assert isinstance(result, NLIResult)

    def test_evidence_is_premise_claim_is_hypothesis(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        verifier.verify_single("The claim.", "The evidence.")
        pairs = mock_cross_encoder.predict.call_args.args[0]
        assert pairs == [("The evidence.", "The claim.")]

    def test_entailment_is_predicted(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.9, 0.05, 0.05]])
        result = verifier.verify_single("Claim.", "Supporting evidence.")
        assert result.predicted_label == "entailment"
        assert result.entailment_score == pytest.approx(0.9)
        assert result.support == pytest.approx(0.85)

    def test_contradiction_is_predicted(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.1, 0.05, 0.85]])
        result = verifier.verify_single("Claim.", "Contradicting evidence.")
        assert result.predicted_label == "contradiction"
        assert result.support == pytest.approx(-0.75)

    def test_label_order_follows_label2id(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.config.label2id = {
            "contradiction": 0, "entailment": 1, "neutral": 2
        }
        mock_cross_encoder.predict.return_value = np.array([[0.7, 0.2, 0.1]])
        result = verifier.verify_single("Claim.", "Evidence.")
        assert result.predicted_label == "contradiction"
        assert result.contradiction_score == pytest.approx(0.7)

    def test_raises_nli_error_on_failure(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.side_effect = RuntimeError("OOM")
        with pytest.raises(NLIError):
            verifier.verify_single("claim", "evidence")


class TestLabelMap:
    def test_reads_nested_model_config(self) -> None:
        model = SimpleNamespace(
            config=None,
            model=SimpleNamespace(config=SimpleNamespace(label2id={"ENTAILMENT": 2, "Neutral": 1, "contradiction": 0})),
        )
        assert NLIVerifier._get_label_map(model) == {
            "entailment": 2, "neutral": 1, "contradiction": 0
        }

    def test_missing_label2id_raises(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(label2id=None))
        with pytest.raises(NLIError, match="label2id"):
            NLIVerifier._get_label_map(model)

    def test_generic_labels_raise(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(label2id={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2})
        )
        with pytest.raises(NLIError):
            NLIVerifier._get_label_map(model)

    def test_verify_raises_without_label2id(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.config = SimpleNamespace(label2id={})
        mock_cross_encoder.model = SimpleNamespace(config=SimpleNamespace(label2id={}))
        with pytest.raises(NLIError):
            verifier.verify_single("claim", "evidence")


class TestVerifyBatch:
    def test_returns_results_in_passage_order(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.1, 0.7],
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
        ])
        evidences = [_make_evidence(f"E{i}") for i in range(4)]
        results = verifier.verify_batch("Claim", evidences)
        assert [r.evidence_text for r in results] == ["E0", "E1", "E2", "E3"]
        assert [r.predicted_label for r in results] == [
            "entailment", "contradiction", "neutral", "neutral"
        ]
        for r in results:
            total = r.entailment_score + r.neutral_score + r.contradiction_score
            assert total == pytest.approx(1.0, abs=0.01)

    def test_pairs_are_evidence_then_claim(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.8, 0.1, 0.1]] * 2)
        verifier.verify_batch("Claim", [_make_evidence("E1"), _make_evidence("E2")])
        pairs = mock_cross_encoder.predict.call_args.args[0]
        assert pairs == [("E1", "Claim"), ("E2", "Claim")]

    def test_empty_evidences_returns_empty(self, verifier: NLIVerifier) -> None:
        results = verifier.verify_batch("Claim", [])
        assert results == []

    def test_results_carry_trust(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.8, 0.1, 0.1]] * 3)
        evidences = [
            _make_evidence("E0", trust=1.0),
            _make_evidence("E1", trust=0.8),
            _make_evidence("E2", trust=0.3),
        ]
        results = verifier.verify_batch("Claim", evidences)
        assert [r.trust for r in results] == [1.0, 0.8, 0.3]

    def test_verify_single_uses_full_trust(self, verifier: NLIVerifier) -> None:
        assert verifier.verify_single("Claim", "E").trust == 1.0


class TestAggregateNliScore:
    def test_is_static(self) -> None:
        assert NLIVerifier.aggregate_nli_score([]) == 0.5

    def test_one_strongly_contradicting_passage(self) -> None:
        results = [_result(0.05), _result(-0.80), _result(0.10)]
        assert NLIVerifier.aggregate_nli_score(results) == pytest.approx(0.10, abs=0.001)

    def test_strong_entailment(self) -> None:
        results = [_result(0.9), _result(0.2)]
        assert NLIVerifier.aggregate_nli_score(results) == pytest.approx(0.95)

    def test_neutral_returns_mid_score(self) -> None:
        results = [NLIResult("c", "e", 0.1, 0.1, 0.8, "neutral", 0.8)]
        assert NLIVerifier.aggregate_nli_score(results) == pytest.approx(0.5)

    def test_empty_results_returns_neutral(self) -> None:
        assert NLIVerifier.aggregate_nli_score([]) == 0.5

    def test_low_trust_passage_cannot_dominate(self) -> None:
        results = [_result(-0.90, trust=0.3), _result(0.60, trust=0.95)]
        assert NLIVerifier.aggregate_nli_score(results) == pytest.approx(0.785, abs=0.001)

    def test_trust_scales_toward_neutral(self) -> None:
        results = [_result(0.90, trust=0.8)]
        assert NLIVerifier.aggregate_nli_score(results) == pytest.approx(0.86, abs=0.001)

    def test_default_trust_is_one(self) -> None:
        assert NLIResult("c", "e", 0.5, 0.1, 0.4, "entailment", 0.5).trust == 1.0

    def test_support_defaults_from_probabilities(self) -> None:
        r = NLIResult("c", "e", 0.7, 0.2, 0.1, "entailment", 0.7)
        assert r.support == pytest.approx(0.5)
