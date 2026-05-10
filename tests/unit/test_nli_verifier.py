"""Unit tests for verification.nli_verifier.NLIVerifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from exceptions import NLIError
from retrieval.vector_store import RetrievedEvidence
from verification.nli_verifier import NLIResult, NLIVerifier


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.verification.nli_model = "cross-encoder/nli-deberta-v3-base"
    return settings


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    model = MagicMock()
    # Returns [contradiction, entailment, neutral] scores
    model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
    model.model.config.label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return model


@pytest.fixture
def verifier(mock_settings: MagicMock, mock_cross_encoder: MagicMock) -> NLIVerifier:
    v = NLIVerifier(mock_settings)
    v._model = mock_cross_encoder
    return v


def _make_evidence(text: str = "Evidence text.", score: float = 0.9) -> RetrievedEvidence:
    return RetrievedEvidence(
        vector_id="test-id",
        score=score,
        claim_text="Claim",
        evidence_text=text,
        label="Supported",
        language="EN",
        explanation="Explanation",
    )


class TestVerifySingle:
    def test_returns_nli_result(self, verifier: NLIVerifier) -> None:
        result = verifier.verify_single("Russia attacked Ukraine.", "Evidence text.")
        assert isinstance(result, NLIResult)

    def test_entailment_is_predicted(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.05, 0.9, 0.05]])
        result = verifier.verify_single("Claim.", "Supporting evidence.")
        assert result.predicted_label == "entailment"
        assert result.entailment_score == pytest.approx(0.9)

    def test_contradiction_is_predicted(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([[0.85, 0.1, 0.05]])
        result = verifier.verify_single("Claim.", "Contradicting evidence.")
        assert result.predicted_label == "contradiction"

    def test_raises_nli_error_on_failure(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.side_effect = RuntimeError("OOM")
        with pytest.raises(NLIError):
            verifier.verify_single("claim", "evidence")


class TestVerifyBatch:
    def test_returns_list_of_nli_results(
        self, verifier: NLIVerifier, mock_cross_encoder: MagicMock
    ) -> None:
        mock_cross_encoder.predict.return_value = np.array([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ])
        evidences = [_make_evidence("E1"), _make_evidence("E2")]
        results = verifier.verify_batch("Claim", evidences)
        assert len(results) == 2

    def test_empty_evidences_returns_empty(self, verifier: NLIVerifier) -> None:
        results = verifier.verify_batch("Claim", [])
        assert results == []


class TestAggregateNliScore:
    def test_all_entailment_returns_high_score(
        self, verifier: NLIVerifier
    ) -> None:
        results = [
            NLIResult("c", "e", 0.9, 0.05, 0.05, "entailment", 0.9),
            NLIResult("c", "e", 0.85, 0.1, 0.05, "entailment", 0.85),
        ]
        score = verifier.aggregate_nli_score(results)
        assert score > 0.8

    def test_all_contradiction_returns_low_score(
        self, verifier: NLIVerifier
    ) -> None:
        results = [
            NLIResult("c", "e", 0.05, 0.9, 0.05, "contradiction", 0.9),
        ]
        score = verifier.aggregate_nli_score(results)
        assert score < 0.2

    def test_neutral_returns_mid_score(self, verifier: NLIVerifier) -> None:
        results = [
            NLIResult("c", "e", 0.1, 0.1, 0.8, "neutral", 0.8),
        ]
        score = verifier.aggregate_nli_score(results)
        assert 0.4 <= score <= 0.6

    def test_empty_results_returns_neutral(self, verifier: NLIVerifier) -> None:
        assert verifier.aggregate_nli_score([]) == 0.5
