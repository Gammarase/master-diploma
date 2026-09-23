"""Unit tests for verification.stance_detector.StanceDetector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verification.stance_detector import StanceDetector, StanceResult


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.claim_extraction.classifier_model = "typeform/distilbert-base-uncased-mnli"
    return settings


@pytest.fixture
def mock_pipeline() -> MagicMock:
    clf = MagicMock()
    clf.return_value = {
        "labels": ["agree", "disagree", "discuss", "unrelated"],
        "scores": [0.75, 0.1, 0.1, 0.05],
    }
    return clf


@pytest.fixture
def detector(mock_settings: MagicMock, mock_pipeline: MagicMock) -> StanceDetector:
    d = StanceDetector(mock_settings)
    d._pipeline = mock_pipeline
    return d


class TestDetect:
    def test_returns_stance_result(self, detector: StanceDetector) -> None:
        result = detector.detect("Russia attacked Ukraine.", "Evidence supports the claim.")
        assert isinstance(result, StanceResult)
        assert result.stance == "agree"
        assert result.confidence == pytest.approx(0.75)

    def test_stance_is_top_label(
        self, detector: StanceDetector, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.return_value = {
            "labels": ["disagree", "agree", "discuss", "unrelated"],
            "scores": [0.8, 0.1, 0.05, 0.05],
        }
        result = detector.detect("Claim", "Contradicting evidence.")
        assert result.stance == "disagree"

    def test_phrases_mapped_to_stance_labels(
        self, detector: StanceDetector, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.return_value = {
            "labels": ["contradicts", "supports", "is unrelated to"],
            "scores": [0.7, 0.2, 0.1],
        }
        result = detector.detect("Claim", "Contradicting evidence.")
        assert result.stance == "disagree"
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["candidate_labels"] == ["supports", "contradicts", "is unrelated to"]

    def test_braces_in_claim_are_neutralised(
        self, detector: StanceDetector, mock_pipeline: MagicMock
    ) -> None:
        detector.detect("Budget {2023} cut", "Evidence.")
        template = mock_pipeline.call_args.kwargs["hypothesis_template"]
        assert template == "This text {} the claim that Budget (2023) cut"
        template.format("supports")  # must not raise


class TestToNliCompatibleScore:
    @pytest.fixture
    def detector_bare(self, mock_settings: MagicMock) -> StanceDetector:
        return StanceDetector(mock_settings)

    def test_agree_returns_confidence(self, detector_bare: StanceDetector) -> None:
        score = detector_bare.to_nli_compatible_score("agree", 0.8)
        assert score == pytest.approx(0.8)

    def test_disagree_returns_inverted(self, detector_bare: StanceDetector) -> None:
        score = detector_bare.to_nli_compatible_score("disagree", 0.7)
        assert score == pytest.approx(0.3)

    def test_discuss_returns_neutral(self, detector_bare: StanceDetector) -> None:
        assert detector_bare.to_nli_compatible_score("discuss", 0.9) == 0.5

    def test_unrelated_returns_neutral(self, detector_bare: StanceDetector) -> None:
        assert detector_bare.to_nli_compatible_score("unrelated", 0.6) == 0.5
