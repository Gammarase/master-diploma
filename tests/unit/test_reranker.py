"""Unit tests for retrieval.reranker.Reranker."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from exceptions import VectorStoreError
from retrieval.reranker import Reranker, _sigmoid
from retrieval.evidence import RetrievedEvidence


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.retrieval.reranker_model = "BAAI/bge-reranker-v2-m3"
    settings.retrieval.device = "cpu"
    return settings


def _ev(idx: int, score: float) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=f"v{idx}",
        score=score,
        evidence_text=f"Passage {idx}",
        language="en",
    )


@pytest.fixture
def reranker(mock_settings: MagicMock) -> Reranker:
    r = Reranker(mock_settings)
    r._model = MagicMock()
    return r


class TestSigmoid:
    @pytest.mark.parametrize("x", [-50.0, -2.0, 0.0, 2.0, 50.0])
    def test_matches_formula_and_is_stable(self, x: float) -> None:
        assert _sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)))


class TestRerank:
    def test_sorted_by_sigmoid_score(self, reranker: Reranker) -> None:
        reranker._model.predict.return_value = np.array([-1.0, 3.0, 0.0])
        evidences = [_ev(0, 0.9), _ev(1, 0.8), _ev(2, 0.7)]
        results = reranker.rerank("claim", evidences)
        assert [r.evidence_id for r in results] == ["v1", "v2", "v0"]
        assert results[0].score == pytest.approx(_sigmoid(3.0))
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_keeps_retrieval_score(self, reranker: Reranker) -> None:
        reranker._model.predict.return_value = np.array([5.0])
        result = reranker.rerank("claim", [_ev(0, 0.42)])[0]
        assert result.retrieval_score == pytest.approx(0.42)

    def test_passes_query_passage_pairs(self, reranker: Reranker) -> None:
        reranker._model.predict.return_value = np.array([0.0, 0.0])
        reranker.rerank("the claim", [_ev(0, 0.9), _ev(1, 0.8)])
        pairs = reranker._model.predict.call_args.args[0]
        assert pairs == [("the claim", "Passage 0"), ("the claim", "Passage 1")]

    def test_does_not_mutate_input(self, reranker: Reranker) -> None:
        reranker._model.predict.return_value = np.array([5.0])
        evidences = [_ev(0, 0.3)]
        reranker.rerank("claim", evidences)
        assert evidences[0].score == 0.3

    def test_empty_input(self, reranker: Reranker) -> None:
        assert reranker.rerank("claim", []) == []
        reranker._model.predict.assert_not_called()

    def test_inference_error_wrapped(self, reranker: Reranker) -> None:
        reranker._model.predict.side_effect = RuntimeError("OOM")
        with pytest.raises(VectorStoreError):
            reranker.rerank("claim", [_ev(0, 0.5)])


class TestDisabled:
    @pytest.mark.parametrize("name", [None, ""])
    def test_disabled_when_model_null(
        self, mock_settings: MagicMock, name: str | None
    ) -> None:
        mock_settings.retrieval.reranker_model = name
        r = Reranker(mock_settings)
        assert r.enabled is False
        assert r.model_name is None

    def test_disabled_keeps_score_order(self, mock_settings: MagicMock) -> None:
        mock_settings.retrieval.reranker_model = None
        r = Reranker(mock_settings)
        with patch("sentence_transformers.cross_encoder.CrossEncoder") as ce:
            results = r.rerank("claim", [_ev(0, 0.5), _ev(1, 0.9)])
        ce.assert_not_called()
        assert [x.evidence_id for x in results] == ["v1", "v0"]


class TestLoad:
    def test_lazy_loads_cross_encoder(self, mock_settings: MagicMock) -> None:
        r = Reranker(mock_settings)
        with patch("sentence_transformers.cross_encoder.CrossEncoder") as ce:
            ce.return_value.predict.return_value = np.array([0.0])
            r.rerank("claim", [_ev(0, 0.5)])
            r.rerank("claim", [_ev(0, 0.5)])
        ce.assert_called_once_with("BAAI/bge-reranker-v2-m3", device="cpu")
        ce.return_value.half.assert_not_called()

    def test_half_precision_on_cuda(self, mock_settings: MagicMock) -> None:
        mock_settings.retrieval.device = "cuda"
        r = Reranker(mock_settings)
        with patch("sentence_transformers.cross_encoder.CrossEncoder") as ce:
            r._load()
        ce.return_value.half.assert_called_once()

    def test_load_failure_wrapped(self, mock_settings: MagicMock) -> None:
        r = Reranker(mock_settings)
        with patch(
            "sentence_transformers.cross_encoder.CrossEncoder",
            side_effect=OSError("not found"),
        ):
            with pytest.raises(VectorStoreError, match="bge-reranker"):
                r._load()
