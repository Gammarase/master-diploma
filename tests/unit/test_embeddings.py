"""Unit tests for retrieval.embeddings.EmbeddingModel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exceptions import EmbeddingError
from retrieval.embeddings import EmbeddingModel


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.embeddings.model_name = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    settings.embeddings.device = "cpu"
    settings.embeddings.batch_size = 32
    return settings


@pytest.fixture
def mock_st_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = __import__("numpy").zeros((768,))
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture
def embedding_model(mock_settings: MagicMock, mock_st_model: MagicMock) -> EmbeddingModel:
    em = EmbeddingModel(mock_settings)
    em._model = mock_st_model
    return em


class TestEmbedSingle:
    def test_returns_list_of_floats(self, embedding_model: EmbeddingModel) -> None:
        result = embedding_model.embed_single("Test sentence.")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_raises_on_empty_string(self, embedding_model: EmbeddingModel) -> None:
        with pytest.raises(EmbeddingError):
            embedding_model.embed_single("")

    def test_raises_on_whitespace_only(self, embedding_model: EmbeddingModel) -> None:
        with pytest.raises(EmbeddingError):
            embedding_model.embed_single("   ")

    def test_calls_encode(
        self, embedding_model: EmbeddingModel, mock_st_model: MagicMock
    ) -> None:
        embedding_model.embed_single("Hello world")
        mock_st_model.encode.assert_called_once()


class TestEmbedBatch:
    def test_empty_list_returns_empty(self, embedding_model: EmbeddingModel) -> None:
        result = embedding_model.embed_batch([])
        assert result == []

    def test_returns_one_vector_per_text(
        self, embedding_model: EmbeddingModel, mock_st_model: MagicMock
    ) -> None:
        import numpy as np

        mock_st_model.encode.return_value = np.zeros((3, 768))
        result = embedding_model.embed_batch(["a", "b", "c"])
        assert len(result) == 3

    def test_raises_on_encode_failure(
        self, embedding_model: EmbeddingModel, mock_st_model: MagicMock
    ) -> None:
        mock_st_model.encode.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(EmbeddingError):
            embedding_model.embed_batch(["text"])


class TestEmbedClaim:
    def test_embeds_claim_text(
        self, embedding_model: EmbeddingModel
    ) -> None:
        from claim_extraction.extractor import Claim

        claim = Claim(
            text="Russia launched missiles.",
            original_sentence="Russia launched missiles.",
            sentence_index=0,
            language="en",
        )
        result = embedding_model.embed_claim(claim)
        assert isinstance(result, list)


class TestDimension:
    def test_returns_768(
        self, embedding_model: EmbeddingModel, mock_st_model: MagicMock
    ) -> None:
        assert embedding_model.dimension == 768


class TestLoad:
    def test_raises_embedding_error_on_load_failure(
        self, mock_settings: MagicMock
    ) -> None:
        em = EmbeddingModel(mock_settings)
        with patch(
            "retrieval.embeddings.SentenceTransformer",
            side_effect=OSError("model not found"),
        ):
            with pytest.raises(EmbeddingError):
                em._load()
