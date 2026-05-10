"""Unit tests for retrieval.vector_store.PineconeVectorStore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exceptions import (
    PineconeConnectionError,
    PineconeUpsertError,
    VectorStoreError,
)
from retrieval.vector_store import (
    EvidenceDocument,
    PineconeVectorStore,
    RetrievedEvidence,
)


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.pinecone.api_key = "test-key"
    settings.pinecone.index_name = "test-index"
    settings.pinecone.cloud = "aws"
    settings.pinecone.region = "us-east-1"
    settings.pinecone.top_k = 5
    return settings


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0] * 768
    return embedder


@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    index = MagicMock()
    index.upsert.return_value = None
    index.query.return_value = {
        "matches": [
            {
                "id": "ru22fact-0-en",
                "score": 0.92,
                "metadata": {
                    "claim_id": "0",
                    "claim_text": "Russia launched missiles",
                    "evidence_text": "Evidence text here",
                    "label": "Supported",
                    "language": "EN",
                    "explanation": "Explanation here",
                },
            }
        ]
    }
    return index


@pytest.fixture
def connected_store(
    mock_settings: MagicMock,
    mock_embedder: MagicMock,
    mock_pinecone_index: MagicMock,
) -> PineconeVectorStore:
    store = PineconeVectorStore(mock_settings, mock_embedder)
    store._index = mock_pinecone_index
    store._pc = MagicMock()
    return store


def _make_document(idx: int = 0) -> EvidenceDocument:
    return EvidenceDocument(
        vector_id=f"ru22fact-{idx}-en",
        claim_id=str(idx),
        claim_text="Russia launched missiles",
        evidence_text="Evidence text",
        label="Supported",
        language="EN",
        explanation="Explanation",
        embedding=[0.0] * 768,
    )


class TestConnect:
    def test_raises_if_api_key_empty(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        mock_settings.pinecone.api_key = ""
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with pytest.raises(PineconeConnectionError, match="PINECONE_API_KEY"):
            store.connect()

    def test_raises_on_pinecone_error(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with patch("pinecone.Pinecone", side_effect=Exception("network error")):
            with pytest.raises(PineconeConnectionError):
                store.connect()


class TestUpsertDocuments:
    def test_upserts_all_documents(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        docs = [_make_document(i) for i in range(5)]
        total = connected_store.upsert_documents(docs)
        assert total == 5
        mock_pinecone_index.upsert.assert_called()

    def test_raises_when_not_connected(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with pytest.raises(VectorStoreError):
            store.upsert_documents([_make_document()])

    def test_raises_on_upsert_failure(
        self,
        connected_store: PineconeVectorStore,
        mock_pinecone_index: MagicMock,
    ) -> None:
        mock_pinecone_index.upsert.side_effect = Exception("Pinecone down")
        with pytest.raises(PineconeUpsertError):
            connected_store.upsert_documents([_make_document()])

    def test_batches_over_100(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        docs = [_make_document(i) for i in range(250)]
        connected_store.upsert_documents(docs)
        # Should call upsert 3 times: batches of 100, 100, 50
        assert mock_pinecone_index.upsert.call_count == 3


class TestSimilaritySearch:
    def test_returns_retrieved_evidence(
        self, connected_store: PineconeVectorStore
    ) -> None:
        results = connected_store.similarity_search("Russia attacked Ukraine")
        assert len(results) == 1
        assert isinstance(results[0], RetrievedEvidence)
        assert results[0].score == 0.92

    def test_raises_when_not_connected(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with pytest.raises(VectorStoreError):
            store.similarity_search("query")

    def test_applies_language_filter(
        self,
        connected_store: PineconeVectorStore,
        mock_pinecone_index: MagicMock,
    ) -> None:
        connected_store.similarity_search("query", filter_language="UK")
        call_kwargs = mock_pinecone_index.query.call_args[1]
        assert call_kwargs["filter"] == {"language": {"$eq": "UK"}}

    def test_no_filter_when_language_none(
        self,
        connected_store: PineconeVectorStore,
        mock_pinecone_index: MagicMock,
    ) -> None:
        connected_store.similarity_search("query", filter_language=None)
        call_kwargs = mock_pinecone_index.query.call_args[1]
        assert "filter" not in call_kwargs

    def test_raises_on_query_failure(
        self,
        connected_store: PineconeVectorStore,
        mock_pinecone_index: MagicMock,
    ) -> None:
        mock_pinecone_index.query.side_effect = Exception("timeout")
        with pytest.raises(VectorStoreError):
            connected_store.similarity_search("query")


class TestIsConnected:
    def test_false_when_not_connected(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        store = PineconeVectorStore(mock_settings, mock_embedder)
        assert store.is_connected is False

    def test_true_when_connected(
        self, connected_store: PineconeVectorStore
    ) -> None:
        assert connected_store.is_connected is True
