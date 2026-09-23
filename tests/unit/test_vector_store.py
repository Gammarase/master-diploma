"""Unit tests for retrieval.vector_store.PineconeVectorStore."""

from __future__ import annotations

import dataclasses
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
    settings.embeddings.model_name = "BAAI/bge-m3"
    settings.retrieval.candidate_k = 20
    settings.retrieval.min_relevance = 0.0
    settings.retrieval.max_passages_per_record = 2
    return settings


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0] * 1024
    embedder.dimension = 1024
    return embedder


def _match(
    idx: int,
    score: float,
    claim_id: str | None = None,
    split: str = "test",
    passage_index: int = 0,
) -> dict:
    cid = claim_id if claim_id is not None else str(idx)
    return {
        "id": f"ru22fact-{split}-{cid}-en-p{passage_index}",
        "score": score,
        "metadata": {
            "split": split,
            "claim_id": cid,
            "claim_text": f"Claim {cid}",
            "passage_text": f"Passage {idx}",
            "passage_index": passage_index,
            "label": "Supported",
            "language": "EN",
            "explanation": "Explanation here",
            "date": "2022-09-19",
        },
    }


@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    index = MagicMock()
    index.upsert.return_value = None
    index.query.return_value = {"matches": [_match(0, 0.92)]}
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
        vector_id=f"ru22fact-test-{idx}-en-p0",
        claim_id=str(idx),
        claim_text="Russia launched missiles",
        passage_text="Evidence text",
        label="Supported",
        language="EN",
        explanation="Explanation",
        embedding=[0.0] * 1024,
        split="test",
        passage_index=0,
        date="2022-09-19",
    )


def _mock_pinecone_client(existing: list[str], dimension: int) -> MagicMock:
    pc = MagicMock()
    indexes = []
    for name in existing:
        idx = MagicMock()
        idx.name = name
        indexes.append(idx)
    pc.list_indexes.return_value = indexes
    pc.describe_index.return_value = MagicMock(dimension=dimension)
    return pc


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

    def test_creates_index_with_model_dimension(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        pc = _mock_pinecone_client(existing=[], dimension=1024)
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with patch("pinecone.Pinecone", return_value=pc):
            store.connect()
        assert pc.create_index.call_args.kwargs["dimension"] == 1024
        assert store.is_connected

    def test_dimension_mismatch_raises(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        pc = _mock_pinecone_client(existing=["test-index"], dimension=768)
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with patch("pinecone.Pinecone", return_value=pc):
            with pytest.raises(VectorStoreError) as exc_info:
                store.connect()
        message = str(exc_info.value)
        assert "768" in message and "1024" in message and "test-index" in message
        assert not isinstance(exc_info.value, PineconeConnectionError)
        assert not store.is_connected
        pc.create_index.assert_not_called()

    def test_matching_dimension_connects(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        pc = _mock_pinecone_client(existing=["test-index"], dimension=1024)
        store = PineconeVectorStore(mock_settings, mock_embedder)
        with patch("pinecone.Pinecone", return_value=pc):
            store.connect()
        assert store.is_connected


class TestUpsertDocuments:
    def test_upserts_all_documents(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        docs = [_make_document(i) for i in range(5)]
        total = connected_store.upsert_documents(docs)
        assert total == 5
        mock_pinecone_index.upsert.assert_called()

    def test_metadata_has_passage_schema(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        connected_store.upsert_documents([_make_document(622)])
        vector = mock_pinecone_index.upsert.call_args.kwargs["vectors"][0]
        meta = vector["metadata"]
        assert vector["id"] == "ru22fact-test-622-en-p0"
        assert meta["split"] == "test"
        assert meta["claim_id"] == "622"
        assert meta["passage_text"] == "Evidence text"
        assert meta["passage_index"] == 0
        assert meta["date"] == "2022-09-19"
        assert "evidence_text" not in meta

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


class TestRetrievedEvidence:
    def test_vector_score_defaults_to_score(self) -> None:
        ev = RetrievedEvidence("id", 0.7, "c", "e", "NEI", "EN", "x")
        assert ev.vector_score == 0.7
        assert ev.passage_text == "e"

    def test_record_key_falls_back_to_vector_id(self) -> None:
        ev = RetrievedEvidence("vid", 0.7, "c", "e", "NEI", "EN", "x")
        assert ev.record_key == "vid"


class TestSimilaritySearch:
    def test_returns_retrieved_evidence(
        self, connected_store: PineconeVectorStore
    ) -> None:
        results = connected_store.similarity_search("Russia attacked Ukraine")
        assert len(results) == 1
        ev = results[0]
        assert isinstance(ev, RetrievedEvidence)
        assert ev.score == 0.92
        assert ev.vector_score == 0.92
        assert ev.evidence_text == "Passage 0"
        assert ev.claim_id == "0"
        assert ev.split == "test"
        assert ev.date == "2022-09-19"

    def test_reads_legacy_evidence_text(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        mock_pinecone_index.query.return_value = {
            "matches": [
                {
                    "id": "ru22fact-0-en",
                    "score": 0.9,
                    "metadata": {"claim_id": "0", "evidence_text": "Old schema"},
                }
            ]
        }
        results = connected_store.similarity_search("q")
        assert results[0].evidence_text == "Old schema"

    def test_fetches_candidate_k(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        connected_store.similarity_search("query")
        assert mock_pinecone_index.query.call_args.kwargs["top_k"] == 20

    def test_candidate_k_never_below_top_k(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        connected_store.similarity_search("query", top_k=30, candidate_k=10)
        assert mock_pinecone_index.query.call_args.kwargs["top_k"] == 30

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

    def test_exclusion_filter_sent(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        connected_store.similarity_search("query", exclude=("test", "622"))
        assert mock_pinecone_index.query.call_args.kwargs["filter"] == {
            "$or": [
                {"split": {"$ne": "test"}},
                {"claim_id": {"$ne": "622"}},
            ]
        }

    def test_exclusion_combined_with_language(
        self, connected_store: PineconeVectorStore, mock_pinecone_index: MagicMock
    ) -> None:
        connected_store.similarity_search(
            "query", filter_language="en", exclude=("test", 622)
        )
        flt = mock_pinecone_index.query.call_args.kwargs["filter"]
        assert flt["$and"][0] == {"language": {"$eq": "EN"}}
        assert flt["$and"][1]["$or"][1] == {"claim_id": {"$ne": "622"}}

    def test_raises_on_query_failure(
        self,
        connected_store: PineconeVectorStore,
        mock_pinecone_index: MagicMock,
    ) -> None:
        mock_pinecone_index.query.side_effect = Exception("timeout")
        with pytest.raises(VectorStoreError):
            connected_store.similarity_search("query")


def _reranker_preferring(best_idx: int) -> MagicMock:
    """Reranker mock that scores candidate *best_idx* highest."""
    reranker = MagicMock()
    reranker.enabled = True

    def rerank(query: str, evidences: list[RetrievedEvidence]):
        rescored = []
        for ev in evidences:
            idx = int(ev.evidence_text.split()[-1])
            score = 0.99 if idx == best_idx else 0.9 - idx * 0.01
            rescored.append(dataclasses.replace(ev, score=score))
        return sorted(rescored, key=lambda e: e.score, reverse=True)

    reranker.rerank.side_effect = rerank
    return reranker


class TestRerankingAndFiltering:
    """Scenarios from specs/evidence-retrieval/spec.md."""

    def _store(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        matches: list[dict],
        reranker: MagicMock | None = None,
    ) -> PineconeVectorStore:
        index = MagicMock()
        index.query.return_value = {"matches": matches}
        store = PineconeVectorStore(mock_settings, mock_embedder, reranker=reranker)
        store._index = index
        return store

    def test_reranked_order_returned(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        matches = [_match(i, 0.9 - i * 0.01) for i in range(20)]
        store = self._store(
            mock_settings, mock_embedder, matches, _reranker_preferring(14)
        )
        results = store.similarity_search("claim")
        assert results[0].evidence_text == "Passage 14"
        assert results[0].score == pytest.approx(0.99)
        assert results[0].vector_score == pytest.approx(0.9 - 14 * 0.01)
        assert len(results) <= 5

    def test_reranker_disabled_uses_vector_order(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        matches = [_match(i, s) for i, s in enumerate([0.5, 0.9, 0.7, 0.8, 0.6, 0.4])]
        reranker = MagicMock()
        reranker.enabled = False
        store = self._store(mock_settings, mock_embedder, matches, reranker)
        results = store.similarity_search("claim")
        assert [r.score for r in results] == [0.9, 0.8, 0.7, 0.6, 0.5]
        reranker.rerank.assert_not_called()

    def test_no_relevant_evidence_returns_empty(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        mock_settings.retrieval.min_relevance = 0.95
        matches = [_match(i, 0.9 - i * 0.01) for i in range(20)]
        store = self._store(
            mock_settings, mock_embedder, matches, _reranker_preferring(-1)
        )
        assert store.similarity_search("claim") == []

    def test_min_relevance_drops_low_scores(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        mock_settings.retrieval.min_relevance = 0.75
        matches = [_match(i, s) for i, s in enumerate([0.9, 0.8, 0.7, 0.6])]
        store = self._store(mock_settings, mock_embedder, matches)
        results = store.similarity_search("claim")
        assert [r.score for r in results] == [0.9, 0.8]

    def test_diversity_cap(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        # Top 5 all come from record test-100; the rest from other records.
        matches = [
            _match(i, 0.99 - i * 0.01, claim_id="100", passage_index=i)
            for i in range(5)
        ] + [_match(i, 0.9 - i * 0.01) for i in range(5, 10)]
        store = self._store(mock_settings, mock_embedder, matches)
        results = store.similarity_search("claim")
        assert len(results) == 5
        assert sum(1 for r in results if r.claim_id == "100") == 2
        assert [r.claim_id for r in results[2:]] == ["5", "6", "7"]

    def test_same_id_different_split_not_capped_together(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        matches = [
            _match(0, 0.9, claim_id="7", split="test", passage_index=0),
            _match(1, 0.8, claim_id="7", split="test", passage_index=1),
            _match(2, 0.7, claim_id="7", split="train", passage_index=0),
        ]
        mock_settings.retrieval.max_passages_per_record = 1
        store = self._store(mock_settings, mock_embedder, matches)
        results = store.similarity_search("claim")
        assert [(r.split, r.claim_id) for r in results] == [("test", "7"), ("train", "7")]

    def test_empty_candidates(
        self, mock_settings: MagicMock, mock_embedder: MagicMock
    ) -> None:
        reranker = _reranker_preferring(0)
        store = self._store(mock_settings, mock_embedder, [], reranker)
        assert store.similarity_search("claim") == []


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
