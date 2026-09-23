"""
Pinecone vector store integration for the Disinformation Detection System.

Manages the Pinecone index lifecycle (create, upsert, query, delete) and
provides a similarity search interface for evidence retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from exceptions import PineconeConnectionError, PineconeUpsertError, VectorStoreError
from logging_config import get_logger
from retrieval.embeddings import EmbeddingModel

if TYPE_CHECKING:
    from retrieval.reranker import Reranker

__all__ = ["PineconeVectorStore", "EvidenceDocument", "RetrievedEvidence"]

logger = get_logger(__name__)

_UPSERT_BATCH_SIZE = 100
_METADATA_TEXT_LIMIT = 512


@dataclass
class EvidenceDocument:
    """A single evidence passage ready for indexing in Pinecone.

    Attributes:
        vector_id: Unique identifier, ``ru22fact-{split}-{id}-{lang}-p{n}``.
        claim_id: Dataset ``id`` of the parent record.
        claim_text: Original claim text (truncated).
        passage_text: Cleaned evidence passage (one chunk of the record).
        label: Ground-truth label ("Supported", "Refuted", or "NEI").
        language: Language code ("EN", "UK", "RU", "ZH").
        explanation: Human explanation for the label (truncated).
        embedding: Dense vector representation of ``passage_text``.
        split: Dataset split the record comes from ("train", "test", ...).
        passage_index: Position of this passage within the record.
        date: Claim date from the dataset ("" when unknown).
    """

    vector_id: str
    claim_id: str
    claim_text: str
    passage_text: str
    label: str
    language: str
    explanation: str
    embedding: list[float]
    split: str = ""
    passage_index: int = 0
    date: str = ""


@dataclass
class RetrievedEvidence:
    """A single result from evidence retrieval.

    Attributes:
        vector_id: Pinecone vector ID.
        score: Relevance score: the reranker score when reranking is
            enabled, otherwise the vector similarity.
        claim_text: Claim text of the parent record.
        evidence_text: Passage text used by the verifiers.
        label: Ground-truth label of the parent record.
        language: Language code.
        explanation: Human explanation of the parent record's label.
        claim_id: Dataset ``id`` of the parent record.
        split: Dataset split of the parent record.
        passage_index: Position of the passage within its record.
        date: Claim date of the parent record ("" when unknown).
        vector_score: Raw cosine similarity from Pinecone (defaults to score).
    """

    vector_id: str
    score: float
    claim_text: str
    evidence_text: str
    label: str
    language: str
    explanation: str
    claim_id: str = ""
    split: str = ""
    passage_index: int = 0
    date: str = ""
    vector_score: float | None = None

    def __post_init__(self) -> None:
        if self.vector_score is None:
            self.vector_score = self.score

    @property
    def passage_text(self) -> str:
        """Alias for ``evidence_text`` (the indexed passage)."""
        return self.evidence_text

    @property
    def record_key(self) -> str:
        """Identifier of the parent record, used for the diversity cap."""
        if self.claim_id:
            return f"{self.split}-{self.claim_id}"
        return self.vector_id


class PineconeVectorStore:
    """Manage a Pinecone serverless index for evidence retrieval.

    Args:
        settings: Application settings with Pinecone and retrieval configuration.
        embedding_model: EmbeddingModel for encoding queries.
        reranker: Optional Reranker applied to vector-search candidates.
    """

    def __init__(
        self,
        settings: "Settings",  # noqa: F821
        embedding_model: EmbeddingModel,
        reranker: "Reranker | None" = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedding_model
        self._reranker = reranker
        self._index: Any | None = None
        self._pc: Any | None = None  # Pinecone client

    def connect(self) -> None:
        """Initialize the Pinecone client and obtain the index handle.

        Creates the index if it does not yet exist, then checks that the
        index dimension matches the embedding model.

        Raises:
            PineconeConnectionError: If the API key is missing or the
                connection cannot be established.
            VectorStoreError: If the index dimension differs from the
                embedding model's dimension.
        """
        cfg = self._settings.pinecone
        if not cfg.api_key:
            raise PineconeConnectionError(
                "PINECONE_API_KEY is not set. "
                "Set the environment variable or config value."
            )
        try:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=cfg.api_key)
            self.create_index_if_not_exists()
            self._check_dimension()
            self._index = self._pc.Index(cfg.index_name)
            logger.info("Connected to Pinecone index '%s'.", cfg.index_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise PineconeConnectionError(
                "Failed to connect to Pinecone", original_error=exc
            ) from exc

    def create_index_if_not_exists(self) -> None:
        """Create the serverless Pinecone index if it does not exist.

        Raises:
            PineconeConnectionError: On API failure.
        """
        cfg = self._settings.pinecone
        try:
            from pinecone import ServerlessSpec

            existing = [idx.name for idx in self._pc.list_indexes()]
            if cfg.index_name not in existing:
                dimension = self._embedder.dimension
                self._pc.create_index(
                    name=cfg.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cfg.cloud, region=cfg.region),
                )
                logger.info(
                    "Created Pinecone index '%s' (dim=%d, metric=cosine).",
                    cfg.index_name,
                    dimension,
                )
            else:
                logger.debug("Pinecone index '%s' already exists.", cfg.index_name)
        except Exception as exc:
            raise PineconeConnectionError(
                f"Failed to create/verify index '{cfg.index_name}'",
                original_error=exc,
            ) from exc

    def _check_dimension(self) -> None:
        """Raise VectorStoreError if the index and model dimensions differ."""
        name = self._settings.pinecone.index_name
        description = self._pc.describe_index(name)
        index_dim = getattr(description, "dimension", None)
        if index_dim is None and isinstance(description, dict):
            index_dim = description.get("dimension")
        model_dim = self._embedder.dimension
        if index_dim is not None and int(index_dim) != int(model_dim):
            raise VectorStoreError(
                f"Pinecone index '{name}' has dimension {index_dim}, but the "
                f"embedding model '{self._settings.embeddings.model_name}' "
                f"produces dimension {model_dim}. Use a different "
                "pinecone.index_name or re-create the index."
            )

    def upsert_documents(self, documents: list[EvidenceDocument]) -> int:
        """Batch-upsert evidence documents into the Pinecone index.

        Args:
            documents: List of EvidenceDocument instances to index.

        Returns:
            Total number of vectors successfully upserted.

        Raises:
            VectorStoreError: If the index is not connected.
            PineconeUpsertError: On upsert failure.
        """
        self._ensure_connected()
        total = 0
        for batch_start in range(0, len(documents), _UPSERT_BATCH_SIZE):
            batch = documents[batch_start : batch_start + _UPSERT_BATCH_SIZE]
            vectors = [
                {
                    "id": doc.vector_id,
                    "values": doc.embedding,
                    "metadata": {
                        "split": doc.split,
                        "claim_id": doc.claim_id,
                        "claim_text": doc.claim_text[:_METADATA_TEXT_LIMIT],
                        "passage_text": doc.passage_text,
                        "passage_index": doc.passage_index,
                        "label": doc.label,
                        "language": doc.language,
                        "explanation": doc.explanation[:_METADATA_TEXT_LIMIT],
                        "date": doc.date,
                    },
                }
                for doc in batch
            ]
            try:
                self._index.upsert(vectors=vectors)
                total += len(batch)
                logger.debug(
                    "Upserted batch %d-%d (%d vectors).",
                    batch_start,
                    batch_start + len(batch),
                    len(batch),
                )
            except Exception as exc:
                raise PineconeUpsertError(
                    f"Upsert failed at batch starting index {batch_start}",
                    original_error=exc,
                ) from exc
        logger.info("Upserted %d vectors into Pinecone.", total)
        return total

    def similarity_search(
        self,
        query_text: str,
        top_k: int | None = None,
        filter_language: str | None = None,
        exclude: tuple[str, str] | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve the most relevant evidence passages for *query_text*.

        Steps: fetch ``candidate_k`` candidates by vector similarity (with the
        language and exclusion filters applied server-side), rerank them,
        drop passages below ``retrieval.min_relevance``, keep at most
        ``retrieval.max_passages_per_record`` per parent record, and cut to
        ``top_k``.

        Args:
            query_text: The claim or query string to search for.
            top_k: Number of results to return. Uses config default if None.
            filter_language: Restrict results to a specific language code
                (e.g. "EN", "UK"). No filter applied when None.
            exclude: ``(split, claim_id)`` of a record whose passages must
                not be returned (self-match exclusion).
            candidate_k: Number of vector-search candidates. Uses
                ``retrieval.candidate_k`` if None.

        Returns:
            List of RetrievedEvidence sorted by relevance score descending.
            May be empty.

        Raises:
            VectorStoreError: If the index is not connected or search fails.
            EmbeddingError: If query embedding fails.
        """
        self._ensure_connected()
        rcfg = self._settings.retrieval
        k = top_k if top_k is not None else self._settings.pinecone.top_k
        n_candidates = max(
            k, candidate_k if candidate_k is not None else rcfg.candidate_k
        )

        query_vector = self._embedder.embed_single(query_text)

        query_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": n_candidates,
            "include_metadata": True,
        }
        metadata_filter = self._build_filter(filter_language, exclude)
        if metadata_filter:
            query_kwargs["filter"] = metadata_filter

        try:
            response = self._index.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(
                "Pinecone similarity search failed", original_error=exc
            ) from exc

        candidates = [
            self._match_to_evidence(m) for m in response.get("matches", [])
        ]

        if self._reranker is not None and self._reranker.enabled:
            ranked = self._reranker.rerank(query_text, candidates)
        else:
            ranked = sorted(candidates, key=lambda ev: ev.score, reverse=True)

        results: list[RetrievedEvidence] = []
        per_record: dict[str, int] = {}
        for ev in ranked:
            if ev.score < rcfg.min_relevance:
                continue
            key = ev.record_key
            if per_record.get(key, 0) >= rcfg.max_passages_per_record:
                continue
            per_record[key] = per_record.get(key, 0) + 1
            results.append(ev)
            if len(results) >= k:
                break

        logger.debug(
            "Similarity search: %d candidates -> %d results (query len=%d).",
            len(candidates),
            len(results),
            len(query_text),
        )
        return results

    @staticmethod
    def _build_filter(
        filter_language: str | None, exclude: tuple[str, str] | None
    ) -> dict[str, Any] | None:
        """Combine the language and self-exclusion metadata filters."""
        clauses: list[dict[str, Any]] = []
        if filter_language:
            clauses.append({"language": {"$eq": filter_language.upper()}})
        if exclude is not None:
            split, claim_id = exclude
            clauses.append(
                {
                    "$or": [
                        {"split": {"$ne": str(split)}},
                        {"claim_id": {"$ne": str(claim_id)}},
                    ]
                }
            )
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _match_to_evidence(match: dict[str, Any]) -> RetrievedEvidence:
        """Convert a Pinecone match into RetrievedEvidence.

        Supports both the passage schema (``passage_text``) and the legacy
        whole-record schema (``evidence_text``).
        """
        meta = match.get("metadata", {}) or {}
        score = float(match.get("score", 0.0))
        return RetrievedEvidence(
            vector_id=match["id"],
            score=score,
            claim_text=meta.get("claim_text", ""),
            evidence_text=meta.get("passage_text") or meta.get("evidence_text", ""),
            label=meta.get("label", "NEI"),
            language=meta.get("language", ""),
            explanation=meta.get("explanation", ""),
            claim_id=str(meta.get("claim_id", "")),
            split=str(meta.get("split", "")),
            passage_index=int(meta.get("passage_index", 0) or 0),
            date=str(meta.get("date", "") or ""),
            vector_score=score,
        )

    def delete_index(self) -> None:
        """Delete the Pinecone index. Used for re-indexing and test teardown.

        Raises:
            PineconeConnectionError: If the client is not initialized.
        """
        if self._pc is None:
            raise PineconeConnectionError("Pinecone client not initialized.")
        try:
            self._pc.delete_index(self._settings.pinecone.index_name)
            self._index = None
            logger.info("Deleted Pinecone index '%s'.", self._settings.pinecone.index_name)
        except Exception as exc:
            raise PineconeConnectionError(
                "Failed to delete Pinecone index", original_error=exc
            ) from exc

    @property
    def is_connected(self) -> bool:
        """Return True if the index handle is initialized."""
        return self._index is not None

    def _ensure_connected(self) -> None:
        """Raise VectorStoreError if not connected."""
        if not self.is_connected:
            raise VectorStoreError(
                "PineconeVectorStore is not connected. Call connect() first."
            )
