"""
Pinecone vector store integration for the Disinformation Detection System.

Manages the Pinecone index lifecycle (create, upsert, query, delete) and
provides a similarity search interface for evidence retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exceptions import PineconeConnectionError, PineconeUpsertError, VectorStoreError
from logging_config import get_logger
from retrieval.embeddings import EmbeddingModel

__all__ = ["PineconeVectorStore", "EvidenceDocument", "RetrievedEvidence"]

logger = get_logger(__name__)

_EMBEDDING_DIMENSION = 768
_UPSERT_BATCH_SIZE = 100
_METADATA_TEXT_LIMIT = 512


@dataclass
class EvidenceDocument:
    """A single evidence record ready for indexing in Pinecone.

    Attributes:
        vector_id: Unique identifier for this vector.
        claim_id: Row index from the source dataset.
        claim_text: Original claim text (truncated).
        evidence_text: Full evidence passage.
        label: Ground-truth label ("Supported", "Refuted", or "NEI").
        language: Language code ("EN", "UK", "RU", "ZH").
        explanation: Human explanation for the label (truncated).
        embedding: Dense vector representation.
    """

    vector_id: str
    claim_id: str
    claim_text: str
    evidence_text: str
    label: str
    language: str
    explanation: str
    embedding: list[float]


@dataclass
class RetrievedEvidence:
    """A single result from a Pinecone similarity search.

    Attributes:
        vector_id: Pinecone vector ID.
        score: Cosine similarity score (higher = more similar).
        claim_text: Claim text stored in metadata.
        evidence_text: Evidence passage stored in metadata.
        label: Ground-truth label from the dataset.
        language: Language code.
        explanation: Human explanation.
    """

    vector_id: str
    score: float
    claim_text: str
    evidence_text: str
    label: str
    language: str
    explanation: str


class PineconeVectorStore:
    """Manage a Pinecone serverless index for evidence retrieval.

    Args:
        settings: Application settings with Pinecone configuration.
        embedding_model: EmbeddingModel for encoding queries.
    """

    def __init__(
        self,
        settings: "Settings",  # noqa: F821
        embedding_model: EmbeddingModel,
    ) -> None:
        self._settings = settings
        self._embedder = embedding_model
        self._index: Any | None = None
        self._pc: Any | None = None  # Pinecone client

    def connect(self) -> None:
        """Initialize the Pinecone client and obtain the index handle.

        Creates the index if it does not yet exist.

        Raises:
            PineconeConnectionError: If the API key is missing or the
                connection cannot be established.
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
            self._index = self._pc.Index(cfg.index_name)
            logger.info("Connected to Pinecone index '%s'.", cfg.index_name)
        except PineconeConnectionError:
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
                self._pc.create_index(
                    name=cfg.index_name,
                    dimension=_EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cfg.cloud, region=cfg.region),
                )
                logger.info(
                    "Created Pinecone index '%s' (dim=%d, metric=cosine).",
                    cfg.index_name,
                    _EMBEDDING_DIMENSION,
                )
            else:
                logger.debug("Pinecone index '%s' already exists.", cfg.index_name)
        except Exception as exc:
            raise PineconeConnectionError(
                f"Failed to create/verify index '{cfg.index_name}'",
                original_error=exc,
            ) from exc

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
                        "claim_id": doc.claim_id,
                        "claim_text": doc.claim_text[:_METADATA_TEXT_LIMIT],
                        "evidence_text": doc.evidence_text,
                        "label": doc.label,
                        "language": doc.language,
                        "explanation": doc.explanation[:_METADATA_TEXT_LIMIT],
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
    ) -> list[RetrievedEvidence]:
        """Embed *query_text* and retrieve the top-k most similar evidence passages.

        Args:
            query_text: The claim or query string to search for.
            top_k: Number of results to return. Uses config default if None.
            filter_language: Restrict results to a specific language code
                (e.g. "EN", "UK"). No filter applied when None.

        Returns:
            List of RetrievedEvidence sorted by score descending.

        Raises:
            VectorStoreError: If the index is not connected.
            EmbeddingError: If query embedding fails.
        """
        self._ensure_connected()
        k = top_k if top_k is not None else self._settings.pinecone.top_k

        query_vector = self._embedder.embed_single(query_text)

        query_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": k,
            "include_metadata": True,
        }
        if filter_language:
            query_kwargs["filter"] = {
                "language": {"$eq": filter_language.upper()}
            }

        try:
            response = self._index.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(
                "Pinecone similarity search failed", original_error=exc
            ) from exc

        results: list[RetrievedEvidence] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append(
                RetrievedEvidence(
                    vector_id=match["id"],
                    score=match.get("score", 0.0),
                    claim_text=meta.get("claim_text", ""),
                    evidence_text=meta.get("evidence_text", ""),
                    label=meta.get("label", "NEI"),
                    language=meta.get("language", ""),
                    explanation=meta.get("explanation", ""),
                )
            )

        logger.debug(
            "Similarity search returned %d results for query (len=%d).",
            len(results),
            len(query_text),
        )
        return results

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
