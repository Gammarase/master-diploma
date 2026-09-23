"""
Embedding generation for the Disinformation Detection System.

Uses sentence-transformers to produce dense, L2-normalised vector
representations of text. The configured model must support English,
Ukrainian, Russian, and Chinese — all languages present in RU22Fact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from exceptions import EmbeddingError
from logging_config import get_logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from claim_extraction.extractor import Claim

__all__ = ["EmbeddingModel"]

logger = get_logger(__name__)


class EmbeddingModel:
    """Generate sentence embeddings using sentence-transformers.

    Embeddings are returned as plain Python float lists to ensure
    compatibility with the Pinecone client.

    Args:
        settings: Application settings providing model name, device, batch size.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._model: Any | None = None
        self._model_name: str = settings.embeddings.model_name
        self._device: str = settings.embeddings.device
        self._batch_size: int = settings.embeddings.batch_size
        self._dimension: int | None = None

    def _load(self) -> Any:
        """Lazy-load the SentenceTransformer model.

        Returns:
            Loaded SentenceTransformer instance.

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        if self._model is not None:
            return self._model
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
            if str(self._device).startswith("cuda"):
                model.half()
            self._model = model
            logger.info(
                "Loaded embedding model '%s' on device '%s'",
                self._model_name,
                self._device,
            )
            return self._model
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self._model_name}'",
                original_error=exc,
            ) from exc

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: Input text.

        Returns:
            Embedding as a list of floats.

        Raises:
            EmbeddingError: On encoding failure.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            model = self._load()
            vector = model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return vector.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Single embedding failed", original_error=exc
            ) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of strings.

        Args:
            texts: List of input strings (must be non-empty).

        Returns:
            List of embeddings, one per input string.

        Raises:
            EmbeddingError: If any embedding fails.
        """
        if not texts:
            return []
        try:
            model = self._load()
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self._batch_size,
            )
            return [v.tolist() for v in vectors]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Batch embedding failed", original_error=exc
            ) from exc

    def embed_claim(self, claim: "Claim") -> list[float]:
        """Convenience wrapper: embed the text of a Claim.

        Args:
            claim: A Claim dataclass instance.

        Returns:
            Embedding as a list of floats.
        """
        return self.embed_single(claim.text)

    @property
    def dimension(self) -> int:
        """Return the dimension of vectors this model actually produces.

        This is the source of truth for Pinecone index creation. When the
        model does not report a dimension, a probe sentence is encoded.

        Returns:
            Integer dimension (1024 for BAAI/bge-m3).

        Raises:
            EmbeddingError: If the dimension cannot be determined.
        """
        if self._dimension is not None:
            return self._dimension
        model = self._load()
        dim: int | None = None
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if getter is not None:
            reported = getter()
            if isinstance(reported, int) and reported > 0:
                dim = reported
        if dim is None:
            dim = len(self.embed_single("dimension probe"))
        self._dimension = dim
        return dim
