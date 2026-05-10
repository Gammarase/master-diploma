"""
Embedding generation for the Disinformation Detection System.

Uses sentence-transformers to produce dense vector representations
of text. The default model supports English, Ukrainian, Russian, and
Chinese — all languages present in RU22Fact.
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
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
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
            vector = model.encode(text, convert_to_numpy=True)
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
        """Return the embedding vector dimension.

        Returns:
            Integer dimension (768 for the default multilingual model).
        """
        model = self._load()
        return model.get_sentence_embedding_dimension()
