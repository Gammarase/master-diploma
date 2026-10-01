"""
Cross-encoder reranking for the Disinformation Detection System.

Rescores candidate passages against the claim with a multilingual
cross-encoder (default ``BAAI/bge-reranker-v2-m3``). Scores are the sigmoid
of the model logits, so they lie in [0, 1] and can be compared with
``retrieval.min_relevance``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from exceptions import VectorStoreError
from logging_config import get_logger

if TYPE_CHECKING:
    from retrieval.evidence import RetrievedEvidence

__all__ = ["Reranker"]

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    """Lazy-loaded cross-encoder reranker.

    Disabled when ``retrieval.reranker_model`` is null; ``rerank`` then just
    orders candidates by their existing score.

    Args:
        settings: Application settings providing the reranker model and device.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._model_name: str | None = settings.retrieval.reranker_model or None
        self._device: str = settings.retrieval.device
        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        """True when a reranker model is configured."""
        return self._model_name is not None

    @property
    def model_name(self) -> str | None:
        """Configured reranker model name, or None when disabled."""
        return self._model_name

    def _load(self) -> Any:
        """Lazy-load the CrossEncoder (fp16 on CUDA).

        Raises:
            VectorStoreError: If the model cannot be loaded.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            model = CrossEncoder(self._model_name, device=self._device)
            if str(self._device).startswith("cuda"):
                model.half()
            self._model = model
            logger.info(
                "Loaded reranker '%s' on device '%s'", self._model_name, self._device
            )
            return self._model
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load reranker model '{self._model_name}'",
                original_error=exc,
            ) from exc

    def rerank(
        self, query: str, evidences: list["RetrievedEvidence"]
    ) -> list["RetrievedEvidence"]:
        """Rescore *evidences* against *query* and sort by the new score.

        Each returned item keeps its pre-rerank score in ``retrieval_score``
        and carries the reranker score in ``score``.

        Args:
            query: The claim text.
            evidences: Candidate passages.

        Returns:
            New list sorted by score, highest first.

        Raises:
            VectorStoreError: If inference fails.
        """
        if not evidences:
            return []
        if not self.enabled:
            return sorted(evidences, key=lambda ev: ev.score, reverse=True)

        try:
            import torch

            model = self._load()
            logits = model.predict(
                [(query, ev.evidence_text) for ev in evidences],
                activation_fn=torch.nn.Identity(),
                show_progress_bar=False,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Reranking failed", original_error=exc) from exc

        rescored = [
            dataclasses.replace(ev, score=_sigmoid(float(logit)))
            for ev, logit in zip(evidences, logits)
        ]
        rescored.sort(key=lambda ev: ev.score, reverse=True)
        return rescored
