"""
NLI-based claim verification for the Disinformation Detection System.

Uses a cross-encoder DeBERTa model to classify the logical relationship
(entailment / contradiction / neutral) between a claim and each piece
of retrieved evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exceptions import NLIError
from logging_config import get_logger
from retrieval.vector_store import RetrievedEvidence

__all__ = ["NLIVerifier", "NLIResult"]

logger = get_logger(__name__)

_NLI_LABELS = ["contradiction", "entailment", "neutral"]


@dataclass
class NLIResult:
    """Result of a single NLI inference pass.

    Attributes:
        claim: The claim text.
        evidence_text: The evidence passage.
        entailment_score: Probability of entailment.
        contradiction_score: Probability of contradiction.
        neutral_score: Probability of neutral.
        predicted_label: The argmax label.
        confidence: The highest of the three scores.
    """

    claim: str
    evidence_text: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    predicted_label: str
    confidence: float


class NLIVerifier:
    """Verify claims using a cross-encoder NLI model (DeBERTa).

    Args:
        settings: Application settings providing the NLI model name.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Lazy-load the CrossEncoder NLI model.

        Returns:
            Loaded CrossEncoder instance.

        Raises:
            NLIError: If the model cannot be loaded.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            model_name = self._settings.verification.nli_model
            self._model = CrossEncoder(model_name)
            logger.info("Loaded NLI model: %s", model_name)
            return self._model
        except Exception as exc:
            raise NLIError(
                f"Failed to load NLI model '{self._settings.verification.nli_model}'",
                original_error=exc,
            ) from exc

    def verify_single(self, claim: str, evidence: str) -> NLIResult:
        """Run NLI inference on a single (claim, evidence) pair.

        Args:
            claim: The claim string.
            evidence: The evidence passage.

        Returns:
            NLIResult with per-label probabilities.

        Raises:
            NLIError: On inference failure.
        """
        try:
            model = self._load_model()
            scores = model.predict(
                [(claim, evidence)], apply_softmax=True
            )[0]
            # CrossEncoder NLI models typically output [contradiction, entailment, neutral]
            # but the ordering may vary — we use the model's label2id mapping.
            label_map = self._get_label_map(model)
            contradiction_score = float(scores[label_map.get("contradiction", 0)])
            entailment_score = float(scores[label_map.get("entailment", 1)])
            neutral_score = float(scores[label_map.get("neutral", 2)])

            label_scores = {
                "contradiction": contradiction_score,
                "entailment": entailment_score,
                "neutral": neutral_score,
            }
            predicted_label = max(label_scores, key=label_scores.__getitem__)
            confidence = label_scores[predicted_label]

            return NLIResult(
                claim=claim,
                evidence_text=evidence,
                entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                neutral_score=neutral_score,
                predicted_label=predicted_label,
                confidence=confidence,
            )
        except NLIError:
            raise
        except Exception as exc:
            raise NLIError("NLI inference failed", original_error=exc) from exc

    def verify_batch(
        self, claim: str, evidences: list[RetrievedEvidence]
    ) -> list[NLIResult]:
        """Run NLI for one claim against multiple evidence passages.

        Args:
            claim: The claim string to verify.
            evidences: List of RetrievedEvidence from Pinecone search.

        Returns:
            List of NLIResult, one per evidence passage.

        Raises:
            NLIError: On batch inference failure.
        """
        if not evidences:
            return []
        try:
            model = self._load_model()
            pairs = [(claim, ev.evidence_text) for ev in evidences]
            scores_batch = model.predict(pairs, apply_softmax=True)
            label_map = self._get_label_map(model)
            results: list[NLIResult] = []
            for scores, ev in zip(scores_batch, evidences):
                contradiction_score = float(scores[label_map.get("contradiction", 0)])
                entailment_score = float(scores[label_map.get("entailment", 1)])
                neutral_score = float(scores[label_map.get("neutral", 2)])
                label_scores = {
                    "contradiction": contradiction_score,
                    "entailment": entailment_score,
                    "neutral": neutral_score,
                }
                predicted_label = max(label_scores, key=label_scores.__getitem__)
                results.append(
                    NLIResult(
                        claim=claim,
                        evidence_text=ev.evidence_text,
                        entailment_score=entailment_score,
                        contradiction_score=contradiction_score,
                        neutral_score=neutral_score,
                        predicted_label=predicted_label,
                        confidence=label_scores[predicted_label],
                    )
                )
            return results
        except NLIError:
            raise
        except Exception as exc:
            raise NLIError("Batch NLI inference failed", original_error=exc) from exc

    def aggregate_nli_score(self, results: list[NLIResult]) -> float:
        """Convert multiple NLI results into a single truthfulness score.

        Scoring:
        - entailment  → +1.0
        - neutral     → +0.5
        - contradiction → 0.0

        The per-result contribution is weighted by the model confidence.
        The final score is the average weighted contribution.

        Args:
            results: NLI results for a claim against multiple evidence passages.

        Returns:
            Float in [0, 1] representing the degree of evidential support.
        """
        if not results:
            return 0.5  # neutral fallback

        label_values = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            weight = result.confidence
            value = label_values.get(result.predicted_label, 0.5)
            weighted_sum += weight * value
            total_weight += weight

        if total_weight == 0.0:
            return 0.5
        return weighted_sum / total_weight

    def _get_label_map(self, model: Any) -> dict[str, int]:
        """Return a mapping from label name to score index.

        Falls back to a sensible default if the model has no label2id.

        Args:
            model: A CrossEncoder instance.

        Returns:
            Dict mapping label names to their index in the scores array.
        """
        try:
            if hasattr(model, "model") and hasattr(model.model, "config"):
                label2id = model.model.config.label2id
                if label2id:
                    return {k.lower(): v for k, v in label2id.items()}
        except Exception:
            pass
        # Default order for cross-encoder/nli-deberta-v3-base
        return {"contradiction": 0, "entailment": 1, "neutral": 2}
