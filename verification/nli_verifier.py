"""
NLI-based claim verification for the Disinformation Detection System.

Uses a multilingual cross-encoder NLI model to classify the logical
relationship (entailment / contradiction / neutral) between each piece of
retrieved evidence (premise) and the claim (hypothesis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exceptions import NLIError
from logging_config import get_logger
from retrieval.vector_store import RetrievedEvidence

__all__ = ["NLIVerifier", "NLIResult"]

logger = get_logger(__name__)

_NLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class NLIResult:
    """Result of a single NLI inference pass.

    Attributes:
        claim: The claim text (hypothesis).
        evidence_text: The evidence passage (premise).
        entailment_score: Probability of entailment.
        contradiction_score: Probability of contradiction.
        neutral_score: Probability of neutral.
        predicted_label: The argmax label.
        confidence: The highest of the three scores.
        support: Signed support ``p_entailment - p_contradiction`` in [-1, 1].
    """

    claim: str
    evidence_text: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    predicted_label: str
    confidence: float
    support: float | None = None

    def __post_init__(self) -> None:
        if self.support is None:
            self.support = self.entailment_score - self.contradiction_score


class NLIVerifier:
    """Verify claims using a cross-encoder NLI model.

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
        """Run NLI inference with *evidence* as premise and *claim* as hypothesis.

        Args:
            claim: The claim string (hypothesis).
            evidence: The evidence passage (premise).

        Returns:
            NLIResult with per-label probabilities.

        Raises:
            NLIError: On inference failure or a model without label2id.
        """
        results = self._infer(claim, [evidence])
        return results[0]

    def verify_batch(
        self, claim: str, evidences: list[RetrievedEvidence]
    ) -> list[NLIResult]:
        """Run NLI for one claim against multiple evidence passages.

        Args:
            claim: The claim string to verify.
            evidences: List of RetrievedEvidence from retrieval.

        Returns:
            List of NLIResult, one per evidence passage, in passage order.

        Raises:
            NLIError: On batch inference failure.
        """
        if not evidences:
            return []
        return self._infer(claim, [ev.evidence_text for ev in evidences])

    def _infer(self, claim: str, premises: list[str]) -> list[NLIResult]:
        """Score ``(premise, claim)`` pairs and build NLIResults."""
        try:
            model = self._load_model()
            label_map = self._get_label_map(model)
            pairs = [(premise, claim) for premise in premises]
            scores_batch = model.predict(
                pairs, apply_softmax=True, show_progress_bar=False
            )
        except NLIError:
            raise
        except Exception as exc:
            raise NLIError("NLI inference failed", original_error=exc) from exc

        results: list[NLIResult] = []
        for scores, premise in zip(scores_batch, premises):
            label_scores = {
                label: float(scores[label_map[label]]) for label in _NLI_LABELS
            }
            predicted_label = max(label_scores, key=label_scores.__getitem__)
            results.append(
                NLIResult(
                    claim=claim,
                    evidence_text=premise,
                    entailment_score=label_scores["entailment"],
                    contradiction_score=label_scores["contradiction"],
                    neutral_score=label_scores["neutral"],
                    predicted_label=predicted_label,
                    confidence=label_scores[predicted_label],
                )
            )
        return results

    @staticmethod
    def aggregate_nli_score(results: list[NLIResult]) -> float:
        """Convert per-passage NLI results into one truthfulness score.

        The passage with the strongest signed support ``s`` (largest ``|s|``)
        decides: ``score = (1 + s*) / 2``. One strongly contradicting passage
        is not diluted by merely related ones.

        Args:
            results: NLI results for a claim against its evidence passages.

        Returns:
            Float in [0, 1]; 0.5 when there are no results.
        """
        if not results:
            return 0.5
        strongest = max((r.support for r in results), key=abs)
        return max(0.0, min(1.0, (1.0 + strongest) / 2.0))

    @staticmethod
    def _get_label_map(model: Any) -> dict[str, int]:
        """Return a mapping from NLI label name to score index.

        The model's ``config.label2id`` is the only source of truth; label
        order differs between NLI models.

        Args:
            model: A CrossEncoder instance.

        Returns:
            Dict mapping "entailment", "neutral", "contradiction" to indices.

        Raises:
            NLIError: If the model has no usable label2id mapping.
        """
        for owner in (model, getattr(model, "model", None)):
            config = getattr(owner, "config", None)
            label2id = getattr(config, "label2id", None)
            if isinstance(label2id, dict) and label2id:
                mapping = {str(k).lower(): int(v) for k, v in label2id.items()}
                if all(label in mapping for label in _NLI_LABELS):
                    return mapping
                raise NLIError(
                    f"NLI model label2id {label2id} lacks one of {list(_NLI_LABELS)}."
                )
        raise NLIError("NLI model config has no label2id mapping.")
