"""
Stance detection module for the Disinformation Detection System.

Provides a supplementary stance signal (agree/disagree/discuss/unrelated)
to complement the NLI verifier in the aggregation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exceptions import VerificationError
from logging_config import get_logger

__all__ = ["StanceDetector", "StanceResult", "STANCE_LABELS"]

logger = get_logger(__name__)

STANCE_LABELS: list[str] = ["agree", "disagree", "discuss", "unrelated"]

# Zero-shot phrasing for the mDeBERTa-xnli classifier, mapped back to
# STANCE_LABELS. Grammatical verb phrases ("This text supports the claim
# that …") separated stances better than the bare labels on a multilingual
# probe, but zero-shot stance stays weak (it rarely predicts "disagree");
# NLIVerifier is the primary support/contradiction signal.
_STANCE_PHRASES: dict[str, str] = {
    "supports": "agree",
    "contradicts": "disagree",
    "is unrelated to": "unrelated",
}


@dataclass
class StanceResult:
    """Result of stance detection for a (claim, evidence) pair.

    Attributes:
        claim: The claim text.
        evidence_text: The evidence passage.
        stance: One of STANCE_LABELS.
        confidence: Confidence of the detected stance.
    """

    claim: str
    evidence_text: str
    stance: str
    confidence: float


class StanceDetector:
    """Detect the stance of an evidence passage toward a claim.

    Uses zero-shot classification with predefined stance candidate labels.
    This is a helper module — its output is used only inside the aggregator
    as a supplementary signal and does not produce a standalone score.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        """Lazy-load the zero-shot classification pipeline.

        Returns:
            Loaded HuggingFace pipeline.

        Raises:
            VerificationError: On load failure.
        """
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            # Reuse the claim extraction model for stance as well
            model_name = self._settings.claim_extraction.classifier_model
            self._pipeline = pipeline("zero-shot-classification", model=model_name)
            logger.info("Loaded stance classifier: %s", model_name)
            return self._pipeline
        except Exception as exc:
            raise VerificationError(
                "Failed to load stance classifier", original_error=exc
            ) from exc

    def detect(self, claim: str, evidence: str) -> StanceResult:
        """Detect the stance of *evidence* toward *claim*.

        Args:
            claim: The claim string.
            evidence: The evidence passage string.

        Returns:
            StanceResult with the detected stance and confidence.

        Raises:
            VerificationError: On inference failure.
        """
        # Braces in the claim would break str.format inside the pipeline.
        safe_claim = claim.replace("{", "(").replace("}", ")")
        hypothesis_template = f"This text {{}} the claim that {safe_claim}"
        try:
            clf = self._load_pipeline()
            result = clf(
                evidence,
                candidate_labels=list(_STANCE_PHRASES),
                hypothesis_template=hypothesis_template,
            )
            top_label: str = _STANCE_PHRASES.get(
                result["labels"][0], result["labels"][0]
            )
            top_score: float = float(result["scores"][0])
            return StanceResult(
                claim=claim,
                evidence_text=evidence,
                stance=top_label,
                confidence=top_score,
            )
        except VerificationError:
            raise
        except Exception as exc:
            raise VerificationError(
                "Stance detection failed", original_error=exc
            ) from exc

    def to_nli_compatible_score(self, stance: str, confidence: float) -> float:
        """Map a stance label to a [0, 1] NLI-compatible truthfulness score.

        Mapping:
        - agree       → confidence  (supports the claim)
        - disagree    → 1 - confidence  (contradicts the claim)
        - discuss     → 0.5  (ambiguous)
        - unrelated   → 0.5  (no signal)

        Args:
            stance: One of STANCE_LABELS.
            confidence: Classifier confidence for the stance.

        Returns:
            Float in [0, 1].
        """
        if stance == "agree":
            return confidence
        if stance == "disagree":
            return 1.0 - confidence
        return 0.5
