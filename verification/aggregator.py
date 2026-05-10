"""
Result aggregation module for the Disinformation Detection System.

Combines NLI scores and RAG verdicts into a single truthfulness score
and verdict classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from logging_config import get_logger
from retrieval.vector_store import RetrievedEvidence
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict

if TYPE_CHECKING:
    from claim_extraction.extractor import Claim

__all__ = ["ResultAggregator", "VerificationResult", "VERDICT_DISINFORMATION", "VERDICT_UNCERTAIN", "VERDICT_CONFIRMED"]

logger = get_logger(__name__)

VERDICT_DISINFORMATION = "DISINFORMATION"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICT_CONFIRMED = "CONFIRMED"


@dataclass
class VerificationResult:
    """Aggregated verification result for a single claim.

    Attributes:
        claim: The original Claim dataclass.
        nli_score: Aggregated NLI truthfulness score [0, 1].
        rag_score: RAG verdict truthfulness score [0, 1].
        final_score: Weighted combination of nli_score and rag_score.
        verdict: One of DISINFORMATION, UNCERTAIN, CONFIRMED.
        nli_results: Individual NLI results per evidence passage.
        rag_verdict: The full RAG verdict dataclass.
        retrieved_evidences: Evidence passages used for verification.
        component_weights: Dict of weights used for aggregation.
    """

    claim: "Claim"
    nli_score: float
    rag_score: float
    final_score: float
    verdict: str
    nli_results: list[NLIResult] = field(default_factory=list)
    rag_verdict: RAGVerdict = field(
        default_factory=lambda: RAGVerdict(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.5,
            reasoning="",
        )
    )
    retrieved_evidences: list[RetrievedEvidence] = field(default_factory=list)
    component_weights: dict[str, float] = field(default_factory=dict)


class ResultAggregator:
    """Aggregate NLI and RAG scores into a final verification verdict.

    Formula: ``final_score = nli_weight * nli_score + rag_weight * rag_score``

    When the two methods disagree significantly (|nli - rag| > disagreement_delta),
    the verdict is forced to UNCERTAIN regardless of the numeric score.

    Args:
        settings: Application settings providing weights and thresholds.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        cfg = settings.verification
        self._nli_weight: float = cfg.nli_weight
        self._rag_weight: float = cfg.rag_weight
        self._threshold_disinformation: float = cfg.thresholds.disinformation
        self._threshold_confirmed: float = cfg.thresholds.confirmed
        self._disagreement_delta: float = cfg.disagreement_delta

    def aggregate(
        self,
        claim: "Claim",
        nli_results: list[NLIResult],
        rag_verdict: RAGVerdict,
        retrieved_evidences: list[RetrievedEvidence],
    ) -> VerificationResult:
        """Compute the final verification result for a claim.

        Args:
            claim: The claim being verified.
            nli_results: NLI results from NLIVerifier.verify_batch().
            rag_verdict: Verdict from RAGVerifier.verify().
            retrieved_evidences: Evidence passages from similarity search.

        Returns:
            A fully populated VerificationResult.
        """
        from verification.nli_verifier import NLIVerifier

        # Compute scores
        nli_verifier = NLIVerifier.__new__(NLIVerifier)
        nli_score = nli_verifier.aggregate_nli_score(nli_results)
        rag_score = rag_verdict.to_score()

        final_score = self._compute_final_score(nli_score, rag_score)
        verdict = self._apply_threshold(final_score)

        # Override to UNCERTAIN on strong disagreement between methods
        if self._detect_disagreement(nli_score, rag_score):
            verdict = VERDICT_UNCERTAIN
            logger.debug(
                "Methods disagree (nli=%.2f, rag=%.2f) → verdict forced to UNCERTAIN.",
                nli_score,
                rag_score,
            )

        logger.debug(
            "Aggregation: nli=%.2f, rag=%.2f, final=%.2f, verdict=%s",
            nli_score,
            rag_score,
            final_score,
            verdict,
        )

        return VerificationResult(
            claim=claim,
            nli_score=nli_score,
            rag_score=rag_score,
            final_score=final_score,
            verdict=verdict,
            nli_results=nli_results,
            rag_verdict=rag_verdict,
            retrieved_evidences=retrieved_evidences,
            component_weights={
                "nli": self._nli_weight,
                "rag": self._rag_weight,
            },
        )

    def _compute_final_score(self, nli_score: float, rag_score: float) -> float:
        """Compute the weighted final score.

        Args:
            nli_score: NLI truthfulness score [0, 1].
            rag_score: RAG truthfulness score [0, 1].

        Returns:
            Weighted combination, clamped to [0, 1].
        """
        score = self._nli_weight * nli_score + self._rag_weight * rag_score
        return max(0.0, min(1.0, score))

    def _apply_threshold(self, score: float) -> str:
        """Map a numeric score to a verdict string using configured thresholds.

        Args:
            score: Final score in [0, 1].

        Returns:
            One of VERDICT_DISINFORMATION, VERDICT_UNCERTAIN, VERDICT_CONFIRMED.
        """
        if score < self._threshold_disinformation:
            return VERDICT_DISINFORMATION
        if score >= self._threshold_confirmed:
            return VERDICT_CONFIRMED
        return VERDICT_UNCERTAIN

    def _detect_disagreement(self, nli_score: float, rag_score: float) -> bool:
        """Return True if the two methods disagree beyond the configured delta.

        Args:
            nli_score: NLI truthfulness score.
            rag_score: RAG truthfulness score.

        Returns:
            True if |nli_score - rag_score| > disagreement_delta.
        """
        return abs(nli_score - rag_score) > self._disagreement_delta
