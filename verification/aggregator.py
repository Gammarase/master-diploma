"""
Result aggregation module for the Disinformation Detection System.

Combines NLI scores and RAG verdicts into a single truthfulness score
and verdict classification.

The decision logic is the pure function :func:`decide`, which depends only on
the two component scores, the evidence count and a :class:`DecisionConfig`.
``scripts/calibrate_thresholds.py`` imports it directly, so calibration and
runtime verdicts cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from logging_config import get_logger
from retrieval.evidence import RetrievedEvidence
from verification.nli_verifier import NLIResult, NLIVerifier
from verification.rag_verifier import RAGVerdict

if TYPE_CHECKING:
    from claim_extraction.extractor import Claim

__all__ = [
    "ResultAggregator",
    "VerificationResult",
    "Decision",
    "DecisionConfig",
    "decide",
    "is_decisive",
    "VERDICT_DISINFORMATION",
    "VERDICT_UNCERTAIN",
    "VERDICT_CONFIRMED",
    "REASON_CONFLICT",
    "REASON_LOW_SCORE_MARGIN",
    "REASON_NO_EVIDENCE",
]

logger = get_logger(__name__)

VERDICT_DISINFORMATION = "DISINFORMATION"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICT_CONFIRMED = "CONFIRMED"

REASON_CONFLICT = "conflict"
REASON_LOW_SCORE_MARGIN = "low_score_margin"
REASON_NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True)
class DecisionConfig:
    """Parameters of the verdict decision.

    Attributes:
        nli_weight: Configured NLI weight (normalised with rag_weight).
        rag_weight: Configured RAG weight.
        threshold_disinformation: Scores below this are DISINFORMATION.
        threshold_confirmed: Scores at or above this are CONFIRMED.
        decisiveness_margin: A component ``x`` is decisive when
            ``|x - 0.5| > decisiveness_margin``.
    """

    nli_weight: float
    rag_weight: float
    threshold_disinformation: float
    threshold_confirmed: float
    decisiveness_margin: float

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DecisionConfig":  # noqa: F821
        cfg = settings.verification
        return cls(
            nli_weight=float(cfg.nli_weight),
            rag_weight=float(cfg.rag_weight),
            threshold_disinformation=float(cfg.thresholds.disinformation),
            threshold_confirmed=float(cfg.thresholds.confirmed),
            decisiveness_margin=float(cfg.decisiveness_margin),
        )


@dataclass(frozen=True)
class Decision:
    """Outcome of :func:`decide`.

    Attributes:
        final_score: Weighted truthfulness score in [0, 1].
        verdict: One of DISINFORMATION, UNCERTAIN, CONFIRMED.
        reason: Uncertainty reason for UNCERTAIN verdicts, else None.
        effective_weights: Weights actually applied (``{"nli", "rag"}``).
    """

    final_score: float
    verdict: str
    reason: str | None
    effective_weights: dict[str, float]


def is_decisive(score: float, margin: float) -> bool:
    """True when *score* is further than *margin* from the neutral 0.5."""
    return abs(score - 0.5) > margin


def decide(
    nli: float, rag: float, n_evidence: int, cfg: DecisionConfig
) -> Decision:
    """Combine component scores into a final score and verdict.

    Rules (see specs/verdict-aggregation/spec.md):

    - Weights are normalised to sum to 1. If exactly one component is
      decisive, it receives all the weight; otherwise the configured weights
      apply.
    - With no evidence the verdict is UNCERTAIN (``no_evidence``).
    - If both components are decisive and on opposite sides of 0.5 the
      verdict is UNCERTAIN (``conflict``).
    - Otherwise thresholds apply; the middle band is UNCERTAIN
      (``low_score_margin``).

    Args:
        nli: NLI truthfulness score in [0, 1].
        rag: RAG truthfulness score in [0, 1].
        n_evidence: Number of retrieved evidence passages.
        cfg: Decision parameters.

    Returns:
        A Decision.
    """
    total = cfg.nli_weight + cfg.rag_weight
    if total > 0:
        w_nli, w_rag = cfg.nli_weight / total, cfg.rag_weight / total
    else:
        w_nli = w_rag = 0.5

    nli_decisive = is_decisive(nli, cfg.decisiveness_margin)
    rag_decisive = is_decisive(rag, cfg.decisiveness_margin)
    if nli_decisive and not rag_decisive:
        w_nli, w_rag = 1.0, 0.0
    elif rag_decisive and not nli_decisive:
        w_nli, w_rag = 0.0, 1.0

    final_score = max(0.0, min(1.0, w_nli * nli + w_rag * rag))
    weights = {"nli": w_nli, "rag": w_rag}

    if n_evidence <= 0:
        return Decision(final_score, VERDICT_UNCERTAIN, REASON_NO_EVIDENCE, weights)
    if nli_decisive and rag_decisive and (nli - 0.5) * (rag - 0.5) < 0:
        return Decision(final_score, VERDICT_UNCERTAIN, REASON_CONFLICT, weights)
    if final_score < cfg.threshold_disinformation:
        return Decision(final_score, VERDICT_DISINFORMATION, None, weights)
    if final_score >= cfg.threshold_confirmed:
        return Decision(final_score, VERDICT_CONFIRMED, None, weights)
    return Decision(final_score, VERDICT_UNCERTAIN, REASON_LOW_SCORE_MARGIN, weights)


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
        component_weights: Configured weights.
        uncertainty_reason: ``conflict``, ``low_score_margin`` or
            ``no_evidence`` for UNCERTAIN verdicts, else None.
        effective_weights: Weights actually applied to the scores.
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
    uncertainty_reason: str | None = None
    effective_weights: dict[str, float] = field(default_factory=dict)


class ResultAggregator:
    """Aggregate NLI and RAG scores into a final verification verdict.

    A thin wrapper around :func:`decide`.

    Args:
        settings: Application settings providing weights and thresholds.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._config = DecisionConfig.from_settings(settings)

    @property
    def config(self) -> DecisionConfig:
        """The decision parameters in use."""
        return self._config

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
        nli_score = NLIVerifier.aggregate_nli_score(nli_results)
        rag_score = rag_verdict.to_score()
        decision = decide(
            nli_score, rag_score, len(retrieved_evidences), self._config
        )

        logger.debug(
            "Aggregation: nli=%.2f, rag=%.2f, weights=%s, final=%.2f, "
            "verdict=%s, reason=%s",
            nli_score,
            rag_score,
            decision.effective_weights,
            decision.final_score,
            decision.verdict,
            decision.reason,
        )

        return VerificationResult(
            claim=claim,
            nli_score=nli_score,
            rag_score=rag_score,
            final_score=decision.final_score,
            verdict=decision.verdict,
            nli_results=nli_results,
            rag_verdict=rag_verdict,
            retrieved_evidences=retrieved_evidences,
            component_weights={
                "nli": self._config.nli_weight,
                "rag": self._config.rag_weight,
            },
            uncertainty_reason=decision.reason,
            effective_weights=decision.effective_weights,
        )
