"""
Explainability module for the Disinformation Detection System.

Formats the final verification result into a human-readable, structured
ExplanationOutput with evidence excerpts, source citations, and a
natural-language explanation.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from exceptions import ExplainabilityError
from logging_config import get_logger
from verification.aggregator import (
    REASON_CONFLICT,
    REASON_LOW_SCORE_MARGIN,
    REASON_NO_EVIDENCE,
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    VerificationResult,
)

if TYPE_CHECKING:
    from retrieval.evidence import RetrievalResult

__all__ = ["Explainer", "ExplanationOutput"]

logger = get_logger(__name__)

_EVIDENCE_TEXT_LIMIT = 300

_VERDICT_TEMPLATES = {
    VERDICT_CONFIRMED: (
        "The claim is supported by {n} evidence passage(s) "
        "with an overall confidence score of {score:.2f}. "
        "The fact-checking analysis indicates the information is likely accurate."
    ),
    VERDICT_DISINFORMATION: (
        "The claim appears to be contradicted by {n} evidence passage(s) "
        "with an overall confidence score of {score:.2f}. "
        "The fact-checking analysis indicates this may be disinformation."
    ),
    VERDICT_UNCERTAIN: (
        "The available evidence is insufficient or inconsistent to make a "
        "definitive determination (confidence score: {score:.2f}, "
        "based on {n} evidence passage(s)). "
        "Manual review is recommended."
    ),
}

# UNCERTAIN explanations keyed by VerificationResult.uncertainty_reason.
_UNCERTAIN_REASON_TEMPLATES = {
    REASON_CONFLICT: (
        "The NLI analysis and the LLM analysis point in opposite directions "
        "(NLI score {nli:.2f}, LLM score {rag:.2f}, based on {n} evidence "
        "passage(s)), so no firm verdict can be given. "
        "Manual review is recommended."
    ),
    REASON_LOW_SCORE_MARGIN: (
        "The evidence is too weak or mixed for a firm verdict "
        "(overall score {score:.2f}, based on {n} evidence passage(s)). "
        "Manual review is recommended."
    ),
    REASON_NO_EVIDENCE: (
        "No sufficiently relevant evidence from trusted sources was found for "
        "this claim, so it cannot be verified. Manual review is recommended."
    ),
}

# Used instead of the no_evidence text when the search itself failed.
_SEARCH_UNAVAILABLE_TEMPLATE = (
    "The evidence search could not be performed (the search service was "
    "unreachable or rate-limited), so the claim could not be checked against "
    "any sources. Retry later or review the claim manually."
)
_STATUS_UNAVAILABLE = "unavailable"

_LLM_ASSESSMENT_LABEL = "LLM assessment:"


@dataclass
class ExplanationOutput:
    """Final structured output of the pipeline for a single claim.

    Attributes:
        claim_text: The claim that was verified.
        verdict: Final verdict string.
        final_score: Aggregated truthfulness score [0, 1].
        explanation: 1–3 sentence human-readable explanation.
        component_scores: Per-method scores (``{"nli": float, "rag": float}``).
        evidence_excerpts: Formatted evidence passages with stance info.
        citations: Source metadata for each evidence passage.
        language: Detected language of the claim.
        processing_metadata: Timing and model version info.
    """

    claim_text: str
    verdict: str
    final_score: float
    explanation: str
    component_scores: dict[str, float]
    evidence_excerpts: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    language: str
    processing_metadata: dict[str, Any] = field(default_factory=dict)


class Explainer:
    """Format VerificationResult into a structured ExplanationOutput.

    This class keeps no per-request state and is safe to reuse across
    requests.

    Args:
        settings: Optional application settings, used only to report the
            evidence mode and model names in ``processing_metadata``.
    """

    def __init__(self, settings: "Settings | None" = None) -> None:  # noqa: F821
        self._settings = settings

    def explain(
        self,
        verification_result: VerificationResult,
        retrieval: "RetrievalResult | None" = None,
    ) -> ExplanationOutput:
        """Generate a full explanation from a VerificationResult.

        Args:
            verification_result: Aggregated result from ResultAggregator.
            retrieval: The retrieval result for the claim, used for the
                search metadata and the "search unavailable" explanation.

        Returns:
            ExplanationOutput dataclass instance.

        Raises:
            ExplainabilityError: On unexpected failure.
        """
        try:
            result = verification_result
            claim_text = result.claim.text
            verdict = result.verdict
            final_score = result.final_score
            language = result.claim.language

            search_status = retrieval.status if retrieval is not None else None
            explanation_text = self._generate_explanation_text(result, search_status)
            evidence_excerpts = self._format_evidence_excerpts(
                result.retrieved_evidences, result.nli_results
            )
            citations = self._format_citations(result.retrieved_evidences)

            component_scores: dict[str, float] = {
                "nli": round(result.nli_score, 4),
                "rag": round(result.rag_score, 4),
            }

            processing_metadata: dict[str, Any] = {
                "evidence_count": len(result.retrieved_evidences),
                "nli_results_count": len(result.nli_results),
                "component_weights": result.component_weights,
                "effective_weights": result.effective_weights,
                "uncertainty_reason": result.uncertainty_reason,
                "rag_model_verdict": result.rag_verdict.verdict,
                "rag_confidence": result.rag_verdict.confidence,
                "rag_raw_response": result.rag_verdict.raw_response,
                "evidence_mode": self._evidence_mode(),
                "search_backend": retrieval.backend if retrieval is not None else None,
                "search_status": search_status,
                "search_queries": list(retrieval.queries) if retrieval is not None else [],
                "source_policy_mode": self._source_policy_mode(),
                "models": self._model_names(),
            }

            output = ExplanationOutput(
                claim_text=claim_text,
                verdict=verdict,
                final_score=round(final_score, 4),
                explanation=explanation_text,
                component_scores=component_scores,
                evidence_excerpts=evidence_excerpts,
                citations=citations,
                language=language,
                processing_metadata=processing_metadata,
            )

            logger.debug(
                "Generated explanation for claim '%s...': verdict=%s, score=%.2f",
                claim_text[:50],
                verdict,
                final_score,
            )
            return output

        except ExplainabilityError:
            raise
        except Exception as exc:
            raise ExplainabilityError(
                "Failed to generate explanation", original_error=exc
            ) from exc

    def _generate_explanation_text(
        self, result: VerificationResult, search_status: str | None = None
    ) -> str:
        """Generate a 1–3 sentence explanation for the verdict.

        Incorporates the RAG reasoning if available, otherwise falls back
        to a template-based explanation.

        Args:
            result: VerificationResult to explain.
            search_status: Search status of the retrieval, if known.

        Returns:
            Explanation string.
        """
        n_evidence = len(result.retrieved_evidences)
        score = result.final_score

        template = _VERDICT_TEMPLATES.get(
            result.verdict, _VERDICT_TEMPLATES[VERDICT_UNCERTAIN]
        )
        if result.verdict == VERDICT_UNCERTAIN:
            template = _UNCERTAIN_REASON_TEMPLATES.get(
                result.uncertainty_reason or "", template
            )
            if (
                result.uncertainty_reason == REASON_NO_EVIDENCE
                and search_status == _STATUS_UNAVAILABLE
            ):
                template = _SEARCH_UNAVAILABLE_TEMPLATE
        base_explanation = template.format(
            n=n_evidence, score=score, nli=result.nli_score, rag=result.rag_score
        )

        # Append the LLM's reasoning, labelled so it is not read as the verdict.
        rag_reasoning = result.rag_verdict.reasoning.strip()
        if rag_reasoning and len(rag_reasoning) > 10:
            return f"{base_explanation} {_LLM_ASSESSMENT_LABEL} {rag_reasoning}"
        return base_explanation

    def _format_evidence_excerpts(
        self,
        evidences: list[Any],
        nli_results: list[Any],
    ) -> list[dict[str, Any]]:
        """Format evidence passages into structured excerpt records.

        Args:
            evidences: List of RetrievedEvidence from retrieval.
            nli_results: NLIResult objects in the same order as *evidences*.

        Returns:
            List of dicts with passage index, text, relevance and stance.
        """
        aligned = len(nli_results) == len(evidences)
        nli_map: dict[str, str] = {
            nli.evidence_text: nli.predicted_label for nli in nli_results
        }

        excerpts: list[dict[str, Any]] = []
        for idx, ev in enumerate(evidences):
            if aligned:
                nli_label = nli_results[idx].predicted_label
            else:
                nli_label = nli_map.get(ev.evidence_text, "unknown")

            excerpts.append(
                {
                    "passage_index": idx,
                    "text": ev.evidence_text[:_EVIDENCE_TEXT_LIMIT],
                    "relevance_score": round(ev.score, 4),
                    "stance": self._nli_label_to_stance(nli_label),
                }
            )
        return excerpts

    def _format_citations(self, evidences: list[Any]) -> list[dict[str, Any]]:
        """Format citation records for each evidence passage.

        Args:
            evidences: List of RetrievedEvidence instances.

        Returns:
            List of citation dicts, in excerpt order, with the passage
            index, URL, publisher, title, publication and retrieval dates,
            source tier, trust and language.
        """
        return [
            {
                "passage_index": idx,
                "evidence_id": ev.evidence_id,
                "url": ev.source_url,
                "publisher": ev.publisher,
                "title": ev.title,
                "published_at": ev.published_at,
                "retrieved_at": ev.retrieved_at,
                "tier": ev.source_tier,
                "trust": ev.trust,
                "language": ev.language,
            }
            for idx, ev in enumerate(evidences)
        ]

    def _evidence_mode(self) -> str | None:
        if self._settings is None:
            return None
        return self._settings.ollama.evidence_mode

    def _source_policy_mode(self) -> str | None:
        if self._settings is None:
            return None
        return self._settings.source_policy.mode

    def _model_names(self) -> dict[str, str | None]:
        if self._settings is None:
            return {"reranker": None, "nli": None, "llm": None}
        s = self._settings
        return {
            "reranker": s.retrieval.reranker_model,
            "nli": s.verification.nli_model,
            "llm": s.ollama.model,
        }

    def _nli_label_to_stance(self, nli_label: str) -> str:
        """Map an NLI label to a human-readable stance string.

        Args:
            nli_label: One of "entailment", "contradiction", "neutral".

        Returns:
            One of "supports", "contradicts", "neutral".
        """
        mapping = {
            "entailment": "supports",
            "contradiction": "contradicts",
            "neutral": "neutral",
        }
        return mapping.get(nli_label, "unknown")

    def to_json(self, output: ExplanationOutput) -> str:
        """Serialize an ExplanationOutput to an indented JSON string.

        Args:
            output: ExplanationOutput dataclass instance.

        Returns:
            JSON-formatted string.
        """
        return json.dumps(dataclasses.asdict(output), ensure_ascii=False, indent=2)
