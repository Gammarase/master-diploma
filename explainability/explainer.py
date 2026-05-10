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
from typing import Any

from exceptions import ExplainabilityError
from logging_config import get_logger
from verification.aggregator import (
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    VerificationResult,
)

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

    This class is stateless and has no external dependencies, making it
    trivially testable and safe to reuse across requests.
    """

    def explain(self, verification_result: VerificationResult) -> ExplanationOutput:
        """Generate a full explanation from a VerificationResult.

        Args:
            verification_result: Aggregated result from ResultAggregator.

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

            explanation_text = self._generate_explanation_text(result)
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
                "rag_model_verdict": result.rag_verdict.verdict,
                "nli_results_count": len(result.nli_results),
                "component_weights": result.component_weights,
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

    def _generate_explanation_text(self, result: VerificationResult) -> str:
        """Generate a 1–3 sentence explanation for the verdict.

        Incorporates the RAG reasoning if available, otherwise falls back
        to a template-based explanation.

        Args:
            result: VerificationResult to explain.

        Returns:
            Explanation string.
        """
        n_evidence = len(result.retrieved_evidences)
        score = result.final_score

        template = _VERDICT_TEMPLATES.get(
            result.verdict, _VERDICT_TEMPLATES[VERDICT_UNCERTAIN]
        )
        base_explanation = template.format(n=n_evidence, score=score)

        # Append RAG reasoning if it is non-trivial
        rag_reasoning = result.rag_verdict.reasoning.strip()
        if rag_reasoning and len(rag_reasoning) > 10:
            return f"{base_explanation} {rag_reasoning}"
        return base_explanation

    def _format_evidence_excerpts(
        self,
        evidences: list[Any],
        nli_results: list[Any],
    ) -> list[dict[str, Any]]:
        """Format evidence passages into structured excerpt records.

        Args:
            evidences: List of RetrievedEvidence from similarity search.
            nli_results: Corresponding NLIResult objects.

        Returns:
            List of dicts with passage index, text, relevance, stance, label.
        """
        nli_map: dict[str, str] = {}
        for nli in nli_results:
            nli_map[nli.evidence_text[:50]] = nli.predicted_label

        excerpts: list[dict[str, Any]] = []
        for idx, ev in enumerate(evidences):
            nli_label = nli_map.get(ev.evidence_text[:50], "unknown")
            stance = self._nli_label_to_stance(nli_label)

            excerpts.append(
                {
                    "passage_index": idx,
                    "text": ev.evidence_text[:_EVIDENCE_TEXT_LIMIT],
                    "relevance_score": round(ev.score, 4),
                    "stance": stance,
                    "label_from_dataset": ev.label,
                }
            )
        return excerpts

    def _format_citations(self, evidences: list[Any]) -> list[dict[str, Any]]:
        """Format citation records for each evidence passage.

        Args:
            evidences: List of RetrievedEvidence instances.

        Returns:
            List of citation dicts with vector_id, claim_id, language, dataset.
        """
        return [
            {
                "vector_id": ev.vector_id,
                "claim_id": getattr(ev, "claim_id", ""),
                "language": ev.language,
                "dataset": "RU22Fact",
            }
            for ev in evidences
        ]

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
