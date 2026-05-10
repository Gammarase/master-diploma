"""
RAG + Ollama verification module for the Disinformation Detection System.

Formats a structured prompt with retrieved evidence, calls the local Ollama
LLM via its HTTP API, and parses the JSON verdict response.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from exceptions import OllamaConnectionError, RAGVerifierError
from logging_config import get_logger
from retrieval.vector_store import RetrievedEvidence

if TYPE_CHECKING:
    from claim_extraction.extractor import Claim

__all__ = ["OllamaClient", "RAGVerifier", "RAGVerdict"]

logger = get_logger(__name__)

RAG_PROMPT_TEMPLATE = """\
You are a professional fact-checking assistant. \
Your task is to verify the following claim using ONLY the provided evidence passages.

CLAIM: {claim}

EVIDENCE PASSAGES:
{evidence_passages}

Based solely on the evidence above, provide your verdict as a valid JSON object \
conforming EXACTLY to this schema:
{{
  "verdict": "<SUPPORTED|REFUTED|INSUFFICIENT_EVIDENCE>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<1-3 sentences explaining your verdict>",
  "supporting_evidence_ids": [<list of passage indices that support the claim>],
  "contradicting_evidence_ids": [<list of passage indices that contradict the claim>]
}}

IMPORTANT: Return ONLY the JSON object. Do not include any other text, markdown, or explanation.\
"""

_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.DOTALL)

VALID_VERDICTS = {"SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"}


@dataclass
class RAGVerdict:
    """Structured verdict produced by the RAG verifier.

    Attributes:
        verdict: Classification result.
        confidence: Model confidence in the verdict.
        reasoning: 1-3 sentence explanation.
        supporting_evidence_ids: Passage indices that support the verdict.
        contradicting_evidence_ids: Passage indices that contradict the claim.
        raw_response: Raw LLM response string for debugging.
    """

    verdict: str
    confidence: float
    reasoning: str
    supporting_evidence_ids: list[int] = field(default_factory=list)
    contradicting_evidence_ids: list[int] = field(default_factory=list)
    raw_response: str = ""

    def to_score(self) -> float:
        """Map the verdict to a [0, 1] truthfulness score.

        Mapping:
        - SUPPORTED             → confidence
        - REFUTED               → 1 - confidence
        - INSUFFICIENT_EVIDENCE → 0.5

        Returns:
            Float in [0, 1].
        """
        if self.verdict == "SUPPORTED":
            return max(0.0, min(1.0, self.confidence))
        if self.verdict == "REFUTED":
            return max(0.0, min(1.0, 1.0 - self.confidence))
        return 0.5


class OllamaClient:
    """HTTP client for the Ollama local LLM server.

    Uses ``httpx`` directly against the Ollama REST API instead of the
    ``ollama`` Python package for greater API stability.

    Endpoints used:
    - ``GET  /api/tags``     — health check
    - ``POST /api/generate`` — text generation

    Args:
        settings: Application settings with Ollama configuration.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._base_url: str = settings.ollama.base_url.rstrip("/")
        self._model: str = settings.ollama.model
        self._timeout: int = settings.ollama.timeout_seconds

    def health_check(self) -> bool:
        """Verify that the Ollama server is reachable and responsive.

        Returns:
            True if the server responds successfully.

        Raises:
            OllamaConnectionError: If the server is unreachable.
        """
        url = f"{self._base_url}/api/tags"
        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()
            logger.debug("Ollama health check passed at %s.", self._base_url)
            return True
        except Exception as exc:
            raise OllamaConnectionError(
                f"Ollama server is not reachable at '{self._base_url}'. "
                "Ensure Ollama is running and the model is pulled.",
                original_error=exc,
            ) from exc

    def generate(self, prompt: str, format: str = "json") -> str:
        """Send a generation request to Ollama and return the response text.

        Args:
            prompt: The full prompt string.
            format: Response format hint sent to Ollama (``"json"`` or ``""``).

        Returns:
            Raw response text from the model.

        Raises:
            OllamaConnectionError: If the request fails at the network level.
            RAGVerifierError: If the server returns a non-2xx response.
        """
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if format:
            payload["format"] = format

        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.TimeoutException as exc:
            raise OllamaConnectionError(
                f"Ollama request timed out after {self._timeout}s.",
                original_error=exc,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RAGVerifierError(
                f"Ollama returned HTTP {exc.response.status_code}.",
                original_error=exc,
            ) from exc
        except OllamaConnectionError:
            raise
        except Exception as exc:
            raise OllamaConnectionError(
                "Unexpected error communicating with Ollama.",
                original_error=exc,
            ) from exc


class RAGVerifier:
    """Verify claims using Retrieval-Augmented Generation via Ollama.

    Args:
        ollama_client: Configured OllamaClient instance.
        settings: Application settings.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        settings: "Settings",  # noqa: F821
    ) -> None:
        self._client = ollama_client
        self._settings = settings

    def verify(
        self, claim: "Claim", evidences: list[RetrievedEvidence]
    ) -> RAGVerdict:
        """Verify a claim against retrieved evidence using Ollama.

        Args:
            claim: The Claim dataclass to verify.
            evidences: Retrieved evidence passages from Pinecone.

        Returns:
            RAGVerdict with verdict, confidence, and reasoning.

        Raises:
            RAGVerifierError: If generation or parsing fails.
            OllamaConnectionError: If Ollama is unreachable.
        """
        if not evidences:
            return RAGVerdict(
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning="No evidence passages were retrieved for this claim.",
            )

        evidence_passages = self._format_evidence_passages(evidences)
        prompt = RAG_PROMPT_TEMPLATE.format(
            claim=claim.text,
            evidence_passages=evidence_passages,
        )

        raw = self._client.generate(prompt, format="json")
        verdict = self._parse_response(raw)
        verdict.raw_response = raw
        logger.debug(
            "RAG verdict for claim '%s...': %s (confidence=%.2f)",
            claim.text[:50],
            verdict.verdict,
            verdict.confidence,
        )
        return verdict

    def _format_evidence_passages(
        self, evidences: list[RetrievedEvidence]
    ) -> str:
        """Format evidence passages as a numbered list for the prompt.

        Args:
            evidences: List of RetrievedEvidence.

        Returns:
            Multi-line string with indexed passages.
        """
        lines: list[str] = []
        for idx, ev in enumerate(evidences):
            lines.append(
                f"[{idx}] (relevance={ev.score:.2f}, label={ev.label})\n"
                f"    {ev.evidence_text}"
            )
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> RAGVerdict:
        """Parse the raw LLM response into a RAGVerdict.

        First attempts direct JSON parsing; falls back to regex extraction.

        Args:
            raw: Raw string response from Ollama.

        Returns:
            Parsed RAGVerdict.

        Raises:
            RAGVerifierError: If the response cannot be parsed as valid JSON.
        """
        text = raw.strip()

        # Attempt 1: direct parse
        try:
            data = json.loads(text)
            return self._dict_to_verdict(data)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract JSON block via regex
        match = _JSON_EXTRACT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group())
                return self._dict_to_verdict(data)
            except json.JSONDecodeError:
                pass

        raise RAGVerifierError(
            f"Failed to parse RAG response as JSON. Raw response: {raw[:200]!r}"
        )

    def _dict_to_verdict(self, data: dict[str, Any]) -> RAGVerdict:
        """Convert a parsed JSON dict to a RAGVerdict.

        Args:
            data: Parsed JSON dictionary.

        Returns:
            RAGVerdict instance.
        """
        verdict = str(data.get("verdict", "INSUFFICIENT_EVIDENCE")).upper()
        if verdict not in VALID_VERDICTS:
            verdict = "INSUFFICIENT_EVIDENCE"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return RAGVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=str(data.get("reasoning", "")),
            supporting_evidence_ids=[
                int(x) for x in data.get("supporting_evidence_ids", [])
            ],
            contradicting_evidence_ids=[
                int(x) for x in data.get("contradicting_evidence_ids", [])
            ],
        )
