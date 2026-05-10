"""Unit tests for verification.rag_verifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from exceptions import OllamaConnectionError, RAGVerifierError
from retrieval.vector_store import RetrievedEvidence
from verification.rag_verifier import OllamaClient, RAGVerdict, RAGVerifier


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.ollama.base_url = "http://localhost:11434"
    settings.ollama.model = "llama3"
    settings.ollama.timeout_seconds = 60
    return settings


@pytest.fixture
def ollama_client(mock_settings: MagicMock) -> OllamaClient:
    return OllamaClient(mock_settings)


@pytest.fixture
def mock_claim() -> MagicMock:
    claim = MagicMock()
    claim.text = "Russia launched 45 missiles at Ukraine."
    return claim


def _make_evidence(idx: int = 0, label: str = "Supported") -> RetrievedEvidence:
    return RetrievedEvidence(
        vector_id=f"test-{idx}",
        score=0.9,
        claim_text="claim",
        evidence_text=f"Evidence passage {idx}.",
        label=label,
        language="EN",
        explanation="Explanation.",
    )


_VALID_JSON_RESPONSE = json.dumps({
    "verdict": "SUPPORTED",
    "confidence": 0.85,
    "reasoning": "Evidence strongly supports the claim.",
    "supporting_evidence_ids": [0, 1],
    "contradicting_evidence_ids": [],
})

_REFUTED_JSON_RESPONSE = json.dumps({
    "verdict": "REFUTED",
    "confidence": 0.75,
    "reasoning": "Evidence contradicts the claim.",
    "supporting_evidence_ids": [],
    "contradicting_evidence_ids": [0],
})


# ─── OllamaClient Tests ───────────────────────────────────────────────────────

class TestOllamaClientHealthCheck:
    def test_returns_true_on_success(self, ollama_client: OllamaClient) -> None:
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            assert ollama_client.health_check() is True

    def test_raises_on_connection_error(self, ollama_client: OllamaClient) -> None:
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            with pytest.raises(OllamaConnectionError):
                ollama_client.health_check()


class TestOllamaClientGenerate:
    def test_returns_response_text(self, ollama_client: OllamaClient) -> None:
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": _VALID_JSON_RESPONSE}
            mock_post.return_value = mock_response
            result = ollama_client.generate("Test prompt")
        assert result == _VALID_JSON_RESPONSE

    def test_raises_on_timeout(self, ollama_client: OllamaClient) -> None:
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OllamaConnectionError):
                ollama_client.generate("prompt")

    def test_raises_on_http_error(self, ollama_client: OllamaClient) -> None:
        import httpx

        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            error = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)
            mock_response.raise_for_status.side_effect = error
            mock_post.return_value = mock_response
            with pytest.raises(RAGVerifierError):
                ollama_client.generate("prompt")


# ─── RAGVerdict Tests ─────────────────────────────────────────────────────────

class TestRAGVerdictToScore:
    def test_supported_returns_confidence(self) -> None:
        v = RAGVerdict(verdict="SUPPORTED", confidence=0.8, reasoning="")
        assert v.to_score() == pytest.approx(0.8)

    def test_refuted_returns_inverted(self) -> None:
        v = RAGVerdict(verdict="REFUTED", confidence=0.7, reasoning="")
        assert v.to_score() == pytest.approx(0.3)

    def test_insufficient_returns_neutral(self) -> None:
        v = RAGVerdict(verdict="INSUFFICIENT_EVIDENCE", confidence=0.9, reasoning="")
        assert v.to_score() == pytest.approx(0.5)

    def test_confidence_clamped_at_1(self) -> None:
        v = RAGVerdict(verdict="SUPPORTED", confidence=1.5, reasoning="")
        assert v.to_score() == pytest.approx(1.0)


# ─── RAGVerifier Tests ────────────────────────────────────────────────────────

class TestRAGVerifier:
    @pytest.fixture
    def mock_ollama(self) -> MagicMock:
        client = MagicMock(spec=OllamaClient)
        client.generate.return_value = _VALID_JSON_RESPONSE
        return client

    @pytest.fixture
    def verifier(self, mock_ollama: MagicMock, mock_settings: MagicMock) -> RAGVerifier:
        return RAGVerifier(mock_ollama, mock_settings)

    def test_returns_rag_verdict(
        self, verifier: RAGVerifier, mock_claim: MagicMock
    ) -> None:
        evidences = [_make_evidence(0), _make_evidence(1)]
        result = verifier.verify(mock_claim, evidences)
        assert isinstance(result, RAGVerdict)
        assert result.verdict == "SUPPORTED"

    def test_empty_evidences_returns_insufficient(
        self, verifier: RAGVerifier, mock_claim: MagicMock
    ) -> None:
        result = verifier.verify(mock_claim, [])
        assert result.verdict == "INSUFFICIENT_EVIDENCE"

    def test_parse_response_valid_json(self, verifier: RAGVerifier) -> None:
        verdict = verifier._parse_response(_VALID_JSON_RESPONSE)
        assert verdict.verdict == "SUPPORTED"
        assert verdict.confidence == pytest.approx(0.85)

    def test_parse_response_extracts_json_from_text(
        self, verifier: RAGVerifier
    ) -> None:
        wrapped = f"Here is the result:\n{_VALID_JSON_RESPONSE}\nDone."
        verdict = verifier._parse_response(wrapped)
        assert verdict.verdict == "SUPPORTED"

    def test_parse_response_raises_on_invalid(self, verifier: RAGVerifier) -> None:
        with pytest.raises(RAGVerifierError):
            verifier._parse_response("This is not JSON at all.")

    def test_normalizes_invalid_verdict(self, verifier: RAGVerifier) -> None:
        response = json.dumps({
            "verdict": "UNKNOWN_LABEL",
            "confidence": 0.5,
            "reasoning": "test",
        })
        verdict = verifier._parse_response(response)
        assert verdict.verdict == "INSUFFICIENT_EVIDENCE"
