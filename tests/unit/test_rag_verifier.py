"""Unit tests for verification.rag_verifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from claim_extraction.extractor import Claim
from exceptions import OllamaConnectionError, RAGVerifierError
from retrieval.vector_store import RetrievedEvidence
from verification.rag_verifier import (
    _FORMAT_SCHEMA,
    OllamaClient,
    RAGVerdict,
    RAGVerifier,
    estimate_tokens,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.ollama.base_url = "http://localhost:11434"
    settings.ollama.model = "qwen3:14b"
    settings.ollama.timeout_seconds = 60
    settings.ollama.temperature = 0.0
    settings.ollama.num_ctx = 8192
    settings.ollama.seed = 42
    settings.ollama.think = False
    settings.ollama.evidence_mode = "evidence_only"
    return settings


@pytest.fixture
def ollama_client(mock_settings: MagicMock) -> OllamaClient:
    return OllamaClient(mock_settings)


@pytest.fixture
def mock_claim() -> MagicMock:
    claim = MagicMock()
    claim.text = "Russia launched 45 missiles at Ukraine."
    return claim


def _make_claim(date: str | None = None) -> Claim:
    text = "Russia launched 45 missiles at Ukraine."
    return Claim(
        text=text, original_sentence=text, sentence_index=0, language="en", date=date
    )


def _make_evidence(
    idx: int = 0,
    label: str = "Supported",
    claim_text: str = "claim",
    text: str | None = None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        vector_id=f"test-{idx}",
        score=0.9,
        claim_text=claim_text,
        evidence_text=text if text is not None else f"Evidence passage {idx}.",
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

    def test_sends_options_and_think(self, ollama_client: OllamaClient) -> None:
        with patch("httpx.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "{}"}
            ollama_client.generate("Test prompt", fmt=_FORMAT_SCHEMA)
        body = mock_post.call_args.kwargs["json"]
        assert body["options"] == {"temperature": 0.0, "num_ctx": 8192, "seed": 42}
        assert body["think"] is False
        assert body["model"] == "qwen3:14b"
        assert body["format"] is _FORMAT_SCHEMA
        assert body["stream"] is False

    def test_think_flag_follows_config(self, mock_settings: MagicMock) -> None:
        mock_settings.ollama.think = True
        with patch("httpx.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": ""}
            OllamaClient(mock_settings).generate("p")
        body = mock_post.call_args.kwargs["json"]
        assert body["think"] is True
        assert "format" not in body

    def test_raises_on_unexpected_error(self, ollama_client: OllamaClient) -> None:
        with patch("httpx.post", side_effect=ValueError("boom")):
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

    def test_empty_response_is_insufficient(self, verifier: RAGVerifier) -> None:
        verdict = verifier._parse_response("   ")
        assert verdict.verdict == "INSUFFICIENT_EVIDENCE"
        assert verdict.confidence == pytest.approx(0.5)

    def test_unknown_verdict_mapped(self, verifier: RAGVerifier) -> None:
        response = json.dumps({"verdict": "MOSTLY_TRUE", "confidence": 0.9})
        assert verifier._parse_response(response).verdict == "INSUFFICIENT_EVIDENCE"

    def test_confidence_clamped(self, verifier: RAGVerifier) -> None:
        response = json.dumps({"verdict": "SUPPORTED", "confidence": 1.7})
        assert verifier._parse_response(response).confidence == 1.0

    def test_raw_response_kept(
        self, verifier: RAGVerifier, mock_claim: MagicMock
    ) -> None:
        result = verifier.verify(mock_claim, [_make_evidence(0)])
        assert result.raw_response == _VALID_JSON_RESPONSE


# ─── Prompt contract ──────────────────────────────────────────────────────────

class TestPromptContract:
    @pytest.fixture
    def mock_ollama(self) -> MagicMock:
        client = MagicMock(spec=OllamaClient)
        client.generate.return_value = _VALID_JSON_RESPONSE
        return client

    def _verifier(
        self, mock_ollama: MagicMock, mock_settings: MagicMock, mode: str
    ) -> RAGVerifier:
        mock_settings.ollama.evidence_mode = mode
        return RAGVerifier(mock_ollama, mock_settings)

    def _sent_prompt(self, mock_ollama: MagicMock) -> str:
        return mock_ollama.generate.call_args.args[0]

    def test_reasoning_is_first_schema_property(self) -> None:
        assert next(iter(_FORMAT_SCHEMA["properties"])) == "reasoning"
        assert _FORMAT_SCHEMA["required"][0] == "reasoning"

    def test_schema_sent_with_request(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "evidence_only")
        verifier.verify(_make_claim(), [_make_evidence(0)])
        assert mock_ollama.generate.call_args.kwargs["fmt"] is _FORMAT_SCHEMA

    def test_default_mode_hides_labels(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "evidence_only")
        evidence = _make_evidence(0, label="Refuted", claim_text="Russia captured Kyiv")
        verifier.verify(_make_claim(), [evidence])
        prompt = self._sent_prompt(mock_ollama)
        assert "Refuted" not in prompt
        assert "label=" not in prompt
        assert "Russia captured Kyiv" not in prompt
        assert "Explanation." not in prompt
        assert "[0] Evidence passage 0." in prompt

    def test_fact_checked_mode_pairs_label_with_claim(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "fact_checked_claims")
        evidence = _make_evidence(0, label="Refuted", claim_text="Russia captured Kyiv")
        verifier.verify(_make_claim(), [evidence])
        prompt = self._sent_prompt(mock_ollama)
        claim_pos = prompt.index("Russia captured Kyiv")
        assert prompt.index("FALSE", claim_pos) > claim_pos
        assert "Refuted" not in prompt
        assert "invert" in prompt

    @pytest.mark.parametrize(
        "label,expected",
        [("Supported", "TRUE"), ("NEI", "UNVERIFIED"), ("weird", "UNVERIFIED")],
    )
    def test_fact_checked_label_mapping(
        self,
        mock_ollama: MagicMock,
        mock_settings: MagicMock,
        label: str,
        expected: str,
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "fact_checked_claims")
        verifier.verify(_make_claim(), [_make_evidence(0, label=label)])
        assert f"Fact-check verdict: {expected}" in self._sent_prompt(mock_ollama)

    def test_date_included(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "evidence_only")
        verifier.verify(_make_claim(date="2022-09-19"), [_make_evidence(0)])
        assert "2022-09-19" in self._sent_prompt(mock_ollama)

    def test_no_date_line_when_unknown(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "evidence_only")
        verifier.verify(_make_claim(date=None), [_make_evidence(0)])
        assert "CLAIM DATE" not in self._sent_prompt(mock_ollama)

    def test_unrelated_events_instruction(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "evidence_only")
        verifier.verify(_make_claim(), [_make_evidence(0)])
        prompt = self._sent_prompt(mock_ollama)
        assert "Ignore passages that describe different events" in prompt

    def test_unknown_mode_rejected(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="evidence_mode"):
            self._verifier(mock_ollama, mock_settings, "labels")

    def test_evidence_mode_property(
        self, mock_ollama: MagicMock, mock_settings: MagicMock
    ) -> None:
        verifier = self._verifier(mock_ollama, mock_settings, "fact_checked_claims")
        assert verifier.evidence_mode == "fact_checked_claims"


class TestPromptSizeGuard:
    @pytest.fixture
    def verifier(self, mock_settings: MagicMock) -> RAGVerifier:
        client = MagicMock(spec=OllamaClient)
        client.generate.return_value = _VALID_JSON_RESPONSE
        return RAGVerifier(client, mock_settings)

    def test_estimate_tokens(self) -> None:
        assert estimate_tokens("abc") == pytest.approx(1.0)
        assert estimate_tokens("ж") == pytest.approx(2 / 3)
        assert estimate_tokens("乌") == pytest.approx(1.0)

    def test_small_prompt_untouched(
        self, verifier: RAGVerifier, caplog: pytest.LogCaptureFixture
    ) -> None:
        evidences = [_make_evidence(i) for i in range(5)]
        prompt = verifier.build_prompt(_make_claim(), evidences)
        assert all(f"Evidence passage {i}." in prompt for i in range(5))
        assert "exceeds" not in caplog.text

    def test_oversized_evidence_shortened(
        self, verifier: RAGVerifier, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 5 passages of ~7,200 bytes (Cyrillic, 2 bytes/char) ≈ 12,000 tokens.
        evidences = [
            _make_evidence(i, text=f"P{i} " + "ж" * 3600) for i in range(5)
        ]
        claim = _make_claim(date="2022-09-19")
        unguarded = verifier._render(
            claim, evidences, [e.evidence_text for e in evidences]
        )
        assert estimate_tokens(unguarded) > 12_000

        with caplog.at_level("WARNING"):
            prompt = verifier.build_prompt(claim, evidences)

        assert estimate_tokens(prompt) <= 7373
        assert "exceeds" in caplog.text
        # Claim, date and instructions intact; top-ranked passage kept whole.
        assert f"CLAIM: {claim.text}" in prompt
        assert "CLAIM DATE: 2022-09-19" in prompt
        assert "Return ONLY the JSON object" in prompt
        assert evidences[0].evidence_text in prompt
        # Lowest-ranked passage is the one dropped.
        assert "P4 " not in prompt

    def test_last_passage_shortened_not_dropped(self, verifier: RAGVerifier) -> None:
        evidences = [
            _make_evidence(0, text="a" * 15000),
            _make_evidence(1, text="b" * 10000),
        ]
        prompt = verifier.build_prompt(_make_claim(), evidences)
        assert estimate_tokens(prompt) <= 0.9 * 8192
        assert "a" * 15000 in prompt
        assert "[1] bbb" in prompt and "b …" in prompt

    def test_claim_never_truncated_even_if_too_long(
        self, verifier: RAGVerifier
    ) -> None:
        long_text = "x" * 30000
        claim = Claim(
            text=long_text, original_sentence=long_text, sentence_index=0, language="en"
        )
        prompt = verifier.build_prompt(claim, [_make_evidence(0)])
        assert long_text in prompt
        assert "Evidence passage 0." not in prompt
