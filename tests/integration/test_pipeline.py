"""
Integration tests for DisinformationDetectionPipeline.

All external services (Pinecone, Ollama) are mocked — no network
connections are required to run these tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claim_extraction.extractor import Claim
from explainability.explainer import ExplanationOutput
from pipeline import DisinformationDetectionPipeline, PipelineConfig
from retrieval.vector_store import RetrievedEvidence
from verification.rag_verifier import RAGVerdict


# ─── Helpers ─────────────────────────────────────────────────────────────────

_VALID_RAG_JSON = json.dumps({
    "verdict": "SUPPORTED",
    "confidence": 0.82,
    "reasoning": "The evidence strongly supports the claim.",
    "supporting_evidence_ids": [0],
    "contradicting_evidence_ids": [],
})

_TWO_EVIDENCE = [
    RetrievedEvidence(
        vector_id="ru22fact-0-en",
        score=0.92,
        claim_text="Russia launched missiles.",
        evidence_text="Russia conducted missile strikes on Ukrainian cities.",
        label="Supported",
        language="EN",
        explanation="Confirmed by multiple sources.",
    ),
    RetrievedEvidence(
        vector_id="ru22fact-1-en",
        score=0.85,
        claim_text="Ukraine was attacked.",
        evidence_text="Ukrainian authorities confirmed the attack.",
        label="Supported",
        language="EN",
        explanation="Official confirmation.",
    ),
]


def _make_pipeline(settings: MagicMock) -> DisinformationDetectionPipeline:
    return DisinformationDetectionPipeline(settings=settings)


def _fully_mock_pipeline(
    pipeline: DisinformationDetectionPipeline,
    mock_pinecone_index: MagicMock,
    mock_ollama_client: MagicMock,
) -> None:
    """Inject mocked dependencies directly into pipeline internals."""
    import numpy as np

    # Mock embedder
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0] * 768
    pipeline._embedder = embedder

    # Mock vector store
    from retrieval.vector_store import PineconeVectorStore

    store = MagicMock(spec=PineconeVectorStore)
    store.similarity_search.return_value = _TWO_EVIDENCE
    pipeline._vector_store = store

    # Mock NLI verifier
    from verification.nli_verifier import NLIResult, NLIVerifier

    nli_verifier = MagicMock(spec=NLIVerifier)
    nli_verifier.verify_batch.return_value = [
        NLIResult(
            claim="Russia launched missiles.",
            evidence_text="Russia conducted missile strikes.",
            entailment_score=0.8,
            contradiction_score=0.1,
            neutral_score=0.1,
            predicted_label="entailment",
            confidence=0.8,
        )
    ]
    pipeline._nli_verifier = nli_verifier

    # Mock RAG verifier
    from verification.rag_verifier import RAGVerifier

    rag_verifier = MagicMock(spec=RAGVerifier)
    rag_verifier.verify.return_value = RAGVerdict(
        verdict="SUPPORTED",
        confidence=0.82,
        reasoning="Evidence strongly supports the claim.",
        supporting_evidence_ids=[0],
        contradicting_evidence_ids=[],
    )
    pipeline._rag_verifier = rag_verifier

    # Mock claim extractor
    from claim_extraction.extractor import ClaimExtractor

    extractor = MagicMock(spec=ClaimExtractor)
    extractor.extract_claims.return_value = [
        Claim(
            text="Russia launched 45 missiles at Ukraine.",
            original_sentence="Russia launched 45 missiles at Ukraine.",
            sentence_index=0,
            language="en",
            checkworthy_score=0.85,
        )
    ]
    pipeline._claim_extractor = extractor

    pipeline._initialized = True


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPipelineFullFlow:
    def test_analyze_returns_explanation_outputs(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)

        results = pipeline.analyze(
            "Russia launched 45 missiles at Ukraine. "
            "The attack caused significant civilian casualties."
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, ExplanationOutput) for r in results)

    def test_analyze_returns_valid_verdict(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        results = pipeline.analyze("Russia launched 45 missiles at Ukraine.")
        assert results[0].verdict in {"CONFIRMED", "UNCERTAIN", "DISINFORMATION"}

    def test_analyze_empty_text_returns_empty_list(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        assert pipeline.analyze("") == []
        assert pipeline.analyze("   ") == []

    def test_analyze_no_claims_returns_empty_list(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        # Make extractor return no claims
        pipeline._claim_extractor.extract_claims.return_value = []
        results = pipeline.analyze("Some text with no claims.")
        assert results == []


class TestAnalyzeSingleClaim:
    def test_returns_explanation_output(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        result = pipeline.analyze_single_claim(
            "Russia launched 45 missiles at Ukraine."
        )
        assert isinstance(result, ExplanationOutput)

    def test_claim_text_preserved(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        claim_text = "Russia launched 45 missiles at Ukraine."
        result = pipeline.analyze_single_claim(claim_text)
        assert result.claim_text == claim_text


class TestGracefulDegradation:
    def test_ollama_down_does_not_crash_pipeline(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        from exceptions import OllamaConnectionError

        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)

        # Make RAG verifier raise OllamaConnectionError
        pipeline._rag_verifier.verify.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        results = pipeline.analyze("Russia launched 45 missiles at Ukraine.")
        # Should still return results with neutral RAG score
        assert len(results) >= 1
        assert isinstance(results[0], ExplanationOutput)

    def test_uninitialized_pipeline_raises(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with pytest.raises(RuntimeError, match="initialize"):
            pipeline.analyze("Some text.")


class TestHealthCheck:
    def test_returns_dict_with_pinecone_and_ollama(
        self, settings: MagicMock
    ) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)

        with patch("retrieval.vector_store.PineconeVectorStore.connect"), \
             patch("retrieval.embeddings.EmbeddingModel._load"), \
             patch("verification.rag_verifier.OllamaClient.health_check", return_value=True):

            # Mock Pinecone client
            with patch("pinecone.Pinecone") as mock_pc:
                mock_pc.return_value.list_indexes.return_value = []
                mock_pc.return_value.Index.return_value = MagicMock()
                status = pipeline.health_check()

        assert "pinecone" in status
        assert "ollama" in status

    def test_returns_false_when_services_down(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        settings.pinecone.api_key = ""  # Force Pinecone failure
        status = pipeline.health_check()
        assert status["pinecone"] is False


class TestClaimDateAndExclusion:
    def test_exclude_passed_to_retrieval(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        pipeline.analyze_single_claim(
            "Russia launched 45 missiles at Ukraine.",
            language="en",
            claim_date="2022-09-19",
            exclude=("test", "622"),
        )
        kwargs = pipeline._vector_store.similarity_search.call_args.kwargs
        assert kwargs["exclude"] == ("test", "622")

    def test_claim_date_reaches_rag_verifier(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        pipeline.analyze_single_claim("Some claim text.", claim_date="2022-09-19")
        claim = pipeline._rag_verifier.verify.call_args.args[0]
        assert claim.date == "2022-09-19"

    def test_defaults_without_date_or_exclusion(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        pipeline.analyze_single_claim("Some claim text.")
        assert pipeline._vector_store.similarity_search.call_args.kwargs["exclude"] is None
        assert pipeline._rag_verifier.verify.call_args.args[0].date is None

    def test_output_has_debug_metadata(
        self,
        settings: MagicMock,
        mock_pinecone_index: MagicMock,
        mock_ollama_client: MagicMock,
    ) -> None:
        pipeline = _make_pipeline(settings)
        _fully_mock_pipeline(pipeline, mock_pinecone_index, mock_ollama_client)
        result = pipeline.analyze_single_claim("Russia launched 45 missiles at Ukraine.")
        meta = result.processing_metadata
        for key in (
            "uncertainty_reason", "effective_weights", "rag_model_verdict",
            "rag_confidence", "rag_raw_response", "evidence_mode", "models",
        ):
            assert key in meta
        assert meta["models"]["reranker"] == "BAAI/bge-reranker-v2-m3"
        assert meta["evidence_mode"] == "evidence_only"
        for excerpt in result.evidence_excerpts:
            assert "label_from_dataset" not in excerpt


class TestInitialize:
    def test_reranker_injected_into_vector_store(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch("retrieval.vector_store.PineconeVectorStore.connect"), \
             patch("verification.nli_verifier.NLIVerifier._load_model"), \
             patch("verification.rag_verifier.OllamaClient.health_check", return_value=True), \
             patch("pipeline.setup_logging"):
            pipeline.initialize()
        from retrieval.reranker import Reranker

        assert isinstance(pipeline._reranker, Reranker)
        assert pipeline._vector_store._reranker is pipeline._reranker
        assert pipeline._initialized is True
        # Second call is a no-op.
        pipeline.initialize()
