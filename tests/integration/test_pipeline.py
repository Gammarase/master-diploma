"""
Integration tests for DisinformationDetectionPipeline.

All external services (SearXNG, web pages, Ollama) are mocked — no network
connections are required to run these tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claim_extraction.extractor import Claim
from exceptions import SearchBackendError
from explainability.explainer import ExplanationOutput
from pipeline import DisinformationDetectionPipeline, PipelineConfig
from retrieval.evidence import RetrievalResult
from verification.rag_verifier import RAGVerdict


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_pipeline(settings: MagicMock) -> DisinformationDetectionPipeline:
    return DisinformationDetectionPipeline(settings=settings)


def _fully_mock_pipeline(
    pipeline: DisinformationDetectionPipeline,
    mock_retriever: MagicMock,
) -> None:
    """Inject mocked dependencies directly into pipeline internals."""
    pipeline._retriever = mock_retriever

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


@pytest.fixture
def pipeline(
    settings: MagicMock, mock_retriever: MagicMock
) -> DisinformationDetectionPipeline:
    p = _make_pipeline(settings)
    _fully_mock_pipeline(p, mock_retriever)
    return p


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestPipelineFullFlow:
    def test_analyze_returns_explanation_outputs(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        results = pipeline.analyze(
            "Russia launched 45 missiles at Ukraine. "
            "The attack caused significant civilian casualties."
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, ExplanationOutput) for r in results)

    def test_analyze_returns_valid_verdict(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        results = pipeline.analyze("Russia launched 45 missiles at Ukraine.")
        assert results[0].verdict in {"CONFIRMED", "UNCERTAIN", "DISINFORMATION"}

    def test_analyze_empty_text_returns_empty_list(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        assert pipeline.analyze("") == []
        assert pipeline.analyze("   ") == []

    def test_analyze_no_claims_returns_empty_list(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline._claim_extractor.extract_claims.return_value = []
        assert pipeline.analyze("Some text with no claims.") == []

    def test_top_k_override_reaches_retriever(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline.analyze("Russia launched 45 missiles.", PipelineConfig(top_k=3))
        assert pipeline._retriever.retrieve.call_args.kwargs["top_k"] == 3

    def test_pipeline_config_has_no_filter_language(self) -> None:
        assert not hasattr(PipelineConfig(), "filter_language")


class TestAnalyzeSingleClaim:
    def test_returns_explanation_output(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        result = pipeline.analyze_single_claim("Russia launched 45 missiles at Ukraine.")
        assert isinstance(result, ExplanationOutput)

    def test_claim_text_preserved(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        claim_text = "Russia launched 45 missiles at Ukraine."
        assert pipeline.analyze_single_claim(claim_text).claim_text == claim_text

    def test_no_exclude_parameter(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        with pytest.raises(TypeError):
            pipeline.analyze_single_claim(  # type: ignore[call-arg]
                "Some claim.", exclude=("test", "622")
            )

    def test_language_passed_to_retriever(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline.analyze_single_claim("Росія запустила ракети.", language="uk")
        call = pipeline._retriever.retrieve.call_args
        assert call.args[0] == "Росія запустила ракети."
        assert call.kwargs["language"] == "uk"

    def test_citations_carry_web_sources(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        result = pipeline.analyze_single_claim("Russia launched 45 missiles at Ukraine.")
        first = result.citations[0]
        assert first["url"] == "https://apnews.com/article/russia-ukraine-missiles"
        assert first["publisher"] == "apnews.com"
        assert first["trust"] == 0.95
        assert result.processing_metadata["search_status"] == "ok"
        assert result.processing_metadata["search_backend"] == "searxng"


class TestSearchUnavailable:
    def test_unavailable_gives_uncertain_no_evidence(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline._retriever.retrieve.return_value = RetrievalResult(
            evidences=[], status="unavailable", queries=["q"], backend="searxng"
        )
        pipeline._nli_verifier.verify_batch.return_value = []
        pipeline._rag_verifier.verify.return_value = RAGVerdict(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning="No evidence passages were retrieved for this claim.",
        )
        result = pipeline.analyze_single_claim("Russia launched 45 missiles at Ukraine.")
        assert result.verdict == "UNCERTAIN"
        meta = result.processing_metadata
        assert meta["uncertainty_reason"] == "no_evidence"
        assert meta["search_status"] == "unavailable"
        assert "evidence search could not be performed" in result.explanation
        assert result.citations == []


class TestGracefulDegradation:
    def test_ollama_down_does_not_crash_pipeline(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        from exceptions import OllamaConnectionError

        pipeline._rag_verifier.verify.side_effect = OllamaConnectionError(
            "Connection refused"
        )
        results = pipeline.analyze("Russia launched 45 missiles at Ukraine.")
        assert len(results) >= 1
        assert isinstance(results[0], ExplanationOutput)

    def test_uninitialized_pipeline_raises(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with pytest.raises(RuntimeError, match="initialize"):
            pipeline.analyze("Some text.")


class TestHealthCheck:
    def test_returns_searxng_and_ollama(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch(
            "retrieval.web_search.SearxngBackend.health_check", return_value=True
        ), patch(
            "verification.rag_verifier.OllamaClient.health_check", return_value=True
        ):
            status = pipeline.health_check()
        assert status == {"searxng": True, "ollama": True}

    def test_returns_false_when_services_down(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch(
            "retrieval.web_search.SearxngBackend.health_check",
            side_effect=SearchBackendError("down"),
        ), patch(
            "verification.rag_verifier.OllamaClient.health_check",
            side_effect=RuntimeError("down"),
        ):
            status = pipeline.health_check()
        assert status == {"searxng": False, "ollama": False}


class TestClaimDate:
    def test_claim_date_reaches_rag_verifier(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline.analyze_single_claim("Some claim text.", claim_date="2022-09-19")
        claim = pipeline._rag_verifier.verify.call_args.args[0]
        assert claim.date == "2022-09-19"

    def test_default_without_date(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        pipeline.analyze_single_claim("Some claim text.")
        assert pipeline._rag_verifier.verify.call_args.args[0].date is None

    def test_output_has_debug_metadata(
        self, pipeline: DisinformationDetectionPipeline
    ) -> None:
        result = pipeline.analyze_single_claim("Russia launched 45 missiles at Ukraine.")
        meta = result.processing_metadata
        for key in (
            "uncertainty_reason", "effective_weights", "rag_model_verdict",
            "rag_confidence", "rag_raw_response", "evidence_mode",
            "search_backend", "search_status", "search_queries",
            "source_policy_mode", "models",
        ):
            assert key in meta
        assert meta["models"]["reranker"] == "BAAI/bge-reranker-v2-m3"
        assert "embeddings" not in meta["models"]
        assert meta["evidence_mode"] == "web"
        assert meta["source_policy_mode"] == "strict"
        for citation in result.citations:
            assert not {"dataset", "split", "vector_id"} & set(citation)


class TestInitialize:
    def _init(self, pipeline: DisinformationDetectionPipeline) -> MagicMock:
        with patch(
            "retrieval.web_search.SearxngBackend.health_check", return_value=True
        ) as searx_hc, patch(
            "verification.nli_verifier.NLIVerifier._load_model"
        ), patch(
            "verification.rag_verifier.OllamaClient.health_check", return_value=True
        ), patch("pipeline.setup_logging"):
            pipeline.initialize()
        return searx_hc

    def test_builds_web_retriever(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        searx_hc = self._init(pipeline)
        from retrieval.reranker import Reranker
        from retrieval.web_retriever import WebEvidenceRetriever

        assert isinstance(pipeline._reranker, Reranker)
        assert isinstance(pipeline._retriever, WebEvidenceRetriever)
        assert pipeline._retriever._reranker is pipeline._reranker
        assert pipeline._retriever.backend_name == "searxng"
        assert pipeline._initialized is True
        searx_hc.assert_called_once()
        # Second call is a no-op.
        pipeline.initialize()

    def test_query_builder_uses_ollama(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        self._init(pipeline)
        assert pipeline._retriever._queries.enabled is True

    def test_read_only_cache_skips_searxng_health_check(
        self, settings: MagicMock, tmp_path
    ) -> None:
        settings.retrieval.cache_mode = "read_only"
        settings.retrieval.cache_dir = str(tmp_path)
        pipeline = DisinformationDetectionPipeline(settings=settings)
        searx_hc = self._init(pipeline)
        searx_hc.assert_not_called()

    def test_searxng_down_fails_initialize(self, settings: MagicMock) -> None:
        pipeline = DisinformationDetectionPipeline(settings=settings)
        with patch(
            "retrieval.web_search.SearxngBackend.health_check",
            side_effect=SearchBackendError("enable the json format"),
        ), patch("pipeline.setup_logging"):
            with pytest.raises(SearchBackendError, match="json format"):
                pipeline.initialize()
        assert pipeline._initialized is False
