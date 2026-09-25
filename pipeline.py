"""
Main pipeline for the Disinformation Detection System.

Orchestrates all modules — preprocessing, claim extraction, retrieval,
verification, and explainability — into a single public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from configs import Settings, get_settings
from explainability.explainer import ExplanationOutput, Explainer
from logging_config import get_logger, setup_logging
from preprocessing import PreprocessingPipeline, PreprocessedText
from verification.aggregator import ResultAggregator

__all__ = ["DisinformationDetectionPipeline", "PipelineConfig"]

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline invocation.

    Attributes:
        run_ner: Whether to run NER-based claim scoring (default True).
        top_k: Override the default top-k for evidence retrieval.
    """

    run_ner: bool = True
    top_k: int | None = None


class DisinformationDetectionPipeline:
    """End-to-end disinformation detection pipeline.

    All heavy components (ML models, SearXNG, Ollama) are lazy-loaded
    inside ``initialize()``. The pipeline can be instantiated cheaply
    for testing by never calling ``initialize()``.

    Args:
        settings: Application settings. Uses ``get_settings()`` when None.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._preprocessing = PreprocessingPipeline(self._settings)
        self._aggregator = ResultAggregator(self._settings)
        self._explainer = Explainer(self._settings)

        # Lazily-initialized components
        self._reranker: Any | None = None
        self._retriever: Any | None = None
        self._nli_verifier: Any | None = None
        self._rag_verifier: Any | None = None
        self._claim_extractor: Any | None = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """Load all models and connect to SearXNG and Ollama.

        Call this once before using ``analyze()`` or ``analyze_single_claim()``.

        Raises:
            RetrievalError: If the source policy file is missing or invalid.
            SearchBackendError: If SearXNG is unreachable or does not serve
                the JSON format (skipped when the cache is ``read_only``).
            OllamaConnectionError: If the Ollama server is not running.
            NLIError: If the NLI model cannot be loaded.
        """
        if self._initialized:
            return

        logger.info("Initializing DisinformationDetectionPipeline...")

        # Setup logging from settings
        setup_logging(
            level=self._settings.logging.level,
            log_file=self._settings.logging.log_file,
        )

        # Reranker (lazy-loaded on first query; disabled when not configured)
        from retrieval.reranker import Reranker

        self._reranker = Reranker(self._settings)

        # Source policy, cache, search backend and page fetcher
        from retrieval.cache import WebCache
        from retrieval.page_fetcher import PageFetcher
        from retrieval.source_policy import SourcePolicy
        from retrieval.web_search import SearxngBackend

        policy = SourcePolicy.from_settings(self._settings)
        cache = WebCache.from_settings(self._settings)
        backend = SearxngBackend.from_settings(self._settings)
        if cache.offline:
            logger.info("Cache is read_only: skipping the SearXNG health check.")
        else:
            backend.health_check()
        fetcher = PageFetcher.from_settings(self._settings, policy, cache)

        # NLI verifier
        from verification.nli_verifier import NLIVerifier

        self._nli_verifier = NLIVerifier(self._settings)
        self._nli_verifier._load_model()

        # Ollama client + RAG verifier
        from verification.rag_verifier import OllamaClient, RAGVerifier

        ollama_client = OllamaClient(self._settings)
        ollama_client.health_check()
        self._rag_verifier = RAGVerifier(ollama_client, self._settings)

        # Query builder + web evidence retriever
        from retrieval.query_builder import QueryBuilder
        from retrieval.web_retriever import WebEvidenceRetriever

        query_builder = QueryBuilder(
            ollama_client.generate,
            enabled=self._settings.retrieval.query_generation,
        )
        self._retriever = WebEvidenceRetriever(
            self._settings,
            backend,
            fetcher,
            policy,
            query_builder,
            self._reranker,
            cache=cache,
        )

        # Claim extractor (lazy model load on first use)
        from claim_extraction.extractor import ClaimExtractor
        from claim_extraction.ner_module import NERModule

        ner_module = NERModule(self._preprocessing.tokenizer)
        self._claim_extractor = ClaimExtractor(
            ner_module, self._preprocessing.tokenizer, self._settings
        )

        self._initialized = True
        logger.info("Pipeline initialization complete.")

    def analyze(
        self,
        text: str,
        config: PipelineConfig | None = None,
    ) -> list[ExplanationOutput]:
        """Run the full disinformation detection pipeline on *text*.

        Steps:
        1. Preprocess → PreprocessedText
        2. Extract claims → list[Claim]
        3. For each claim: retrieve evidence, NLI verify, RAG verify, aggregate
        4. Generate explanations

        Args:
            text: Raw input text to analyze.
            config: Optional per-call configuration overrides.

        Returns:
            List of ExplanationOutput, one per extracted claim.
            Returns an empty list if no claims are found.
        """
        self._ensure_initialized()
        cfg = config or PipelineConfig()

        if not text or not text.strip():
            logger.debug("Empty text passed to analyze() — returning empty list.")
            return []

        preprocessed = self._preprocessing.process(text)
        claims = self._claim_extractor.extract_claims(preprocessed)

        if not claims:
            logger.info("No check-worthy claims found in the input text.")
            return []

        logger.info(
            "Processing %d claims extracted from input text.", len(claims)
        )
        outputs: list[ExplanationOutput] = []

        for claim in claims:
            try:
                output = self._process_claim(claim, cfg)
                outputs.append(output)
            except Exception as exc:
                logger.error(
                    "Failed to process claim '%s...': %s",
                    claim.text[:50],
                    exc,
                )
                continue

        return outputs

    def analyze_single_claim(
        self,
        claim_text: str,
        language: str = "en",
        claim_date: str | None = None,
    ) -> ExplanationOutput:
        """Verify a single claim string directly, skipping extraction.

        Useful for interactive use and evaluation when the claim text is
        already known.

        Args:
            claim_text: The claim to verify.
            language: Language code for the claim.
            claim_date: When the claim was made, if known; shown to the LLM.

        Returns:
            ExplanationOutput for the claim.
        """
        from claim_extraction.extractor import Claim

        self._ensure_initialized()
        claim = Claim(
            text=claim_text,
            original_sentence=claim_text,
            sentence_index=0,
            language=language,
            date=claim_date,
        )
        return self._process_claim(claim, PipelineConfig())

    def health_check(self) -> dict[str, bool]:
        """Check connectivity for all external services.

        Returns:
            Dict with keys ``"searxng"`` and ``"ollama"``, values True/False.
        """
        status: dict[str, bool] = {"searxng": False, "ollama": False}

        try:
            from retrieval.web_search import SearxngBackend

            status["searxng"] = SearxngBackend.from_settings(
                self._settings
            ).health_check()
        except Exception as exc:
            logger.warning("SearXNG health check failed: %s", exc)

        try:
            from verification.rag_verifier import OllamaClient

            client = OllamaClient(self._settings)
            status["ollama"] = client.health_check()
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)

        return status

    def _process_claim(
        self,
        claim: Any,
        cfg: PipelineConfig,
    ) -> ExplanationOutput:
        """Run the retrieval → verification → explanation steps for one claim.

        Gracefully degrades to NLI-only if Ollama is unreachable. A failed
        search yields no evidence (status ``unavailable``), not an error.

        Args:
            claim: A Claim dataclass instance (its ``date`` reaches the LLM).
            cfg: PipelineConfig for this invocation.

        Returns:
            ExplanationOutput.
        """
        # Retrieve evidence from the web
        retrieval = self._retriever.retrieve(
            claim.text, language=claim.language, top_k=cfg.top_k
        )
        evidences = retrieval.evidences

        # NLI verification
        nli_results = self._nli_verifier.verify_batch(claim.text, evidences)

        # RAG verification with graceful degradation
        from exceptions import RAGVerifierError
        from verification.rag_verifier import RAGVerdict

        try:
            rag_verdict = self._rag_verifier.verify(claim, evidences)
        except RAGVerifierError as exc:
            logger.warning(
                "RAG verification failed for claim '%s...'. Using neutral RAG score. Error: %s",
                claim.text[:50],
                exc,
            )
            rag_verdict = RAGVerdict(
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.5,
                reasoning="RAG verification was unavailable or produced an unparseable response.",
            )

        # Aggregate
        verification_result = self._aggregator.aggregate(
            claim=claim,
            nli_results=nli_results,
            rag_verdict=rag_verdict,
            retrieved_evidences=evidences,
        )

        # Explain
        return self._explainer.explain(verification_result, retrieval)

    def _ensure_initialized(self) -> None:
        """Raise RuntimeError if ``initialize()`` has not been called."""
        if not self._initialized:
            raise RuntimeError(
                "Pipeline is not initialized. Call pipeline.initialize() first."
            )
