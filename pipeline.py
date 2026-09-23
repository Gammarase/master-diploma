"""
Main pipeline for the Disinformation Detection System.

Orchestrates all modules — preprocessing, claim extraction, retrieval,
verification, and explainability — into a single public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from configs import Settings, get_settings
from exceptions import OllamaConnectionError, VectorStoreError
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
        filter_language: If set, restrict retrieval to this language code.
        top_k: Override the default top-k for evidence retrieval.
    """

    run_ner: bool = True
    filter_language: str | None = None
    top_k: int | None = None


class DisinformationDetectionPipeline:
    """End-to-end disinformation detection pipeline.

    All heavy components (ML models, Pinecone, Ollama) are lazy-loaded
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
        self._embedder: Any | None = None
        self._reranker: Any | None = None
        self._vector_store: Any | None = None
        self._nli_verifier: Any | None = None
        self._rag_verifier: Any | None = None
        self._claim_extractor: Any | None = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """Load all models and establish connections to Pinecone and Ollama.

        Call this once before using ``analyze()`` or ``analyze_single_claim()``.

        Raises:
            EmbeddingError: If the embedding model cannot be loaded.
            PineconeConnectionError: If Pinecone is unreachable or misconfigured.
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

        # Embedding model
        from retrieval.embeddings import EmbeddingModel

        self._embedder = EmbeddingModel(self._settings)

        # Reranker (lazy-loaded on first query; disabled when not configured)
        from retrieval.reranker import Reranker

        self._reranker = Reranker(self._settings)

        # Pinecone vector store
        from retrieval.vector_store import PineconeVectorStore

        self._vector_store = PineconeVectorStore(
            self._settings, self._embedder, reranker=self._reranker
        )
        self._vector_store.connect()

        # NLI verifier
        from verification.nli_verifier import NLIVerifier

        self._nli_verifier = NLIVerifier(self._settings)
        self._nli_verifier._load_model()

        # Ollama client + RAG verifier
        from verification.rag_verifier import OllamaClient, RAGVerifier

        ollama_client = OllamaClient(self._settings)
        ollama_client.health_check()
        self._rag_verifier = RAGVerifier(ollama_client, self._settings)

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
        exclude: tuple[str, str] | None = None,
    ) -> ExplanationOutput:
        """Verify a single claim string directly, skipping extraction.

        Useful for interactive use and evaluation when the claim text is
        already known.

        Args:
            claim_text: The claim to verify.
            language: Language code for the claim.
            claim_date: When the claim was made, if known; shown to the LLM.
            exclude: ``(split, dataset_id)`` of the claim's own record, so its
                evidence is not retrieved when the claim itself is indexed.

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
        return self._process_claim(claim, PipelineConfig(), exclude=exclude)

    def health_check(self) -> dict[str, bool]:
        """Check connectivity for all external services.

        Returns:
            Dict with keys ``"pinecone"`` and ``"ollama"``, values True/False.
        """
        status: dict[str, bool] = {"pinecone": False, "ollama": False}

        try:
            from retrieval.embeddings import EmbeddingModel
            from retrieval.vector_store import PineconeVectorStore

            embedder = EmbeddingModel(self._settings)
            store = PineconeVectorStore(self._settings, embedder)
            store.connect()
            status["pinecone"] = True
        except Exception as exc:
            logger.warning("Pinecone health check failed: %s", exc)

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
        exclude: tuple[str, str] | None = None,
    ) -> ExplanationOutput:
        """Run the retrieval → verification → explanation steps for one claim.

        Gracefully degrades to NLI-only if Ollama is unreachable.

        Args:
            claim: A Claim dataclass instance (its ``date`` reaches the LLM).
            cfg: PipelineConfig for this invocation.
            exclude: ``(split, dataset_id)`` of a record to exclude from
                retrieval.

        Returns:
            ExplanationOutput.
        """
        # Retrieve evidence
        evidences = self._vector_store.similarity_search(
            claim.text,
            top_k=cfg.top_k,
            filter_language=cfg.filter_language,
            exclude=exclude,
        )

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
        return self._explainer.explain(verification_result)

    def _ensure_initialized(self) -> None:
        """Raise RuntimeError if ``initialize()`` has not been called."""
        if not self._initialized:
            raise RuntimeError(
                "Pipeline is not initialized. Call pipeline.initialize() first."
            )
