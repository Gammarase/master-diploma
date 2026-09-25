"""
Shared pytest fixtures for the Disinformation Detection System test suite.

All external service interactions (SearXNG, web pages, Ollama) are mocked so that
tests can run without network access or running services.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from claim_extraction.extractor import Claim
from claim_extraction.ner_module import NamedEntity
from preprocessing import PreprocessedText
from retrieval.evidence import RetrievalResult, RetrievedEvidence
from verification.aggregator import VerificationResult
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict


# ─── Slow- and live-test gating ──────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (downloads real models).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Live tests hit SearXNG and the web; run them only when selected with -m live.
    if "live" not in (config.getoption("-m") or ""):
        skip_live = pytest.mark.skip(reason="live test: use -m live to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ─── Settings ────────────────────────────────────────────────────────────────

@pytest.fixture
def settings() -> MagicMock:
    """Return a fully-mocked Settings object with safe test defaults."""
    s = MagicMock()
    # Ollama
    s.ollama.base_url = "http://localhost:11434"
    s.ollama.model = "qwen3:14b"
    s.ollama.temperature = 0.0
    s.ollama.timeout_seconds = 30
    s.ollama.num_ctx = 8192
    s.ollama.seed = 42
    s.ollama.think = False
    s.ollama.evidence_mode = "web"
    # Retrieval
    s.retrieval.top_k = 5
    s.retrieval.candidate_k = 40
    s.retrieval.reranker_model = "BAAI/bge-reranker-v2-m3"
    s.retrieval.device = "cpu"
    s.retrieval.min_relevance = 0.2
    s.retrieval.max_passages_per_source = 2
    s.retrieval.max_passage_chars = 800
    s.retrieval.query_generation = True
    s.retrieval.max_search_requests = 8
    s.retrieval.max_pages_per_claim = 8
    s.retrieval.cache_dir = "cache"
    s.retrieval.cache_mode = "off"
    s.retrieval.cache_max_age_days = None
    # Web search
    s.search.backend = "searxng"
    s.search.base_url = "http://localhost:8080"
    s.search.timeout_seconds = 5
    s.search.min_interval_seconds = 0.0
    s.search.results_per_query = 10
    s.search.site_filter = "grouped"
    s.search.site_group_size = 10
    s.search.fetch_timeout_seconds = 5
    s.search.max_page_bytes = 2_000_000
    s.search.user_agent = "DisinfoDetection-Test/1.0"
    s.search.respect_robots = True
    # Source policy
    s.source_policy.path = "configs/source_policy.yaml"
    s.source_policy.mode = "strict"
    s.source_policy.unknown_trust = 0.3
    # Preprocessing
    s.preprocessing.supported_languages = ["en", "uk"]
    s.preprocessing.spacy_models = {"en": "en_core_web_sm", "uk": "uk_core_news_sm"}
    # Claim extraction
    s.claim_extraction.min_claim_length = 10
    s.claim_extraction.max_claim_length = 512
    s.claim_extraction.checkworthy_threshold = 0.5
    s.claim_extraction.classifier_model = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    s.claim_extraction.ner_weight = 0.3
    s.claim_extraction.classifier_weight = 0.7
    # Verification
    s.verification.nli_model = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    s.verification.nli_weight = 0.3
    s.verification.rag_weight = 0.7
    s.verification.thresholds.disinformation = 0.35
    s.verification.thresholds.confirmed = 0.65
    s.verification.decisiveness_margin = 0.15
    # Logging
    s.logging.level = "DEBUG"
    s.logging.log_file = "logs/test.log"
    s.logging.max_bytes = 1_048_576
    s.logging.backup_count = 1
    return s


# ─── Retrieval mock ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_retriever(sample_evidences: list[RetrievedEvidence]) -> MagicMock:
    """Mock WebEvidenceRetriever returning the sample evidences with status ok."""
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        evidences=list(sample_evidences),
        status="ok",
        queries=["Russia launched 45 missiles at Ukraine."],
        backend="searxng",
    )
    return retriever


# ─── Ollama mock ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_response() -> str:
    """A canned valid JSON response from Ollama."""
    return json.dumps({
        "verdict": "SUPPORTED",
        "confidence": 0.82,
        "reasoning": "The evidence strongly supports the claim.",
        "supporting_evidence_ids": [0],
        "contradicting_evidence_ids": [],
    })


@pytest.fixture
def mock_ollama_client(mock_ollama_response: str) -> MagicMock:
    """Mock OllamaClient with health_check and generate methods."""
    client = MagicMock()
    client.health_check.return_value = True
    client.generate.return_value = mock_ollama_response
    return client


# ─── Domain objects ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_claim() -> Claim:
    """A single Claim instance representing a Russia-Ukraine conflict claim."""
    return Claim(
        text="Russia launched 45 missiles at Ukraine.",
        original_sentence="Russia launched 45 missiles at Ukraine.",
        sentence_index=0,
        language="en",
        entities=[
            NamedEntity("Russia", "GPE", 0, 6),
            NamedEntity("45", "CARDINAL", 15, 17),
            NamedEntity("Ukraine", "GPE", 29, 36),
        ],
        checkworthy_score=0.82,
    )


@pytest.fixture
def sample_evidences() -> list[RetrievedEvidence]:
    """Three web evidence passages from trusted sources."""
    return [
        RetrievedEvidence(
            evidence_id="0f1e2d3c4b5a6978-p0",
            score=0.92,
            evidence_text="Russia conducted a large-scale missile attack on Ukrainian cities.",
            source_url="https://apnews.com/article/russia-ukraine-missiles",
            publisher="apnews.com",
            title="Russia fires missiles at Ukrainian cities",
            published_at="2022-10-10",
            retrieved_at="2026-09-25T10:00:00Z",
            source_tier="wire_agencies",
            trust=0.95,
            language="en",
        ),
        RetrievedEvidence(
            evidence_id="8a7b6c5d4e3f2011-p0",
            score=0.85,
            evidence_text="No evidence of attacks was found in the reviewed period.",
            source_url="https://www.bbc.com/news/world-europe-1",
            publisher="bbc.com",
            title="Ukraine war latest",
            published_at="",
            retrieved_at="2026-09-25T10:00:01Z",
            source_tier="major_outlets",
            trust=0.8,
            language="en",
        ),
        RetrievedEvidence(
            evidence_id="1122334455667788-p1",
            score=0.78,
            evidence_text="Офіційні джерела підтвердили ракетний удар.",
            source_url="https://suspilne.media/123-raketnyi-udar/",
            publisher="suspilne.media",
            title="Ракетний удар",
            published_at="2022-10-10",
            retrieved_at="2026-09-25T10:00:02Z",
            source_tier="major_outlets",
            trust=0.8,
            language="uk",
            passage_index=1,
        ),
    ]


@pytest.fixture
def sample_preprocessed_text() -> PreprocessedText:
    """A PreprocessedText representing an English disinformation claim."""
    return PreprocessedText(
        raw_text=(
            "Russia launched 45 missiles at Ukraine. "
            "The attack caused significant civilian casualties."
        ),
        clean_text=(
            "Russia launched 45 missiles at Ukraine. "
            "The attack caused significant civilian casualties."
        ),
        normalized_text="russia launched 45 missiles ukraine attack caused significant civilian casualties",
        sentences=[
            "Russia launched 45 missiles at Ukraine.",
            "The attack caused significant civilian casualties.",
        ],
        language="en",
        tokens=[
            "Russia", "launched", "45", "missiles", "Ukraine",
            "attack", "caused", "significant", "civilian", "casualties",
        ],
    )
