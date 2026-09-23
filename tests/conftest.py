"""
Shared pytest fixtures for the Disinformation Detection System test suite.

All external service interactions (Pinecone, Ollama) are mocked so that
tests can run without network access or running services.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from claim_extraction.extractor import Claim
from claim_extraction.ner_module import NamedEntity
from preprocessing import PreprocessedText
from retrieval.vector_store import RetrievedEvidence
from verification.aggregator import VerificationResult
from verification.nli_verifier import NLIResult
from verification.rag_verifier import RAGVerdict


# ─── Slow-test gating ────────────────────────────────────────────────────────

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
    # Pinecone
    s.pinecone.api_key = "test-pinecone-key"
    s.pinecone.index_name = "test-index"
    s.pinecone.cloud = "aws"
    s.pinecone.region = "us-east-1"
    s.pinecone.top_k = 3
    # Ollama
    s.ollama.base_url = "http://localhost:11434"
    s.ollama.model = "qwen3:14b"
    s.ollama.temperature = 0.0
    s.ollama.timeout_seconds = 30
    s.ollama.num_ctx = 8192
    s.ollama.seed = 42
    s.ollama.think = False
    s.ollama.evidence_mode = "evidence_only"
    # Embeddings
    s.embeddings.model_name = "BAAI/bge-m3"
    s.embeddings.device = "cpu"
    s.embeddings.batch_size = 32
    # Retrieval / indexing
    s.retrieval.candidate_k = 20
    s.retrieval.reranker_model = "BAAI/bge-reranker-v2-m3"
    s.retrieval.min_relevance = 0.2
    s.retrieval.max_passages_per_record = 2
    s.indexing.max_passage_chars = 800
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


# ─── Pinecone mock ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    """Mock pinecone.Index with upsert and query methods."""
    index = MagicMock()
    index.upsert.return_value = None
    index.query.return_value = {
        "matches": [
            {
                "id": "ru22fact-0-en",
                "score": 0.92,
                "metadata": {
                    "claim_id": "0",
                    "claim_text": "Russia launched 45 missiles at Ukraine.",
                    "evidence_text": (
                        "According to official Ukrainian reports, "
                        "Russia launched a series of missile strikes."
                    ),
                    "label": "Supported",
                    "language": "EN",
                    "explanation": "Multiple sources confirm the attack.",
                },
            },
            {
                "id": "ru22fact-1-en",
                "score": 0.85,
                "metadata": {
                    "claim_id": "1",
                    "claim_text": "NATO did not respond to the attack.",
                    "evidence_text": (
                        "NATO condemned the attacks and pledged additional support."
                    ),
                    "label": "Refuted",
                    "language": "EN",
                    "explanation": "NATO did respond with statements and aid.",
                },
            },
        ]
    }
    return index


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
    """Three RetrievedEvidence instances with mixed labels."""
    return [
        RetrievedEvidence(
            vector_id="ru22fact-0-en",
            score=0.92,
            claim_text="Russia launched missiles at Ukraine.",
            evidence_text="Russia conducted a large-scale missile attack on Ukrainian cities.",
            label="Supported",
            language="EN",
            explanation="Multiple official sources confirmed the missile strikes.",
        ),
        RetrievedEvidence(
            vector_id="ru22fact-1-en",
            score=0.85,
            claim_text="Ukraine was not attacked.",
            evidence_text="No evidence of attacks was found in the reviewed period.",
            label="Refuted",
            language="EN",
            explanation="Contradicts verified reports.",
        ),
        RetrievedEvidence(
            vector_id="ru22fact-2-uk",
            score=0.78,
            claim_text="Росія атакувала Україну.",
            evidence_text="Офіційні джерела підтвердили ракетний удар.",
            label="Supported",
            language="UK",
            explanation="Підтверджено офіційними джерелами.",
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
