"""Unit tests for claim_extraction.extractor.ClaimExtractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claim_extraction.extractor import Claim, ClaimExtractor
from claim_extraction.ner_module import NERModule, NamedEntity
from preprocessing import PreprocessedText


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.claim_extraction.checkworthy_threshold = 0.5
    settings.claim_extraction.min_claim_length = 10
    settings.claim_extraction.max_claim_length = 512
    settings.claim_extraction.classifier_model = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    settings.claim_extraction.ner_weight = 0.3
    settings.claim_extraction.classifier_weight = 0.7
    return settings


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    tokenizer = MagicMock()
    tokenizer.get_doc.side_effect = lambda text, lang="en": nlp(text)
    return tokenizer


@pytest.fixture
def mock_ner() -> MagicMock:
    ner = MagicMock(spec=NERModule)
    ner.extract_entities.return_value = [
        NamedEntity("Russia", "GPE", 0, 6),
        NamedEntity("45", "CARDINAL", 15, 17),
    ]
    ner.has_checkworthy_entities.return_value = True
    return ner


@pytest.fixture
def extractor(
    mock_ner: MagicMock, mock_tokenizer: MagicMock, mock_settings: MagicMock
) -> ClaimExtractor:
    extractor = ClaimExtractor(mock_ner, mock_tokenizer, mock_settings)
    # Provide a mock classifier that returns high score
    mock_clf = MagicMock()
    mock_clf.return_value = {"labels": ["news", "opinion"], "scores": [0.8, 0.2]}
    extractor._classifier = mock_clf
    return extractor


@pytest.fixture
def sample_preprocessed() -> PreprocessedText:
    return PreprocessedText(
        raw_text="Russia launched 45 missiles at Ukraine. The weather is nice today.",
        clean_text="Russia launched 45 missiles at Ukraine. The weather is nice today.",
        normalized_text="russia launched 45 missiles ukraine weather nice today",
        sentences=[
            "Russia launched 45 missiles at Ukraine.",
            "The weather is nice today.",
        ],
        language="en",
        tokens=["Russia", "launched", "45", "missiles", "Ukraine", "weather", "nice", "today"],
    )


class TestExtractClaims:
    def test_returns_list_of_claims(
        self, extractor: ClaimExtractor, sample_preprocessed: PreprocessedText
    ) -> None:
        claims = extractor.extract_claims(sample_preprocessed)
        assert isinstance(claims, list)
        assert all(isinstance(c, Claim) for c in claims)

    def test_empty_sentences_returns_empty(
        self, extractor: ClaimExtractor
    ) -> None:
        preprocessed = PreprocessedText(
            raw_text="",
            clean_text="",
            normalized_text="",
            sentences=[],
            language="en",
        )
        assert extractor.extract_claims(preprocessed) == []

    def test_sorted_by_score_descending(
        self, extractor: ClaimExtractor, sample_preprocessed: PreprocessedText
    ) -> None:
        claims = extractor.extract_claims(sample_preprocessed)
        scores = [c.checkworthy_score for c in claims]
        assert scores == sorted(scores, reverse=True)

    def test_filters_by_threshold(
        self, extractor: ClaimExtractor, mock_settings: MagicMock
    ) -> None:
        mock_settings.claim_extraction.checkworthy_threshold = 0.99
        preprocessed = PreprocessedText(
            raw_text="Some text.",
            clean_text="Some text.",
            normalized_text="some text",
            sentences=["Some short text here."],
            language="en",
        )
        # Score will be below 0.99 → no claims
        claims = extractor.extract_claims(preprocessed)
        assert isinstance(claims, list)


class TestIsFactualSentence:
    def test_question_rejected(
        self, extractor: ClaimExtractor, mock_settings: MagicMock
    ) -> None:
        cfg = mock_settings.claim_extraction
        assert extractor._is_factual_sentence("Did Russia attack Ukraine?", cfg) is False

    def test_too_short_rejected(
        self, extractor: ClaimExtractor, mock_settings: MagicMock
    ) -> None:
        cfg = mock_settings.claim_extraction
        assert extractor._is_factual_sentence("Hi.", cfg) is False

    def test_normal_sentence_accepted(
        self, extractor: ClaimExtractor, mock_settings: MagicMock
    ) -> None:
        cfg = mock_settings.claim_extraction
        assert (
            extractor._is_factual_sentence(
                "Russia launched 45 missiles at Ukraine.", cfg
            )
            is True
        )

    def test_no_alphabetic_rejected(
        self, extractor: ClaimExtractor, mock_settings: MagicMock
    ) -> None:
        cfg = mock_settings.claim_extraction
        assert extractor._is_factual_sentence("123 456 789 000", cfg) is False


class TestScoreSentence:
    def test_uses_news_opinion_labels_and_template(
        self, extractor: ClaimExtractor
    ) -> None:
        extractor._score_sentence("Russia launched 45 missiles.", [])
        kwargs = extractor._classifier.call_args.kwargs
        assert kwargs["candidate_labels"] == ["news", "opinion"]
        assert kwargs["hypothesis_template"] == "This text is {}."

    def test_combines_ner_and_classifier(
        self, extractor: ClaimExtractor, mock_ner: MagicMock
    ) -> None:
        extractor._classifier.return_value = {"labels": ["opinion", "news"], "scores": [0.6, 0.4]}
        mock_ner.has_checkworthy_entities.return_value = False
        assert extractor._score_sentence("text", []) == pytest.approx(0.7 * 0.4)
        mock_ner.has_checkworthy_entities.return_value = True
        assert extractor._score_sentence("text", []) == pytest.approx(0.3 + 0.7 * 0.4)

    @pytest.mark.parametrize("has_entities,expected", [(True, 1.0), (False, 0.0)])
    def test_ner_only_without_classifier(
        self,
        extractor: ClaimExtractor,
        mock_ner: MagicMock,
        has_entities: bool,
        expected: float,
    ) -> None:
        extractor._classifier = None
        mock_ner.has_checkworthy_entities.return_value = has_entities
        assert extractor._score_sentence("text", []) == expected

    def test_ner_only_when_classifier_fails(
        self, extractor: ClaimExtractor, mock_ner: MagicMock
    ) -> None:
        extractor._classifier.side_effect = RuntimeError("OOM")
        mock_ner.has_checkworthy_entities.return_value = False
        assert extractor._score_sentence("text", []) == 0.0
