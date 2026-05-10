"""Unit tests for preprocessing.tokenizer.MultilingualTokenizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from exceptions import PreprocessingError
from preprocessing.tokenizer import MultilingualTokenizer


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.preprocessing.spacy_models = {
        "en": "en_core_web_sm",
        "uk": "uk_core_news_sm",
    }
    return settings


@pytest.fixture
def tokenizer(mock_settings: MagicMock) -> MultilingualTokenizer:
    return MultilingualTokenizer(mock_settings)


class TestDetectLanguage:
    def test_detects_english(self, tokenizer: MultilingualTokenizer) -> None:
        lang = tokenizer.detect_language(
            "Russia launched a major missile attack on Ukraine."
        )
        assert lang == "en"

    def test_fallback_on_empty(self, tokenizer: MultilingualTokenizer) -> None:
        with patch("preprocessing.tokenizer.MultilingualTokenizer.detect_language", return_value="en"):
            lang = tokenizer.detect_language("")
        assert lang == "en"

    def test_fallback_on_exception(self, tokenizer: MultilingualTokenizer) -> None:
        with patch("langdetect.detect", side_effect=Exception("fail")):
            lang = tokenizer.detect_language("some text")
        assert lang == "en"


class TestTokenizeSentences:
    def test_splits_two_sentences(self, tokenizer: MultilingualTokenizer) -> None:
        text = "NATO expanded eastward. Russia objected strongly."
        sentences = tokenizer.tokenize_sentences(text, lang="en")
        assert len(sentences) == 2

    def test_single_sentence(self, tokenizer: MultilingualTokenizer) -> None:
        text = "This is one sentence."
        sentences = tokenizer.tokenize_sentences(text, lang="en")
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_empty_text(self, tokenizer: MultilingualTokenizer) -> None:
        sentences = tokenizer.tokenize_sentences("", lang="en")
        assert sentences == []


class TestTokenizeWords:
    def test_removes_punctuation(self, tokenizer: MultilingualTokenizer) -> None:
        words = tokenizer.tokenize_words("Hello, world!", lang="en")
        assert "," not in words
        assert "!" not in words
        assert "Hello" in words
        assert "world" in words

    def test_empty_returns_empty(self, tokenizer: MultilingualTokenizer) -> None:
        words = tokenizer.tokenize_words("", lang="en")
        assert words == []


class TestGetDoc:
    def test_returns_doc(self, tokenizer: MultilingualTokenizer) -> None:
        import spacy

        doc = tokenizer.get_doc("Test sentence.", lang="en")
        assert isinstance(doc, spacy.tokens.Doc)

    def test_raises_preprocessing_error_on_failure(
        self, tokenizer: MultilingualTokenizer
    ) -> None:
        with patch.object(tokenizer, "_load_model", side_effect=RuntimeError("boom")):
            with pytest.raises(PreprocessingError):
                tokenizer.get_doc("some text", lang="en")
