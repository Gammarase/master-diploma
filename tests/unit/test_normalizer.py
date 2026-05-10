"""Unit tests for preprocessing.normalizer.TextNormalizer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from preprocessing.normalizer import TextNormalizer


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.preprocessing.spacy_models = {
        "en": "en_core_web_sm",
        "uk": "uk_core_news_sm",
    }
    return settings


@pytest.fixture
def normalizer(mock_settings: MagicMock) -> TextNormalizer:
    return TextNormalizer(mock_settings)


class TestLowercase:
    def test_lowercases_text(self, normalizer: TextNormalizer) -> None:
        assert normalizer.lowercase("HELLO World") == "hello world"

    def test_already_lowercase(self, normalizer: TextNormalizer) -> None:
        assert normalizer.lowercase("hello") == "hello"


class TestExpandAbbreviations:
    def test_expands_us(self, normalizer: TextNormalizer) -> None:
        result = normalizer.expand_abbreviations("the u.s. president", "en")
        assert "united states" in result

    def test_expands_nato_en(self, normalizer: TextNormalizer) -> None:
        # NATO is kept as-is in English abbreviation map
        result = normalizer.expand_abbreviations("nato forces", "en")
        assert "nato" in result

    def test_no_match_returns_original(self, normalizer: TextNormalizer) -> None:
        text = "some random text"
        assert normalizer.expand_abbreviations(text, "en") == text

    def test_unknown_lang_no_error(self, normalizer: TextNormalizer) -> None:
        result = normalizer.expand_abbreviations("text", "zh")
        assert result == "text"


class TestRemoveStopwords:
    def test_removes_common_english_stopwords(
        self, normalizer: TextNormalizer
    ) -> None:
        tokens = ["the", "missile", "was", "launched", "at", "ukraine"]
        result = normalizer.remove_stopwords(tokens, "en")
        assert "the" not in result
        assert "missile" in result
        assert "ukraine" in result

    def test_empty_list(self, normalizer: TextNormalizer) -> None:
        assert normalizer.remove_stopwords([], "en") == []

    def test_unknown_lang_returns_all(self, normalizer: TextNormalizer) -> None:
        tokens = ["hello", "world"]
        # No stopwords for unknown lang → returns all tokens
        with patch.object(normalizer, "_get_stopwords", return_value=set()):
            result = normalizer.remove_stopwords(tokens, "xx")
        assert result == tokens


class TestNormalize:
    def test_full_pipeline(self, normalizer: TextNormalizer) -> None:
        result = normalizer.normalize("The U.S. launched a strike.", "en")
        assert isinstance(result, str)
        assert len(result) > 0
        # "The" should be removed as stopword; "u.s." expanded
        assert "the" not in result.split()

    def test_empty_string(self, normalizer: TextNormalizer) -> None:
        result = normalizer.normalize("", "en")
        assert result == ""
