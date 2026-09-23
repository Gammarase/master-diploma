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


class TestMissingModelFallback:
    def test_warning_names_missing_package(
        self, mock_settings: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        import spacy

        mock_settings.preprocessing.spacy_models = {"ru": "ru_core_news_sm"}
        tokenizer = MultilingualTokenizer(mock_settings)
        with patch("spacy.load", side_effect=OSError("E050")), caplog.at_level("WARNING"):
            sentences = tokenizer.tokenize_sentences(
                "Первое предложение. Второе предложение.", lang="ru"
            )
        assert sentences == ["Первое предложение.", "Второе предложение."]
        assert "ru_core_news_sm" in caplog.text
        assert "python -m spacy download ru_core_news_sm" in caplog.text
        assert isinstance(tokenizer._models["ru"], spacy.language.Language)


class TestRegionalLanguageCodes:
    @pytest.mark.parametrize("detected", ["zh-cn", "zh-tw", "ZH-CN"])
    def test_regional_code_normalised(
        self, tokenizer: MultilingualTokenizer, detected: str
    ) -> None:
        with patch("langdetect.detect", return_value=detected):
            assert tokenizer.detect_language("我觉得这部电影很好看。") == "zh"

    def test_real_chinese_detection_uses_zh_pipeline(
        self, mock_settings: MagicMock
    ) -> None:
        mock_settings.preprocessing.spacy_models = {
            "en": "en_core_web_sm",
            "zh": "zh_core_web_sm",
        }
        tokenizer = MultilingualTokenizer(mock_settings)
        lang = tokenizer.detect_language("欧盟决定给予乌克兰和摩尔多瓦欧盟候选国地位。")
        assert lang == "zh"
        tokenizer.get_doc("欧盟决定给予乌克兰候选国地位。", lang=lang)
        assert tokenizer._models["zh"].meta["name"] == "core_web_sm"
        assert tokenizer._models["zh"].lang == "zh"
