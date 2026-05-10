"""
Language-aware tokenization for the Disinformation Detection System.

Uses spaCy for sentence splitting and word tokenization with per-language
model loading. Provides language detection via langdetect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exceptions import PreprocessingError
from logging_config import get_logger

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc

__all__ = ["MultilingualTokenizer"]

logger = get_logger(__name__)

_FALLBACK_LANGUAGE = "en"


class MultilingualTokenizer:
    """Tokenize text using spaCy with lazy per-language model loading.

    Args:
        settings: Application settings providing spacy_models mapping.
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._models: dict[str, "spacy.Language"] = {}

    def detect_language(self, text: str) -> str:
        """Detect the language of *text* using langdetect.

        Args:
            text: Input string (should be at least a few words).

        Returns:
            ISO 639-1 language code (e.g. "en", "uk"). Falls back to "en".
        """
        try:
            from langdetect import detect, LangDetectException

            lang = detect(text)
            return lang
        except Exception as exc:
            logger.debug(
                "Language detection failed, defaulting to '%s': %s",
                _FALLBACK_LANGUAGE,
                exc,
            )
            return _FALLBACK_LANGUAGE

    def _load_model(self, lang: str) -> "spacy.Language":
        """Load and cache a spaCy model for *lang*.

        Falls back to the English model when the requested language model
        is not installed or not configured.

        Args:
            lang: ISO 639-1 language code.

        Returns:
            Loaded spaCy Language object.
        """
        if lang in self._models:
            return self._models[lang]

        import spacy

        model_map: dict[str, str] = self._settings.preprocessing.spacy_models
        model_name = model_map.get(lang) or model_map.get(_FALLBACK_LANGUAGE, "en_core_web_sm")

        try:
            nlp = spacy.load(model_name)
            logger.debug("Loaded spaCy model '%s' for language '%s'", model_name, lang)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found; falling back to blank '%s' model.",
                model_name,
                lang,
            )
            nlp = spacy.blank(lang if len(lang) == 2 else "xx")
            nlp.add_pipe("sentencizer")

        self._models[lang] = nlp
        return nlp

    def tokenize_sentences(
        self, text: str, lang: str | None = None
    ) -> list[str]:
        """Split *text* into sentences.

        Args:
            text: Input string.
            lang: Language code. Auto-detected if None.

        Returns:
            List of sentence strings.

        Raises:
            PreprocessingError: On tokenization failure.
        """
        try:
            resolved_lang = lang or self.detect_language(text)
            doc = self.get_doc(text, lang=resolved_lang)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "Sentence tokenization failed", original_error=exc
            ) from exc

    def tokenize_words(
        self, text: str, lang: str | None = None
    ) -> list[str]:
        """Return word tokens (no punctuation or whitespace tokens).

        Args:
            text: Input string.
            lang: Language code. Auto-detected if None.

        Returns:
            List of word token strings.

        Raises:
            PreprocessingError: On tokenization failure.
        """
        try:
            resolved_lang = lang or self.detect_language(text)
            doc = self.get_doc(text, lang=resolved_lang)
            return [
                token.text
                for token in doc
                if not token.is_space and not token.is_punct
            ]
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "Word tokenization failed", original_error=exc
            ) from exc

    def get_doc(self, text: str, lang: str | None = None) -> "Doc":
        """Run spaCy NLP pipeline and return a Doc object.

        This is the shared entry point for both tokenization and NER — other
        modules receive a pre-computed Doc to avoid double-processing.

        Args:
            text: Input string.
            lang: Language code. Auto-detected if None.

        Returns:
            spaCy Doc object.

        Raises:
            PreprocessingError: On NLP processing failure.
        """
        try:
            resolved_lang = lang or self.detect_language(text)
            nlp = self._load_model(resolved_lang)
            return nlp(text)
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "spaCy processing failed", original_error=exc
            ) from exc
