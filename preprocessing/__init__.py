"""
Preprocessing module for the Disinformation Detection System.

Provides the PreprocessingPipeline convenience class that wraps TextCleaner,
MultilingualTokenizer, and TextNormalizer into a single processing step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exceptions import PreprocessingError
from logging_config import get_logger
from preprocessing.cleaner import TextCleaner
from preprocessing.normalizer import TextNormalizer
from preprocessing.tokenizer import MultilingualTokenizer

__all__ = ["PreprocessingPipeline", "PreprocessedText"]

logger = get_logger(__name__)


@dataclass
class PreprocessedText:
    """Result of the preprocessing pipeline for a single input text.

    Attributes:
        raw_text: Original unmodified input.
        clean_text: HTML-stripped, URL-removed, whitespace-normalized text.
        normalized_text: clean_text after lowercase, stopword removal.
        sentences: List of sentence strings from clean_text.
        language: Detected ISO 639-1 language code.
        tokens: List of word tokens from clean_text.
    """

    raw_text: str
    clean_text: str
    normalized_text: str
    sentences: list[str]
    language: str
    tokens: list[str] = field(default_factory=list)


class PreprocessingPipeline:
    """Orchestrate cleaning, tokenization, and normalization.

    Args:
        settings: Application settings.
        remove_urls: Passed to TextCleaner.
        remove_emojis: Passed to TextCleaner.
    """

    def __init__(
        self,
        settings: "Settings",  # noqa: F821
        remove_urls: bool = True,
        remove_emojis: bool = False,
    ) -> None:
        self._settings = settings
        self.cleaner = TextCleaner(
            remove_urls=remove_urls,
            remove_emojis=remove_emojis,
        )
        self.tokenizer = MultilingualTokenizer(settings)
        self.normalizer = TextNormalizer(settings)

    def process(self, raw_text: str) -> PreprocessedText:
        """Run the full preprocessing pipeline on *raw_text*.

        Args:
            raw_text: Unprocessed input string.

        Returns:
            PreprocessedText dataclass with all processed fields.

        Raises:
            PreprocessingError: If any stage of preprocessing fails.
        """
        if not raw_text or not raw_text.strip():
            logger.debug("Empty input text received by preprocessing pipeline.")
            return PreprocessedText(
                raw_text=raw_text,
                clean_text="",
                normalized_text="",
                sentences=[],
                language="en",
                tokens=[],
            )

        try:
            clean_text = self.cleaner.clean(raw_text)
            language = self.tokenizer.detect_language(clean_text)
            sentences = self.tokenizer.tokenize_sentences(clean_text, lang=language)
            tokens = self.tokenizer.tokenize_words(clean_text, lang=language)
            normalized_text = self.normalizer.normalize(clean_text, lang=language)

            logger.debug(
                "Preprocessed text: lang=%s, sentences=%d, tokens=%d",
                language,
                len(sentences),
                len(tokens),
            )

            return PreprocessedText(
                raw_text=raw_text,
                clean_text=clean_text,
                normalized_text=normalized_text,
                sentences=sentences,
                language=language,
                tokens=tokens,
            )
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "Preprocessing pipeline failed", original_error=exc
            ) from exc
