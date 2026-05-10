"""
Text normalization for the Disinformation Detection System.

Handles lowercasing, abbreviation expansion, stopword removal, and
lemmatization. Designed for NLI/embedding input where normalized text
improves model performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exceptions import PreprocessingError
from logging_config import get_logger

if TYPE_CHECKING:
    from spacy.tokens import Doc

__all__ = ["TextNormalizer", "ABBREVIATIONS"]

logger = get_logger(__name__)

ABBREVIATIONS: dict[str, dict[str, str]] = {
    "en": {
        "u.s.": "united states",
        "u.s.a.": "united states of america",
        "nato": "nato",
        "un": "united nations",
        "eu": "european union",
        "u.k.": "united kingdom",
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "et cetera",
        "vs.": "versus",
        "mr.": "mister",
        "mrs.": "missus",
        "dr.": "doctor",
    },
    "uk": {
        "сша": "сполучені штати америки",
        "нато": "нато",
        "оон": "організація обʼєднаних націй",
        "єс": "європейський союз",
        "рф": "російська федерація",
        "зсу": "збройні сили україни",
        "тощо": "та інше",
        "напр.": "наприклад",
        "ін.": "інші",
    },
    "ru": {
        "сша": "соединённые штаты америки",
        "нато": "нато",
        "оон": "организация объединённых наций",
        "ес": "европейский союз",
        "рф": "российская федерация",
        "напр.": "например",
    },
}


class TextNormalizer:
    """Normalize text for downstream NLI/embedding models.

    Performs lowercase conversion, abbreviation expansion, stopword
    removal, and lemmatization using a pre-processed spaCy Doc.

    Args:
        settings: Application settings (used for language config).
    """

    def __init__(self, settings: "Settings") -> None:  # noqa: F821
        self._settings = settings
        self._stopwords: dict[str, set[str]] = {}

    def normalize(self, text: str, lang: str = "en") -> str:
        """Apply the full normalization pipeline to *text*.

        Steps: lowercase → abbreviation expansion → stopword removal
        (token-level, after split on spaces) → return joined string.

        Note: For lemmatization use ``lemmatize(doc)`` directly on a
        spaCy Doc for best results.

        Args:
            text: Cleaned input string.
            lang: ISO 639-1 language code.

        Returns:
            Normalized string.

        Raises:
            PreprocessingError: On normalization failure.
        """
        try:
            text = self.lowercase(text)
            text = self.expand_abbreviations(text, lang)
            tokens = text.split()
            tokens = self.remove_stopwords(tokens, lang)
            return " ".join(tokens)
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "Text normalization failed", original_error=exc
            ) from exc

    def lowercase(self, text: str) -> str:
        """Convert *text* to lowercase.

        Args:
            text: Input string.

        Returns:
            Lowercased string.
        """
        return text.lower()

    def expand_abbreviations(self, text: str, lang: str) -> str:
        """Replace known abbreviations in *text* with their full forms.

        Matching is case-insensitive and whole-word only.

        Args:
            text: Lowercased input string.
            lang: Language code for selecting the abbreviation dictionary.

        Returns:
            String with abbreviations expanded.
        """
        abbrevs = ABBREVIATIONS.get(lang, {})
        for abbrev, expansion in abbrevs.items():
            # Whole-word replacement (abbrev may contain dots)
            escaped = abbrev.replace(".", r"\.")
            import re
            text = re.sub(
                r"(?<!\w)" + escaped + r"(?!\w)",
                expansion,
                text,
                flags=re.IGNORECASE,
            )
        return text

    def remove_stopwords(self, tokens: list[str], lang: str) -> list[str]:
        """Remove stopwords from *tokens*.

        Uses spaCy's built-in stopword list for the given language.

        Args:
            tokens: List of word strings (already lowercased).
            lang: Language code.

        Returns:
            Filtered list of tokens.
        """
        stopwords = self._get_stopwords(lang)
        return [t for t in tokens if t not in stopwords]

    def lemmatize(self, doc: "Doc") -> list[str]:
        """Extract lemmas from a spaCy Doc, filtering punctuation and spaces.

        Args:
            doc: A processed spaCy Doc object.

        Returns:
            List of lemma strings (lowercased).
        """
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_space and not token.is_punct and token.lemma_ != "-PRON-"
        ]

    def _get_stopwords(self, lang: str) -> set[str]:
        """Return the stopword set for *lang*, loading lazily.

        Args:
            lang: Language code.

        Returns:
            Set of stopword strings.
        """
        if lang in self._stopwords:
            return self._stopwords[lang]

        try:
            import spacy

            model_map = self._settings.preprocessing.spacy_models
            model_name = model_map.get(lang, model_map.get("en", "en_core_web_sm"))
            nlp = spacy.load(model_name)
            stopwords = nlp.Defaults.stop_words
        except Exception as exc:
            logger.debug(
                "Could not load spaCy stopwords for '%s': %s. Using empty set.",
                lang,
                exc,
            )
            stopwords = set()

        self._stopwords[lang] = stopwords
        return stopwords
