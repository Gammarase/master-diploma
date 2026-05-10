"""
Text cleaning utilities for the Disinformation Detection System.

Removes HTML tags, URLs, control characters, and normalizes whitespace.
Language-agnostic — can be applied before language detection.
"""

from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup

from exceptions import PreprocessingError
from logging_config import get_logger

__all__ = ["TextCleaner"]

logger = get_logger(__name__)

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


class TextCleaner:
    """Clean raw text by stripping HTML, URLs, and normalizing whitespace.

    Args:
        remove_urls: Whether to strip HTTP/HTTPS/www URLs.
        remove_emojis: Whether to strip emoji characters.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emojis: bool = False,
    ) -> None:
        self.remove_urls = remove_urls
        self.remove_emojis = remove_emojis

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on *text*.

        Steps: HTML decode → HTML tag removal → URL removal →
        control character removal → emoji removal → whitespace normalization.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string.

        Raises:
            PreprocessingError: If cleaning fails unexpectedly.
        """
        if not isinstance(text, str):
            raise PreprocessingError(
                f"Expected str, got {type(text).__name__}"
            )
        try:
            text = html.unescape(text)
            text = self.remove_html_tags(text)
            if self.remove_urls:
                text = self._remove_urls(text)
            text = self.remove_control_characters(text)
            if self.remove_emojis:
                text = self._remove_emojis(text)
            text = self.normalize_whitespace(text)
            return text
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                "Text cleaning failed", original_error=exc
            ) from exc

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML markup and return plain text.

        Uses BeautifulSoup with the built-in html.parser for robustness.

        Args:
            text: String possibly containing HTML.

        Returns:
            Plain text with HTML tags stripped.
        """
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        except Exception as exc:
            logger.warning("BeautifulSoup failed, falling back to regex: %s", exc)
            return re.sub(r"<[^>]+>", " ", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse runs of whitespace to a single space and strip edges.

        Args:
            text: Input string.

        Returns:
            String with normalized whitespace.
        """
        return _WHITESPACE_RE.sub(" ", text).strip()

    def remove_control_characters(self, text: str) -> str:
        """Remove non-printable Unicode control characters.

        Keeps common whitespace characters (space, tab, newline).

        Args:
            text: Input string.

        Returns:
            String with control characters removed.
        """
        return _CONTROL_RE.sub("", text)

    def _remove_urls(self, text: str) -> str:
        """Replace HTTP/HTTPS/www URLs with a single space.

        Args:
            text: Input string.

        Returns:
            String with URLs removed.
        """
        return _URL_RE.sub(" ", text)

    def _remove_emojis(self, text: str) -> str:
        """Remove emoji and other symbol characters via Unicode categories.

        Args:
            text: Input string.

        Returns:
            String with emoji characters removed.
        """
        return "".join(
            ch
            for ch in text
            if unicodedata.category(ch) not in ("So", "Sm", "Sk", "Sc")
        )
