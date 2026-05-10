"""Unit tests for preprocessing.cleaner.TextCleaner."""

from __future__ import annotations

import pytest

from exceptions import PreprocessingError
from preprocessing.cleaner import TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner(remove_urls=True, remove_emojis=False)


class TestTextCleanerClean:
    def test_strips_html_tags(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_http_urls(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("Visit https://example.com for more info.")
        assert "https://" not in result
        assert "example.com" not in result

    def test_removes_www_urls(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("See www.example.com for details.")
        assert "www." not in result

    def test_normalizes_whitespace(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("Hello   \t  world\n\nfoo")
        assert "  " not in result
        assert result == "Hello world foo"

    def test_html_entities_decoded(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("&amp; &lt;tag&gt; &#39;quote&#39;")
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_empty_string(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean("")
        assert result == ""

    def test_plain_text_unchanged(self, cleaner: TextCleaner) -> None:
        text = "Russia launched missiles at Ukraine."
        result = cleaner.clean(text)
        assert result == text

    def test_raises_on_non_string(self, cleaner: TextCleaner) -> None:
        with pytest.raises(PreprocessingError):
            cleaner.clean(123)  # type: ignore[arg-type]

    def test_url_removal_disabled(self) -> None:
        c = TextCleaner(remove_urls=False)
        result = c.clean("See https://example.com for info.")
        assert "https://example.com" in result

    def test_emoji_removal_enabled(self) -> None:
        c = TextCleaner(remove_emojis=True)
        result = c.clean("Hello 🌍 world")
        assert "🌍" not in result
        assert "Hello" in result
        assert "world" in result


class TestRemoveHtmlTags:
    def test_nested_tags(self, cleaner: TextCleaner) -> None:
        result = cleaner.remove_html_tags("<div><p>text</p></div>")
        assert "text" in result
        assert "<" not in result

    def test_no_html(self, cleaner: TextCleaner) -> None:
        text = "plain text"
        assert cleaner.remove_html_tags(text) == text

    def test_script_tag_content_removed(self, cleaner: TextCleaner) -> None:
        result = cleaner.remove_html_tags("<script>alert('xss')</script>Hello")
        assert "alert" not in result
        assert "Hello" in result


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self, cleaner: TextCleaner) -> None:
        assert cleaner.normalize_whitespace("a  b   c") == "a b c"

    def test_strips_leading_trailing(self, cleaner: TextCleaner) -> None:
        assert cleaner.normalize_whitespace("  hello  ") == "hello"

    def test_newlines_become_space(self, cleaner: TextCleaner) -> None:
        assert cleaner.normalize_whitespace("a\nb") == "a b"


class TestRemoveControlCharacters:
    def test_removes_null_bytes(self, cleaner: TextCleaner) -> None:
        result = cleaner.remove_control_characters("hello\x00world")
        assert "\x00" not in result
        assert "helloworld" in result

    def test_keeps_normal_text(self, cleaner: TextCleaner) -> None:
        text = "Normal text with spaces."
        assert cleaner.remove_control_characters(text) == text
