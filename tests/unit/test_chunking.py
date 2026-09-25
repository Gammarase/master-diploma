"""Unit tests for retrieval.chunking."""

from __future__ import annotations

import pytest

from retrieval.chunking import chunk_evidence


class TestChunkEvidence:
    def test_2000_chars_gives_at_least_3_chunks(self) -> None:
        sentence = "Russian forces shelled the city again overnight, officials said. "
        text = (sentence * 40)[:2000]
        chunks = chunk_evidence(text, 800)
        assert len(chunks) >= 3
        assert all(len(c) <= 800 for c in chunks)

    def test_empty_input(self) -> None:
        assert chunk_evidence("", 800) == []
        assert chunk_evidence("   ", 800) == []

    def test_short_text_single_chunk(self) -> None:
        assert chunk_evidence("One short passage.", 800) == ["One short passage."]

    def test_splits_on_item_markers(self) -> None:
        text = "- **First**: " + "a" * 50 + " - **Second**: " + "b" * 50
        chunks = chunk_evidence(text, 70)
        assert len(chunks) == 2
        assert chunks[0].startswith("- **First**")
        assert chunks[1].startswith("- **Second**")

    def test_numbered_items_and_decimals(self) -> None:
        text = "News:1. Price rose 3.5 percent today. 2. Second item here."
        chunks = chunk_evidence(text, 40)
        # "3.5" is not an item boundary; the short "News:" lead packs with item 1.
        assert chunks == ["News: 1. Price rose 3.5 percent today.", "2. Second item here."]

    def test_unbroken_cjk_text_hard_split(self) -> None:
        text = "乌" * 1000
        chunks = chunk_evidence(text, 300)
        assert all(len(c) <= 300 for c in chunks)
        assert "".join(chunks) == text

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError):
            chunk_evidence("text", 0)
