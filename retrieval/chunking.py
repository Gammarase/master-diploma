"""
Passage chunking for web evidence.

Splits extracted page text into passages of bounded length, breaking at
list-item markers and sentence ends where possible. Pure function, so it
can be tested without any model or network access.
"""

from __future__ import annotations

import re

__all__ = ["chunk_evidence"]

_WHITESPACE_RE = re.compile(r"\s+")

# List/item boundaries: "[1]: ...", "- **Title**", "1. ..." (number must follow
# start, whitespace or a colon/full stop so decimals like "3.5" are not split).
_ITEM_SPLIT_RE = re.compile(
    r"(?=\[\d+\]:)|(?=-\s*\*\*)|(?:(?<=^)|(?<=[\s:：。]))(?=\d{1,2}\.\s)"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _hard_split(piece: str, max_chars: int) -> list[str]:
    """Split an over-long piece at word boundaries (or raw chars for CJK)."""
    parts: list[str] = []
    while len(piece) > max_chars:
        cut = piece.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


def chunk_evidence(text: str, max_chars: int) -> list[str]:
    """Split evidence text into passages of at most *max_chars* characters.

    The text is split on list-item markers first, then on sentence ends; the
    resulting pieces are packed greedily into passages.

    Args:
        text: Evidence text (e.g. extracted page text).
        max_chars: Maximum passage length in characters.

    Returns:
        List of passages; empty when *text* is blank.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = _normalize_whitespace(text or "")
    if not text:
        return []

    pieces: list[str] = []
    for item in _ITEM_SPLIT_RE.split(text):
        item = item.strip()
        if not item:
            continue
        if len(item) <= max_chars:
            pieces.append(item)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(item):
            sentence = sentence.strip()
            if sentence:
                pieces.extend(_hard_split(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
