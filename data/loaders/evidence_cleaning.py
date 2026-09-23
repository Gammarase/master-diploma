"""
Evidence cleaning and chunking for RU22Fact.

RU22Fact evidence was collected through Bing Chat and still carries the
assistant's greetings, "here are five news articles" lead-ins, apologies and
``[^n^][m]`` citation markers. These functions strip that noise and split
the remaining text into passage-sized chunks for embedding. Both are pure,
so they can be tested without Pinecone or any model.
"""

from __future__ import annotations

import re

__all__ = ["clean_evidence", "chunk_evidence"]

# Each pattern is anchored to an exact assistant phrase so that ordinary news
# text is never matched. Order does not matter; all are applied in turn.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # ── English ──
        r"Hello, this is Bing\.",
        r"Here are [^:\n]{0,200}?\b(?:news|articles?|sources|headlines)\b[^:\n]{0,200}?:",
        r"I found [^.:\n]{0,120}?\b(?:news|articles?)\b[^.:\n]{0,120}?[.:]"
        r"(?:\s*Here are [^:.\n]{0,60}?:)?",
        r"Hmm\W+let\W*s try a different topic\.\s*Sorry about that\.\s*"
        r"What else is on your mind\?",
        r"My mistake, I can.t give a response to that right now\.\s*"
        r"Let.s try a different topic\.",
        r"I(?:'|’)?m sorry,? (?:but )?I (?:could not|couldn.t|can.t|cannot|did not|didn.t)"
        r" find[^.\n]*\.",
        r"(?:Is there anything else|Can I help you with anything else)[^?\n]*\?",
        r"Could you please provide [^?\n]*\?",
        r"I hope this helps\.",
        # ── Ukrainian ──
        r"Привіт, це Bing\.",
        r"Ось [^:.\n]{0,120}?(?:новин|статт|повідомлен)[^:.\n]{0,120}?:",
        r"На жаль, я [^.\n]*\.",
        r"Чи можу я (?:ще )?(?:чимось )?допомогти[^?\n]*\?",
        # ── Russian ──
        r"Здравствуйте(?:, это Bing\.|!)",
        r"Я (?:могу|нашел|нашёл|нашла)[^.:\n]{0,150}?(?:запрос\w*|интересует)[.:]",
        r"Вот [^:.\n]{0,120}?(?:новост|сообщени|стать)[^:.\n]{0,120}?:",
        r"Для вас я нашла [^.:\n]{0,120}?[.:]",
        r"К сожалению, я [^.\n]*\.",
        r"Я не могу найти [^.\n]*\.",
        r"Могу ли я помочь [Вв]ам [^?\n]*\?",
        # ── Chinese ──
        r"[您你]好，(?:这是必应。)?",
        r"根据[^：:。\n]{0,30}，(?:以下是|我(?:为您)?找到了以下)[^：:。\n]{0,60}[：:]",
        r"(?:这是)?我(?:为您)?找到了(?:以下|一些)[^：:。\n]{0,60}(?:新闻|消息)[^：:。\n]{0,30}[：:。]",
        r"以下是[^：:。\n]{0,60}(?:新闻|消息)[^：:。\n]{0,40}[：:]",
        r"我没有找到[^。\n]*。",
        r"您能否提供更多的信息[^？?\n]*[？?]\W*",
    )
]

_CITATION_MARKER_RE = re.compile(r"\[\^\d+\^\](?:\[\d+\])?")
_WHITESPACE_RE = re.compile(r"\s+")

# List/item boundaries: "[1]: ...", "- **Title**", "1. ..." (number must follow
# start, whitespace or a colon/full stop so decimals like "3.5" are not split).
_ITEM_SPLIT_RE = re.compile(
    r"(?=\[\d+\]:)|(?=-\s*\*\*)|(?:(?<=^)|(?<=[\s:：。]))(?=\d{1,2}\.\s)"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_evidence(text: str) -> str:
    """Remove search-assistant boilerplate and citation markers from *text*.

    Everything else is kept; whitespace is normalised to single spaces.

    Args:
        text: Raw RU22Fact evidence string.

    Returns:
        Cleaned evidence (may be empty if the row held only boilerplate).
    """
    if not text:
        return ""
    cleaned = _CITATION_MARKER_RE.sub("", text)
    for pattern in _BOILERPLATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return _normalize_whitespace(cleaned)


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
    """Split cleaned evidence into passages of at most *max_chars* characters.

    The text is split on list-item markers first, then on sentence ends; the
    resulting pieces are packed greedily into passages.

    Args:
        text: Cleaned evidence text.
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
