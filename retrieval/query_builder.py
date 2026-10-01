"""
Search query generation for web evidence retrieval.

Every claim is searched with exactly two queries. A non-English claim gets a
neutral query in its own language and one in English, both written by the
LLM; an English claim is searched verbatim plus one neutral English query.
The LLM is injected as a ``generate(prompt, format)`` callable so that
``retrieval`` does not depend on ``verification``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from logging_config import get_logger

__all__ = ["QueryBuilder", "GenerateFn"]

logger = get_logger(__name__)

GenerateFn = Callable[[str, dict[str, Any]], str]

_MIN_QUERY_CHARS = 3
_MAX_QUERY_CHARS = 200
_WHITESPACE_RE = re.compile(r"\s+")

_LANGUAGE_NAMES = {
    "uk": "Ukrainian",
    "ru": "Russian",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pl": "Polish",
}

_RULES = """\
Rules for every query:
- 3 to 12 words, like a query typed into a news search engine.
- Name the key entities, the event, the place and the time of the claim.
- Stay neutral: do not assert that the claim is true or false, and do not \
use words such as "fake", "hoax", "confirmed" or "debunked".
- No quotation marks, no search operators, no explanations."""

_NATIVE_PROMPT = """\
Write two web search queries for finding news reports and fact-checks about \
the claim below. The claim is in {language}.
- "native_query": a query in {language}.
- "english_query": a query in English.

{rules}

CLAIM: {claim}

Return only the JSON object."""

_ENGLISH_PROMPT = """\
Write one web search query in English for finding news reports and \
fact-checks about the claim below. It must differ from the claim's wording.
- "english_query": the query.

{rules}

CLAIM: {claim}

Return only the JSON object."""

_NATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "native_query": {"type": "string"},
        "english_query": {"type": "string"},
    },
    "required": ["native_query", "english_query"],
}

_ENGLISH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"english_query": {"type": "string"}},
    "required": ["english_query"],
}


def _normalize(value: Any) -> str:
    """Strip and collapse whitespace; "" for unusable queries."""
    if not isinstance(value, str):
        return ""
    text = _WHITESPACE_RE.sub(" ", value).strip().strip('"').strip()
    if not _MIN_QUERY_CHARS <= len(text) <= _MAX_QUERY_CHARS:
        return ""
    return text


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = _WHITESPACE_RE.sub(" ", q).strip().casefold()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def _is_english(language: str) -> bool:
    lang = (language or "").strip().lower()
    return lang in ("", "en") or lang.startswith("en-")


class QueryBuilder:
    """Build the search queries for a claim.

    Args:
        generate: ``generate(prompt, json_schema) -> str`` (e.g.
            ``OllamaClient.generate``); None disables generation.
        enabled: ``retrieval.query_generation``.
    """

    def __init__(self, generate: GenerateFn | None, enabled: bool = True) -> None:
        self._generate = generate
        self._enabled = bool(enabled) and generate is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def build(self, claim_text: str, language: str) -> list[str]:
        """Return the queries for *claim_text*, in search order.

        Args:
            claim_text: The claim.
            language: The claim's language code (e.g. ``en``, ``uk``).

        Returns:
            One or two distinct queries; ``[claim_text]`` when generation is
            disabled or fails.
        """
        claim = _WHITESPACE_RE.sub(" ", claim_text).strip()
        if not self._enabled:
            return [claim]

        english = _is_english(language)
        data = self._ask(claim, language, english)
        if data is None:
            return [claim]

        english_query = _normalize(data.get("english_query"))
        if english:
            if not english_query:
                logger.warning("Query generation gave no usable English query.")
            return _dedupe([claim, english_query])

        native_query = _normalize(data.get("native_query"))
        if not native_query and not english_query:
            logger.warning("Query generation gave no usable queries.")
            return [claim]
        if not native_query or not english_query:
            logger.warning(
                "Query generation gave only one usable query; using the claim "
                "text for the other."
            )
        return _dedupe([native_query or claim, english_query or claim])

    def _ask(self, claim: str, language: str, english: bool) -> dict[str, Any] | None:
        if english:
            prompt = _ENGLISH_PROMPT.format(rules=_RULES, claim=claim)
            schema = _ENGLISH_SCHEMA
        else:
            name = _LANGUAGE_NAMES.get(language.lower(), language)
            prompt = _NATIVE_PROMPT.format(language=name, rules=_RULES, claim=claim)
            schema = _NATIVE_SCHEMA
        assert self._generate is not None
        try:
            raw = self._generate(prompt, schema)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Query generation failed (%s); searching with the claim text.", exc
            )
            return None
        if not isinstance(data, dict):
            logger.warning("Query generation returned non-object JSON; using the claim.")
            return None
        return data
