"""
Named Entity Recognition module for the Disinformation Detection System.

Uses spaCy NER to identify entities and applies rule-based heuristics to
score sentences for check-worthiness based on entity type combinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from exceptions import ClaimExtractionError
from logging_config import get_logger

if TYPE_CHECKING:
    from spacy.tokens import Doc

__all__ = ["NERModule", "NamedEntity"]

logger = get_logger(__name__)

# Entity types that indicate a politically/factually relevant subject.
# "PER" is the person label of the uk/ru spaCy pipelines.
_SUBJECT_TYPES: frozenset[str] = frozenset(
    {"ORG", "GPE", "PERSON", "PER", "FAC", "NORP", "LOC", "EVENT"}
)

# Entity types that indicate a quantifiable or time-bound fact
_QUANTIFIER_TYPES: frozenset[str] = frozenset(
    {"CARDINAL", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY", "ORDINAL"}
)

# Regex fallback for pipelines without numeric/date entity types
# (uk_core_news_sm and ru_core_news_sm only emit LOC/ORG/PER).
_MONTH_WORDS = (
    # English ("march"/"may" omitted: also common verbs; en spaCy finds dates)
    "january|february|april|june|july|august|september|october|"
    "november|december|"
    # Ukrainian: nominative, genitive, locative
    "січень|лютий|березень|квітень|травень|червень|липень|серпень|вересень|"
    "жовтень|листопад|грудень|"
    "січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|"
    "листопада|грудня|"
    "січні|лютому|березні|квітні|травні|червні|липні|серпні|вересні|жовтні|"
    "листопаді|грудні|"
    # Russian: nominative, genitive, prepositional
    "январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|"
    "декабрь|"
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|"
    "ноября|декабря|"
    "январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|"
    "ноябре|декабре"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTH_WORDS})\b"
    r"|\d{1,4}\s*[年月日号]"
    r"|[一二三四五六七八九十]{1,3}月",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class NamedEntity:
    """A single named entity extracted from text.

    Attributes:
        text: The surface form of the entity.
        label: spaCy entity label (e.g. "PERSON", "GPE").
        start_char: Character offset start in the original text.
        end_char: Character offset end in the original text.
    """

    text: str
    label: str
    start_char: int
    end_char: int


class NERModule:
    """Extract named entities and compute check-worthiness signals.

    This module reuses a pre-processed spaCy Doc to avoid double-parsing.
    It does not load models itself — the ``MultilingualTokenizer`` owns
    model loading and passes ``Doc`` objects here.

    Args:
        tokenizer: A MultilingualTokenizer instance used to obtain Docs.
    """

    def __init__(self, tokenizer: "MultilingualTokenizer") -> None:  # noqa: F821
        self._tokenizer = tokenizer

    def extract_entities(self, doc: "Doc") -> list[NamedEntity]:
        """Extract named entities from a spaCy Doc.

        When spaCy finds no numeric or date entity (always the case for the
        uk/ru pipelines, which lack those types), regex-detected ``DATE`` and
        ``CARDINAL`` pseudo-entities are added so the check-worthiness
        heuristic works in every language.

        Args:
            doc: A processed spaCy Doc with NER annotations.

        Returns:
            List of NamedEntity objects, ordered by position.

        Raises:
            ClaimExtractionError: On unexpected failure.
        """
        try:
            entities = [
                NamedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
                for ent in doc.ents
            ]
            if not any(e.label in _QUANTIFIER_TYPES for e in entities):
                entities.extend(self._regex_quantifiers(doc.text, entities))
                entities.sort(key=lambda e: e.start_char)
            return entities
        except Exception as exc:
            raise ClaimExtractionError(
                "Entity extraction failed", original_error=exc
            ) from exc

    @staticmethod
    def _regex_quantifiers(
        text: str, existing: list[NamedEntity]
    ) -> list[NamedEntity]:
        """Find dates and numbers that do not overlap *existing* entities."""
        taken = [(e.start_char, e.end_char) for e in existing]
        found: list[NamedEntity] = []

        def add(match: re.Match[str], label: str) -> None:
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                return
            taken.append((start, end))
            found.append(NamedEntity(match.group(), label, start, end))

        for match in _DATE_RE.finditer(text):
            add(match, "DATE")
        for match in _NUMBER_RE.finditer(text):
            add(match, "CARDINAL")
        return found

    def has_checkworthy_entities(self, entities: list[NamedEntity]) -> bool:
        """Return True if *entities* suggest a check-worthy factual claim.

        A sentence is considered check-worthy when it contains at least one
        subject entity (ORG, GPE, PERSON …) AND at least one quantifier
        entity (CARDINAL, PERCENT, DATE …).

        Args:
            entities: List of NamedEntity objects for a sentence.

        Returns:
            True if the combination satisfies the heuristic.
        """
        labels = {e.label for e in entities}
        has_subject = bool(labels & _SUBJECT_TYPES)
        has_quantifier = bool(labels & _QUANTIFIER_TYPES)
        return has_subject and has_quantifier

    def compute_entity_density(
        self, entities: list[NamedEntity], text_len: int
    ) -> float:
        """Compute entity density as entities-per-character.

        Higher density correlates with more factually dense sentences.

        Args:
            entities: Entities extracted from the sentence.
            text_len: Character length of the sentence.

        Returns:
            Float in [0, 1] (capped at 1).
        """
        if text_len == 0:
            return 0.0
        density = len(entities) / text_len
        return min(density * 10, 1.0)  # scale: ~1 entity per 10 chars → 1.0

    def extract_entities_from_text(
        self, text: str, lang: str = "en"
    ) -> list[NamedEntity]:
        """Convenience: obtain a Doc from text and extract entities.

        Args:
            text: Raw sentence string.
            lang: Language code passed to the tokenizer.

        Returns:
            List of NamedEntity objects.
        """
        doc = self._tokenizer.get_doc(text, lang=lang)
        return self.extract_entities(doc)
