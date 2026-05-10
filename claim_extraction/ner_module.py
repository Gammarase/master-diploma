"""
Named Entity Recognition module for the Disinformation Detection System.

Uses spaCy NER to identify entities and applies rule-based heuristics to
score sentences for check-worthiness based on entity type combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from exceptions import ClaimExtractionError
from logging_config import get_logger

if TYPE_CHECKING:
    from spacy.tokens import Doc

__all__ = ["NERModule", "NamedEntity"]

logger = get_logger(__name__)

# Entity types that indicate a politically/factually relevant subject
_SUBJECT_TYPES: frozenset[str] = frozenset(
    {"ORG", "GPE", "PERSON", "FAC", "NORP", "LOC", "EVENT"}
)

# Entity types that indicate a quantifiable or time-bound fact
_QUANTIFIER_TYPES: frozenset[str] = frozenset(
    {"CARDINAL", "PERCENT", "QUANTITY", "DATE", "TIME", "MONEY", "ORDINAL"}
)


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

        Args:
            doc: A processed spaCy Doc with NER annotations.

        Returns:
            List of NamedEntity objects.

        Raises:
            ClaimExtractionError: On unexpected failure.
        """
        try:
            return [
                NamedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
                for ent in doc.ents
            ]
        except Exception as exc:
            raise ClaimExtractionError(
                "Entity extraction failed", original_error=exc
            ) from exc

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
