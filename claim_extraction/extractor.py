"""
Claim extraction module for the Disinformation Detection System.

Identifies check-worthy factual sentences from preprocessed text using a
combination of NER-based heuristics and a zero-shot text classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from claim_extraction.ner_module import NERModule, NamedEntity
from exceptions import ClaimExtractionError
from logging_config import get_logger

if TYPE_CHECKING:
    from preprocessing import PreprocessedText

__all__ = ["ClaimExtractor", "Claim"]

logger = get_logger(__name__)

_CHECKWORTHY_LABEL = "checkworthy factual claim"
_CANDIDATE_LABELS = [
    _CHECKWORTHY_LABEL,
    "opinion or speculation",
    "question",
    "irrelevant",
]


@dataclass
class Claim:
    """A single check-worthy factual claim extracted from text.

    Attributes:
        text: The claim sentence text.
        original_sentence: Unmodified sentence as it appears in the source.
        sentence_index: Index of the sentence in the source document.
        language: ISO 639-1 language code.
        entities: Named entities found in this claim.
        checkworthy_score: Combined score in [0, 1] (higher = more check-worthy).
    """

    text: str
    original_sentence: str
    sentence_index: int
    language: str
    entities: list[NamedEntity] = field(default_factory=list)
    checkworthy_score: float = 0.0


class ClaimExtractor:
    """Identify check-worthy factual claims in preprocessed text.

    Combines two signals:
    - NER rule score (0 or 1): does the sentence have subject + quantifier entities?
    - Zero-shot classifier score (0–1): probability of "checkworthy factual claim".

    The final score is: ``ner_weight * ner_score + classifier_weight * classifier_score``.

    Args:
        ner_module: NERModule for entity extraction.
        tokenizer: MultilingualTokenizer for obtaining spaCy Docs.
        settings: Application settings.
    """

    def __init__(
        self,
        ner_module: NERModule,
        tokenizer: "MultilingualTokenizer",  # noqa: F821
        settings: "Settings",  # noqa: F821
    ) -> None:
        self._ner = ner_module
        self._tokenizer = tokenizer
        self._settings = settings
        self._classifier: Any | None = None  # lazy-loaded HuggingFace pipeline

    def _load_classifier(self) -> None:
        """Lazy-load the zero-shot classification pipeline."""
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline

            model_name = self._settings.claim_extraction.classifier_model
            self._classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
            )
            logger.info("Loaded zero-shot classifier: %s", model_name)
        except Exception as exc:
            raise ClaimExtractionError(
                "Failed to load claim classifier", original_error=exc
            ) from exc

    def extract_claims(
        self, preprocessed: "PreprocessedText"
    ) -> list[Claim]:
        """Extract check-worthy claims from *preprocessed* text.

        Args:
            preprocessed: Output of PreprocessingPipeline.process().

        Returns:
            List of Claim objects sorted by checkworthy_score descending.

        Raises:
            ClaimExtractionError: On failure.
        """
        if not preprocessed.sentences:
            return []

        try:
            self._load_classifier()
        except ClaimExtractionError as exc:
            logger.warning(
                "Classifier unavailable, falling back to NER-only scoring: %s", exc
            )

        claims: list[Claim] = []
        cfg = self._settings.claim_extraction
        threshold = cfg.checkworthy_threshold

        for idx, sentence in enumerate(preprocessed.sentences):
            sentence = sentence.strip()
            if not self._is_factual_sentence(sentence, cfg):
                continue

            try:
                doc = self._tokenizer.get_doc(sentence, lang=preprocessed.language)
                entities = self._ner.extract_entities(doc)
                score = self._score_sentence(sentence, entities)

                if score >= threshold:
                    claims.append(
                        Claim(
                            text=sentence,
                            original_sentence=sentence,
                            sentence_index=idx,
                            language=preprocessed.language,
                            entities=entities,
                            checkworthy_score=score,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Skipping sentence %d due to error: %s", idx, exc
                )
                continue

        claims.sort(key=lambda c: c.checkworthy_score, reverse=True)
        logger.debug(
            "Extracted %d claims from %d sentences.",
            len(claims),
            len(preprocessed.sentences),
        )
        return claims

    def _score_sentence(
        self, sentence: str, entities: list[NamedEntity]
    ) -> float:
        """Compute the combined check-worthiness score for a sentence.

        Args:
            sentence: The sentence string.
            entities: Named entities extracted from the sentence.

        Returns:
            Score in [0, 1].
        """
        cfg = self._settings.claim_extraction
        ner_score = 1.0 if self._ner.has_checkworthy_entities(entities) else 0.0

        classifier_score = 0.5  # neutral fallback
        if self._classifier is not None:
            try:
                result = self._classifier(
                    sentence, candidate_labels=_CANDIDATE_LABELS
                )
                label_scores: dict[str, float] = dict(
                    zip(result["labels"], result["scores"])
                )
                classifier_score = label_scores.get(_CHECKWORTHY_LABEL, 0.5)
            except Exception as exc:
                logger.debug("Classifier scoring failed: %s", exc)

        return cfg.ner_weight * ner_score + cfg.classifier_weight * classifier_score

    def _is_factual_sentence(
        self, sentence: str, cfg: "ClaimExtractionSettings"  # noqa: F821
    ) -> bool:
        """Apply length and structural filters to reject non-factual sentences.

        Args:
            sentence: The sentence to evaluate.
            cfg: Claim extraction settings.

        Returns:
            True if the sentence passes all filters.
        """
        if len(sentence) < cfg.min_claim_length:
            return False
        if len(sentence) > cfg.max_claim_length:
            return False
        if sentence.strip().endswith("?"):
            return False
        # Must contain at least one alphabetic character
        if not any(c.isalpha() for c in sentence):
            return False
        return True
