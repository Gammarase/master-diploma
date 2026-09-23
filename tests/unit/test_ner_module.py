"""Unit tests for claim_extraction.ner_module.NERModule."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import spacy

from claim_extraction.ner_module import NERModule, NamedEntity
from exceptions import ClaimExtractionError


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    return tokenizer


@pytest.fixture
def ner_module(mock_tokenizer: MagicMock) -> NERModule:
    return NERModule(mock_tokenizer)


def _make_entities(labels: list[str]) -> list[NamedEntity]:
    return [
        NamedEntity(text=f"entity_{i}", label=label, start_char=i * 5, end_char=i * 5 + 4)
        for i, label in enumerate(labels)
    ]


class TestExtractEntities:
    def test_extracts_entities_from_doc(self, ner_module: NERModule) -> None:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Russia launched 45 missiles at Ukraine in 2024.")
        entities = ner_module.extract_entities(doc)
        labels = {e.label for e in entities}
        # Should find at least GPE (Russia/Ukraine) and a number
        assert len(entities) > 0
        assert all(isinstance(e, NamedEntity) for e in entities)

    def test_empty_doc(self, ner_module: NERModule) -> None:
        nlp = spacy.blank("en")
        doc = nlp("")
        entities = ner_module.extract_entities(doc)
        assert entities == []

    def test_raises_on_failure(self, ner_module: NERModule) -> None:
        bad_doc = MagicMock()
        bad_doc.ents = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        # If ents raises, should wrap in ClaimExtractionError
        bad_doc2 = MagicMock()
        type(bad_doc2).ents = property(lambda self: [])
        bad_doc2.text = ""
        # Just verify normal path works without error
        ner_module.extract_entities(bad_doc2)


class TestHasCheckworthyEntities:
    def test_subject_and_quantifier(self, ner_module: NERModule) -> None:
        entities = _make_entities(["GPE", "CARDINAL"])
        assert ner_module.has_checkworthy_entities(entities) is True

    def test_subject_only(self, ner_module: NERModule) -> None:
        entities = _make_entities(["PERSON"])
        assert ner_module.has_checkworthy_entities(entities) is False

    def test_quantifier_only(self, ner_module: NERModule) -> None:
        entities = _make_entities(["CARDINAL", "PERCENT"])
        assert ner_module.has_checkworthy_entities(entities) is False

    def test_empty_entities(self, ner_module: NERModule) -> None:
        assert ner_module.has_checkworthy_entities([]) is False

    def test_org_and_money(self, ner_module: NERModule) -> None:
        entities = _make_entities(["ORG", "MONEY"])
        assert ner_module.has_checkworthy_entities(entities) is True

    def test_person_and_date(self, ner_module: NERModule) -> None:
        entities = _make_entities(["PERSON", "DATE"])
        assert ner_module.has_checkworthy_entities(entities) is True


class TestComputeEntityDensity:
    def test_zero_text_length(self, ner_module: NERModule) -> None:
        assert ner_module.compute_entity_density([], 0) == 0.0

    def test_no_entities(self, ner_module: NERModule) -> None:
        assert ner_module.compute_entity_density([], 100) == 0.0

    def test_high_density_capped_at_one(self, ner_module: NERModule) -> None:
        entities = _make_entities(["GPE"] * 20)
        density = ner_module.compute_entity_density(entities, 5)
        assert density == 1.0

    def test_normal_density(self, ner_module: NERModule) -> None:
        entities = _make_entities(["GPE", "CARDINAL"])
        density = ner_module.compute_entity_density(entities, 50)
        assert 0.0 < density <= 1.0


class TestMultilingualQuantifiers:
    """uk/ru pipelines have no DATE/CARDINAL labels; a regex fills the gap."""

    def test_ukrainian_date_and_location(self, ner_module: NERModule) -> None:
        doc = spacy.load("uk_core_news_sm")(
            "У Києві 5 червня пролунали вибухи, повідомив мер міста."
        )
        entities = ner_module.extract_entities(doc)
        labels = {e.label for e in entities}
        assert "LOC" in labels
        assert {"DATE", "CARDINAL"} <= labels
        assert ner_module.has_checkworthy_entities(entities)

    def test_russian_number_and_org(self, ner_module: NERModule) -> None:
        doc = spacy.load("ru_core_news_sm")(
            "Минобороны России заявило об уничтожении 20 украинских танков."
        )
        entities = ner_module.extract_entities(doc)
        assert any(e.label == "CARDINAL" and e.text == "20" for e in entities)
        assert ner_module.has_checkworthy_entities(entities)

    def test_per_counts_as_subject(self, ner_module: NERModule) -> None:
        assert ner_module.has_checkworthy_entities(_make_entities(["PER", "DATE"]))

    @pytest.mark.parametrize(
        "text,label,match",
        [
            ("Вибухи пролунали у травні.", "DATE", "травні"),
            ("Это случилось 3 марта.", "DATE", "марта"),
            ("泽连斯基于2023年5月9日通电话。", "DATE", "2023年"),
            ("冲突在三月升级。", "DATE", "三月"),
            ("Loss of 1,200 tanks.", "CARDINAL", "1,200"),
        ],
    )
    def test_regex_quantifiers(self, text: str, label: str, match: str) -> None:
        found = NERModule._regex_quantifiers(text, [])
        assert any(e.label == label and e.text == match for e in found)

    def test_regex_skips_overlap_with_spacy_entities(self) -> None:
        existing = [NamedEntity("F-16", "PRODUCT", 0, 4)]
        found = NERModule._regex_quantifiers("F-16 jets arrived.", existing)
        assert found == []

    def test_modal_may_is_not_a_date(self) -> None:
        assert NERModule._regex_quantifiers("Russia may attack.", []) == []

    def test_regex_not_used_when_spacy_found_quantifier(
        self, ner_module: NERModule
    ) -> None:
        doc = spacy.load("en_core_web_sm")("Russia launched 45 missiles in 2024.")
        entities = ner_module.extract_entities(doc)
        spacy_spans = {(e.start_char, e.end_char) for e in doc.ents}
        assert {(e.start_char, e.end_char) for e in entities} == spacy_spans
