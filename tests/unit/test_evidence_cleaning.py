"""Unit tests for data.loaders.evidence_cleaning."""

from __future__ import annotations

import pytest

from data.loaders.evidence_cleaning import chunk_evidence, clean_evidence

# Verbatim prefixes of real rows from ru22fact_test.csv.
_ROW_5 = (
    '[1]: "September 1, 2022 Russia-Ukraine news - CNN International"'
    '[2]: "Russia-appointed official tells state media that IAEA mission will stay ..."'
    "Here are five relevant news articles about the topic:"
    "- **Russia-appointed official tells state media that IAEA mission will stay "
    "until Sept. 3**[^1^][2] [^2^][1]: International Atomic Energy Agency (IAEA) "
    "mission is planning on inspecting operational parts of the Zaporizhzhia "
    "nuclear power plant."
)
_ROW_2900 = (
    '[3]: "UN says nearly 9m people have fled Ukraine since Russian invasion"'
    "Hello, this is Bing. Here are some relevant news articles based on your query:"
    '- **More civilians now ready to flee Donetsk with casualties suffered "almost '
    'every day," says official**[^1^][1] [^2^][2]: increasing number of civilians '
    "are now ready to evacuate from the Ukrainian-controlled Donetsk region."
)
_ROW_139_RU = (
    '[3]: "В Ставрополе семьям военных при покупке двух гробов третий обещают ..."  '
    "К сожалению, я не нашел пяти новостей о предоставлении семьям российских "
    "военных бесплатного третьего гроба при покупке двух. Однако, я нашел "
    "информацию о том, что объявление оказалось фейком[^1^][1].   "
    "Могу ли я помочь Вам найти что-то еще?"
)


class TestCleanEvidence:
    def test_spec_bing_example(self) -> None:
        raw = (
            "Hello, this is Bing. Here are some relevant news articles based on "
            "your query:- **Headline**[^1^][1]: body text"
        )
        cleaned = clean_evidence(raw)
        assert "Headline" in cleaned
        assert "body text" in cleaned
        assert "Hello, this is Bing" not in cleaned
        assert "[^1^][1]" not in cleaned

    def test_row_5_lead_in_and_markers_removed(self) -> None:
        cleaned = clean_evidence(_ROW_5)
        assert "Here are five relevant news articles" not in cleaned
        assert "[^" not in cleaned
        assert "until Sept. 3" in cleaned
        assert "Zaporizhzhia nuclear power plant." in cleaned
        assert '[1]: "September 1, 2022' in cleaned

    def test_row_2900_greeting_removed(self) -> None:
        cleaned = clean_evidence(_ROW_2900)
        assert "Bing" not in cleaned
        assert "based on your query" not in cleaned
        assert "[^2^][2]" not in cleaned
        assert "Ukrainian-controlled Donetsk region." in cleaned

    def test_russian_apology_and_offer_removed(self) -> None:
        cleaned = clean_evidence(_ROW_139_RU)
        assert "К сожалению" not in cleaned
        assert "Могу ли я помочь" not in cleaned
        assert "объявление оказалось фейком." in cleaned
        assert "[^1^]" not in cleaned

    def test_chinese_greeting_removed(self) -> None:
        raw = "您好，这是必应。根据网络搜索，以下是五条相关消息：- 新华社报道，卢布走强[^1^][1]。"
        cleaned = clean_evidence(raw)
        assert "必应" not in cleaned
        assert "以下是五条相关消息" not in cleaned
        assert "新华社报道，卢布走强。" in cleaned

    def test_ukrainian_boilerplate_removed(self) -> None:
        raw = "Привіт, це Bing. На жаль, я не знайшов новин про це. ЗСУ звільнили Херсон."
        assert clean_evidence(raw) == "ЗСУ звільнили Херсон."

    def test_clean_text_unchanged_modulo_whitespace(self) -> None:
        raw = "Russia launched   missiles at Kyiv.\n\nOfficials said 3.5 km of road was hit."
        assert clean_evidence(raw) == (
            "Russia launched missiles at Kyiv. Officials said 3.5 km of road was hit."
        )

    def test_legit_sorry_in_news_kept(self) -> None:
        raw = 'The minister said "we are sorry for the losses" on Monday.'
        assert clean_evidence(raw) == raw

    @pytest.mark.parametrize("raw", ["", "   ", "Hello, this is Bing."])
    def test_empty_or_boilerplate_only(self, raw: str) -> None:
        assert clean_evidence(raw) == ""


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

    def test_real_row_chunks_within_limit(self) -> None:
        chunks = chunk_evidence(clean_evidence(_ROW_5 * 5), 300)
        assert len(chunks) >= 2
        assert all(len(c) <= 300 for c in chunks)

    def test_invalid_max_chars(self) -> None:
        with pytest.raises(ValueError):
            chunk_evidence("text", 0)
