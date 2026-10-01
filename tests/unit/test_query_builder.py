"""Unit tests for retrieval.query_builder.QueryBuilder."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from retrieval.query_builder import QueryBuilder

_UK_CLAIM = "Росія обстріляла Запорізьку АЕС 5 серпня 2022 року."
_EN_CLAIM = "The IAEA confirmed shelling at the Zaporizhzhia plant in August 2022."


def _fake(payload: Any) -> MagicMock:
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return MagicMock(return_value=raw)


class TestNonEnglish:
    def test_ukrainian_claim_gets_native_then_english(self) -> None:
        gen = _fake(
            {
                "native_query": "обстріл Запорізької АЕС серпень 2022",
                "english_query": "Zaporizhzhia nuclear plant shelling August 2022",
            }
        )
        queries = QueryBuilder(gen).build(_UK_CLAIM, "uk")
        assert queries == [
            "обстріл Запорізької АЕС серпень 2022",
            "Zaporizhzhia nuclear plant shelling August 2022",
        ]
        assert _UK_CLAIM not in queries
        gen.assert_called_once()
        prompt, schema = gen.call_args.args
        assert "Ukrainian" in prompt and _UK_CLAIM in prompt
        assert list(schema["properties"]) == ["native_query", "english_query"]

    def test_only_english_usable(self) -> None:
        gen = _fake({"native_query": "", "english_query": "Zaporizhzhia shelling 2022"})
        assert QueryBuilder(gen).build(_UK_CLAIM, "uk") == [
            _UK_CLAIM,
            "Zaporizhzhia shelling 2022",
        ]

    def test_only_native_usable(self) -> None:
        gen = _fake({"native_query": "обстріл ЗАЕС 2022", "english_query": "x"})
        assert QueryBuilder(gen).build(_UK_CLAIM, "uk") == ["обстріл ЗАЕС 2022", _UK_CLAIM]

    def test_neither_usable(self) -> None:
        gen = _fake({"native_query": " ", "english_query": None})
        assert QueryBuilder(gen).build(_UK_CLAIM, "uk") == [_UK_CLAIM]


class TestEnglish:
    def test_english_claim_first_then_generated(self) -> None:
        gen = _fake({"english_query": "IAEA Zaporizhzhia shelling August 2022"})
        assert QueryBuilder(gen).build(_EN_CLAIM, "en") == [
            _EN_CLAIM,
            "IAEA Zaporizhzhia shelling August 2022",
        ]
        _, schema = gen.call_args.args
        assert list(schema["properties"]) == ["english_query"]

    def test_generated_equal_to_claim_deduped(self) -> None:
        gen = _fake({"english_query": "  " + _EN_CLAIM.upper() + " "})
        assert QueryBuilder(gen).build(_EN_CLAIM, "en") == [_EN_CLAIM]

    def test_empty_language_treated_as_english(self) -> None:
        gen = _fake({"english_query": "IAEA Zaporizhzhia 2022"})
        assert QueryBuilder(gen).build(_EN_CLAIM, "")[0] == _EN_CLAIM


class TestNormalisation:
    def test_whitespace_collapsed(self) -> None:
        gen = _fake({"english_query": "IAEA\n  Zaporizhzhia\tplant"})
        assert QueryBuilder(gen).build(_EN_CLAIM, "en")[1] == "IAEA Zaporizhzhia plant"

    @pytest.mark.parametrize("bad", ["ab", "x" * 201])
    def test_length_limits(self, bad: str) -> None:
        gen = _fake({"english_query": bad})
        assert QueryBuilder(gen).build(_EN_CLAIM, "en") == [_EN_CLAIM]


class TestFallbacks:
    def test_exception_gives_claim(self) -> None:
        gen = MagicMock(side_effect=RuntimeError("ollama down"))
        assert QueryBuilder(gen).build(_UK_CLAIM, "uk") == [_UK_CLAIM]

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
    def test_garbage_output_gives_claim(self, raw: str) -> None:
        assert QueryBuilder(_fake(raw)).build(_EN_CLAIM, "en") == [_EN_CLAIM]

    def test_disabled_no_llm_call(self) -> None:
        gen = _fake({"english_query": "x y z"})
        builder = QueryBuilder(gen, enabled=False)
        assert builder.enabled is False
        assert builder.build(_UK_CLAIM, "uk") == [_UK_CLAIM]
        gen.assert_not_called()

    def test_no_generator(self) -> None:
        assert QueryBuilder(None).build(_EN_CLAIM, "en") == [_EN_CLAIM]
