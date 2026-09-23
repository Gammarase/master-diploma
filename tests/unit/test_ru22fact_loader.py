"""Unit tests for data.loaders.ru22fact_loader.RU22FactLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.loaders.ru22fact_loader import RU22FactLoader
from retrieval.vector_store import EvidenceDocument


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.indexing.max_passage_chars = 800
    return settings


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[0.0] * 1024 for _ in texts]
    return embedder


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.upsert_documents.side_effect = lambda docs: len(docs)
    return store


@pytest.fixture
def loader(
    mock_store: MagicMock, mock_embedder: MagicMock, mock_settings: MagicMock
) -> RU22FactLoader:
    return RU22FactLoader(mock_store, mock_embedder, mock_settings)


def _upserted_docs(mock_store: MagicMock) -> list[EvidenceDocument]:
    return [d for call in mock_store.upsert_documents.call_args_list for d in call.args[0]]


def _make_tsv_file(tmp_path: Path, name: str = "ru22fact_test.tsv") -> Path:
    content = (
        "id\tclaim\tevidence\tlabel\texplanation\tlanguage\tdate\n"
        "622\tRussia launched missiles.\tEvidence passage 1.\tSupported\tExplanation 1.\tEnglish\t2022-09-19\n"
        "623\tNATO expanded east.\tEvidence passage 2.\tRefuted\tExplanation 2.\tEN\t\n"
        "624\tUkraine troops advanced.\tEvidence passage 3.\tNEI\tExplanation 3.\tUkrainian\t2023-01-02\n"
    )
    file = tmp_path / name
    file.write_text(content, encoding="utf-8")
    return file


def _row(**overrides: object) -> pd.Series:
    data = {
        "id": 622,
        "claim": "Russia launched missiles.",
        "evidence": "Evidence passage.",
        "label": "Supported",
        "explanation": "Explanation.",
        "language": "EN",
        "date": "2022-09-19",
    }
    data.update(overrides)
    return pd.Series(data)


class TestLoadFromFile:
    def test_loads_tsv_file(self, loader: RU22FactLoader, tmp_path: Path) -> None:
        tsv = _make_tsv_file(tmp_path)
        df = loader.load_from_file(tsv)
        assert len(df) == 3
        assert "claim" in df.columns
        assert list(df["language"]) == ["EN", "EN", "UK"]

    def test_raises_file_not_found(self, loader: RU22FactLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path/data.tsv")

    def test_raises_on_missing_columns(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad.tsv"
        bad_file.write_text("col1\tcol2\nval1\tval2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing expected columns"):
            loader.load_from_file(bad_file)

    def test_normalizes_column_names(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        content = "Claim\tEvidence\tLabel\tExplanation\tLanguage\ntext\tevid\tSupported\texpl\tEN\n"
        file = tmp_path / "data.tsv"
        file.write_text(content, encoding="utf-8")
        df = loader.load_from_file(file)
        assert "claim" in df.columns

    def test_accepts_referenced_explanation(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        file = tmp_path / "ru22fact_test.csv"
        file.write_text(
            "id,claim,evidence,referenced_explanation,label,language\n"
            "1,c,e,x,Supported,Russian\n",
            encoding="utf-8",
        )
        df = loader.load_from_file(file)
        assert "explanation" in df.columns
        assert df["language"][0] == "RU"


class TestSplitFromPath:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data/ru22fact_test.csv", "test"),
            ("ru22fact_train.tsv", "train"),
            ("RU22Fact_Validate.csv", "validate"),
            ("ru22fact.tsv", None),
        ],
    )
    def test_split_from_path(self, path: str, expected: str | None) -> None:
        assert RU22FactLoader.split_from_path(path) == expected


class TestRowToPassages:
    def test_basic_conversion(self, loader: RU22FactLoader) -> None:
        passages = loader._row_to_passages(_row(), "test", 0)
        assert passages is not None and len(passages) == 1
        p = passages[0]
        assert p.vector_id == "ru22fact-test-622-en-p0"
        assert p.claim_id == "622"
        assert p.label == "Supported"
        assert p.language == "EN"
        assert p.date == "2022-09-19"

    def test_normalizes_invalid_label(self, loader: RU22FactLoader) -> None:
        passages = loader._row_to_passages(_row(label="UnknownLabel"), "test", 1)
        assert passages[0].label == "NEI"

    @pytest.mark.parametrize("language", [None, float("nan"), "", "XX"])
    def test_skips_missing_or_unknown_language(
        self, loader: RU22FactLoader, language: object
    ) -> None:
        assert loader._row_to_passages(_row(language=language), "test", 2) is None

    def test_skips_boilerplate_only_evidence(self, loader: RU22FactLoader) -> None:
        row = _row(evidence="Hello, this is Bing. [^1^][1]")
        assert loader._row_to_passages(row, "test", 0) is None

    def test_evidence_is_cleaned(self, loader: RU22FactLoader) -> None:
        row = _row(
            evidence=(
                "Hello, this is Bing. Here are some relevant news articles based on "
                "your query:- **Headline**[^1^][1]: body text"
            )
        )
        text = loader._row_to_passages(row, "test", 0)[0].passage_text
        assert "Headline" in text and "body text" in text
        assert "Bing" not in text and "[^1^]" not in text

    def test_long_evidence_gives_several_passages(
        self, loader: RU22FactLoader
    ) -> None:
        evidence = ("Shelling continued near the city overnight, officials said. " * 40)[:2000]
        passages = loader._row_to_passages(_row(evidence=evidence), "test", 0)
        assert len(passages) >= 3
        assert all(len(p.passage_text) <= 800 for p in passages)
        assert [p.passage_index for p in passages] == list(range(len(passages)))
        assert passages[2].vector_id == "ru22fact-test-622-en-p2"

    def test_missing_id_uses_fallback(self, loader: RU22FactLoader) -> None:
        passages = loader._row_to_passages(_row(id=float("nan")), "test", 17)
        assert passages[0].claim_id == "17"

    def test_float_id_is_rendered_as_int(self, loader: RU22FactLoader) -> None:
        passages = loader._row_to_passages(_row(id=622.0), "test", 0)
        assert passages[0].claim_id == "622"


class TestIndexDataset:
    def test_returns_stats(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path)
        stats = loader.index_dataset(tsv)
        assert stats["total"] == 3
        assert stats["indexed_rows"] == 3
        assert stats["upserted"] == 3
        assert stats["skipped"] == 0

    def test_id_and_metadata_from_test_split(
        self, loader: RU22FactLoader, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        loader.index_dataset(_make_tsv_file(tmp_path))
        doc = _upserted_docs(mock_store)[0]
        assert doc.vector_id == "ru22fact-test-622-en-p0"
        assert doc.split == "test"
        assert doc.claim_id == "622"
        assert doc.date == "2022-09-19"
        assert len(doc.embedding) == 1024

    def test_empty_date_stored_as_empty_string(
        self, loader: RU22FactLoader, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        loader.index_dataset(_make_tsv_file(tmp_path))
        assert _upserted_docs(mock_store)[1].date == ""

    def test_nan_language_skipped(
        self, loader: RU22FactLoader, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        file = tmp_path / "ru22fact_test.csv"
        file.write_text(
            "id,claim,evidence,explanation,label,language\n"
            "1,Claim one,Evidence one.,x,Supported,English\n"
            "40,Claim two,Evidence two.,x,Refuted,\n",
            encoding="utf-8",
        )
        stats = loader.index_dataset(file)
        assert stats == {"total": 2, "indexed_rows": 1, "upserted": 1, "skipped": 1}
        assert all(d.claim_id != "40" for d in _upserted_docs(mock_store))

    def test_one_vector_per_passage(
        self,
        loader: RU22FactLoader,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
        tmp_path: Path,
    ) -> None:
        long_evidence = ("Shelling continued near the city overnight, officials said. " * 40)[:2000]
        file = tmp_path / "ru22fact_train.csv"
        pd.DataFrame(
            [
                {"id": 5, "claim": "c", "evidence": long_evidence, "label": "NEI",
                 "explanation": "x", "language": "English"},
                {"id": 6, "claim": "c", "evidence": "Short.", "label": "NEI",
                 "explanation": "x", "language": "Chinese"},
            ]
        ).to_csv(file, index=False)
        stats = loader.index_dataset(file)
        docs = _upserted_docs(mock_store)
        row5 = [d for d in docs if d.claim_id == "5"]
        assert len(row5) >= 3
        assert stats["upserted"] == len(docs) == len(row5) + 1
        assert docs[-1].vector_id == "ru22fact-train-6-zh-p0"
        # Passages are embedded in batches, not one call per row.
        assert mock_embedder.embed_batch.call_count == 1
        mock_embedder.embed_single.assert_not_called()

    def test_language_filter(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path)
        stats = loader.index_dataset(tsv, language_filter="en")
        assert stats["total"] == 2  # Only EN records

    def test_explicit_split_overrides_filename(
        self, loader: RU22FactLoader, mock_store: MagicMock, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path, name="custom.tsv")
        loader.index_dataset(tsv, split="kb")
        assert _upserted_docs(mock_store)[0].vector_id.startswith("ru22fact-kb-622-")

    def test_underivable_split_raises(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path, name="custom.tsv")
        with pytest.raises(ValueError, match="split"):
            loader.index_dataset(tsv)

    def test_make_vector_id(self) -> None:
        assert RU22FactLoader._make_vector_id("test", "622", "EN", 0) == "ru22fact-test-622-en-p0"
        assert RU22FactLoader._make_vector_id("train", "42", "UK", 3) == "ru22fact-train-42-uk-p3"
