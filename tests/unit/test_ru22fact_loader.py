"""Unit tests for data.loaders.ru22fact_loader.RU22FactLoader."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.loaders.ru22fact_loader import RU22FactLoader
from retrieval.vector_store import EvidenceDocument


@pytest.fixture
def mock_settings() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0] * 768
    return embedder


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.upsert_documents.return_value = 5
    return store


@pytest.fixture
def loader(
    mock_store: MagicMock, mock_embedder: MagicMock, mock_settings: MagicMock
) -> RU22FactLoader:
    return RU22FactLoader(mock_store, mock_embedder, mock_settings)


def _make_tsv_file(tmp_path: Path) -> Path:
    content = (
        "claim\tevidence\tlabel\texplanation\tlanguage\n"
        "Russia launched missiles.\tEvidence passage 1.\tSupported\tExplanation 1.\tEN\n"
        "NATO expanded east.\tEvidence passage 2.\tRefuted\tExplanation 2.\tEN\n"
        "Ukraine troops advanced.\tEvidence passage 3.\tNEI\tExplanation 3.\tUK\n"
    )
    file = tmp_path / "ru22fact.tsv"
    file.write_text(content, encoding="utf-8")
    return file


class TestLoadFromFile:
    def test_loads_tsv_file(self, loader: RU22FactLoader, tmp_path: Path) -> None:
        tsv = _make_tsv_file(tmp_path)
        df = loader.load_from_file(tsv)
        assert len(df) == 3
        assert "claim" in df.columns

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


class TestRowToDocument:
    def test_basic_conversion(
        self, loader: RU22FactLoader, mock_embedder: MagicMock
    ) -> None:
        row = pd.Series(
            {
                "claim": "Russia launched missiles.",
                "evidence": "Evidence passage.",
                "label": "Supported",
                "explanation": "Explanation.",
                "language": "EN",
            }
        )
        doc = loader._row_to_document(row, 0)
        assert isinstance(doc, EvidenceDocument)
        assert doc.label == "Supported"
        assert doc.language == "EN"
        assert doc.vector_id == "ru22fact-0-en"

    def test_normalizes_invalid_label(self, loader: RU22FactLoader) -> None:
        row = pd.Series(
            {
                "claim": "claim",
                "evidence": "evidence",
                "label": "UnknownLabel",
                "explanation": "expl",
                "language": "EN",
            }
        )
        doc = loader._row_to_document(row, 1)
        assert doc.label == "NEI"

    def test_normalizes_invalid_language(self, loader: RU22FactLoader) -> None:
        row = pd.Series(
            {
                "claim": "claim",
                "evidence": "evidence",
                "label": "Supported",
                "explanation": "expl",
                "language": "XX",
            }
        )
        doc = loader._row_to_document(row, 2)
        assert doc.language == "EN"

    def test_embed_text_is_combined(
        self, loader: RU22FactLoader, mock_embedder: MagicMock
    ) -> None:
        row = pd.Series(
            {
                "claim": "Claim text",
                "evidence": "Evidence text",
                "label": "Supported",
                "explanation": "Expl",
                "language": "EN",
            }
        )
        loader._row_to_document(row, 0)
        mock_embedder.embed_single.assert_called_with("Claim text [SEP] Evidence text")


class TestIndexDataset:
    def test_returns_stats(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path)
        stats = loader.index_dataset(tsv)
        assert "total" in stats
        assert stats["total"] == 3
        assert stats["upserted"] == 3
        assert stats["skipped"] == 0

    def test_language_filter(
        self, loader: RU22FactLoader, tmp_path: Path
    ) -> None:
        tsv = _make_tsv_file(tmp_path)
        stats = loader.index_dataset(tsv, language_filter="EN")
        assert stats["total"] == 2  # Only EN records

    def test_make_vector_id(self, loader: RU22FactLoader) -> None:
        assert loader._make_vector_id(0, "EN") == "ru22fact-0-en"
        assert loader._make_vector_id(42, "UK") == "ru22fact-42-uk"
