"""
RU22Fact dataset loader for the Disinformation Detection System.

Loads a RU22Fact CSV/TSV split (https://github.com/zeng-yirong/ru22fact),
cleans and chunks each record's evidence into passages, batch-embeds the
passages, and upserts one vector per passage into Pinecone.

Dataset fields: id, claim, evidence, label, explanation, language, date
Labels: Supported, Refuted, NEI
Languages: EN, UK, RU, ZH
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.loaders.evidence_cleaning import chunk_evidence, clean_evidence
from exceptions import VectorStoreError
from logging_config import get_logger
from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import EvidenceDocument, PineconeVectorStore

__all__ = ["RU22FactLoader"]

logger = get_logger(__name__)

RU22FACT_COLUMNS = ["claim", "evidence", "label", "explanation", "language"]
VALID_LABELS = {"Supported", "Refuted", "NEI"}
VALID_LANGUAGES = {"EN", "UK", "RU", "ZH"}

_LANGUAGE_NAME_MAP = {
    "english": "EN",
    "ukrainian": "UK",
    "russian": "RU",
    "chinese": "ZH",
}

_SPLIT_FROM_FILENAME_RE = re.compile(r"^ru22fact_(\w+)$", re.IGNORECASE)

# Text length limits for metadata storage
_CLAIM_META_LIMIT = 512
_EXPL_META_LIMIT = 512


def _is_missing(value: Any) -> bool:
    """True for None, NaN and blank strings."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def _text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


@dataclass
class _Passage:
    """A passage awaiting embedding."""

    vector_id: str
    claim_id: str
    claim_text: str
    passage_text: str
    passage_index: int
    label: str
    language: str
    explanation: str
    date: str


class RU22FactLoader:
    """Load a RU22Fact split and index its evidence passages into Pinecone.

    Each record's evidence is cleaned of search-assistant boilerplate and
    split into passages of at most ``indexing.max_passage_chars``. Every
    passage is embedded on its own (the claim is *not* prepended), because
    incoming queries are claims and should match evidence text.

    Args:
        vector_store: Connected PineconeVectorStore instance.
        embedding_model: EmbeddingModel for generating embeddings.
        settings: Application settings.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embedding_model: EmbeddingModel,
        settings: "Settings",  # noqa: F821
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_model
        self._settings = settings

    def load_from_file(self, file_path: str | Path) -> "pd.DataFrame":  # noqa: F821
        """Load the RU22Fact dataset from a TSV or CSV file.

        Auto-detects separator based on file extension (.tsv → tab, else comma).
        Language names are mapped to codes; missing values are left missing.

        Args:
            file_path: Path to the dataset file.

        Returns:
            DataFrame with validated columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If expected columns are missing.
        """
        import pandas as pd

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep, encoding="utf-8")

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Accept referenced_explanation as an alias for explanation
        if "explanation" not in df.columns and "referenced_explanation" in df.columns:
            df = df.rename(columns={"referenced_explanation": "explanation"})

        # Normalize full language names to codes (e.g. "English" → "EN")
        if "language" in df.columns:
            df["language"] = df["language"].apply(
                lambda v: None
                if _is_missing(v)
                else _LANGUAGE_NAME_MAP.get(str(v).strip().lower(), str(v).strip().upper())
            )

        missing = [c for c in RU22FACT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"RU22Fact dataset missing expected columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        logger.info(
            "Loaded RU22Fact dataset: %d rows from '%s'.", len(df), file_path
        )
        return df

    @staticmethod
    def split_from_path(file_path: str | Path) -> str | None:
        """Derive the split name from ``ru22fact_{split}.csv``, else None."""
        match = _SPLIT_FROM_FILENAME_RE.match(Path(file_path).stem)
        return match.group(1).lower() if match else None

    def _row_to_passages(
        self, row: "pd.Series", split: str, fallback_id: int  # noqa: F821
    ) -> list[_Passage] | None:
        """Clean and chunk one row into passages.

        Returns:
            List of passages, or None if the row must be skipped (unknown
            language or no evidence left after cleaning).
        """
        raw_id = row.get("id")
        if _is_missing(raw_id):
            claim_id = str(fallback_id)
        elif isinstance(raw_id, float) and raw_id.is_integer():
            claim_id = str(int(raw_id))  # pandas upcasts ids to float when any is NaN
        else:
            claim_id = _text(raw_id)
        language = _text(row.get("language")).upper()
        if language not in VALID_LANGUAGES:
            logger.warning(
                "Skipping row id=%s: missing or unsupported language %r.",
                claim_id,
                row.get("language"),
            )
            return None

        raw_evidence = _text(row.get("evidence"))
        cleaned = clean_evidence(raw_evidence)
        removed = len(raw_evidence) - len(cleaned)
        if removed:
            logger.debug("Row id=%s: cleaning removed %d characters.", claim_id, removed)

        chunks = chunk_evidence(cleaned, self._settings.indexing.max_passage_chars)
        if not chunks:
            logger.warning("Skipping row id=%s: no evidence after cleaning.", claim_id)
            return None

        label = _text(row.get("label"))
        if label not in VALID_LABELS:
            label = "NEI"
        claim = _text(row.get("claim"))
        explanation = _text(row.get("explanation"))
        date = _text(row.get("date"))

        return [
            _Passage(
                vector_id=self._make_vector_id(split, claim_id, language, n),
                claim_id=claim_id,
                claim_text=claim[:_CLAIM_META_LIMIT],
                passage_text=chunk,
                passage_index=n,
                label=label,
                language=language,
                explanation=explanation[:_EXPL_META_LIMIT],
                date=date,
            )
            for n, chunk in enumerate(chunks)
        ]

    def _embed_passages(
        self, passages: list[_Passage], split: str
    ) -> list[EvidenceDocument]:
        """Batch-embed passages and wrap them as EvidenceDocuments."""
        embeddings = self._embedder.embed_batch([p.passage_text for p in passages])
        return [
            EvidenceDocument(
                vector_id=p.vector_id,
                claim_id=p.claim_id,
                claim_text=p.claim_text,
                passage_text=p.passage_text,
                label=p.label,
                language=p.language,
                explanation=p.explanation,
                embedding=emb,
                split=split,
                passage_index=p.passage_index,
                date=p.date,
            )
            for p, emb in zip(passages, embeddings)
        ]

    def index_dataset(
        self,
        file_path: str | Path,
        batch_size: int = 100,
        skip_existing: bool = True,
        language_filter: str | None = None,
        split: str | None = None,
    ) -> dict[str, int]:
        """Load, clean, chunk, embed, and index a RU22Fact split into Pinecone.

        Args:
            file_path: Path to the dataset file.
            batch_size: Number of rows to process per embed/upsert batch.
            skip_existing: Kept for CLI compatibility; upserts are idempotent
                because vector IDs are deterministic.
            language_filter: If set, only index records with this language
                (e.g. "EN", "UK"). Indexes all languages when None.
            split: Dataset split name. Derived from ``ru22fact_{split}.csv``
                when None.

        Returns:
            Dict with ``"total"`` (rows), ``"indexed_rows"``, ``"upserted"``
            (vectors) and ``"skipped"`` (rows).

        Raises:
            FileNotFoundError: If the dataset file is missing.
            ValueError: If the split cannot be determined.
            VectorStoreError: On Pinecone upsert failure.
        """
        resolved_split = split or self.split_from_path(file_path)
        if not resolved_split:
            raise ValueError(
                f"Cannot derive the dataset split from '{file_path}'. "
                "Name the file ru22fact_{split}.csv or pass split explicitly."
            )

        df = self.load_from_file(file_path)
        if "id" not in df.columns:
            logger.warning("Dataset has no 'id' column; using row positions as ids.")

        if language_filter:
            lang = language_filter.upper()
            df = df[df["language"] == lang].reset_index(drop=True)
            logger.info(
                "Filtered to language '%s': %d rows remain.", lang, len(df)
            )

        total = len(df)
        upserted = 0
        skipped = 0
        indexed_rows = 0

        for batch_start in range(0, total, batch_size):
            batch_df = df.iloc[batch_start : batch_start + batch_size]
            passages: list[_Passage] = []

            for local_idx, (_, row) in enumerate(batch_df.iterrows()):
                row_passages = self._row_to_passages(
                    row, resolved_split, batch_start + local_idx
                )
                if row_passages is None:
                    skipped += 1
                    continue
                passages.extend(row_passages)
                indexed_rows += 1

            if passages:
                documents = self._embed_passages(passages, resolved_split)
                try:
                    upserted += self._store.upsert_documents(documents)
                except VectorStoreError as exc:
                    logger.error(
                        "Upsert failed for rows %d-%d: %s",
                        batch_start,
                        batch_start + len(batch_df),
                        exc,
                    )
                    raise

            logger.info(
                "Progress: %d/%d rows processed.", batch_start + len(batch_df), total
            )

        stats = {
            "total": total,
            "indexed_rows": indexed_rows,
            "upserted": upserted,
            "skipped": skipped,
        }
        logger.info("Indexing complete: %s", stats)
        return stats

    @staticmethod
    def _make_vector_id(split: str, claim_id: str, language: str, n: int) -> str:
        """Generate a deterministic Pinecone vector ID.

        Returns:
            String of the form ``"ru22fact-{split}-{id}-{lang}-p{n}"``.
        """
        return f"ru22fact-{split}-{claim_id}-{language.lower()}-p{n}"


def main() -> None:
    """Entry point for the ``disinfo-index`` CLI command.

    Usage::

        disinfo-index --file data/datasets/ru22fact/ru22fact_test.csv [--language EN]
    """
    import argparse

    from configs import get_settings
    from logging_config import setup_logging
    from retrieval.embeddings import EmbeddingModel
    from retrieval.vector_store import PineconeVectorStore

    parser = argparse.ArgumentParser(
        description="Index RU22Fact evidence passages into Pinecone."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the RU22Fact TSV/CSV file.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split name (default: derived from ru22fact_{split}.csv).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Filter to a specific language (EN, UK, RU, ZH). Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows per embed/upsert batch (default: 100).",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(
        level=settings.logging.level,
        log_file=settings.logging.log_file,
    )

    embedder = EmbeddingModel(settings)
    store = PineconeVectorStore(settings, embedder)
    store.connect()

    loader = RU22FactLoader(store, embedder, settings)
    stats = loader.index_dataset(
        file_path=args.file,
        batch_size=args.batch_size,
        language_filter=args.language,
        split=args.split,
    )
    print(f"Indexing complete: {stats}")


if __name__ == "__main__":
    main()
