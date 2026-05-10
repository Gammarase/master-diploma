"""
RU22Fact dataset loader for the Disinformation Detection System.

Loads the RU22Fact TSV/CSV file (https://github.com/zeng-yirong/ru22fact),
converts records to EvidenceDocument objects, embeds them, and upserts
into the Pinecone vector index.

Dataset fields: claim, evidence, label, explanation, language
Labels: Supported, Refuted, NEI
Languages: EN, UK, RU, ZH
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

# Text length limits for metadata storage
_CLAIM_META_LIMIT = 512
_EXPL_META_LIMIT = 512


class RU22FactLoader:
    """Load the RU22Fact dataset and index it into Pinecone.

    The embedding for each record is computed from the concatenation:
        ``f"{claim} [SEP] {evidence}"``
    This combined representation improves retrieval because incoming
    queries (new claims) are matched against known claim+evidence pairs.

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
                lambda v: _LANGUAGE_NAME_MAP.get(str(v).strip().lower(), str(v).strip())
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

    def _row_to_document(
        self, row: "pd.Series", idx: int  # noqa: F821
    ) -> EvidenceDocument:
        """Convert a single DataFrame row to an EvidenceDocument.

        The embedding text is ``"{claim} [SEP] {evidence}"``.

        Args:
            row: A pandas Series with RU22FACT_COLUMNS fields.
            idx: Integer row index used for ID generation.

        Returns:
            EvidenceDocument with embedding populated.
        """
        claim = str(row.get("claim", "")).strip()
        evidence = str(row.get("evidence", "")).strip()
        label = str(row.get("label", "NEI")).strip()
        explanation = str(row.get("explanation", "")).strip()
        language = str(row.get("language", "EN")).strip().upper()

        # Normalize label and language to known values
        if label not in VALID_LABELS:
            label = "NEI"
        if language not in VALID_LANGUAGES:
            language = "EN"

        embed_text = f"{claim} [SEP] {evidence}"
        embedding = self._embedder.embed_single(embed_text)

        return EvidenceDocument(
            vector_id=self._make_vector_id(idx, language),
            claim_id=str(idx),
            claim_text=claim[:_CLAIM_META_LIMIT],
            evidence_text=evidence,
            label=label,
            language=language,
            explanation=explanation[:_EXPL_META_LIMIT],
            embedding=embedding,
        )

    def index_dataset(
        self,
        file_path: str | Path,
        batch_size: int = 100,
        skip_existing: bool = True,
        language_filter: str | None = None,
    ) -> dict[str, int]:
        """Load, embed, and index the entire RU22Fact dataset into Pinecone.

        Args:
            file_path: Path to the dataset file.
            batch_size: Number of documents to embed and upsert per batch.
            skip_existing: If True, attempt to detect already-indexed IDs
                and skip them (best-effort — Pinecone upsert is idempotent).
            language_filter: If set, only index records with this language
                (e.g. "EN", "UK"). Indexes all languages when None.

        Returns:
            Dict with keys ``"total"``, ``"upserted"``, ``"skipped"``.

        Raises:
            FileNotFoundError: If the dataset file is missing.
            VectorStoreError: On Pinecone upsert failure.
        """
        import pandas as pd

        df = self.load_from_file(file_path)

        if language_filter:
            lang = language_filter.upper()
            df = df[df["language"].str.upper() == lang].reset_index(drop=True)
            logger.info(
                "Filtered to language '%s': %d rows remain.", lang, len(df)
            )

        total = len(df)
        upserted = 0
        skipped = 0

        for batch_start in range(0, total, batch_size):
            batch_df = df.iloc[batch_start : batch_start + batch_size]
            documents: list[EvidenceDocument] = []

            for local_idx, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = batch_start + local_idx
                try:
                    doc = self._row_to_document(row, global_idx)
                    documents.append(doc)
                except Exception as exc:
                    logger.warning(
                        "Skipping row %d due to error: %s", global_idx, exc
                    )
                    skipped += 1
                    continue

            if documents:
                try:
                    self._store.upsert_documents(documents)
                    upserted += len(documents)
                except VectorStoreError as exc:
                    logger.error(
                        "Upsert failed for batch %d-%d: %s",
                        batch_start,
                        batch_start + len(documents),
                        exc,
                    )
                    raise

            logger.info(
                "Progress: %d/%d rows processed.", batch_start + len(batch_df), total
            )

        stats = {"total": total, "upserted": upserted, "skipped": skipped}
        logger.info("Indexing complete: %s", stats)
        return stats

    def _make_vector_id(self, idx: int, language: str) -> str:
        """Generate a deterministic Pinecone vector ID.

        Args:
            idx: Row index in the dataset.
            language: Language code (uppercase).

        Returns:
            String of the form ``"ru22fact-{idx}-{language.lower()}"``.
        """
        return f"ru22fact-{idx}-{language.lower()}"


def main() -> None:
    """Entry point for the ``disinfo-index`` CLI command.

    Usage::

        disinfo-index --file data/ru22fact.tsv [--language EN]
    """
    import argparse

    from configs import get_settings
    from logging_config import setup_logging
    from retrieval.embeddings import EmbeddingModel
    from retrieval.vector_store import PineconeVectorStore

    parser = argparse.ArgumentParser(
        description="Index RU22Fact dataset into Pinecone."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the RU22Fact TSV/CSV file.",
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
        help="Upsert batch size (default: 100).",
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
    )
    print(f"Indexing complete: {stats}")


if __name__ == "__main__":
    main()
