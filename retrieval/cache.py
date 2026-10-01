"""
SQLite cache for web evidence retrieval.

Stores search responses (keyed by backend, full query string and language)
and extracted page content (keyed by URL), each with its retrieval time.
A frozen cache file makes a run replayable offline (``read_only`` mode).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logging_config import get_logger

if TYPE_CHECKING:
    from configs import Settings

__all__ = ["WebCache", "CacheMode", "CACHE_FILENAME", "utc_now_iso"]

logger = get_logger(__name__)

CacheMode = Literal["read_write", "read_only", "off"]
CACHE_FILENAME = "web_cache.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search (
    backend TEXT NOT NULL,
    query TEXT NOT NULL,
    language TEXT NOT NULL,
    response_json TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (backend, query, language)
);
CREATE TABLE IF NOT EXISTS page (
    url TEXT PRIMARY KEY,
    final_url TEXT NOT NULL,
    status TEXT NOT NULL,
    content_json TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a ``Z`` suffix, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class WebCache:
    """Disk cache of search responses and extracted pages.

    Args:
        cache_dir: Directory holding ``web_cache.sqlite``.
        mode: ``read_write`` (default), ``read_only`` or ``off``.
        max_age_days: Entries older than this are ignored; None = no expiry.
        now: Clock returning the current ISO-8601 UTC time (for tests).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        mode: CacheMode = "read_write",
        max_age_days: float | None = None,
        now: Callable[[], str] = utc_now_iso,
    ) -> None:
        if mode not in ("read_write", "read_only", "off"):
            raise ValueError(f"Unknown cache mode {mode!r}")
        self._mode: CacheMode = mode
        self._max_age = (
            timedelta(days=max_age_days) if max_age_days is not None else None
        )
        self._now = now
        self._path = Path(cache_dir) / CACHE_FILENAME
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if mode == "read_write":
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        elif mode == "read_only" and self._path.is_file():
            uri = f"{self._path.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        elif mode == "read_only":
            logger.warning(
                "Cache file '%s' does not exist; read_only cache is empty.",
                self._path,
            )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "WebCache":
        r = settings.retrieval
        return cls(r.cache_dir, r.cache_mode, r.cache_max_age_days)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def mode(self) -> CacheMode:
        return self._mode

    @property
    def offline(self) -> bool:
        """True in ``read_only`` mode: no network requests may be sent."""
        return self._mode == "read_only"

    @property
    def path(self) -> Path:
        return self._path

    # ── helpers ─────────────────────────────────────────────────────────────

    def _fresh(self, retrieved_at: str) -> bool:
        if self._max_age is None:
            return True
        try:
            age = _parse_iso(self._now()) - _parse_iso(retrieved_at)
        except ValueError:
            return False
        return age <= self._max_age

    def _read(self, sql: str, params: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if self._mode == "off" or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None or not self._fresh(row[-1]):
            return None
        return tuple(row)

    def _write(self, sql: str, params: tuple[Any, ...]) -> None:
        if self._mode != "read_write" or self._conn is None:
            return
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ── search ──────────────────────────────────────────────────────────────

    def get_search(
        self, backend: str, query: str, language: str
    ) -> dict[str, Any] | None:
        """Cached search response dict, or None on a miss."""
        row = self._read(
            "SELECT response_json, retrieved_at FROM search "
            "WHERE backend=? AND query=? AND language=?",
            (backend, query, language),
        )
        return json.loads(row[0]) if row else None

    def put_search(
        self,
        backend: str,
        query: str,
        language: str,
        response: dict[str, Any],
        failed: bool = False,
    ) -> None:
        """Store a search response. Failed searches are never stored."""
        if failed:
            return
        self._write(
            "INSERT OR REPLACE INTO search VALUES (?, ?, ?, ?, ?)",
            (
                backend,
                query,
                language,
                json.dumps(response, ensure_ascii=False),
                self._now(),
            ),
        )

    # ── pages ───────────────────────────────────────────────────────────────

    def get_page(self, url: str) -> dict[str, Any] | None:
        """Cached page entry ``{final_url, status, content, retrieved_at}``."""
        row = self._read(
            "SELECT final_url, status, content_json, retrieved_at FROM page "
            "WHERE url=?",
            (url,),
        )
        if row is None:
            return None
        return {
            "final_url": row[0],
            "status": row[1],
            "content": json.loads(row[2]),
            "retrieved_at": row[3],
        }

    def put_page(
        self,
        url: str,
        final_url: str,
        status: str,
        content: dict[str, Any] | None,
        retrieved_at: str | None = None,
    ) -> None:
        """Store extracted content, or a negative result (``content=None``).

        Args:
            url: Requested (normalised) URL.
            final_url: URL after redirects.
            status: ``ok`` or a failure reason such as ``403`` or
                ``robots_disallowed``.
            content: Extracted text and metadata; None for failures.
            retrieved_at: Fetch time; defaults to now.
        """
        self._write(
            "INSERT OR REPLACE INTO page VALUES (?, ?, ?, ?, ?)",
            (
                url,
                final_url,
                status,
                json.dumps(content, ensure_ascii=False),
                retrieved_at or self._now(),
            ),
        )

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None
