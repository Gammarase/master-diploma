"""
SQLite store for submitted checks.

One table holds every check with its status, text and results. The
``seq`` column gives the queue order; ``id`` is the public random UUID.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["CheckRecord", "CheckStore", "TERMINAL_STATUSES"]

TERMINAL_STATUSES = frozenset({"completed", "failed"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT NOT NULL UNIQUE,
    text         TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    results_json TEXT,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS checks_status_seq ON checks(status, seq);
"""

_COLUMNS = (
    "seq, id, text, status, created_at, started_at, finished_at, results_json, error"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CheckRecord:
    """One stored check.

    Attributes:
        seq: Queue order (monotonic).
        id: Public UUID of the check.
        text: Submitted news text.
        status: ``queued``, ``running``, ``completed`` or ``failed``.
        created_at: Submission time (ISO 8601 UTC).
        started_at: Time the analysis started, or None.
        finished_at: Time the analysis ended, or None.
        results_json: JSON list of per-claim results once completed.
        error: Short error message once failed.
    """

    seq: int
    id: str
    text: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    results_json: str | None
    error: str | None


class CheckStore:
    """Durable queue of checks backed by one SQLite file.

    Args:
        db_path: Database file; its parent directory is created if missing.
    """

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, text: str, max_queued: int) -> CheckRecord | None:
        """Queue a new check; return None when ``max_queued`` is reached."""
        check_id = str(uuid.uuid4())
        with self._lock:
            (queued,) = self._conn.execute(
                "SELECT COUNT(*) FROM checks WHERE status = 'queued'"
            ).fetchone()
            if queued >= max_queued:
                return None
            self._conn.execute(
                "INSERT INTO checks (id, text, status, created_at) "
                "VALUES (?, ?, 'queued', ?)",
                (check_id, text, _now()),
            )
            self._conn.commit()
            return self._get_locked(check_id)

    def get(self, check_id: str) -> CheckRecord | None:
        with self._lock:
            return self._get_locked(check_id)

    def queue_position(self, check_id: str) -> int | None:
        """1-based position among queued checks, or None if not queued."""
        with self._lock:
            row = self._conn.execute(
                "SELECT seq, status FROM checks WHERE id = ?", (check_id,)
            ).fetchone()
            if row is None or row[1] != "queued":
                return None
            (position,) = self._conn.execute(
                "SELECT COUNT(*) FROM checks WHERE status = 'queued' AND seq <= ?",
                (row[0],),
            ).fetchone()
            return int(position)

    def count_queued(self) -> int:
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM checks WHERE status = 'queued'"
            ).fetchone()
            return int(count)

    def claim_next(self) -> CheckRecord | None:
        """Move the oldest queued check to ``running`` and return it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM checks WHERE status = 'queued' ORDER BY seq LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE checks SET status = 'running', started_at = ? WHERE id = ?",
                (_now(), row[0]),
            )
            self._conn.commit()
            return self._get_locked(row[0])

    def complete(self, check_id: str, results_json: str) -> None:
        self._finish(check_id, "completed", results_json=results_json)

    def fail(self, check_id: str, error: str) -> None:
        self._finish(check_id, "failed", error=error)

    def requeue_running(self) -> int:
        """Put checks left ``running`` by a stopped process back in the queue."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE checks SET status = 'queued', started_at = NULL "
                "WHERE status = 'running'"
            )
            self._conn.commit()
            return cur.rowcount

    def _finish(
        self,
        check_id: str,
        status: str,
        results_json: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE checks SET status = ?, finished_at = ?, results_json = ?, "
                "error = ? WHERE id = ?",
                (status, _now(), results_json, error, check_id),
            )
            self._conn.commit()

    def _get_locked(self, check_id: str) -> CheckRecord | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM checks WHERE id = ?", (check_id,)
        ).fetchone()
        return CheckRecord(*row) if row else None
