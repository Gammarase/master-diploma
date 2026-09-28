"""
Background worker that runs queued checks through the pipeline.

The pipeline is initialized with retries (Ollama may still be pulling its
model, SearXNG may still be starting), then checks are processed one at a
time in submission order.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import Any, Protocol

from api.store import CheckStore
from logging_config import get_logger

__all__ = ["CheckWorker", "PipelineLike"]

logger = get_logger(__name__)

_MAX_ERROR_CHARS = 500


class PipelineLike(Protocol):
    def initialize(self) -> None: ...

    def analyze(self, text: str) -> list[Any]: ...


def _to_jsonable(item: Any) -> Any:
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.asdict(item)
    return item


class CheckWorker:
    """Initializes the pipeline and processes checks on a daemon thread.

    Args:
        store: Check store to claim work from.
        pipeline: Object with ``initialize()`` and ``analyze(text)``.
        init_retry_seconds: Delay between failed initialization attempts.
    """

    def __init__(
        self,
        store: CheckStore,
        pipeline: PipelineLike,
        init_retry_seconds: float = 15,
    ) -> None:
        self._store = store
        self._pipeline = pipeline
        self._init_retry_seconds = init_retry_seconds
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.state: str = "initializing"
        self.last_error: str | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="check-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5) -> None:
        """Ask the loop to exit; a running analysis is not interrupted."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def notify(self) -> None:
        """Wake the loop after a new submission."""
        self._wake.set()

    # ── Work ─────────────────────────────────────────────────────────────

    def ensure_initialized(self) -> bool:
        """Initialize the pipeline, retrying until it succeeds or stop is set."""
        while self.state != "ready":
            try:
                self._pipeline.initialize()
            except Exception as exc:
                self.last_error = _short_error(exc)
                logger.warning(
                    "Pipeline initialization failed (%s); retrying in %ss",
                    self.last_error,
                    self._init_retry_seconds,
                )
                if self._stop.wait(self._init_retry_seconds):
                    return False
            else:
                self.state = "ready"
                self.last_error = None
                logger.info("Pipeline ready; processing checks.")
        return True

    def run_once(self) -> bool:
        """Process the oldest queued check; return False if none was queued."""
        record = self._store.claim_next()
        if record is None:
            return False
        logger.info("Check %s started", record.id)
        try:
            results = self._pipeline.analyze(record.text)
            payload = json.dumps(
                [_to_jsonable(r) for r in results], ensure_ascii=False, default=str
            )
        except Exception as exc:
            logger.exception("Check %s failed", record.id)
            self._store.fail(record.id, _short_error(exc))
        else:
            self._store.complete(record.id, payload)
            logger.info("Check %s completed with %d result(s)", record.id, len(results))
        return True

    def _run(self) -> None:
        if not self.ensure_initialized():
            return
        while not self._stop.is_set():
            if self.run_once():
                continue
            self._wake.wait(1.0)
            self._wake.clear()


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
