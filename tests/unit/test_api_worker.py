"""Unit tests for api.worker.CheckWorker (fake pipeline, no models)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from api.store import CheckStore
from api.worker import CheckWorker
from explainability.explainer import ExplanationOutput


def make_output(claim: str = "claim") -> ExplanationOutput:
    return ExplanationOutput(
        claim_text=claim,
        verdict="CONFIRMED",
        final_score=0.9,
        explanation="Supported by sources.",
        component_scores={"nli": 0.8, "rag": 0.95},
        evidence_excerpts=[{"text": "x", "stance": "support"}],
        citations=[{"url": "https://example.org/a", "title": "A"}],
        language="en",
        processing_metadata={"model": "qwen3:14b"},
    )


class FakePipeline:
    def __init__(self, results=None, init_failures: int = 0) -> None:
        self.results = results if results is not None else [make_output()]
        self.init_failures = init_failures
        self.init_calls = 0
        self.analyzed: list[str] = []

    def initialize(self) -> None:
        self.init_calls += 1
        if self.init_calls <= self.init_failures:
            raise ConnectionError("ollama down")

    def analyze(self, text: str):
        self.analyzed.append(text)
        if text == "boom":
            raise RuntimeError("x" * 1000)
        return self.results


@pytest.fixture
def store(tmp_path: Path) -> CheckStore:
    s = CheckStore(tmp_path / "checks.sqlite")
    yield s
    s.close()


def test_run_once_completes_with_serialized_fields(store: CheckStore) -> None:
    rec = store.create("text", 10)
    worker = CheckWorker(store, FakePipeline())
    assert worker.run_once() is True
    got = store.get(rec.id)
    assert got.status == "completed"
    results = json.loads(got.results_json)
    assert results[0]["claim_text"] == "claim"
    assert results[0]["verdict"] == "CONFIRMED"
    assert results[0]["citations"][0]["url"] == "https://example.org/a"
    assert set(results[0]) >= {
        "final_score", "explanation", "component_scores",
        "evidence_excerpts", "language", "processing_metadata",
    }


def test_run_once_without_queued_check(store: CheckStore) -> None:
    assert CheckWorker(store, FakePipeline()).run_once() is False


def test_empty_results_complete(store: CheckStore) -> None:
    rec = store.create("text", 10)
    CheckWorker(store, FakePipeline(results=[])).run_once()
    got = store.get(rec.id)
    assert got.status == "completed" and json.loads(got.results_json) == []


def test_analysis_error_fails_and_next_runs(store: CheckStore) -> None:
    bad = store.create("boom", 10)
    good = store.create("fine", 10)
    worker = CheckWorker(store, FakePipeline())
    worker.run_once()
    worker.run_once()
    failed = store.get(bad.id)
    assert failed.status == "failed"
    assert failed.error.startswith("RuntimeError: ")
    assert len(failed.error) == 500
    assert "Traceback" not in failed.error
    assert store.get(good.id).status == "completed"


def test_initialize_retries_until_ready(store: CheckStore) -> None:
    pipeline = FakePipeline(init_failures=2)
    worker = CheckWorker(store, pipeline, init_retry_seconds=0)
    assert worker.state == "initializing"
    assert worker.ensure_initialized() is True
    assert worker.state == "ready"
    assert worker.last_error is None
    assert pipeline.init_calls == 3


def test_initialize_records_last_error(store: CheckStore) -> None:
    worker = CheckWorker(store, FakePipeline(init_failures=100), init_retry_seconds=0.05)
    worker.start()
    time.sleep(0.2)
    assert worker.state == "initializing"
    assert worker.last_error == "ConnectionError: ollama down"
    worker.stop()


def test_start_stop_processes_queued_check(store: CheckStore) -> None:
    rec = store.create("text", 10)
    worker = CheckWorker(store, FakePipeline(), init_retry_seconds=0)
    worker.start()
    deadline = time.monotonic() + 5
    while store.get(rec.id).status != "completed" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert store.get(rec.id).status == "completed"
    second = store.create("more", 10)
    worker.notify()
    deadline = time.monotonic() + 5
    while store.get(second.id).status != "completed" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert store.get(second.id).status == "completed"
    worker.stop()
    assert not worker.is_alive
