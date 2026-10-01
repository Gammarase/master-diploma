"""Unit tests for api.store.CheckStore."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from api.store import CheckStore


@pytest.fixture
def store(tmp_path: Path) -> CheckStore:
    s = CheckStore(tmp_path / "nested" / "checks.sqlite")
    yield s
    s.close()


def test_create_get_round_trip(store: CheckStore) -> None:
    rec = store.create("some text", max_queued=10)
    assert rec is not None
    assert uuid.UUID(rec.id)
    assert rec.status == "queued"
    assert rec.started_at is None and rec.finished_at is None
    got = store.get(rec.id)
    assert got == rec
    assert got.text == "some text"


def test_get_unknown_returns_none(store: CheckStore) -> None:
    assert store.get("missing") is None
    assert store.queue_position("missing") is None


def test_claim_next_in_submission_order(store: CheckStore) -> None:
    a = store.create("a", 10)
    b = store.create("b", 10)
    first = store.claim_next()
    assert first.id == a.id
    assert first.status == "running"
    assert first.started_at is not None
    assert store.claim_next().id == b.id
    assert store.claim_next() is None


def test_queue_positions(store: CheckStore) -> None:
    a = store.create("a", 10)
    b = store.create("b", 10)
    assert store.queue_position(a.id) == 1
    assert store.queue_position(b.id) == 2
    assert store.count_queued() == 2
    store.claim_next()
    assert store.queue_position(a.id) is None
    assert store.queue_position(b.id) == 1


def test_complete_and_fail(store: CheckStore) -> None:
    a = store.create("a", 10)
    b = store.create("b", 10)
    store.claim_next()
    store.complete(a.id, json.dumps([{"verdict": "CONFIRMED"}]))
    store.claim_next()
    store.fail(b.id, "ValueError: boom")
    ra, rb = store.get(a.id), store.get(b.id)
    assert ra.status == "completed" and ra.finished_at is not None
    assert json.loads(ra.results_json) == [{"verdict": "CONFIRMED"}]
    assert ra.error is None
    assert rb.status == "failed" and rb.finished_at is not None
    assert rb.error == "ValueError: boom" and rb.results_json is None


def test_create_rejects_when_queue_full(store: CheckStore) -> None:
    assert store.create("a", 2) is not None
    assert store.create("b", 2) is not None
    assert store.create("c", 2) is None
    assert store.count_queued() == 2
    store.claim_next()
    assert store.create("d", 2) is not None


def test_results_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "checks.sqlite"
    s1 = CheckStore(path)
    rec = s1.create("a", 10)
    s1.claim_next()
    s1.complete(rec.id, "[]")
    s1.close()
    s2 = CheckStore(path)
    got = s2.get(rec.id)
    assert got.status == "completed" and got.results_json == "[]"
    s2.close()


def test_requeue_running_keeps_order(store: CheckStore) -> None:
    a = store.create("a", 10)
    store.claim_next()  # a is running when the process "stops"
    b = store.create("b", 10)
    assert store.requeue_running() == 1
    ra = store.get(a.id)
    assert ra.status == "queued" and ra.started_at is None
    assert store.queue_position(a.id) == 1
    assert store.queue_position(b.id) == 2
    assert store.claim_next().id == a.id
