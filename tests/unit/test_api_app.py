"""Unit tests for the FastAPI app (fake pipeline, tmp_path database)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from api.app import create_app
from api.settings import ApiSettings
from explainability.explainer import ExplanationOutput

SEARXNG_URL = "http://searxng.test/healthz"
OLLAMA_URL = "http://ollama.test/api/tags"


def make_output(claim: str, verdict: str = "CONFIRMED") -> ExplanationOutput:
    return ExplanationOutput(
        claim_text=claim,
        verdict=verdict,
        final_score=0.8,
        explanation="Because.",
        component_scores={"nli": 0.7, "rag": 0.9},
        evidence_excerpts=[],
        citations=[{"url": "https://example.org", "title": "Example"}],
        language="en",
        processing_metadata={"model": "qwen3:14b"},
    )


class FakePipeline:
    """Blocks in analyze() while ``gate`` is clear; ``ready`` gates initialize()."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.ready = threading.Event()
        self.ready.set()
        self.started = threading.Event()

    def initialize(self) -> None:
        if not self.ready.is_set():
            raise ConnectionError("not yet")

    def analyze(self, text: str):
        self.started.set()
        self.gate.wait(10)
        if text == "boom":
            raise ValueError("pipeline exploded")
        if text == "nothing":
            return []
        return [make_output("claim one", "DISINFORMATION"), make_output("claim two")]


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def settings(tmp_path: Path) -> ApiSettings:
    return ApiSettings(
        db_path=str(tmp_path / "var" / "checks.sqlite"),
        max_queued=3,
        init_retry_seconds=0.05,
        poll_interval_seconds=0.05,
    )


@pytest.fixture
def client(settings: ApiSettings, pipeline: FakePipeline):
    app = create_app(settings, lambda: pipeline, lambda: (SEARXNG_URL, OLLAMA_URL))
    with TestClient(app) as c:
        yield c
        pipeline.gate.set()  # let a blocked analysis finish before shutdown


def submit(client: TestClient, text: str = "The Kakhovka dam was destroyed on 6 June 2023."):
    return client.post("/api/v1/checks", json={"text": text})


def wait_status(client: TestClient, check_id: str, status: str) -> dict:
    deadline = time.monotonic() + 5
    while True:
        body = client.get(f"/api/v1/checks/{check_id}").json()
        if body["status"] == status or time.monotonic() > deadline:
            return body
        time.sleep(0.02)


# ── Submission ───────────────────────────────────────────────────────────────


def test_valid_submission(client: TestClient) -> None:
    r = submit(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["id"] and body["created_at"]
    assert r.headers["location"] == f"/api/v1/checks/{body['id']}"


def test_submission_does_not_wait_for_analysis(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    t0 = time.monotonic()
    assert submit(client).status_code == 202
    assert submit(client).status_code == 202
    assert time.monotonic() - t0 < 2


@pytest.mark.parametrize("payload", [{"text": "   "}, {"text": 5}, {}, ["text"]])
def test_invalid_body_rejected(client: TestClient, settings: ApiSettings, payload) -> None:
    r = client.post("/api/v1/checks", json=payload)
    assert r.status_code == 422


def test_whitespace_text_creates_no_check(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    assert client.post("/api/v1/checks", json={"text": " \n\t "}).status_code == 422
    # Nothing was queued: three valid submissions still fit under max_queued=3
    # (the first one is taken by the worker, so four fit in total).
    for _ in range(3):
        assert submit(client).status_code == 202


def test_too_long_text_rejected_naming_limit(tmp_path: Path, pipeline: FakePipeline) -> None:
    app = create_app(
        ApiSettings(db_path=str(tmp_path / "c.sqlite")),
        lambda: pipeline,
        lambda: (SEARXNG_URL, OLLAMA_URL),
    )
    with TestClient(app) as c:
        r = c.post("/api/v1/checks", json={"text": "a" * 20001})
        assert r.status_code == 422
        assert "20000" in r.json()["detail"]
        assert c.post("/api/v1/checks", json={"text": "a" * 20000}).status_code == 202


def test_full_queue_returns_503(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    first = submit(client).json()["id"]
    assert pipeline.started.wait(5)
    assert wait_status(client, first, "running")["status"] == "running"
    for _ in range(3):
        assert submit(client).status_code == 202
    r = submit(client)
    assert r.status_code == 503
    assert r.headers["retry-after"] == "60"
    assert "full" in r.json()["detail"]


# ── Status and long polling ─────────────────────────────────────────────────


def test_unknown_id_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/checks/does-not-exist").status_code == 404


def test_negative_wait_rejected(client: TestClient) -> None:
    check_id = submit(client).json()["id"]
    assert client.get(f"/api/v1/checks/{check_id}?wait=-1").status_code == 422


def test_finished_check_returns_immediately(client: TestClient) -> None:
    check_id = submit(client).json()["id"]
    assert wait_status(client, check_id, "completed")["status"] == "completed"
    t0 = time.monotonic()
    r = client.get(f"/api/v1/checks/{check_id}?wait=30")
    assert time.monotonic() - t0 < 1
    assert r.json()["status"] == "completed"


def test_wait_expires_while_running(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    check_id = submit(client).json()["id"]
    assert pipeline.started.wait(5)
    t0 = time.monotonic()
    body = client.get(f"/api/v1/checks/{check_id}?wait=1").json()
    elapsed = time.monotonic() - t0
    assert 0.9 <= elapsed < 3
    assert body["status"] == "running"
    assert body["results"] is None


def test_wait_is_clamped_to_max(tmp_path: Path, pipeline: FakePipeline) -> None:
    settings = ApiSettings(
        db_path=str(tmp_path / "c.sqlite"), max_wait_seconds=0.5, poll_interval_seconds=0.05
    )
    app = create_app(settings, lambda: pipeline, lambda: (SEARXNG_URL, OLLAMA_URL))
    pipeline.gate.clear()
    with TestClient(app) as c:
        check_id = c.post("/api/v1/checks", json={"text": "x"}).json()["id"]
        t0 = time.monotonic()
        c.get(f"/api/v1/checks/{check_id}?wait=100")
        assert time.monotonic() - t0 < 2
        pipeline.gate.set()


def test_result_arrives_during_wait(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    check_id = submit(client).json()["id"]
    assert pipeline.started.wait(5)
    threading.Timer(0.5, pipeline.gate.set).start()
    t0 = time.monotonic()
    body = client.get(f"/api/v1/checks/{check_id}?wait=10").json()
    assert time.monotonic() - t0 < 3
    assert body["status"] == "completed"
    assert len(body["results"]) == 2


def test_queue_position_for_second_check(client: TestClient, pipeline: FakePipeline) -> None:
    pipeline.gate.clear()
    first = submit(client).json()["id"]
    assert pipeline.started.wait(5)
    wait_status(client, first, "running")
    second = submit(client).json()["id"]
    third = submit(client).json()["id"]
    body = client.get(f"/api/v1/checks/{third}").json()
    assert body["status"] == "queued"
    assert body["queue_position"] == 2
    assert body["results"] is None
    assert client.get(f"/api/v1/checks/{second}").json()["queue_position"] == 1
    assert client.get(f"/api/v1/checks/{first}").json()["queue_position"] is None


def test_completed_response_shape(client: TestClient) -> None:
    check_id = submit(client).json()["id"]
    body = wait_status(client, check_id, "completed")
    assert body["status"] == "completed"
    assert body["error"] is None
    assert body["queue_position"] is None
    assert body["started_at"] and body["finished_at"]
    assert len(body["results"]) == 2
    for result in body["results"]:
        assert {"claim_text", "verdict", "citations", "final_score", "explanation",
                "component_scores", "evidence_excerpts", "language",
                "processing_metadata"} <= set(result)
    assert body["results"][0]["verdict"] == "DISINFORMATION"


def test_no_claims_completes_with_empty_list(client: TestClient) -> None:
    check_id = submit(client, "nothing").json()["id"]
    body = wait_status(client, check_id, "completed")
    assert body["results"] == []


def test_failed_response_has_error(client: TestClient) -> None:
    bad = submit(client, "boom").json()["id"]
    good = submit(client).json()["id"]
    body = wait_status(client, bad, "failed")
    assert body["status"] == "failed"
    assert body["error"] == "ValueError: pipeline exploded"
    assert body["results"] is None
    assert wait_status(client, good, "completed")["status"] == "completed"


def test_results_survive_restart(settings: ApiSettings, pipeline: FakePipeline) -> None:
    urls = lambda: (SEARXNG_URL, OLLAMA_URL)  # noqa: E731
    with TestClient(create_app(settings, lambda: pipeline, urls)) as c:
        check_id = submit(c).json()["id"]
        first = wait_status(c, check_id, "completed")
    with TestClient(create_app(settings, lambda: pipeline, urls)) as c:
        assert c.get(f"/api/v1/checks/{check_id}").json() == first


def test_submission_accepted_while_initializing(settings: ApiSettings) -> None:
    pipeline = FakePipeline()
    pipeline.ready.clear()
    app = create_app(settings, lambda: pipeline, lambda: (SEARXNG_URL, OLLAMA_URL))
    with TestClient(app) as c:
        r = submit(c)
        assert r.status_code == 202
        check_id = r.json()["id"]
        time.sleep(0.2)
        assert c.get(f"/api/v1/checks/{check_id}").json()["status"] == "queued"
        pipeline.ready.set()
        assert wait_status(c, check_id, "completed")["status"] == "completed"


# ── Health ──────────────────────────────────────────────────────────────────


@pytest.fixture
def http_calls(monkeypatch: pytest.MonkeyPatch):
    """Route the health probes through an httpx.MockTransport."""
    state = {"up": {SEARXNG_URL: True, OLLAMA_URL: True}, "calls": []}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        state["calls"].append(url)
        if not state["up"].get(url, False):
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={})

    real_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(api_app.httpx, "AsyncClient", fake_client)
    return state


def _wait_ready(c: TestClient) -> dict:
    deadline = time.monotonic() + 5
    while True:
        body = c.get("/api/v1/health").json()
        if body["pipeline"] == "ready" or time.monotonic() > deadline:
            return body
        time.sleep(0.05)


def test_health_both_services_up(client: TestClient, http_calls) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = _wait_ready(client)
    assert body == {"pipeline": "ready", "last_error": None, "searxng": True, "ollama": True}


def test_health_one_service_down(client: TestClient, http_calls) -> None:
    http_calls["up"][OLLAMA_URL] = False
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["searxng"] is True
    assert r.json()["ollama"] is False


def test_health_initializing_before_ready(settings: ApiSettings, http_calls) -> None:
    pipeline = FakePipeline()
    pipeline.ready.clear()
    app = create_app(settings, lambda: pipeline, lambda: (SEARXNG_URL, OLLAMA_URL))
    with TestClient(app) as c:
        time.sleep(0.2)
        body = c.get("/api/v1/health").json()
        assert body["pipeline"] == "initializing"
        assert body["last_error"] == "ConnectionError: not yet"
        pipeline.ready.set()
        assert _wait_ready(c)["pipeline"] == "ready"


def test_health_probes_are_cached(client: TestClient, http_calls) -> None:
    client.get("/api/v1/health")
    assert len(http_calls["calls"]) == 2
    client.get("/api/v1/health")
    assert len(http_calls["calls"]) == 2


# ── Demo page ────────────────────────────────────────────────────────────────


def test_index_page_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<textarea" in r.text
    assert 'id="verify"' in r.text


def test_static_assets_served(client: TestClient) -> None:
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "textContent" in r.text
    assert "innerHTML" not in r.text
    assert client.get("/static/style.css").status_code == 200


# ── OpenAPI document ─────────────────────────────────────────────────────────


def test_openapi_lists_only_check_endpoints() -> None:
    spec = create_app().openapi()
    assert set(spec["paths"]) == {"/api/v1/checks", "/api/v1/checks/{check_id}"}
    post = spec["paths"]["/api/v1/checks"]["post"]
    assert {"202", "422", "503"} <= set(post["responses"])
    get = spec["paths"]["/api/v1/checks/{check_id}"]["get"]
    assert {"200", "404", "422"} <= set(get["responses"])


def test_exported_openapi_is_current() -> None:
    import yaml

    from scripts.export_openapi import OUTPUT, render

    assert yaml.safe_load(OUTPUT.read_text(encoding="utf-8")) == yaml.safe_load(render())
