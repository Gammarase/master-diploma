"""
FastAPI application: check submission, long-polled status, health and
the demo page.

Run with ``python -m api``; the module-level ``app`` is what uvicorn loads.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import BasicAuthMiddleware
from api.schemas import (
    CheckStatusResponse,
    ErrorResponse,
    HealthResponse,
    SubmitRequest,
    SubmitResponse,
)
from api.settings import ApiSettings
from api.store import TERMINAL_STATUSES, CheckRecord, CheckStore
from api.worker import CheckWorker, PipelineLike
from logging_config import get_logger

__all__ = ["create_app", "app"]

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_RETRY_AFTER_SECONDS = "60"
_PROBE_TIMEOUT_SECONDS = 3.0
_HEALTH_TTL_SECONDS = 30.0

API_VERSION = "1.0.0"
API_DESCRIPTION = """Checks news texts for disinformation. The service extracts the check-worthy
claims from a text, searches trusted web sources for evidence, and gives each
claim a verdict (`CONFIRMED`, `DISINFORMATION` or `UNCERTAIN`) with an
explanation and the cited sources.

Analysis takes minutes, so the API is asynchronous:

1. `POST /api/v1/checks` submits a text and returns a check `id` at once.
2. `GET /api/v1/checks/{id}?wait=25` returns the status. The request is held
   until the check finishes or `wait` seconds pass (long polling), so repeat
   it until `status` is `completed` or `failed`.

Checks run one at a time in submission order and are kept across restarts.
When `API_BASIC_AUTH_USERNAME` and `API_BASIC_AUTH_PASSWORD` are set, every
endpoint requires HTTP Basic authentication.
"""


def _default_pipeline_factory() -> PipelineLike:
    from pipeline import DisinformationDetectionPipeline

    return DisinformationDetectionPipeline()


def _service_urls() -> tuple[str, str]:
    """Health probe URLs of SearXNG and Ollama, from the pipeline settings."""
    from configs import get_settings

    s = get_settings()
    return (
        s.search.base_url.rstrip("/") + "/healthz",
        s.ollama.base_url.rstrip("/") + "/api/tags",
    )


class _ServiceHealth:
    """Reachability of SearXNG and Ollama, cached for ``ttl`` seconds.

    Uses SearXNG's own ``/healthz`` rather than a real search, so probes do
    not spend upstream engine rate limits.
    """

    def __init__(self, urls: Callable[[], tuple[str, str]], ttl: float) -> None:
        self._urls = urls
        self._ttl = ttl
        self._cached: tuple[float, bool, bool] | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> tuple[bool, bool]:
        async with self._lock:
            now = time.monotonic()
            if self._cached is None or now - self._cached[0] >= self._ttl:
                searxng_url, ollama_url = self._urls()
                async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                    searxng, ollama = await asyncio.gather(
                        _probe(client, searxng_url), _probe(client, ollama_url)
                    )
                self._cached = (now, searxng, ollama)
            return self._cached[1], self._cached[2]


async def _probe(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def create_app(
    settings: ApiSettings | None = None,
    pipeline_factory: Callable[[], PipelineLike] | None = None,
    service_urls: Callable[[], tuple[str, str]] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: API settings; read from the environment when None.
        pipeline_factory: Builds the pipeline in the lifespan; defaults to
            ``DisinformationDetectionPipeline``.
        service_urls: Returns the SearXNG and Ollama health URLs; defaults
            to the pipeline settings.
    """
    cfg = settings or ApiSettings()
    factory = pipeline_factory or _default_pipeline_factory
    health = _ServiceHealth(service_urls or _service_urls, _HEALTH_TTL_SECONDS)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = CheckStore(cfg.db_path)
        requeued = store.requeue_running()
        if requeued:
            logger.info("Re-queued %d interrupted check(s)", requeued)
        worker = CheckWorker(store, factory(), cfg.init_retry_seconds)
        app.state.store = store
        app.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            worker.stop()
            # A check still in analysis keeps the thread (and the store) busy;
            # the process exits and requeue_running() resumes it next start.
            if not worker.is_alive:
                store.close()

    app = FastAPI(
        title="Disinformation Detection API",
        version=API_VERSION,
        description=API_DESCRIPTION,
        openapi_tags=[{"name": "checks", "description": "Submit texts and fetch results."}],
        lifespan=lifespan,
    )

    def to_status(record: CheckRecord, store: CheckStore) -> CheckStatusResponse:
        results: list[dict[str, Any]] | None = None
        if record.status == "completed" and record.results_json is not None:
            results = json.loads(record.results_json)
        return CheckStatusResponse(
            id=record.id,
            status=record.status,
            text=record.text,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            queue_position=(
                store.queue_position(record.id) if record.status == "queued" else None
            ),
            results=results,
            error=record.error if record.status == "failed" else None,
        )

    @app.post(
        "/api/v1/checks",
        status_code=202,
        response_model=SubmitResponse,
        tags=["checks"],
        operation_id="submitCheck",
        summary="Submit a text for checking",
        description=(
            "Queues the text and returns at once, without waiting for the analysis. "
            "The `Location` header holds the URL of the new check."
        ),
        responses={
            202: {
                "description": "Check queued.",
                "headers": {
                    "Location": {
                        "description": "Status URL of the check, `/api/v1/checks/{id}`.",
                        "schema": {"type": "string"},
                    }
                },
            },
            422: {
                "model": ErrorResponse,
                "description": "Text is missing, not a string, empty, or longer than the limit.",
            },
            503: {
                "model": ErrorResponse,
                "description": "The queue is full. Retry after the `Retry-After` seconds.",
                "headers": {
                    "Retry-After": {
                        "description": "Seconds to wait before retrying.",
                        "schema": {"type": "integer"},
                    }
                },
            },
        },
    )
    async def submit_check(
        body: SubmitRequest, request: Request, response: Response
    ) -> SubmitResponse:
        text = body.text.strip()
        if not text:
            raise HTTPException(422, detail="Text must not be empty.")
        if len(text) > cfg.max_text_chars:
            raise HTTPException(
                422,
                detail=f"Text is too long: the limit is {cfg.max_text_chars} characters.",
            )
        store: CheckStore = request.app.state.store
        record = await run_in_threadpool(store.create, text, cfg.max_queued)
        if record is None:
            raise HTTPException(
                503,
                detail="The check queue is full. Try again later.",
                headers={"Retry-After": _RETRY_AFTER_SECONDS},
            )
        request.app.state.worker.notify()
        response.headers["Location"] = f"/api/v1/checks/{record.id}"
        return SubmitResponse(
            id=record.id, status=record.status, created_at=record.created_at
        )

    @app.get(
        "/api/v1/checks/{check_id}",
        response_model=CheckStatusResponse,
        tags=["checks"],
        operation_id="getCheck",
        summary="Get the status and results of a check",
        description=(
            "Returns the check at once when it is finished or `wait` is 0. "
            "Otherwise holds the request until the check finishes or `wait` "
            "seconds pass, then returns the current state (long polling)."
        ),
        responses={
            404: {"model": ErrorResponse, "description": "No check with this id."},
            422: {"model": ErrorResponse, "description": "`wait` is negative or not a number."},
        },
    )
    async def get_check(
        request: Request,
        check_id: str = PathParam(description="Check id returned by `POST /api/v1/checks`."),
        wait: float = Query(
            0,
            ge=0,
            description=(
                "Seconds to wait for the check to finish. Values above "
                "`API_MAX_WAIT_SECONDS` (default 30) are lowered to it."
            ),
        ),
    ) -> CheckStatusResponse:
        store: CheckStore = request.app.state.store
        record = await run_in_threadpool(store.get, check_id)
        if record is None:
            raise HTTPException(404, detail="Check not found.")
        deadline = time.monotonic() + min(wait, cfg.max_wait_seconds)
        while record.status not in TERMINAL_STATUSES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(cfg.poll_interval_seconds, remaining))
            record = await run_in_threadpool(store.get, check_id)
        return await run_in_threadpool(to_status, record, store)

    # Operational endpoint for container health checks; not part of the public docs.
    @app.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
    async def get_health(request: Request) -> HealthResponse:
        worker: CheckWorker | None = getattr(request.app.state, "worker", None)
        searxng, ollama = await health.get()
        return HealthResponse(
            pipeline="ready" if worker and worker.state == "ready" else "initializing",
            last_error=worker.last_error if worker else None,
            searxng=searxng,
            ollama=ollama,
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    username = cfg.basic_auth_username
    password = cfg.basic_auth_password.get_secret_value()
    if bool(username) != bool(password):
        raise ValueError(
            "Set both API_BASIC_AUTH_USERNAME and API_BASIC_AUTH_PASSWORD, or neither."
        )
    if username:
        app.add_middleware(BasicAuthMiddleware, username=username, password=password)
        logger.info("Basic auth enabled")
    return app


app = create_app()
