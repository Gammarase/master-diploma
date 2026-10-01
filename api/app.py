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
The API has no authentication; expose it only on a trusted network.
"""


def _default_pipeline_factory() -> PipelineLike:
    from pipeline import DisinformationDetectionPipeline

    return DisinformationDetectionPipeline()


ServiceUrls = tuple[str, str, str]


def _service_urls() -> ServiceUrls:
    """Active search backend name, its probe URL and the Ollama LLM probe URL.

    SearXNG is probed at ``/healthz``; Ollama web search at the root of its
    web URL. Neither probe runs a search, so no quota is used.
    """
    from configs import get_settings

    s = get_settings()
    if s.search.backend == "ollama":
        search_url = s.search.ollama_web_url.rstrip("/")
    else:
        search_url = s.search.base_url.rstrip("/") + "/healthz"
    return (
        s.search.backend,
        search_url,
        s.ollama.base_url.rstrip("/") + "/api/tags",
    )


class _ServiceHealth:
    """Reachability of the search backend and Ollama, cached for ``ttl`` seconds.

    Probes health URLs rather than running a real search, so they do not
    spend upstream rate limits or search quota.
    """

    def __init__(self, urls: Callable[[], ServiceUrls], ttl: float) -> None:
        self._urls = urls
        self._ttl = ttl
        self._cached: tuple[float, str, bool, bool] | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> tuple[str, bool, bool]:
        """Return ``(search_backend, search_up, ollama_up)``."""
        async with self._lock:
            now = time.monotonic()
            if self._cached is None or now - self._cached[0] >= self._ttl:
                backend, search_url, ollama_url = self._urls()
                async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                    search, ollama = await asyncio.gather(
                        _probe(client, search_url), _probe(client, ollama_url)
                    )
                self._cached = (now, backend, search, ollama)
            return self._cached[1], self._cached[2], self._cached[3]


async def _probe(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def create_app(
    settings: ApiSettings | None = None,
    pipeline_factory: Callable[[], PipelineLike] | None = None,
    service_urls: Callable[[], ServiceUrls] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: API settings; read from the environment when None.
        pipeline_factory: Builds the pipeline in the lifespan; defaults to
            ``DisinformationDetectionPipeline``.
        service_urls: Returns the search backend name, its health URL and the
            Ollama health URL; defaults to the pipeline settings.
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
        backend, search, ollama = await health.get()
        return HealthResponse(
            pipeline="ready" if worker and worker.state == "ready" else "initializing",
            last_error=worker.last_error if worker else None,
            search_backend=backend,
            search=search,
            searxng=search if backend == "searxng" else None,
            ollama=ollama,
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
