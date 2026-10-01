"""Unit tests for the HTTP Basic auth middleware of the API."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import ApiSettings
from tests.unit.test_api_app import (  # noqa: F401  (http_calls is a fixture)
    OLLAMA_URL,
    SEARXNG_URL,
    FakePipeline,
    http_calls,
)

USER, PASSWORD = "admin", "s3cret:pass"


def basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def make_app(tmp_path: Path, user: str = USER, password: str = PASSWORD):
    settings = ApiSettings(
        db_path=str(tmp_path / "c.sqlite"),
        basic_auth_username=user,
        basic_auth_password=password,
    )
    return create_app(settings, FakePipeline, lambda: (SEARXNG_URL, OLLAMA_URL))


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(make_app(tmp_path)) as c:
        yield c


@pytest.mark.parametrize(
    "method, path",
    [("GET", "/"), ("GET", "/static/app.js"), ("GET", "/api/v1/checks/x"),
     ("GET", "/openapi.json"), ("GET", "/docs"), ("POST", "/api/v1/checks")],
)
def test_missing_credentials_rejected(client: TestClient, method: str, path: str) -> None:
    r = client.request(method, path, json={"text": "hi"} if method == "POST" else None)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Basic realm=")
    assert r.json() == {"detail": "Authentication required."}


@pytest.mark.parametrize(
    "headers",
    [
        basic(USER, "wrong"),
        basic("other", PASSWORD),
        basic(USER, ""),
        {"Authorization": "Basic !!!"},
        {"Authorization": "Basic " + base64.b64encode(b"\xff\xfe").decode()},
        {"Authorization": "Basic " + base64.b64encode(b"no-colon").decode()},
        {"Authorization": "Bearer token"},
    ],
)
def test_bad_credentials_rejected(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/api/v1/checks/x", headers=headers).status_code == 401


def test_valid_credentials_accepted(client: TestClient) -> None:
    auth = basic(USER, PASSWORD)
    assert client.get("/", headers=auth).status_code == 200
    r = client.post("/api/v1/checks", json={"text": "Some news."}, headers=auth)
    assert r.status_code == 202
    assert client.get(f"/api/v1/checks/{r.json()['id']}", headers=auth).status_code == 200


def test_health_needs_no_credentials(client: TestClient, http_calls) -> None:  # noqa: F811
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("user, password", [(USER, ""), ("", PASSWORD)])
def test_half_configured_auth_refused(tmp_path: Path, user: str, password: str) -> None:
    with pytest.raises(ValueError, match="or neither"):
        make_app(tmp_path, user, password)


def test_auth_off_when_not_configured(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, "", "")) as c:
        assert c.get("/api/v1/checks/x").status_code == 404
