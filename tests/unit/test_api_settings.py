"""Unit tests for api.settings.ApiSettings."""

from __future__ import annotations

import pytest

from api.settings import ApiSettings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("API_DB_PATH", "API_MAX_QUEUED", "API_PORT"):
        monkeypatch.delenv(name, raising=False)
    s = ApiSettings()
    assert s.db_path == "var/checks.sqlite"
    assert s.max_text_chars == 20000
    assert s.max_queued == 20
    assert s.max_wait_seconds == 30
    assert s.init_retry_seconds == 15
    assert s.poll_interval_seconds == 0.5
    assert s.host == "0.0.0.0"
    assert s.port == 8000


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_MAX_QUEUED", "5")
    assert ApiSettings().max_queued == 5
