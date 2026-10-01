"""
Settings for the web API (environment prefix ``API_``).

Kept apart from ``configs.Settings`` so the pipeline configuration stays
untouched.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ApiSettings"]


class ApiSettings(BaseSettings):
    """Runtime settings of the API server.

    Attributes:
        db_path: SQLite file holding the checks.
        max_text_chars: Longest accepted text, in characters.
        max_queued: Submissions are rejected once this many checks are queued.
        max_wait_seconds: Upper bound for the long-poll ``wait`` parameter.
        init_retry_seconds: Delay between failed pipeline initializations.
        poll_interval_seconds: How often a long-poll request re-reads the store.
        host: Interface the server binds to.
        port: Port the server listens on.
    """

    model_config = SettingsConfigDict(env_prefix="API_")

    db_path: str = "var/checks.sqlite"
    max_text_chars: int = 20000
    max_queued: int = 20
    max_wait_seconds: float = 30
    init_retry_seconds: float = 15
    poll_interval_seconds: float = 0.5
    host: str = "0.0.0.0"
    port: int = 8000
