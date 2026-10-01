"""Run the API server: ``python -m api``."""

from __future__ import annotations

import uvicorn

from api.settings import ApiSettings


def main() -> None:
    settings = ApiSettings()
    # One worker: each process would load its own models and job runner.
    uvicorn.run("api.app:app", host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
