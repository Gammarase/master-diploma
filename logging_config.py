"""
Logging configuration for the Disinformation Detection System.

Provides a rotating file handler with JSON-formatted lines and a
stream handler for console output.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from typing import Any

__all__ = ["setup_logging", "get_logger"]

NAMESPACE = "disinformation"


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_obj["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure root logger with rotating file + stream handlers.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        log_file: Path to the rotating log file.
        max_bytes: Maximum size in bytes before rotating (default 10 MB).
        backup_count: Number of rotated backup files to keep.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger(NAMESPACE)
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Rotating file handler — JSON format
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_JsonFormatter())

    # Stream handler — human-readable format
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'disinformation' namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A Logger instance named ``disinformation.<name>``.
    """
    if name.startswith(NAMESPACE):
        return logging.getLogger(name)
    # Strip leading package path to keep names short
    short_name = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"{NAMESPACE}.{short_name}")
