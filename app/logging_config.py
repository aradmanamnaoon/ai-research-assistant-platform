"""Logging configuration for the application and libraries."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Logging level name (e.g. ``"INFO"``).
    """
    root = logging.getLogger()
    if getattr(root, "_ara_configured", False):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._ara_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(f"ara.{name}")