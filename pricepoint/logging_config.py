"""Structured logging setup for the PricePoint pipeline."""

from __future__ import annotations

import logging
import sys

from pricepoint.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure the root logger based on application settings.

    Parameters
    ----------
    settings : Settings
        Application settings containing logging configuration.
    """
    # Windows terminals commonly default stdout/stderr to a legacy codepage
    # (e.g. cp1252) that cannot encode the unicode characters (✓, →, …) used
    # in CLI/log messages throughout this project, crashing the process
    # after the underlying work has already succeeded. Force UTF-8 so the
    # pipeline actually runs to completion on Windows, not just Linux CI.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001 -- best-effort, never fatal
                pass

    log_cfg = settings.logging
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_cfg.file:
        handlers.append(logging.FileHandler(log_cfg.file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=log_cfg.format,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
