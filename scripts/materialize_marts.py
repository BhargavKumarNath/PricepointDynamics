#!/usr/bin/env python3
"""Standalone CLI for materializing the analytics marts (data/03_marts/).

Thin wrapper around ``pricepoint.marts.run_materialize_marts`` -- the same
function ``python run.py marts`` calls -- so the marts can also be
rebuilt without going through the full CLI (e.g. in a cron job or a CI
step that only needs this one stage).

Usage:
    uv run python scripts/materialize_marts.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pricepoint.config import load_settings  # noqa: E402
from pricepoint.logging_config import setup_logging  # noqa: E402
from pricepoint.marts import run_materialize_marts  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the analytics marts from 02_processed/ via DuckDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every mart even if the source data is unchanged since the last run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    settings = load_settings()
    setup_logging(settings)

    paths = run_materialize_marts(settings, force=args.force)

    logger.info("All marts materialized:")
    for name, path in paths.items():
        logger.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
