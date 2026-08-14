"""Per-stage JSON run reports (project_refactor.md §14).

Distinct from ``manifest.py``'s ``<artifact>.manifest.json`` sidecars: a
manifest describes the *current* output artifact and is overwritten every
run (it exists to answer "what produced this file" and to drive
skip-if-unchanged checks). A run report is a timestamped, append-only
historical record of one execution of a stage -- rows in/out, null
counts, duration, schema hash -- the actual data-quality artifact §14
calls for. Deliberately a lightweight JSON file per run, not a
dashboard-shaped tool (Great Expectations Data Docs, Monte Carlo, Soda):
disproportionate for a solo-maintained pipeline at this scale.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pricepoint.config import Settings
from pricepoint.manifest import get_git_sha, schema_hash

logger = logging.getLogger(__name__)

# A stage unexpectedly losing more than this fraction of its input rows is
# surfaced as a warning rather than passed through silently (§14: "flag --
# log a warning, don't silently continue -- if a stage drops more than an
# expected threshold of rows"). Deliberately generous: ingestion routinely
# drops a small, known fraction of genuinely bad rows (missing name, null
# price) as part of its normal, correct behaviour -- this threshold is a
# backstop against a *stage regressing*, not a tight per-stage budget.
_UNEXPECTED_ROW_LOSS_WARN_THRESHOLD = 0.20


def default_report_dir(settings: Settings) -> Path:
    """The standard run-report directory for this project: ``data/_run_reports/``."""
    return settings.data.raw_dir.parent / "_run_reports"


def write_run_report(
    stage: str,
    report_dir: Path,
    *,
    rows_in: int,
    df_out: pd.DataFrame,
    duration_seconds: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a timestamped JSON run report for one pipeline stage execution.

    Parameters
    ----------
    stage : str
        Pipeline stage name (e.g. ``"ingestion"``), matching the
        ``stage`` values already used in ``manifest.py::write_manifest``.
    report_dir : Path
        Directory the report is written into, as
        ``<report_dir>/<stage>_<timestamp>.json``. Use
        :func:`default_report_dir` for the project's standard location
        unless a caller has a specific reason to deviate (e.g. tests).
    rows_in : int
        Row count of the stage's input, before this run's processing.
    df_out : pd.DataFrame
        The stage's output data, used to compute row count, per-column
        null counts, and a schema fingerprint.
    duration_seconds : float
        Wall-clock duration of the stage's processing (excluding any
        skip-if-unchanged short-circuit -- a skipped run isn't a
        "processing" event worth reporting on).
    extra : dict, optional
        Additional stage-specific fields (e.g. training's row/feature
        counts already captured in ``metrics.json``) to merge in verbatim.

    Returns
    -------
    Path
        Path to the written report file.
    """
    rows_out = len(df_out)
    null_counts = {col: int(n) for col, n in df_out.isna().sum().items() if n > 0}

    report: dict[str, Any] = {
        "stage": stage,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "row_delta": rows_out - rows_in,
        "null_counts": null_counts,
        "duration_seconds": round(duration_seconds, 3),
        "schema_hash": schema_hash(df_out),
        "git_sha": get_git_sha(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        report.update(extra)

    report_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond-resolution timestamp in the filename: pipeline stages can
    # legitimately run twice in the same second (e.g. this project's own
    # test suite, or `--force` re-runs during debugging), and reports are
    # additive/historical, not overwritten like a manifest -- a filename
    # collision would silently drop one run's report.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = report_dir / f"{stage}_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info(
        "Wrote run report: %s (rows_in=%s, rows_out=%s, duration=%.2fs)",
        report_path,
        f"{rows_in:,}",
        f"{rows_out:,}",
        duration_seconds,
    )

    if rows_in > 0:
        loss_fraction = max(0.0, (rows_in - rows_out) / rows_in)
        if loss_fraction > _UNEXPECTED_ROW_LOSS_WARN_THRESHOLD:
            logger.warning(
                "Stage '%s' dropped %.1f%% of its input rows (%s -> %s) -- "
                "exceeds the %.0f%% sanity threshold; investigate before trusting this output.",
                stage,
                loss_fraction * 100,
                f"{rows_in:,}",
                f"{rows_out:,}",
                _UNEXPECTED_ROW_LOSS_WARN_THRESHOLD * 100,
            )

    return report_path
