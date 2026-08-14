"""Data manifest generation for pipeline artifacts.

Every Gold-layer (and Silver-layer) write is accompanied by a
``<artifact>.manifest.json`` sidecar describing exactly what produced it —
row count, a schema fingerprint, the source file(s) it was derived from, and
the git commit of the code that produced it — so "what is this file, and can
I trust it" is answerable without re-running the pipeline.

This is deliberately lightweight rather than a DVC/lakeFS dependency
(ADR-0004 in project_v2.md): a single JSON sidecar file is enough
traceability for a single-contributor project with no concurrent branches
touching data. If this project ever gains a second contributor or a second
machine producing data, that decision should be revisited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> dict | None:
    """Load and parse an existing manifest file.

    Parameters
    ----------
    manifest_path : Path
        Path to the `<artifact>.manifest.json` file.

    Returns
    -------
    dict or None
        The manifest as a dictionary, or None if the file doesn't exist
        or is invalid.
    """
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load manifest from %s: %s", manifest_path, e)
        return None


def has_sources_changed(
    source_files: list[Path],
    manifest_path: Path,
) -> bool:
    """Check if any source files have changed since the last manifest was written.

    Compares row counts and modification times of source files against what's
    recorded in the manifest. Returns True if any file is missing, has a
    different row count, or has a more recent modification time.

    Parameters
    ----------
    source_files : list[Path]
        List of source files (e.g., input Parquets).
    manifest_path : Path
        Path to the existing `<artifact>.manifest.json`.

    Returns
    -------
    bool
        True if sources have changed (re-processing needed), False if unchanged.
    """
    manifest = load_manifest(manifest_path)
    if manifest is None:
        logger.debug("No existing manifest at %s; sources considered changed.", manifest_path)
        return True

    manifest_sources = {Path(p).name: p for p in manifest.get("source_files", [])}
    current_sources = {p.name: str(p) for p in source_files}

    # Check if source file list changed
    if set(manifest_sources.keys()) != set(current_sources.keys()):
        logger.debug("Source file list changed; re-processing needed.")
        return True

    # Check if any source file modification time is newer than the manifest
    manifest_time_str = manifest.get("generated_at")
    if manifest_time_str:
        try:
            manifest_time = datetime.fromisoformat(manifest_time_str)
            for source_file in source_files:
                if source_file.exists():
                    source_mtime = datetime.fromtimestamp(source_file.stat().st_mtime, tz=UTC)
                    if source_mtime > manifest_time:
                        logger.debug(
                            "Source file %s modified after manifest was written; re-processing needed.",
                            source_file.name,
                        )
                        return True
        except (ValueError, OSError):
            logger.debug("Could not compare modification times; assuming sources changed.")
            return True

    logger.debug("Sources unchanged since last manifest; re-processing can be skipped.")
    return False


def get_git_sha() -> str | None:
    """Best-effort git SHA of the current checkout.

    Returns ``None`` (rather than raising) outside a git repo or if git is
    unavailable -- artifact traceability (manifests, metrics.json) must
    never break the pipeline stage it's describing.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        logger.warning("Could not determine git SHA (not a git repo?).")
        return None


def schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of column names + dtypes.

    Lets two manifests (or a manifest and a run report, see
    ``run_reports.py``) be compared to detect schema drift between
    pipeline runs without needing a full data diff.
    """
    schema_repr = "|".join(f"{col}:{dtype}" for col, dtype in sorted(df.dtypes.astype(str).items()))
    return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()[:16]


def write_manifest(
    output_path: Path,
    df: pd.DataFrame,
    source_files: Sequence[Path | str],
    stage: str,
) -> Path:
    """Write a manifest.json sidecar describing a pipeline output artifact.

    Parameters
    ----------
    output_path : Path
        Path to the artifact the manifest describes (e.g. a ``.parquet``
        file). The manifest is written alongside it as
        ``<output_path>.manifest.json``.
    df : pd.DataFrame
        The DataFrame that was written to ``output_path``.
    source_files : Sequence[Path | str]
        The input file(s) this artifact was derived from. ``Sequence``
        (covariant) rather than ``list`` (invariant) so a ``list[Path]``
        caller-side argument type-checks without a cast -- see
        https://mypy.readthedocs.io/en/stable/common_issues.html#variance.
    stage : str
        Pipeline stage name (e.g. ``"ingestion"``, ``"feature_engineering"``).

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest = {
        "stage": stage,
        "output_file": str(output_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "schema_hash": schema_hash(df),
        "source_files": [str(p) for p in source_files],
        "git_sha": get_git_sha(),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "Wrote manifest: %s (%s rows, schema %s)",
        manifest_path,
        f"{len(df):,}",
        manifest["schema_hash"],
    )
    return manifest_path
