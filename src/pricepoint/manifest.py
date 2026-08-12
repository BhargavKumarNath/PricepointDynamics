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


def _get_git_sha() -> str | None:
    """Best-effort git SHA of the current checkout.

    Returns ``None`` (rather than raising) outside a git repo or if git is
    unavailable -- manifest generation must never break the pipeline it's
    describing.
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
        logger.warning("Could not determine git SHA for manifest (not a git repo?).")
        return None


def _schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of column names + dtypes.

    Lets two manifests be compared to detect schema drift between pipeline
    runs without needing a full data diff.
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
        "schema_hash": _schema_hash(df),
        "source_files": [str(p) for p in source_files],
        "git_sha": _get_git_sha(),
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
