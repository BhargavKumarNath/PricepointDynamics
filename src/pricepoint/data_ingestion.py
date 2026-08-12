"""Data ingestion pipeline.

Reads raw retailer CSVs, cleans them, validates against the Pandera
schema, and writes the cleaned interim dataset to Parquet.

Loading and cleaning (``load_raw_csvs``/``clean_raw_data``) are Polars-
native: Polars' multi-threaded, Rust-implemented CSV reader and string
expressions are substantially faster than ``pd.read_csv`` + per-column
pandas string ops on the real ~791MB / 9.5M-row raw dataset
(project_refactor.md §1/§6/Phase 2). Conversion to pandas happens exactly
once, immediately before Pandera validation (Pandera only validates
pandas DataFrames) -- "convert to pandas only at the ... Pandera boundary".
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import polars as pl

from pricepoint.config import Settings
from pricepoint.manifest import has_sources_changed, write_manifest
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory
from pricepoint.schemas import RAW_DATA_SCHEMA

logger = logging.getLogger(__name__)

# The raw retailer CSVs use messy, retailer-inconsistent column headers
# (e.g. "prices_(£)", "names"). Cleaning the header text alone (lowercase,
# strip non-word characters) still leaves "names" rather than the
# "product_name" that RAW_DATA_SCHEMA and downstream modules
# (product_matching.py, etc.) actually expect -- rename explicitly rather
# than silently carrying two different names for the same field.
_COLUMN_RENAME_MAP = {"names": "product_name"}

_COLUMN_NAME_PATTERN = re.compile(r"[^\w]+")


def load_raw_csvs(settings: Settings) -> pl.DataFrame:
    """Load and concatenate all raw retailer CSV files.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    pl.DataFrame
        Concatenated raw data.
    """
    raw_dir = settings.data.raw_dir
    frames: list[pl.DataFrame] = []

    for filename in settings.data.raw_files:
        filepath = raw_dir / filename
        if not filepath.exists():
            logger.warning("Raw file not found, skipping: %s", filepath)
            continue

        logger.info("Loading %s …", filepath.name)
        # infer_schema_length=None scans the whole file for dtype
        # inference rather than just the first N rows -- the default,
        # sample-based inference can mis-detect a column's type (e.g. an
        # int column that only starts containing decimals partway through
        # a 2M+ row file) and raise a parse error partway through reading.
        frame = pl.read_csv(filepath, infer_schema_length=None)
        frames.append(frame)
        logger.info("  → %s rows loaded.", f"{len(frame):,}")

    if not frames:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}. Expected: {settings.data.raw_files}")

    # diagonal_relaxed: tolerates the retailer CSVs having slightly
    # different column sets or dtypes for the same-named column (matching
    # pd.concat's permissive default of filling missing columns with
    # nulls and upcasting to a common dtype, rather than requiring an
    # exact schema match across all 5 files).
    combined = pl.concat(frames, how="diagonal_relaxed")
    del frames
    collect_garbage()
    logger.info("Total raw records: %s", f"{len(combined):,}")
    log_memory("after load_raw_csvs")
    return combined


def clean_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """Apply cleaning transformations to raw data.

    - Normalise column names, rename to the schema's expected names
    - Coerce date column
    - Strip whitespace from string columns
    - Remove rows with null prices
    - Remove rows with a missing product name

    Parameters
    ----------
    df : pl.DataFrame
        Raw concatenated data.

    Returns
    -------
    pl.DataFrame
        Cleaned data.
    """
    logger.info("Cleaning raw data …")

    # Normalise column names: lowercase, collapse runs of non-word
    # characters to a single underscore, strip leading/trailing
    # underscores. Turns e.g. "prices_(£)" into "prices". This step
    # previously only existed in the research notebooks and was never
    # migrated into the package -- without it, RAW_DATA_SCHEMA validation
    # fails on every real run because the raw column names never match
    # what the schema expects.
    rename_map = {col: _COLUMN_NAME_PATTERN.sub("_", col.lower()).strip("_") for col in df.columns}
    df = df.rename(rename_map)
    df = df.rename({old: new for old, new in _COLUMN_RENAME_MAP.items() if old in df.columns})

    # Coerce dates (raw values are e.g. 20240413, an int; strict=False
    # nulls out anything unparseable rather than raising, matching
    # pandas' errors="coerce").
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False))

    # Strip whitespace from string columns. Polars infers each raw CSV
    # column's dtype independently per file (no pandas-style "object
    # dtype secretly holding booleans after concat" ambiguity -- e.g.
    # own_brand reads as native Boolean and is simply not selected here),
    # so this only ever touches genuinely string-valued columns.
    string_cols = [name for name, dtype in zip(df.columns, df.dtypes, strict=True) if dtype == pl.Utf8]
    if string_cols:
        df = df.with_columns([pl.col(c).str.strip_chars() for c in string_cols])

    # Coerce prices
    if "prices" in df.columns:
        df = df.with_columns(pl.col("prices").cast(pl.Float64, strict=False))
        before = len(df)
        df = df.drop_nulls(subset=["prices"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %s rows with null/invalid prices.", f"{dropped:,}")

    # Drop rows with a missing product name -- RAW_DATA_SCHEMA requires
    # product_name to be non-nullable (a nameless product can't be
    # normalised or matched downstream), and a small number of real
    # scraping rows genuinely have no name.
    if "product_name" in df.columns:
        before = len(df)
        df = df.drop_nulls(subset=["product_name"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %s rows with null product_name.", f"{dropped:,}")

    logger.info("Cleaning complete. %s rows remaining.", f"{len(df):,}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the raw data schema.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data.

    Returns
    -------
    pd.DataFrame
        Validated data (unchanged if valid).

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails.
    """
    logger.info("Validating against RAW_DATA_SCHEMA …")
    validated = RAW_DATA_SCHEMA.validate(df, lazy=True)
    logger.info("Validation passed. ✓")
    return validated


def run_ingestion(settings: Settings, force: bool = False) -> Path:
    """Execute the full ingestion pipeline.

    1. Load raw CSVs
    2. Clean
    3. Validate
    4. Save to interim Parquet

    Skips re-running ingestion if none of the raw source CSVs have
    changed since the last successful run, per this stage's
    `<output>.manifest.json` sidecar -- unless ``force`` is set.

    Parameters
    ----------
    settings : Settings
        Application settings.
    force : bool
        If True, re-run ingestion even if the raw sources are unchanged
        since the last recorded manifest.

    Returns
    -------
    Path
        Path to the output Parquet file.
    """
    source_files = [
        settings.data.raw_dir / filename
        for filename in settings.data.raw_files
        if (settings.data.raw_dir / filename).exists()
    ]
    output_path = settings.data.interim_dir / "cleaned_supermarket_data.parquet"
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    if not force and output_path.exists() and not has_sources_changed(source_files, manifest_path):
        logger.info("Raw source files unchanged since last run; skipping ingestion. Output: %s", output_path)
        return output_path

    pl_df = load_raw_csvs(settings)
    pl_df = clean_raw_data(pl_df)

    df = pl_df.to_pandas()
    del pl_df
    collect_garbage()
    # Normalise the date column's time unit: Polars' Date -> pandas
    # conversion can yield datetime64[ms] rather than pandas' own
    # [us]/[ns]-unit output from pd.to_datetime -- both are valid and
    # value-equal, but re-coercing here keeps this stage's output dtype
    # identical to what the pre-Polars-rewrite pipeline produced, and
    # matches the same defensive re-coercion feature_engineering.py
    # already does on load.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df = validate_data(df)
    downcast_dtypes(df)  # cleaning re-coerces prices/dates, which can undo earlier downcasting
    log_memory("before writing interim Parquet")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing cleaned data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Ingestion complete. Output: %s", output_path)

    write_manifest(output_path, df, source_files, stage="ingestion")

    del df
    collect_garbage()
    return output_path
