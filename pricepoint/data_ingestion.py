"""Data ingestion pipeline.

Reads raw retailer CSVs, cleans them, validates against the Pandera
schema, and writes the cleaned interim dataset to Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pricepoint.config import Settings
from pricepoint.manifest import write_manifest
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


def load_raw_csvs(settings: Settings) -> pd.DataFrame:
    """Load and concatenate all raw retailer CSV files.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    pd.DataFrame
        Concatenated raw data.
    """
    raw_dir = settings.data.raw_dir
    frames: list[pd.DataFrame] = []

    for filename in settings.data.raw_files:
        filepath = raw_dir / filename
        if not filepath.exists():
            logger.warning("Raw file not found, skipping: %s", filepath)
            continue

        logger.info("Loading %s …", filepath.name)
        df = pd.read_csv(filepath, low_memory=False)
        frames.append(df)
        logger.info("  → %s rows loaded.", f"{len(df):,}")

    if not frames:
        raise FileNotFoundError(
            f"No raw CSV files found in {raw_dir}. "
            f"Expected: {settings.data.raw_files}"
        )

    combined = pd.concat(frames, ignore_index=True)
    del frames
    collect_garbage()
    logger.info("Total raw records: %s", f"{len(combined):,}")

    # Downcast as early as possible: pd.read_csv defaults to float64/object,
    # which is roughly 2x the memory this data actually needs (prices only
    # need penny precision; supermarket/category/unit have single-digit
    # cardinality). Doing this immediately after concat means every
    # downstream stage in this process works on the smaller frame, rather
    # than paying the full float64/object cost throughout. This was added
    # after real pipeline runs against the full dataset repeatedly froze a
    # 15 GB development machine -- see project_v2.md Phase 1 Progress Log.
    downcast_dtypes(combined)
    log_memory("after load_raw_csvs")
    return combined


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transformations to raw data.

    - Coerce date column
    - Strip whitespace from string columns
    - Remove rows with null prices
    - Standardise supermarket names

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated data.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    logger.info("Cleaning raw data …")
    df = df.copy()

    # Normalise column names: lowercase, collapse runs of non-word
    # characters to a single underscore, strip leading/trailing
    # underscores. Turns e.g. "prices_(£)" into "prices". This step
    # previously only existed in the research notebooks and was never
    # migrated into the package -- without it, RAW_DATA_SCHEMA validation
    # fails on every real run because the raw column names never match
    # what the schema expects.
    df.columns = (
        df.columns.str.lower().str.replace(r"[^\w]+", "_", regex=True).str.strip("_")
    )
    df = df.rename(columns=_COLUMN_RENAME_MAP)

    # Coerce dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # Strip string columns. Some "object"-dtype columns are not actually
    # strings -- e.g. `own_brand` is stored inconsistently across the 5
    # source retailers (native bool in some, blank/NaN in others), so after
    # concatenation the combined column can end up as object-dtype holding
    # real booleans rather than text. `.str.strip()` only applies to
    # genuinely string-valued columns; skip anything else rather than
    # hardcoding a column name that may change.
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            logger.debug(
                "Column %r is object-dtype but not string-valued; skipping strip().", col
            )

    # Coerce prices
    if "prices" in df.columns:
        df["prices"] = pd.to_numeric(df["prices"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["prices"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %s rows with null/invalid prices.", f"{dropped:,}")

    # Drop rows with a missing product name -- RAW_DATA_SCHEMA requires
    # product_name to be non-nullable (a nameless product can't be
    # normalised or matched downstream), and a small number of real
    # scraping rows genuinely have no name.
    if "product_name" in df.columns:
        before = len(df)
        df = df.dropna(subset=["product_name"])
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


def run_ingestion(settings: Settings) -> Path:
    """Execute the full ingestion pipeline.

    1. Load raw CSVs
    2. Clean
    3. Validate
    4. Save to interim Parquet

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output Parquet file.
    """
    df = load_raw_csvs(settings)
    df = clean_raw_data(df)
    df = validate_data(df)
    downcast_dtypes(df)  # cleaning re-coerces prices/dates, which can undo earlier downcasting
    log_memory("before writing interim Parquet")

    output_dir = settings.data.interim_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_supermarket_data.parquet"

    logger.info("Writing cleaned data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Ingestion complete. Output: %s", output_path)

    source_files = [
        settings.data.raw_dir / filename
        for filename in settings.data.raw_files
        if (settings.data.raw_dir / filename).exists()
    ]
    write_manifest(output_path, df, source_files, stage="ingestion")

    del df
    collect_garbage()
    return output_path
