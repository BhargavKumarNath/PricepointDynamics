"""Feature engineering pipeline.

Computes lag, rolling, momentum and competitive features for one price
series per listing. Every feature on day t uses only prices from before
day t (docs/forecast_leak_fix.md).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pricepoint.config import Settings
from pricepoint.manifest import has_sources_changed, write_manifest
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory
from pricepoint.run_reports import default_report_dir, write_run_report
from pricepoint.schemas import FEATURE_DATA_SCHEMA

logger = logging.getLogger(__name__)

# One price series per listing. canonical_name merges listings (pack sizes)
# across a product family, so it is only used for cross-retailer market
# comparisons, never as a series key. See docs/forecast_leak_fix.md.
_GROUP_COLS = ["supermarket", "product_name"]

# Columns that only describe the target day itself; everything else a row
# carries must be computable from earlier days.
_TARGET_COL = "prices"


def aggregate_to_listing_day(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (supermarket, product_name, date).

    The raw scrape is already almost unique at this grain (23,830 exact
    duplicate rows in 9.5M, 2 with differing prices); duplicates are
    averaged. Descriptive columns are taken from the first row, since they
    do not vary within a listing-day.
    """
    keys = [*_GROUP_COLS, "date"]
    descriptive = [c for c in df.columns if c not in keys and c not in (_TARGET_COL, "prices_unit")]
    agg = {_TARGET_COL: "mean", **{c: "first" for c in descriptive}}
    if "prices_unit" in df.columns:
        agg["prices_unit"] = "mean"
    out = df.groupby(keys, observed=True, sort=False).agg(agg).reset_index()
    logger.info("Aggregated %s rows to %s listing-days.", f"{len(df):,}", f"{len(out):,}")
    return out


def add_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lag_days: list[int],
) -> pd.DataFrame:
    """Add lag, rolling and momentum features from *earlier* observations only.

    Every feature on the row for day t is computed from the same listing's
    prices on days before t; the row's own price (the training target) never
    enters any of them. Concretely, all statistics are taken over the
    series shifted by one observation:

    - ``price_lag_{k}d``: the price k observations back
    - ``price_rol_{stat}_{w}d``: mean/std/max/min of the previous w
      observations (row-count windows, so a gap in the dates does not
      shorten the window)
    - ``price_diff_1d``: previous price minus the one before it (the most
      recent known move, not today's)
    - ``days_since_last_obs``: calendar days since the previous observation

    Computed in Polars (native rolling expressions per group); pandas in,
    pandas out.

    Returns
    -------
    pd.DataFrame
        Data with new temporal columns, sorted by
        ``[supermarket, product_name, date]`` -- the order the shifted
        statistics are defined in.
    """
    logger.info("Adding temporal features (windows=%s, lags=%s), past-only …", rolling_windows, lag_days)
    df = df.sort_values(_GROUP_COLS + ["date"]).reset_index(drop=True)

    pl_df = pl.from_pandas(df[[*_GROUP_COLS, "date", _TARGET_COL]])
    prev = pl.col(_TARGET_COL).shift(1)

    exprs: list[pl.Expr] = []
    for window in rolling_windows:
        exprs.append(prev.rolling_mean(window, min_samples=1).over(_GROUP_COLS).alias(f"price_rol_mean_{window}d"))
        exprs.append(prev.rolling_std(window, min_samples=1).over(_GROUP_COLS).alias(f"price_rol_std_{window}d"))
        exprs.append(prev.rolling_max(window, min_samples=1).over(_GROUP_COLS).alias(f"price_rol_max_{window}d"))
        exprs.append(prev.rolling_min(window, min_samples=1).over(_GROUP_COLS).alias(f"price_rol_min_{window}d"))

    for lag in lag_days:
        exprs.append(pl.col(_TARGET_COL).shift(lag).over(_GROUP_COLS).alias(f"price_lag_{lag}d"))

    exprs.append((pl.col(_TARGET_COL).shift(1) - pl.col(_TARGET_COL).shift(2)).over(_GROUP_COLS).alias("price_diff_1d"))
    exprs.append(
        (pl.col("date") - pl.col("date").shift(1))
        .over(_GROUP_COLS)
        .dt.total_days()
        .cast(pl.Float64)
        .alias("days_since_last_obs")
    )

    new_cols = pl_df.select(exprs).to_pandas()
    return pd.concat([df, new_cols], axis=1)


def add_competitive_features(df: pd.DataFrame, price_col: str = "price_lag_1d") -> pd.DataFrame:
    """Add cross-retailer market-position features from previous prices.

    For each (canonical_name, date) group, every row is compared with the
    *other* rows of the group (leave-one-out), using ``price_col`` -- by
    default each listing's previous observed price, so nothing here
    depends on any price on the target day:

    - ``price_vs_market_avg``: own previous price minus the mean previous
      price of the other listings in the group
    - ``price_rank``: 1 + the number of other listings with a lower
      previous price (1 = cheapest)
    - ``is_cheapest_in_market``: ``price_rank == 1``

    Only the delta is persisted, never the market average itself (a raw
    average plus the delta would reconstruct ``price_col`` exactly). Rows
    with no other listing to compare against (or no previous price of
    their own) get ``NaN`` market stats and ``is_cheapest_in_market = 0``.
    """
    logger.info("Adding competitive features (leave-one-out on %s) …", price_col)
    df = df.copy()

    grp = df.groupby(["canonical_name", "date"], observed=True)[price_col]
    own = df[price_col]
    group_sum = grp.transform("sum")
    group_count = grp.transform("count")  # non-null values only
    other_count = group_count - own.notna().astype(int)
    has_competitors = own.notna() & (other_count > 0)

    market_avg_others = (group_sum - own.fillna(0)) / other_count.where(has_competitors)
    df["price_vs_market_avg"] = own - market_avg_others

    # rank(method="min") == 1 + count of strictly lower values in the group,
    # which is the same whether or not the row itself is counted.
    df["price_rank"] = grp.rank(method="min")
    df.loc[~has_competitors, "price_rank"] = np.nan

    df["is_cheapest_in_market"] = (df["price_rank"] == 1).astype(int)
    df.loc[df["price_rank"].isna(), "is_cheapest_in_market"] = 0

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical date encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Data with cyclical features.
    """
    logger.info("Adding cyclical date features …")
    df = df.copy()
    dt = df["date"].dt

    df["day_of_week_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * dt.day / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * dt.day / 31)
    df["week_of_year_sin"] = np.sin(2 * np.pi * dt.isocalendar().week.astype(int) / 52)
    df["week_of_year_cos"] = np.cos(2 * np.pi * dt.isocalendar().week.astype(int) / 52)

    return df


def run_feature_engineering(settings: Settings, force: bool = False) -> Path:
    """Execute the full feature engineering pipeline.

    Skips re-running feature engineering if the canonical products input
    hasn't changed since the last successful run, per this stage's
    `<output>.manifest.json` sidecar -- unless ``force`` is set.

    Parameters
    ----------
    settings : Settings
        Application settings.
    force : bool
        If True, re-run feature engineering even if the canonical
        products input is unchanged since the last recorded manifest.

    Returns
    -------
    Path
        Path to the output feature-engineered Parquet file.
    """
    canonical_path = settings.data.processed_dir / settings.matching.output_filename
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical products not found at {canonical_path}. Run matching first.")

    output_path = settings.data.processed_dir / settings.features.output_filename
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    if not force and output_path.exists() and not has_sources_changed([canonical_path], manifest_path):
        logger.info(
            "Canonical products unchanged since last run; skipping feature engineering. Output: %s", output_path
        )
        return output_path

    start_time = time.perf_counter()
    logger.info("Loading canonical products from %s …", canonical_path)
    df = pd.read_parquet(canonical_path, engine="pyarrow")
    rows_in = len(df)
    df["date"] = pd.to_datetime(df["date"])
    downcast_dtypes(df)
    log_memory("after loading canonical products")

    # Rolling-window and groupby.transform operations (below) are the most
    # memory-intensive step in the whole pipeline -- each one can create a
    # full-size intermediate copy. Downcasting once up front and logging
    # memory around each stage was added after real runs against the full
    # 9.5M-row dataset repeatedly froze a 15 GB development machine -- see
    # project_v2.md Phase 1 Progress Log, 2026-07-02.
    df = aggregate_to_listing_day(df)
    df = add_temporal_features(df, settings.features.rolling_windows, settings.features.lag_days)
    downcast_dtypes(df)
    log_memory("after temporal features")

    df = add_competitive_features(df)
    log_memory("after competitive features")

    df = add_cyclical_features(df)
    downcast_dtypes(df)
    log_memory("after cyclical features")

    # Validate against the feature data schema before persisting
    logger.info("Validating against FEATURE_DATA_SCHEMA …")
    df = FEATURE_DATA_SCHEMA.validate(df, lazy=False)
    logger.info("Validation passed. ✓")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing feature data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(
        "Feature engineering complete. %s rows × %s columns. Output: %s",
        f"{len(df):,}",
        len(df.columns),
        output_path,
    )

    write_manifest(output_path, df, [canonical_path], stage="feature_engineering")
    write_run_report(
        "feature_engineering",
        default_report_dir(settings),
        rows_in=rows_in,
        df_out=df,
        duration_seconds=time.perf_counter() - start_time,
    )

    del df
    collect_garbage()
    return output_path
