"""Feature engineering pipeline.

Computes temporal, momentum, and competitive features from
canonical product price data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pricepoint.config import Settings
from pricepoint.manifest import write_manifest
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory

logger = logging.getLogger(__name__)


def add_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lag_days: list[int],
) -> pd.DataFrame:
    """Add rolling statistics and lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Price data sorted by product and date.
    rolling_windows : list[int]
        Window sizes for rolling statistics (e.g., [7, 14, 30]).
    lag_days : list[int]
        Lag periods in days (e.g., [1, 7]).

    Returns
    -------
    pd.DataFrame
        Data with new temporal columns.
    """
    logger.info("Adding temporal features (windows=%s, lags=%s) …", rolling_windows, lag_days)
    df = df.sort_values(["canonical_name", "supermarket", "date"]).copy()

    group_cols = ["canonical_name", "supermarket"]

    for window in rolling_windows:
        grp = df.groupby(group_cols, observed=True)["prices"]
        # ruff's B023 ("function uses loop variable") is a false positive
        # here: each lambda is passed straight into `.transform()` and both
        # called and discarded within this same loop iteration, never
        # stored for later -- there is no closure that outlives `window`'s
        # current value, which is what B023 actually guards against.
        df[f"price_rol_mean_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()  # noqa: B023
        )
        df[f"price_rol_std_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std()  # noqa: B023
        )
        df[f"price_rol_max_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).max()  # noqa: B023
        )
        df[f"price_rol_min_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).min()  # noqa: B023
        )

    for lag in lag_days:
        df[f"price_lag_{lag}d"] = df.groupby(group_cols, observed=True)["prices"].shift(lag)

    # Momentum: daily price change
    df["price_diff_1d"] = df.groupby(group_cols, observed=True)["prices"].diff(1)

    return df


def add_competitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-retailer competitive context features.

    All statistics are computed *leave-one-out*: a row's own price is
    excluded from the same-day, same-product aggregate used to describe
    "the market" around it. This matters because ``prices`` is the model's
    training target — a same-day aggregate that includes the row's own
    price lets the model partially recover its own target from its own
    inputs (ADR-0006 / ENGINEERING_AUDIT_REPORT.md §1.5).

    Only the *derived* comparison (``price_vs_market_avg``) is persisted,
    never the raw market average itself. This is deliberate and separate
    from the leave-one-out fix: persisting both a raw aggregate and
    ``price - that aggregate`` as simultaneous features would let the
    target be reconstructed exactly via simple addition, regardless of how
    the aggregate is computed. Exposing only the delta closes that door
    structurally rather than relying on downstream feature selection to
    catch it.

    Products sold by only one retailer on a given day have no "market" to
    compare against — their competitive features are ``NaN`` rather than a
    vacuous self-comparison (e.g. "rank 1 of 1").

    Parameters
    ----------
    df : pd.DataFrame
        Price data with canonical_name and date columns.

    Returns
    -------
    pd.DataFrame
        Data with competitive features.
    """
    logger.info("Adding competitive features (leave-one-out) …")
    df = df.copy()

    group_cols = ["canonical_name", "date"]
    grp = df.groupby(group_cols, observed=True)["prices"]

    group_sum = grp.transform("sum")
    group_count = grp.transform("count")
    other_count = group_count - 1
    has_competitors = other_count > 0

    # Leave-one-out market average: (sum of the group minus this row's own
    # price) / (count of the group minus this row). NaN when there are no
    # other retailers to compare against.
    market_avg_others = (group_sum - df["prices"]) / other_count.where(has_competitors)
    df["price_vs_market_avg"] = df["prices"] - market_avg_others

    # Dense rank of a value relative to the OTHER values in its group is
    # mathematically identical whether or not the value itself is included
    # in the comparison set — nothing is ever "less than itself" — so no
    # separate leave-one-out computation is needed here, only the
    # single-retailer edge case (no competitors to rank against).
    df["price_rank"] = grp.rank(method="dense")
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


def run_feature_engineering(settings: Settings) -> Path:
    """Execute the full feature engineering pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output feature-engineered Parquet file.
    """
    canonical_path = settings.data.processed_dir / settings.matching.output_filename
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical products not found at {canonical_path}. Run matching first.")

    logger.info("Loading canonical products from %s …", canonical_path)
    df = pd.read_parquet(canonical_path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    downcast_dtypes(df)
    log_memory("after loading canonical products")

    # Rolling-window and groupby.transform operations (below) are the most
    # memory-intensive step in the whole pipeline -- each one can create a
    # full-size intermediate copy. Downcasting once up front and logging
    # memory around each stage was added after real runs against the full
    # 9.5M-row dataset repeatedly froze a 15 GB development machine -- see
    # project_v2.md Phase 1 Progress Log, 2026-07-02.
    df = add_temporal_features(df, settings.features.rolling_windows, settings.features.lag_days)
    downcast_dtypes(df)
    log_memory("after temporal features")

    df = add_competitive_features(df)
    log_memory("after competitive features")

    df = add_cyclical_features(df)
    downcast_dtypes(df)
    log_memory("after cyclical features")

    output_dir = settings.data.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.features.output_filename

    logger.info("Writing feature data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(
        "Feature engineering complete. %s rows × %s columns. Output: %s",
        f"{len(df):,}",
        len(df.columns),
        output_path,
    )

    write_manifest(output_path, df, [canonical_path], stage="feature_engineering")

    del df
    collect_garbage()
    return output_path
