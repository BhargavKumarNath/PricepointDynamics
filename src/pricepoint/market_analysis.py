"""Market analysis: HHI, price dispersion, leadership, and SHAP precomputation.

Consolidates logic from ``precompute_market_dynamics.py`` and
``precompute_shap_values.py`` into a single, professionally
structured module.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pricepoint.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Herfindahl-Hirschman Index (HHI) — NEW feature
# ---------------------------------------------------------------------------


def calculate_hhi(df: pd.DataFrame, group_col: str = "category") -> pd.DataFrame:
    """Calculate the Herfindahl-Hirschman Index per product category.

    The HHI measures market concentration.  It is computed as the sum
    of squared market shares of each retailer within a category.

    * HHI < 1500  → competitive market
    * 1500–2500   → moderately concentrated
    * HHI > 2500  → highly concentrated

    Parameters
    ----------
    df : pd.DataFrame
        Canonical products data with ``supermarket`` and ``group_col`` columns.
    group_col : str
        Column to group by (default: ``category``).

    Returns
    -------
    pd.DataFrame
        HHI per category with columns: ``category``, ``hhi``, ``concentration``.
    """
    logger.info("Calculating HHI by '%s' …", group_col)

    if group_col not in df.columns:
        logger.warning("Column '%s' not found. Using canonical_name instead.", group_col)
        group_col = "canonical_name"

    # Count listings per retailer per category as a proxy for market share
    counts = df.groupby([group_col, "supermarket"], observed=True).size().reset_index(name="n_listings")
    totals = counts.groupby(group_col, observed=True)["n_listings"].transform("sum")
    counts["share"] = counts["n_listings"] / totals
    counts["share_sq"] = (counts["share"] * 100) ** 2  # HHI uses percentage shares

    hhi = counts.groupby(group_col, observed=True)["share_sq"].sum().reset_index().rename(columns={"share_sq": "hhi"})
    hhi["hhi"] = hhi["hhi"].round(0).astype(int)
    hhi["concentration"] = pd.cut(
        hhi["hhi"],
        bins=[0, 1500, 2500, 10001],
        labels=["Competitive", "Moderate", "Highly Concentrated"],
    )

    logger.info("HHI calculated for %s categories.", f"{len(hhi):,}")
    return hhi


# ---------------------------------------------------------------------------
# Market dispersion
# ---------------------------------------------------------------------------


def compute_market_dispersion(df: pd.DataFrame) -> pd.Series:
    """Compute daily market-wide price dispersion (coefficient of variation).

    Parameters
    ----------
    df : pd.DataFrame
        Canonical products data with ``canonical_name``, ``date``, ``prices``.

    Returns
    -------
    pd.Series
        Daily mean dispersion indexed by date.
    """
    logger.info("Computing market dispersion …")
    daily = df.groupby(["canonical_name", "date"], observed=True)["prices"].agg(["mean", "std"]).reset_index()
    daily["dispersion"] = np.where(daily["mean"] > 0, daily["std"] / daily["mean"], 0)
    series = daily.groupby("date", observed=True)["dispersion"].mean().sort_index()
    logger.info("Dispersion computed. %s data points.", f"{len(series):,}")
    return series


# ---------------------------------------------------------------------------
# Price leadership cross-correlation
# ---------------------------------------------------------------------------

# A price series that never changes should never produce a "leadership"
# signal -- but on real data, a bit-exact `variance == 0` check (what the
# pre-vectorization implementation used) is not reliable: pandas' and
# numpy's variance computation over a large-ish run (empirically, ~90+
# rows) of an identical, non-power-of-2-representable float value (e.g.
# 1.95) accumulates floating-point rounding error and returns a spuriously
# tiny *nonzero* value like 7.98e-31, not exactly 0.0 -- verified directly
# against this project's real data (Phase 2 benchmarking). A `> 0` check
# then incorrectly treats these degenerate, no-real-variation products as
# having a signal, and since every lag's correlation is then ~1.0 (an
# essentially arbitrary, ~15-way tie), which lag "wins" becomes sensitive
# to sub-1e-15 floating-point differences between computation paths (e.g.
# a single pandas.Series.corr() call vs. this module's batched numpy
# computation), producing different-but-equally-meaningless results.
# 1e-6 is chosen empirically: on the real dataset, every *genuine*
# price-variation product's variance is >= ~1.1e-6, while every
# genuinely-constant product's variance (whether exactly 0.0 or a
# floating-point residual near it) is many orders of magnitude below
# that -- so this threshold cleanly separates "real signal" from
# "floating-point noise" with wide margin on real data, restoring the
# original code's clear intent (skip products with no real price
# variation) rather than just its literal, unreliable implementation.
_MIN_VARIANCE_FOR_CORRELATION = 1e-6


def _lagged_column_correlations(x: np.ndarray, y: np.ndarray, lag: int) -> np.ndarray:
    """Vectorized Pearson correlation between each column of ``x`` and the
    same-indexed column of ``y`` shifted by ``lag`` rows, matching
    ``pandas.Series.corr()`` applied to ``x_col`` vs. ``y_col.shift(lag)``.

    Positive ``lag`` mirrors ``pandas.Series.shift(lag)``: the first
    ``lag`` rows of the shifted series are undefined, so this function
    aligns ``x``/``y`` to the overlapping valid row range instead of
    shifting-and-padding with NaN, then computes correlation from that
    range's own mean/std -- exactly what ``pandas.Series.corr()`` does
    internally after dropping the NaN-shifted edge rows.

    Parameters
    ----------
    x : np.ndarray
        Shape ``(n_dates, n_products)``, no NaNs.
    y : np.ndarray
        Shape ``(n_dates, n_products)``, no NaNs, column-aligned with ``x``.
    lag : int
        Number of rows to shift ``y`` relative to ``x`` (may be negative).

    Returns
    -------
    np.ndarray
        Shape ``(n_products,)``. NaN for any column where the aligned
        range is degenerate (too short, or zero variance in that range) --
        matching pandas' own NaN-on-zero-variance behaviour.
    """
    n = x.shape[0]
    if lag > 0:
        xs, ys = x[lag:], y[: n - lag]
    elif lag < 0:
        xs, ys = x[: n + lag], y[-lag:]
    else:
        xs, ys = x, y

    if xs.shape[0] < 2:
        return np.full(x.shape[1], np.nan)

    xc = xs - xs.mean(axis=0)
    yc = ys - ys.mean(axis=0)
    cov = (xc * yc).sum(axis=0)
    denom = np.sqrt((xc**2).sum(axis=0)) * np.sqrt((yc**2).sum(axis=0))
    return np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)


def compute_price_leadership(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Analyse cross-correlation to identify price leaders and followers.

    For each ordered pair of supermarkets, finds -- per sampled common
    product -- the lag (in days, within ``+/- max_lag_days``) that
    maximises the absolute cross-correlation between the two retailers'
    price series, then reports the median lag across all products that
    clear ``min_correlation``.

    Vectorized over products and lags via :func:`_lagged_column_correlations`
    rather than looping ``leader x follower x product x lag`` in pure
    Python: the original nested-loop implementation called
    ``pandas.Series.corr()`` once per (leader, follower, product, lag)
    combination -- with the default config (5 stores, 1000 sampled
    products, +/-7 day lag) that is 20 pairs x 1000 products x 15 lags =
    300,000 individual pandas calls, each with per-call overhead
    disproportionate to the O(n_dates) work it does. This version still
    loops over the (leader, follower) pairs (a small, fixed 5x4=20
    iterations) and over lags (2*max_lag_days+1, typically 15), but each
    iteration computes correlations for *every* sampled product at once
    via array operations -- 20 x 15 = 300 vectorized calls instead of
    300,000 scalar ones. Produces numerically identical results to the
    pre-vectorization implementation (see tests/test_market_analysis.py).

    Parameters
    ----------
    df : pd.DataFrame
        Canonical products data.
    settings : Settings
        Application settings (sample_size, max_lag, min_correlation).

    Returns
    -------
    pd.DataFrame
        Leadership pairs with columns: leader, follower, median_lag_days,
        n_products_analyzed.
    """
    cfg = settings.market_dynamics
    logger.info("Computing price leadership (sample=%s) …", cfg.sample_size)

    # Products in 3+ stores
    product_counts = df.groupby("canonical_name", observed=True)["supermarket"].nunique()
    common = product_counts[product_counts >= cfg.min_stores_for_common].index

    logger.info("Common products (≥%s stores): %s", cfg.min_stores_for_common, f"{len(common):,}")
    if len(common) == 0:
        logger.warning("No common products found.")
        return pd.DataFrame(columns=["leader", "follower", "median_lag_days", "n_products_analyzed"])

    sampled = np.random.choice(common, min(cfg.sample_size, len(common)), replace=False)

    pivot = (
        df[df["canonical_name"].isin(sampled)]
        .pivot_table(index="date", columns=["supermarket", "canonical_name"], values="prices")
        .ffill()
    )

    supermarkets = df["supermarket"].unique()
    lags_array = np.arange(-cfg.max_lag_days, cfg.max_lag_days + 1)
    results: list[dict] = []

    with warnings.catch_warnings():
        # Degenerate (too-short / zero-variance) aligned ranges legitimately
        # produce NaN via 0/0 in _lagged_column_correlations -- handled
        # explicitly there, not a sign of a real problem here.
        warnings.simplefilter("ignore", RuntimeWarning)

        for leader in supermarkets:
            for follower in supermarkets:
                if leader == follower:
                    continue

                # Only products where BOTH stores have a pivot column at all
                valid_products = [p for p in sampled if (leader, p) in pivot.columns and (follower, p) in pivot.columns]
                if not valid_products:
                    continue

                leader_mat = pivot[leader][valid_products].to_numpy()
                follower_mat = pivot[follower][valid_products].to_numpy()

                # Full-range validity: matches the original per-product
                # `s1.isnull().any() or s2.isnull().any() or s1.var()==0 or
                # s2.var()==0` skip condition, computed for all products at
                # once instead of one Series at a time -- with one
                # deliberate fix (see _MIN_VARIANCE_FOR_CORRELATION).
                no_nan = ~(np.isnan(leader_mat).any(axis=0) | np.isnan(follower_mat).any(axis=0))
                nonzero_var = (np.nanvar(leader_mat, axis=0, ddof=1) > _MIN_VARIANCE_FOR_CORRELATION) & (
                    np.nanvar(follower_mat, axis=0, ddof=1) > _MIN_VARIANCE_FOR_CORRELATION
                )
                col_mask = no_nan & nonzero_var
                if not col_mask.any():
                    continue

                lm, fm = leader_mat[:, col_mask], follower_mat[:, col_mask]

                # (n_lags, n_valid_products) correlation matrix, one
                # vectorized call per lag across all valid products.
                corr_matrix = np.stack([_lagged_column_correlations(lm, fm, int(lag)) for lag in lags_array])
                abs_corr = np.abs(corr_matrix)

                # lag=0 always has a full-length, non-degenerate aligned
                # range for every valid column (col_mask already guarantees
                # non-zero variance and no NaNs), so nanargmax always has a
                # non-NaN candidate to find here.
                best_lag_idx = np.nanargmax(abs_corr, axis=0)
                best_corr = abs_corr[best_lag_idx, np.arange(abs_corr.shape[1])]

                passes = best_corr > cfg.min_correlation
                if not passes.any():
                    continue

                selected_lags = lags_array[best_lag_idx][passes]
                results.append(
                    {
                        "leader": leader,
                        "follower": follower,
                        "median_lag_days": float(np.median(selected_lags)),
                        "n_products_analyzed": int(passes.sum()),
                    }
                )

    # Fixed columns even when `results` is empty: pd.DataFrame([]) has zero
    # columns, which would make `result_df["median_lag_days"]` raise
    # KeyError below (a latent bug in the pre-vectorization implementation,
    # only reachable when common products exist but none pass
    # min_correlation for any pair -- fixed here since this function is
    # already being rewritten this phase).
    result_df = pd.DataFrame(results, columns=["leader", "follower", "median_lag_days", "n_products_analyzed"])
    result_df = result_df[result_df["median_lag_days"] != 0].copy()
    logger.info("Leadership analysis complete. %s pairs found.", len(result_df))
    return result_df


# ---------------------------------------------------------------------------
# SHAP precomputation
# ---------------------------------------------------------------------------


def precompute_shap(settings: Settings) -> Path:
    """Pre-compute SHAP values for dashboard consumption.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Output directory containing SHAP artifacts.
    """
    import shap

    cfg = settings.shap
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = settings.model.model_path
    feature_path = settings.data.processed_dir / settings.features.output_filename

    logger.info("Loading model from %s …", model_path)
    model = joblib.load(model_path)

    logger.info("Loading feature data from %s …", feature_path)
    df = pd.read_parquet(feature_path, engine="pyarrow")

    # Sample
    sample_size = min(cfg.sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=cfg.random_seed)
    logger.info("Sampled %s rows for SHAP.", f"{sample_size:,}")

    # Encode categoricals. "str" is included alongside "object" for the same
    # pandas >= 3.0 forward-compatibility reason as training.py's identical
    # select_dtypes call -- see the comment there.
    cat_cols = df_sample.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    if cat_cols:
        df_sample = pd.get_dummies(df_sample, columns=cat_cols, drop_first=True)

    df_sample = df_sample.select_dtypes(include=["number"])

    # Align with model features
    model_features = model.feature_name_
    for col in set(model_features) - set(df_sample.columns):
        df_sample[col] = 0
    df_sample = df_sample[model_features].dropna()

    logger.info("Computing SHAP values for %s samples …", f"{len(df_sample):,}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_sample)

    # Save artifacts
    df_sample.to_parquet(output_dir / "shap_sample_data.parquet", compression="snappy")
    np.save(output_dir / "shap_values.npy", shap_values)
    with open(output_dir / "shap_base_value.txt", "w") as f:
        f.write(str(explainer.expected_value))
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(model_features))

    logger.info("SHAP precomputation complete. Output: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Orchestrator: all precomputation
# ---------------------------------------------------------------------------


def run_precompute(settings: Settings) -> None:
    """Run all precomputation pipelines (SHAP + market dynamics).

    Parameters
    ----------
    settings : Settings
        Application settings.
    """
    canonical_path = settings.data.processed_dir / settings.matching.output_filename
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical products not found at {canonical_path}.")

    df = pd.read_parquet(canonical_path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])

    # Market dispersion
    dispersion = compute_market_dispersion(df)
    md_dir = settings.market_dynamics.output_dir
    md_dir.mkdir(parents=True, exist_ok=True)
    dispersion.to_frame("dispersion").to_parquet(md_dir / "market_dispersion.parquet", compression="snappy")

    # Price leadership
    leadership = compute_price_leadership(df, settings)
    leadership.to_parquet(md_dir / "price_leadership.parquet", compression="snappy")

    # SHAP
    precompute_shap(settings)

    logger.info("All precomputation complete. ✓")


def run_hhi(settings: Settings) -> Path:
    """Calculate and save HHI market concentration index.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the HHI output file.
    """
    canonical_path = settings.data.processed_dir / settings.matching.output_filename
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical products not found at {canonical_path}.")

    df = pd.read_parquet(canonical_path, engine="pyarrow")
    hhi = calculate_hhi(df)

    output_path = settings.market_dynamics.output_dir / "hhi_index.parquet"
    settings.market_dynamics.output_dir.mkdir(parents=True, exist_ok=True)
    hhi.to_parquet(output_path, compression="snappy", index=False)

    logger.info("HHI saved to %s", output_path)
    return output_path
