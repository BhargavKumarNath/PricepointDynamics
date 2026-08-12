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


def compute_price_leadership(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Analyse cross-correlation to identify price leaders and followers.

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
    results: list[dict] = []

    for leader in supermarkets:
        for follower in supermarkets:
            if leader == follower:
                continue

            lags: list[int] = []
            for product in sampled:
                try:
                    if (leader, product) not in pivot.columns or (follower, product) not in pivot.columns:
                        continue
                    s1 = pivot[leader][product]
                    s2 = pivot[follower][product]
                    if s1.isnull().any() or s2.isnull().any() or s1.var() == 0 or s2.var() == 0:
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corrs = [s1.corr(s2.shift(lag)) for lag in range(-cfg.max_lag_days, cfg.max_lag_days + 1)]

                    if np.nanmax(np.abs(corrs)) > cfg.min_correlation:
                        lag_val = np.arange(-cfg.max_lag_days, cfg.max_lag_days + 1)[np.nanargmax(np.abs(corrs))]
                        lags.append(int(lag_val))
                except (KeyError, ValueError):
                    continue

            if lags:
                results.append(
                    {
                        "leader": leader,
                        "follower": follower,
                        "median_lag_days": float(np.median(lags)),
                        "n_products_analyzed": len(lags),
                    }
                )

    result_df = pd.DataFrame(results)
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
