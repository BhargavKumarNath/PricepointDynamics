"""Export precomputed, page-shaped artifacts for the Next.js frontend
(project_refactor.md §25.3).

5 of 6 dashboard pages need nothing but a static file fetch (§25.1) --
this module produces exactly those files: small pre-aggregated JSON for
Home/Market Overview/Basket Analysis/Market Dynamics (destined for
`frontend/public/data/`, small enough to commit to git), and two larger
Parquet artifacts for the Predictor's product selector and the SHAP
explorer (destined for Cloudflare R2 in production; written to
`frontend/public/data-r2/` locally, see `WebArtifactsConfig`'s docstring).

Every number here traces to a real mart/model artifact -- never a
fabricated or hardcoded value (closes defect #7, the hardcoded dashboard
metric strings this project started with).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from pricepoint.config import Settings
from pricepoint.manifest import load_manifest
from pricepoint.warehouse import Warehouse

logger = logging.getLogger(__name__)


def _load_baskets(settings: Settings) -> dict[str, list[str]]:
    """`configs/baskets.yaml` -- the same basket definitions `GET
    /v1/baskets` would have served under §8.1's original (now-superseded)
    API surface; under §25.2 the frontend reads them precomputed instead."""
    # configs/ sits at the project root, two levels up from data/00_raw.
    baskets_path = settings.data.raw_dir.parents[1] / "configs" / "baskets.yaml"
    if not baskets_path.exists():
        raise FileNotFoundError(f"Basket definitions not found at {baskets_path}.")
    with open(baskets_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def export_home_metrics(settings: Settings) -> dict[str, Any]:
    """Home page's 3 headline metrics + Model Insights' performance tiles
    -- one artifact backs both pages, since they show the same numbers
    (closes defect #7: replaces `"9.5 Million"`, `"67,000+"`, `"£0.14"`,
    `"£0.37"` literal strings with real, traceable values).
    """
    metrics_path = settings.model.output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No trained model metrics at {metrics_path}. Run `python run.py train` first.")
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)

    ingestion_manifest_path = settings.data.interim_dir / "cleaned_supermarket_data.parquet.manifest.json"
    ingestion_manifest = load_manifest(ingestion_manifest_path)
    if ingestion_manifest is None:
        raise FileNotFoundError(
            f"No ingestion manifest at {ingestion_manifest_path}. Run `python run.py ingest` first."
        )
    total_records = ingestion_manifest["row_count"]

    dim_product_path = settings.marts.output_dir / "dim_product.parquet"
    if not dim_product_path.exists():
        raise FileNotFoundError(f"dim_product mart not found at {dim_product_path}. Run `python run.py marts` first.")
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(dim_product_path)]).fetchone()
    finally:
        conn.close()
    if row is None:
        raise FileNotFoundError(f"dim_product mart at {dim_product_path} contains no rows.")
    canonical_products = row[0]

    return {
        "total_records": total_records,
        "canonical_products": canonical_products,
        "mae": metrics["MAE"],
        "rmse": metrics["RMSE"],
        "r2": metrics["R2"],
        "n_train_rows": metrics["n_train_rows"],
        "n_test_rows": metrics["n_test_rows"],
        "n_features": metrics["n_features"],
        "trained_at": metrics["trained_at"],
    }


def export_market_overview(warehouse: Warehouse) -> dict[str, Any]:
    """Per-retailer price distribution (5-number summary, not raw
    per-row outliers -- §25.3 explicitly rejects shipping row-level data
    to the browser) + portfolio/own-brand mix."""
    df = warehouse.get_market_overview()
    return {"supermarkets": df.to_dict(orient="records")}


def export_basket_analysis(warehouse: Warehouse, baskets: dict[str, list[str]]) -> dict[str, Any]:
    """Per-basket cost/coverage per retailer, plus the item-level price
    grid backing the "detailed breakdown" expander -- for all 13 baskets
    at once (each basket is small; all 13 together are still comfortably
    under §25.3's <10KB-ish JSON budget)."""
    result: dict[str, Any] = {}
    for name, items in baskets.items():
        cost_df = warehouse.get_basket_cost(items)
        cost_df["coverage_pct"] = (cost_df["items_found"] / len(items)) * 100
        item_prices_df = warehouse.get_basket_item_prices(items)

        result[name] = {
            "total_items": len(items),
            "rows": cost_df.to_dict(orient="records"),
            "items": item_prices_df.to_dict(orient="records"),
        }
    return {"baskets": result}


def export_market_dynamics(settings: Settings) -> dict[str, Any]:
    """Full dispersion time series (~90 points, negligible size) + price
    leadership pairs -- both already small, precomputed by
    `market_analysis.py::run_precompute`, just reshaped for JSON."""
    dynamics_dir = settings.market_dynamics.output_dir
    dispersion_path = dynamics_dir / "market_dispersion.parquet"
    leadership_path = dynamics_dir / "price_leadership.parquet"
    if not dispersion_path.exists() or not leadership_path.exists():
        raise FileNotFoundError(
            f"Market dynamics artifacts not found in {dynamics_dir}. Run `python run.py precompute` first."
        )

    dispersion_df = pd.read_parquet(dispersion_path, engine="pyarrow")
    dispersion_reset = dispersion_df.reset_index().rename(columns={dispersion_df.index.name or "index": "date"})
    dispersion_reset["date"] = dispersion_reset["date"].astype(str)

    leadership_df = pd.read_parquet(leadership_path, engine="pyarrow")

    dispersion_series = dispersion_df["dispersion"]
    latest_dispersion = float(dispersion_series.iloc[-1])
    avg_dispersion = float(dispersion_series.mean())

    top_leader = None
    fastest_follower = None
    if not leadership_df.empty:
        top_leader = leadership_df["leader"].mode().iloc[0]
        fastest_row = leadership_df.loc[leadership_df["median_lag_days"].abs().idxmin()]
        fastest_follower = {
            "follower": fastest_row["follower"],
            "leader": fastest_row["leader"],
            "median_lag_days": float(fastest_row["median_lag_days"]),
        }

    return {
        "dispersion": dispersion_reset[["date", "dispersion"]].to_dict(orient="records"),
        "leadership": leadership_df.to_dict(orient="records"),
        "latest_dispersion": latest_dispersion,
        "avg_dispersion": avg_dispersion,
        "top_leader": top_leader,
        "fastest_follower": fastest_follower,
    }


def export_predictor_context(settings: Settings) -> pd.DataFrame:
    """Latest feature snapshot per (canonical_name, supermarket) -- backs
    the Predictor page's product/store selector and history-context view,
    **not** the prediction itself (that's always the live `POST
    /v1/predict` call). Reads `feature_engineered_data.parquet` directly
    (not a mart -- marts deliberately exclude ML columns), via a
    window-function "latest row per group" query so the full 9.5M-row
    file is never pulled into pandas."""
    feature_path = settings.data.processed_dir / settings.features.output_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found at {feature_path}. Run `python run.py features` first.")

    conn = duckdb.connect(":memory:")
    try:
        df = conn.execute(
            """
            SELECT canonical_name, supermarket, category, date, prices, price_lag_1d
            FROM (
                SELECT
                    canonical_name, supermarket, category, date, prices, price_lag_1d,
                    ROW_NUMBER() OVER (
                        PARTITION BY canonical_name, supermarket ORDER BY date DESC
                    ) AS rn
                FROM read_parquet(?)
            )
            WHERE rn = 1
            ORDER BY canonical_name, supermarket
            """,
            [str(feature_path)],
        ).fetchdf()
    finally:
        conn.close()
    return df


def export_shap_explorer(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The precomputed SHAP sample (features + per-feature SHAP
    contributions) as two same-shaped numeric grids, plus small metadata
    -- backs Model Insights' global importance charts and per-sample
    local explanation, without a live SHAP computation per request (v1
    scope, project_refactor.md §8.1)."""
    shap_dir = settings.shap.output_dir
    sample_path = shap_dir / "shap_sample_data.parquet"
    values_path = shap_dir / "shap_values.npy"
    base_value_path = shap_dir / "shap_base_value.txt"
    for path in (sample_path, values_path, base_value_path):
        if not path.exists():
            raise FileNotFoundError(f"SHAP artifact not found at {path}. Run `python run.py precompute` first.")

    features_df = pd.read_parquet(sample_path, engine="pyarrow")
    shap_values = np.load(values_path)
    base_value = float(open(base_value_path, encoding="utf-8").read().strip())
    values_df = pd.DataFrame(shap_values, columns=features_df.columns, index=features_df.index)

    meta = {
        "base_value": base_value,
        "feature_names": features_df.columns.tolist(),
        "n_samples": len(features_df),
    }
    return features_df, values_df, meta


def run_export_web_artifacts(settings: Settings) -> dict[str, Path]:
    """Run every exporter and write its output to `settings.web_artifacts`'
    directories. Returns a mapping of artifact name to written path.

    Raises
    ------
    FileNotFoundError
        If any upstream artifact (model, marts, precomputed SHAP/market
        dynamics) hasn't been built yet -- every exporter fails loudly
        and specifically, naming exactly which pipeline stage to run.
    """
    json_dir = settings.web_artifacts.json_dir
    parquet_dir = settings.web_artifacts.parquet_dir
    json_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    def _write_json(name: str, payload: Any) -> None:
        path = json_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        written[name] = path
        logger.info("Wrote %s (%.1f KB)", path, path.stat().st_size / 1024)

    logger.info("Exporting home_metrics.json …")
    _write_json("home_metrics", export_home_metrics(settings))

    baskets = _load_baskets(settings)
    with Warehouse(settings.marts.output_dir) as warehouse:
        logger.info("Exporting market_overview.json …")
        _write_json("market_overview", export_market_overview(warehouse))

        logger.info("Exporting basket_analysis.json …")
        _write_json("basket_analysis", export_basket_analysis(warehouse, baskets))

    logger.info("Exporting market_dynamics.json …")
    _write_json("market_dynamics", export_market_dynamics(settings))

    logger.info("Exporting predictor_context.parquet …")
    predictor_context_path = parquet_dir / "predictor_context.parquet"
    export_predictor_context(settings).to_parquet(predictor_context_path, compression="snappy", index=False)
    written["predictor_context"] = predictor_context_path
    logger.info("Wrote %s (%.1f MB)", predictor_context_path, predictor_context_path.stat().st_size / 1024**2)

    logger.info("Exporting shap_explorer parquet + metadata …")
    features_df, values_df, meta = export_shap_explorer(settings)
    features_path = parquet_dir / "shap_explorer_features.parquet"
    values_path = parquet_dir / "shap_explorer_values.parquet"
    features_df.to_parquet(features_path, compression="snappy", index=False)
    values_df.to_parquet(values_path, compression="snappy", index=False)
    written["shap_explorer_features"] = features_path
    written["shap_explorer_values"] = values_path
    _write_json("shap_explorer_meta", meta)
    logger.info(
        "Wrote SHAP explorer parquet (%.1f MB combined)",
        (features_path.stat().st_size + values_path.stat().st_size) / 1024**2,
    )

    return written
