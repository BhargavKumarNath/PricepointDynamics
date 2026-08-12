"""FastAPI dependency providers: settings, model, static lookups.

Everything here is cheap to call per-request because the expensive parts
(loading the model, computing the dataset's date bounds) are cached at
process level (project_refactor.md §8.3: "a cached model singleton,
loaded once at startup, not per-request").

Only `GET /health` and `POST /v1/predict` are live endpoints
(project_refactor.md §25.2 -- the locked, narrowed API surface); nothing
here touches `Warehouse`/marts or `configs/baskets.yaml`, since the
endpoints that needed those (products search/history, market
overview/basket/dynamics/hhi, model card/explain) are superseded by
precomputed static/R2 artifacts for the dashboard's current scope, not
live queries.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

import duckdb
import joblib
from lightgbm import LGBMRegressor

from pricepoint.config import Settings, load_settings
from pricepoint.exceptions import ArtifactNotFoundError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Application settings, loaded once per process."""
    return load_settings()


@lru_cache(maxsize=1)
def get_model() -> LGBMRegressor:
    """The trained price-prediction model, loaded once per process.

    Raises
    ------
    ArtifactNotFoundError
        If no trained model exists at ``settings.model.model_path`` --
        `POST /v1/predict` and `GET /v1/model/*` cannot function without
        one, and this must fail loudly (404, via the router) rather than
        the process crashing on first request.
    """
    settings = get_settings()
    model_path = settings.model.model_path
    if not model_path.exists():
        raise ArtifactNotFoundError(
            f"No trained model found at {model_path}. Run `python run.py train` first.",
            context={"model_path": str(model_path)},
        )
    logger.info("Loading model from %s …", model_path)
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_feature_data_path() -> Path:
    """Path to `feature_engineered_data.parquet` -- the full-feature source
    `/v1/predict` resolves real historical feature vectors from (not a
    mart: marts deliberately exclude the ~40 ML-training columns, see
    project_refactor.md §5)."""
    settings = get_settings()
    path = settings.data.processed_dir / settings.features.output_filename
    if not path.exists():
        raise ArtifactNotFoundError(
            f"Feature data not found at {path}. Run `python run.py features` first.",
            context={"feature_data_path": str(path)},
        )
    return path


@lru_cache(maxsize=1)
def get_date_bounds() -> tuple[date, date]:
    """The dataset's observed (min, max) date, computed once from the
    `fact_price_daily` mart.

    Used to validate `POST /v1/predict`'s requested date falls within the
    range real history actually exists for (project_refactor.md §8.2) --
    a cheap aggregate query, not a full-file scan, so it's fine to run
    once at first access rather than needing it precomputed at startup.
    """
    settings = get_settings()
    mart_path = settings.marts.output_dir / "fact_price_daily.parquet"
    if not mart_path.exists():
        raise ArtifactNotFoundError(
            f"Marts not found at {mart_path}. Run `python run.py marts` first.",
            context={"mart_path": str(mart_path)},
        )
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute("SELECT MIN(date), MAX(date) FROM read_parquet(?)", [str(mart_path)]).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None or row[1] is None:
        raise ArtifactNotFoundError(
            f"fact_price_daily mart at {mart_path} contains no rows.",
            context={"mart_path": str(mart_path)},
        )
    # The mart's `date` column is stored as datetime64 (inherited from
    # feature_engineered_data.parquet), so DuckDB returns `datetime`
    # objects here even though the values are always midnight -- normalise
    # to `date` to match this function's contract and what request
    # payloads compare against.
    min_date, max_date = row
    return min_date.date(), max_date.date()


def reset_caches() -> None:
    """Clear all cached dependencies -- for tests that need to point the
    API at fixture data/model between test cases, since `lru_cache` would
    otherwise leak state from whichever test ran first."""
    get_settings.cache_clear()
    get_model.cache_clear()
    get_feature_data_path.cache_clear()
    get_date_bounds.cache_clear()


__all__ = [
    "get_settings",
    "get_model",
    "get_feature_data_path",
    "get_date_bounds",
    "reset_caches",
]
