"""Fixtures for API contract tests.

Builds a small but genuinely-trained model + feature dataset so
`tests/api/*` never touches real data under `data/`/`models/` -- those
are gitignored and absent in a fresh CI checkout. All FastAPI
dependencies are swapped via `app.dependency_overrides`, not by
monkeypatching paths, so the app under test is the real `api.main.app`.

Live endpoints: `GET /health`, `POST /v1/predict`, and
`GET /v1/products/{id}/history` (project_refactor.md §25.2's locked,
narrowed API surface, plus the one promotion UI_refactor.md made) -- so
these fixtures cover both the feature-data path `/v1/predict` needs and
a small marts fixture for the history endpoint's `Warehouse` dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_date_bounds, get_feature_data_path, get_model, get_settings, get_warehouse
from api.main import app
from pricepoint.training import prepare_training_data, train_model
from pricepoint.warehouse import Warehouse

PRODUCTS = ["test bananas", "test bread"]
STORES = ["TestMart", "OtherMart"]


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """A synthetic but internally-consistent feature-engineered dataset:
    2 products x 2 stores x 20 days, with real lag/rolling columns derived
    from the same synthetic price series (not just present-but-meaningless
    columns) so a model trained on it behaves like a real one."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rows = []
    for product in PRODUCTS:
        for store in STORES:
            base = 1.20 if product == "test bananas" else 0.85
            for i, d in enumerate(dates):
                price = round(base + 0.01 * np.sin(i / 3) + (0.05 if store == "OtherMart" else 0.0), 4)
                rows.append(
                    {
                        "canonical_name": product,
                        "supermarket": store,
                        "date": d,
                        "prices": price,
                        "own_brand": product == "test bread",
                        "day_of_week_sin": float(np.sin(2 * np.pi * d.dayofweek / 7)),
                        "day_of_week_cos": float(np.cos(2 * np.pi * d.dayofweek / 7)),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["canonical_name", "supermarket", "date"]).reset_index(drop=True)

    grouped = df.groupby(["canonical_name", "supermarket"])["prices"]
    df["price_lag_1d"] = grouped.shift(1)
    df["price_diff_1d"] = df["prices"] - df["price_lag_1d"]
    df["price_rol_mean_7d"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["price_rol_max_7d"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).max())
    df["price_rol_min_7d"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).min())
    df["price_vs_market_avg"] = df["prices"] - df.groupby("date")["prices"].transform("mean")
    return df


@pytest.fixture
def fixture_model(feature_df):
    """A genuinely-trained (not mocked) tiny LightGBM model, via the same
    `prepare_training_data` training uses -- so `model.feature_name_`
    matches what `build_feature_vector` (serving) reindexes against, the
    same train/serve contract the real pipeline relies on."""
    x_train, y_train, _, _ = prepare_training_data(feature_df)
    return train_model(
        x_train,
        y_train,
        {"objective": "mae", "n_estimators": 30, "num_leaves": 7, "min_child_samples": 1, "verbose": -1},
    )


@pytest.fixture
def feature_data_path(tmp_path, feature_df):
    path = tmp_path / "feature_engineered_data.parquet"
    feature_df.to_parquet(path)
    return path


@pytest.fixture
def marts_dir(tmp_path):
    """A small fixture `fact_price_daily` mart -- backs the history
    endpoint's `Warehouse.get_product_history` dependency, mirroring the
    shape already established in `tests/test_warehouse.py`."""
    fact_price_daily = pd.DataFrame(
        {
            "canonical_name": ["test bananas", "test bananas", "test bread"],
            "supermarket": ["TestMart", "OtherMart", "TestMart"],
            "date": pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-10"]),
            "avg_price": [1.19, 1.25, 0.85],
            "min_price": [1.19, 1.25, 0.85],
            "max_price": [1.19, 1.25, 0.85],
            "n_listings": [1, 1, 1],
        }
    )
    fact_price_daily.to_parquet(tmp_path / "fact_price_daily.parquet")
    return tmp_path


@pytest.fixture
def client(fixture_model, feature_data_path, marts_dir):
    real_settings = get_settings()
    real_settings.api.predict_rate_limit = "1000/minute"

    min_date = pd.Timestamp("2024-01-01").date()
    max_date = pd.Timestamp("2024-01-20").date()

    def _warehouse():
        warehouse = Warehouse(marts_dir)
        try:
            yield warehouse
        finally:
            warehouse.close()

    app.dependency_overrides[get_settings] = lambda: real_settings
    app.dependency_overrides[get_model] = lambda: fixture_model
    app.dependency_overrides[get_feature_data_path] = lambda: feature_data_path
    app.dependency_overrides[get_date_bounds] = lambda: (min_date, max_date)
    app.dependency_overrides[get_warehouse] = _warehouse

    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()
