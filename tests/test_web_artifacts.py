"""Tests for pricepoint/web_artifacts.py -- the precomputed frontend
artifact exporters (project_refactor.md §25.3).

Builds a small fixture universe (marts, model metrics, an ingestion
manifest, market dynamics/SHAP precompute artifacts, feature data, a
baskets.yaml) so these tests never touch real data -- gitignored and
absent in a fresh CI checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from pricepoint.warehouse import Warehouse
from pricepoint.web_artifacts import (
    export_basket_analysis,
    export_home_metrics,
    export_market_dynamics,
    export_market_overview,
    export_predictor_context,
    export_shap_explorer,
    run_export_web_artifacts,
)


@pytest.fixture
def marts_dir(tmp_path) -> object:
    dim_product = pd.DataFrame(
        {
            "canonical_name": ["bananas", "milk"],
            "category": ["fresh_food", "fresh_food"],
            "own_brand": [False, True],
            "n_retailers": [2, 2],
            "first_seen_date": pd.to_datetime(["2024-01-01"] * 2),
            "last_seen_date": pd.to_datetime(["2024-01-10"] * 2),
        }
    )
    dim_product.to_parquet(tmp_path / "dim_product.parquet")

    fact_price_daily = pd.DataFrame(
        {
            "canonical_name": ["bananas", "bananas", "milk", "milk"],
            "supermarket": ["Tesco", "ASDA", "Tesco", "ASDA"],
            "date": pd.to_datetime(["2024-01-10"] * 4),
            "avg_price": [0.99, 1.05, 1.20, 1.15],
            "min_price": [0.99, 1.05, 1.20, 1.15],
            "max_price": [0.99, 1.05, 1.20, 1.15],
            "n_listings": [1, 1, 1, 1],
        }
    )
    fact_price_daily.to_parquet(tmp_path / "fact_price_daily.parquet")
    return tmp_path


@pytest.fixture
def baskets() -> dict[str, list[str]]:
    return {"Test Basket": ["bananas", "milk"]}


class _FakeSubConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeSettings:
    """Duck-typed stand-in for `pricepoint.config.Settings`, mirroring the
    convention already established in `tests/test_marts.py` -- exposes
    exactly the attributes `web_artifacts.py`'s functions read."""

    def __init__(self, tmp_path):
        self.data = _FakeSubConfig(
            raw_dir=tmp_path / "data" / "00_raw",
            interim_dir=tmp_path / "data" / "01_interim",
            processed_dir=tmp_path / "data" / "02_processed",
        )
        self.model = _FakeSubConfig(output_dir=tmp_path / "models")
        self.marts = _FakeSubConfig(output_dir=tmp_path / "marts")
        self.market_dynamics = _FakeSubConfig(output_dir=tmp_path / "market_dynamics")
        self.shap = _FakeSubConfig(output_dir=tmp_path / "shap")
        self.features = _FakeSubConfig(output_filename="feature_engineered_data.parquet")
        self.web_artifacts = _FakeSubConfig(
            json_dir=tmp_path / "frontend" / "public" / "data",
            parquet_dir=tmp_path / "frontend" / "public" / "data-r2",
        )


@pytest.fixture
def full_settings(tmp_path, marts_dir, baskets) -> _FakeSettings:
    """A fully-populated fixture settings object -- every artifact
    `run_export_web_artifacts` needs, written to a tmp_path tree."""
    settings = _FakeSettings(tmp_path)

    settings.model.output_dir.mkdir(parents=True)
    (settings.model.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "MAE": 0.15,
                "RMSE": 1.25,
                "R2": 0.96,
                "n_train_rows": 100,
                "n_test_rows": 20,
                "n_features": 5,
                "trained_at": "2024-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    settings.data.interim_dir.mkdir(parents=True)
    (settings.data.interim_dir / "cleaned_supermarket_data.parquet.manifest.json").write_text(
        json.dumps({"row_count": 12345}), encoding="utf-8"
    )

    # marts_dir fixture already wrote dim_product/fact_price_daily into
    # its own tmp_path -- point settings.marts.output_dir at it directly
    # rather than duplicating the fixture data.
    settings.marts = _FakeSubConfig(output_dir=marts_dir)

    settings.market_dynamics.output_dir.mkdir(parents=True)
    pd.DataFrame(
        {"leader": ["Aldi"], "follower": ["Tesco"], "median_lag_days": [2.0], "n_products_analyzed": [10]}
    ).to_parquet(settings.market_dynamics.output_dir / "price_leadership.parquet")
    pd.DataFrame(
        {"dispersion": [0.10, 0.12, 0.11]}, index=pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"])
    ).rename_axis("date").to_parquet(settings.market_dynamics.output_dir / "market_dispersion.parquet")

    settings.data.processed_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "canonical_name": ["bananas", "bananas", "milk"],
            "supermarket": ["Tesco", "Tesco", "ASDA"],
            "category": ["fresh_food", "fresh_food", "fresh_food"],
            "date": pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-10"]),
            "prices": [0.95, 0.99, 1.15],
            "price_lag_1d": [0.90, 0.95, 1.10],
        }
    ).to_parquet(settings.data.processed_dir / "feature_engineered_data.parquet")

    settings.shap.output_dir.mkdir(parents=True)
    x_sample = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    x_sample.to_parquet(settings.shap.output_dir / "shap_sample_data.parquet")
    np.save(settings.shap.output_dir / "shap_values.npy", np.array([[0.1, -0.2], [0.3, -0.1]]))
    (settings.shap.output_dir / "shap_base_value.txt").write_text("2.5", encoding="utf-8")

    settings.data.raw_dir.mkdir(parents=True)
    configs_dir = settings.data.raw_dir.parents[1] / "configs"
    configs_dir.mkdir(parents=True)
    with open(configs_dir / "baskets.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(baskets, fh)

    return settings


class TestExportHomeMetrics:
    def test_returns_real_traced_values(self, full_settings):
        result = export_home_metrics(full_settings)
        assert result["total_records"] == 12345
        assert result["canonical_products"] == 2
        assert result["mae"] == 0.15

    def test_missing_metrics_raises(self, tmp_path):
        settings = _FakeSettings(tmp_path)
        settings.model.output_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            export_home_metrics(settings)


class TestExportMarketOverview:
    def test_wraps_warehouse_output(self, marts_dir):
        with Warehouse(marts_dir) as warehouse:
            result = export_market_overview(warehouse)
        assert "supermarkets" in result
        assert len(result["supermarkets"]) == 2


class TestExportBasketAnalysis:
    def test_returns_cost_and_item_rows_per_basket(self, marts_dir, baskets):
        with Warehouse(marts_dir) as warehouse:
            result = export_basket_analysis(warehouse, baskets)
        basket = result["baskets"]["Test Basket"]
        assert basket["total_items"] == 2
        assert len(basket["rows"]) == 2  # Tesco + ASDA
        assert len(basket["items"]) == 4  # 2 products x 2 stores

    def test_coverage_pct_computed(self, marts_dir):
        with Warehouse(marts_dir) as warehouse:
            result = export_basket_analysis(warehouse, {"Partial": ["bananas", "not_a_real_product"]})
        rows = result["baskets"]["Partial"]["rows"]
        # Only 1 of 2 basket items exists -> 50% coverage for any store that stocks it
        assert all(row["coverage_pct"] == pytest.approx(50.0) for row in rows)


class TestExportMarketDynamics:
    def test_returns_dispersion_and_leadership(self, full_settings):
        result = export_market_dynamics(full_settings)
        assert len(result["dispersion"]) == 3
        assert len(result["leadership"]) == 1
        assert result["top_leader"] == "Aldi"
        assert result["fastest_follower"]["leader"] == "Aldi"

    def test_missing_artifacts_raises(self, tmp_path):
        settings = _FakeSettings(tmp_path)
        settings.market_dynamics.output_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            export_market_dynamics(settings)


class TestExportPredictorContext:
    def test_returns_latest_row_per_product_store(self, full_settings):
        result = export_predictor_context(full_settings)
        assert len(result) == 2  # (bananas, Tesco), (milk, ASDA)
        tesco_bananas = result[(result["canonical_name"] == "bananas") & (result["supermarket"] == "Tesco")]
        # Latest date for bananas/Tesco is 2024-01-10 (price 0.99), not 01-09 (0.95)
        assert tesco_bananas.iloc[0]["prices"] == pytest.approx(0.99)


class TestExportShapExplorer:
    def test_returns_matching_shape_grids_and_meta(self, full_settings):
        features_df, values_df, meta = export_shap_explorer(full_settings)
        assert features_df.shape == values_df.shape == (2, 2)
        assert meta["base_value"] == pytest.approx(2.5)
        assert meta["n_samples"] == 2
        assert meta["feature_names"] == ["f1", "f2"]


class TestRunExportWebArtifacts:
    def test_writes_all_artifacts(self, full_settings):
        paths = run_export_web_artifacts(full_settings)
        for path in paths.values():
            assert path.exists()
        assert (full_settings.web_artifacts.json_dir / "home_metrics.json").exists()
        assert (full_settings.web_artifacts.json_dir / "market_overview.json").exists()
        assert (full_settings.web_artifacts.json_dir / "basket_analysis.json").exists()
        assert (full_settings.web_artifacts.json_dir / "market_dynamics.json").exists()
        assert (full_settings.web_artifacts.json_dir / "shap_explorer_meta.json").exists()
        assert (full_settings.web_artifacts.parquet_dir / "predictor_context.parquet").exists()
        assert (full_settings.web_artifacts.parquet_dir / "shap_explorer_features.parquet").exists()
        assert (full_settings.web_artifacts.parquet_dir / "shap_explorer_values.parquet").exists()
