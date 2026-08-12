"""Tests for pricepoint/training.py::prepare_training_data.

training.py previously had zero test coverage (ENGINEERING_AUDIT_REPORT.md
§5). These tests specifically guard the two real bugs found and fixed while
producing the first honest post-leakage-fix MAE in Phase 1 (see
project_v2.md Progress Log, 2026-07-02):

1. pd.get_dummies() produces bool-dtype columns in modern pandas, and a
   numeric-only select_dtypes(["number"]) silently drops them -- every
   one-hot-encoded categorical feature (and any raw bool column like
   own_brand) would vanish from training with no error.
2. `prices_unit` is a same-row rescaling of the target (`prices`) and must
   never be used as a training feature.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pricepoint.training import evaluate_model, prepare_training_data, run_training, train_model


@pytest.fixture
def feature_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        for store, cat, own_brand in [
            ("Tesco", "drinks", False),
            ("ASDA", "fresh_food", True),
        ]:
            rows.append(
                {
                    "date": date,
                    "product_name": "Coke 500ml",
                    "canonical_name": "coke 500ml",
                    "normalised_name": "coke 500ml",
                    "supermarket": store,
                    "category": cat,
                    "own_brand": own_brand,
                    "prices": 1.50,
                    "prices_unit": 3.00,  # exactly 2x prices -- the leak this test guards
                    "price_lag_1d": 1.45,
                }
            )
    return pd.DataFrame(rows)


class TestPrepareTrainingData:
    def test_boolean_and_one_hot_columns_are_retained(self, feature_df):
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        assert "own_brand" in X_train.columns
        assert any(col.startswith("supermarket_") for col in X_train.columns)
        assert any(col.startswith("category_") for col in X_train.columns)

    def test_boolean_columns_are_numeric_dtype(self, feature_df):
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        assert X_train["own_brand"].dtype != bool
        assert np.issubdtype(X_train["own_brand"].dtype, np.integer)

    def test_prices_unit_is_excluded(self, feature_df):
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        assert "prices_unit" not in X_train.columns
        assert "prices_unit" not in X_test.columns

    def test_identifier_and_target_columns_are_excluded(self, feature_df):
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        for leaky_col in ["prices", "date", "product_name", "canonical_name", "normalised_name"]:
            assert leaky_col not in X_train.columns

    def test_test_columns_match_train_columns(self, feature_df):
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        assert list(X_train.columns) == list(X_test.columns)


class TestTrainingPipeline:
    """Smoke tests for the end-to-end training pipeline."""

    def test_model_trains_and_predicts(self, feature_df):
        """Verify a LightGBM model can be trained and makes predictions."""
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        # Ensure we have data to train on
        assert len(X_train) > 0
        assert len(X_test) > 0

        # Train the model
        model = train_model(X_train, y_train, {"n_estimators": 10, "random_state": 42})
        assert model is not None

        # Verify predictions work
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert all(isinstance(p, (float, np.floating)) for p in preds)

    def test_model_evaluation_computes_metrics(self, feature_df):
        """Verify model evaluation produces valid metrics."""
        X_train, y_train, X_test, y_test = prepare_training_data(feature_df)

        model = train_model(X_train, y_train, {"n_estimators": 10, "random_state": 42})
        metrics = evaluate_model(model, X_test, y_test)

        # Check all expected metrics are present
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

        # Verify metrics are reasonable (MAE and RMSE should be non-negative)
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0
        # R² can be NaN, negative, or positive depending on data; just verify it's a number
        assert isinstance(metrics["R2"], (int, float))


class _FakeDataConfig:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir


class _FakeFeaturesConfig:
    output_filename = "feature_engineered_data.parquet"


class _FakeModelConfig:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.model_filename = "price_predictor_lgbm.joblib"
        self.lgbm_params = {"n_estimators": 10, "random_state": 42}

    @property
    def model_path(self):
        return self.output_dir / self.model_filename


class _FakeTrainingSettings:
    def __init__(self, processed_dir, model_dir):
        self.data = _FakeDataConfig(processed_dir)
        self.features = _FakeFeaturesConfig()
        self.model = _FakeModelConfig(model_dir)


class TestRunTrainingArtifacts:
    """Integration tests for run_training's artifact outputs -- Phase 1's
    task list explicitly called for `metrics.json` to be written alongside
    the model, which the original implementation computed but never
    persisted (only logged)."""

    @pytest.fixture
    def settings(self, tmp_path, feature_df):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        feature_df.to_parquet(processed_dir / "feature_engineered_data.parquet")
        model_dir = tmp_path / "models"
        return _FakeTrainingSettings(processed_dir=processed_dir, model_dir=model_dir)

    def test_model_and_metrics_written(self, settings):
        model_path = run_training(settings)
        assert model_path.exists()

        metrics_path = settings.model.output_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        for key in ("MAE", "RMSE", "R2", "n_train_rows", "n_test_rows", "n_features", "lgbm_params", "trained_at"):
            assert key in metrics, f"metrics.json missing expected key {key!r}"

    def test_saved_model_reloads_and_predicts(self, settings):
        model_path = run_training(settings)
        model = joblib.load(model_path)

        saved_feature_df = pd.read_parquet(settings.data.processed_dir / "feature_engineered_data.parquet")
        _, _, X_test, _ = prepare_training_data(saved_feature_df)
        preds = model.predict(X_test.reindex(columns=model.feature_name_, fill_value=0))
        assert len(preds) == len(X_test)
        assert np.all(np.isfinite(preds))
