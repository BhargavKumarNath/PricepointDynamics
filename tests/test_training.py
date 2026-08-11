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

import numpy as np
import pandas as pd
import pytest

from pricepoint.training import prepare_training_data


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
