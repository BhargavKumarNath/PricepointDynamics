"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricepoint.feature_engineering import (
    add_competitive_features,
    add_cyclical_features,
    add_temporal_features,
)


@pytest.fixture
def price_series_df() -> pd.DataFrame:
    """A multi-day, multi-retailer price dataset for testing."""
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    rows = []
    for date in dates:
        for store in ["Tesco", "ASDA", "Aldi"]:
            rows.append(
                {
                    "canonical_name": "bananas",
                    "supermarket": store,
                    "date": date,
                    "prices": round(1.0 + np.random.rand() * 0.5, 2),
                }
            )
    return pd.DataFrame(rows)


class TestTemporalFeatures:
    def test_rolling_mean_columns_created(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[7], lag_days=[1])
        assert "price_rol_mean_7d" in result.columns
        assert "price_rol_std_7d" in result.columns
        assert "price_rol_max_7d" in result.columns
        assert "price_rol_min_7d" in result.columns

    def test_lag_columns_created(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[], lag_days=[1, 7])
        assert "price_lag_1d" in result.columns
        assert "price_lag_7d" in result.columns

    def test_momentum_column_created(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[], lag_days=[])
        assert "price_diff_1d" in result.columns

    def test_output_same_length(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[7], lag_days=[1])
        assert len(result) == len(price_series_df)


class TestCompetitiveFeatures:
    def test_columns_added(self, price_series_df):
        result = add_competitive_features(price_series_df)
        assert "price_vs_market_avg" in result.columns
        assert "price_rank" in result.columns
        assert "is_cheapest_in_market" in result.columns

    def test_cheapest_flag_values(self, price_series_df):
        result = add_competitive_features(price_series_df)
        assert set(result["is_cheapest_in_market"].unique()).issubset({0, 1})


class TestCyclicalFeatures:
    def test_columns_added(self, price_series_df):
        result = add_cyclical_features(price_series_df)
        expected = [
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "week_of_year_sin",
            "week_of_year_cos",
        ]
        for col in expected:
            assert col in result.columns

    def test_values_bounded(self, price_series_df):
        result = add_cyclical_features(price_series_df)
        for col in ["day_of_week_sin", "day_of_week_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0
