"""Tests for feature engineering pipeline."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

import pricepoint.feature_engineering as feature_engineering_module
from pricepoint.feature_engineering import (
    add_competitive_features,
    add_cyclical_features,
    add_temporal_features,
    aggregate_to_listing_day,
    run_feature_engineering,
)


@pytest.fixture
def price_series_df() -> pd.DataFrame:
    """A multi-day, multi-retailer price dataset for testing."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    rows = []
    for date in dates:
        for store in ["Tesco", "ASDA", "Aldi"]:
            rows.append(
                {
                    "canonical_name": "bananas",
                    "supermarket": store,
                    "product_name": f"{store} Bananas 5pk",
                    "date": date,
                    "prices": round(1.0 + rng.random() * 0.5, 2),
                }
            )
    return pd.DataFrame(rows)


def _reference_add_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lag_days: list[int],
) -> pd.DataFrame:
    """Plain pandas reference for the past-only definitions.

    Written independently of the Polars implementation: every statistic is
    taken over each listing's price series shifted by one observation, so
    the row's own price never enters it (docs/forecast_leak_fix.md).
    """
    group_cols = ["supermarket", "product_name"]
    df = df.sort_values([*group_cols, "date"]).copy()
    prev = df.groupby(group_cols, observed=True)["prices"].shift(1)
    grp_prev = prev.groupby([df[c] for c in group_cols], observed=True)

    for window in rolling_windows:
        df[f"price_rol_mean_{window}d"] = grp_prev.transform(lambda x, w=window: x.rolling(w, min_periods=1).mean())
        df[f"price_rol_std_{window}d"] = grp_prev.transform(lambda x, w=window: x.rolling(w, min_periods=1).std())
        df[f"price_rol_max_{window}d"] = grp_prev.transform(lambda x, w=window: x.rolling(w, min_periods=1).max())
        df[f"price_rol_min_{window}d"] = grp_prev.transform(lambda x, w=window: x.rolling(w, min_periods=1).min())

    for lag in lag_days:
        df[f"price_lag_{lag}d"] = df.groupby(group_cols, observed=True)["prices"].shift(lag)

    df["price_diff_1d"] = prev - df.groupby(group_cols, observed=True)["prices"].shift(2)
    df["days_since_last_obs"] = df.groupby(group_cols, observed=True)["date"].diff().dt.days.astype(float)
    return df


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

    def test_momentum_and_gap_columns_created(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[], lag_days=[])
        assert "price_diff_1d" in result.columns
        assert "days_since_last_obs" in result.columns

    def test_output_same_length(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[7], lag_days=[1])
        assert len(result) == len(price_series_df)


class TestTemporalFeaturesMatchReference:
    """The Polars implementation against the independent pandas reference."""

    @pytest.mark.parametrize(
        "rolling_windows,lag_days",
        [
            ([7], [1]),
            ([7, 14, 30], [1, 7]),
            ([1], [1]),
            ([100], [1]),  # window larger than every group
        ],
    )
    def test_matches_reference_on_multi_group_data(self, price_series_df, rolling_windows, lag_days):
        keys = ["supermarket", "product_name", "date"]
        new_result = add_temporal_features(price_series_df, rolling_windows, lag_days).sort_values(keys)
        old_result = _reference_add_temporal_features(price_series_df, rolling_windows, lag_days).sort_values(keys)
        cols = [c for c in new_result.columns if c.startswith(("price_rol_", "price_lag_", "price_diff_", "days_"))]
        for col in cols:
            np.testing.assert_allclose(
                new_result[col].to_numpy(dtype=float),
                old_result[col].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Mismatch in column {col!r}",
            )

    def test_date_gaps_use_row_count_windows(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11", "2024-01-25"])
        df = pd.DataFrame(
            {
                "canonical_name": ["milk"] * 5,
                "supermarket": ["Tesco"] * 5,
                "product_name": ["Tesco Milk 1L"] * 5,
                "date": dates,
                "prices": [1.0, 1.1, 1.5, 1.4, 2.0],
            }
        )
        result = add_temporal_features(df, rolling_windows=[3], lag_days=[1])
        # Day 5: previous three observations are 1.1, 1.5, 1.4.
        assert result["price_rol_mean_3d"].iloc[4] == pytest.approx((1.1 + 1.5 + 1.4) / 3)
        assert result["price_lag_1d"].iloc[4] == pytest.approx(1.4)
        assert result["price_diff_1d"].iloc[4] == pytest.approx(1.4 - 1.5)
        assert result["days_since_last_obs"].iloc[4] == 14

    def test_first_observation_has_no_history(self):
        df = pd.DataFrame(
            {
                "canonical_name": ["rare_item"],
                "supermarket": ["Aldi"],
                "product_name": ["Aldi Rare Item"],
                "date": pd.to_datetime(["2024-01-01"]),
                "prices": [3.5],
            }
        )
        result = add_temporal_features(df, rolling_windows=[7], lag_days=[1])
        for col in ["price_rol_mean_7d", "price_lag_1d", "price_diff_1d", "days_since_last_obs"]:
            assert pd.isna(result[col].iloc[0]), col

    def test_listings_of_one_product_family_are_separate_series(self):
        """Two pack sizes under one canonical name must not feed each other's lags."""
        df = pd.DataFrame(
            {
                "canonical_name": ["mince"] * 4,
                "supermarket": ["Tesco"] * 4,
                "product_name": ["Mince 500g", "Mince 1kg", "Mince 500g", "Mince 1kg"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                "prices": [2.0, 3.8, 2.1, 3.9],
            }
        )
        result = add_temporal_features(df, rolling_windows=[], lag_days=[1]).set_index(["product_name", "date"])
        assert result.loc[("Mince 500g", pd.Timestamp("2024-01-02")), "price_lag_1d"] == pytest.approx(2.0)
        assert result.loc[("Mince 1kg", pd.Timestamp("2024-01-02")), "price_lag_1d"] == pytest.approx(3.8)

    def test_output_row_order_is_sorted_by_listing_then_date(self, price_series_df):
        shuffled = price_series_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        result = add_temporal_features(shuffled, rolling_windows=[7], lag_days=[1])
        keys = ["supermarket", "product_name", "date"]
        expected = result.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(result.reset_index(drop=True)[keys], expected[keys])


class TestAggregateToListingDay:
    def test_duplicate_rows_are_averaged(self):
        df = pd.DataFrame(
            {
                "canonical_name": ["tea", "tea", "tea"],
                "supermarket": ["ASDA"] * 3,
                "product_name": ["ASDA Tea 80"] * 3,
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "prices": [1.0, 1.2, 1.1],
                "category": ["drinks"] * 3,
            }
        )
        out = aggregate_to_listing_day(df).sort_values("date")
        assert len(out) == 2
        assert out["prices"].iloc[0] == pytest.approx(1.1)
        assert out["category"].iloc[0] == "drinks"


class TestCompetitiveFeatures:
    def test_columns_added(self, price_series_df):
        result = add_competitive_features(add_temporal_features(price_series_df, [], [1]))
        assert "price_vs_market_avg" in result.columns
        assert "price_rank" in result.columns
        assert "is_cheapest_in_market" in result.columns

    def test_cheapest_flag_values(self, price_series_df):
        result = add_competitive_features(add_temporal_features(price_series_df, [], [1]))
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


class _FakeDataConfig:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        # Only `raw_dir.parent` is actually used (run_reports.default_report_dir,
        # to derive data/_run_reports/) -- the directory itself need not exist.
        self.raw_dir = processed_dir.parent / "raw"


class _FakeMatchingConfig:
    output_filename = "canonical_products_e5.parquet"


class _FakeFeaturesConfig:
    output_filename = "feature_engineered_data.parquet"
    rolling_windows = [7]
    lag_days = [1]


class _FakeFeatureEngineeringSettings:
    def __init__(self, processed_dir):
        self.data = _FakeDataConfig(processed_dir)
        self.matching = _FakeMatchingConfig()
        self.features = _FakeFeaturesConfig()


class TestRunFeatureEngineeringSkipIfUnchanged:
    """Integration tests for run_feature_engineering's manifest-based
    skip-if-unchanged wiring -- previously written (write_manifest was
    already called here) but never actually checked before re-running the
    expensive rolling/lag computation."""

    @pytest.fixture
    def settings(self, tmp_path, price_series_df):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        df = price_series_df.copy()
        df["own_brand"] = False
        df.to_parquet(processed_dir / "canonical_products_e5.parquet")
        return _FakeFeatureEngineeringSettings(processed_dir=processed_dir)

    def test_second_run_skips_when_unchanged(self, settings, monkeypatch):
        first_path = run_feature_engineering(settings)
        assert first_path.exists()

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("add_temporal_features was called on an unchanged re-run; skip logic did not trigger")

        monkeypatch.setattr(feature_engineering_module, "add_temporal_features", _fail_if_called)

        second_path = run_feature_engineering(settings)
        assert second_path == first_path

    def test_force_reruns_even_when_unchanged(self, settings, monkeypatch):
        run_feature_engineering(settings)

        calls = []
        original = feature_engineering_module.add_temporal_features
        monkeypatch.setattr(
            feature_engineering_module,
            "add_temporal_features",
            lambda *a, **kw: (calls.append(1), original(*a, **kw))[1],
        )

        run_feature_engineering(settings, force=True)
        assert len(calls) == 1, "force=True must re-run feature engineering even when input is unchanged"

    def test_reruns_when_canonical_data_modified(self, settings, price_series_df):
        first_path = run_feature_engineering(settings)
        first_manifest = json.loads(first_path.with_suffix(first_path.suffix + ".manifest.json").read_text())

        time.sleep(0.01)
        canonical_path = settings.data.processed_dir / "canonical_products_e5.parquet"
        df = price_series_df.copy()
        df["own_brand"] = False
        df.to_parquet(canonical_path)

        second_path = run_feature_engineering(settings)
        second_manifest = json.loads(second_path.with_suffix(second_path.suffix + ".manifest.json").read_text())
        assert second_manifest["generated_at"] != first_manifest["generated_at"]
