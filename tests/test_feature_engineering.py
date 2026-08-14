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
    run_feature_engineering,
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


def _reference_add_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lag_days: list[int],
) -> pd.DataFrame:
    """Pre-Phase-2 pandas reference implementation of add_temporal_features.

    Kept only in this test module (not in production code) as the "old"
    side of a tolerance-based old-vs-new regression comparison for the
    Polars rewrite in feature_engineering.py, per project_refactor.md
    Phase 2's testing requirement. Mirrors the exact
    ``groupby(...).transform(lambda x: x.rolling(...))`` logic that shipped
    before the rewrite.
    """
    df = df.sort_values(["canonical_name", "supermarket", "date"]).copy()
    group_cols = ["canonical_name", "supermarket"]

    for window in rolling_windows:
        grp = df.groupby(group_cols, observed=True)["prices"]
        df[f"price_rol_mean_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).mean())
        df[f"price_rol_std_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).std())
        df[f"price_rol_max_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).max())
        df[f"price_rol_min_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).min())

    for lag in lag_days:
        df[f"price_lag_{lag}d"] = df.groupby(group_cols, observed=True)["prices"].shift(lag)

    df["price_diff_1d"] = df.groupby(group_cols, observed=True)["prices"].diff(1)
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

    def test_momentum_column_created(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[], lag_days=[])
        assert "price_diff_1d" in result.columns

    def test_output_same_length(self, price_series_df):
        result = add_temporal_features(price_series_df, rolling_windows=[7], lag_days=[1])
        assert len(result) == len(price_series_df)


class TestTemporalFeaturesPolarsRewriteRegression:
    """Tolerance-based old-vs-new regression tests for the Phase 2 Polars rewrite.

    Compares the production (Polars-backed) add_temporal_features against
    the pre-rewrite pandas reference implementation on several fixture
    shapes, including edge cases the rewrite must still handle correctly:
    single-row groups, groups with date gaps (row-count, not calendar-day,
    windows), and windows larger than the group size.
    """

    @pytest.mark.parametrize(
        "rolling_windows,lag_days",
        [
            ([7], [1]),
            ([7, 14, 30], [1, 7]),
            ([1], [1]),  # window smaller than any meaningful group
            ([100], [1]),  # window larger than every group (all-partial windows)
        ],
    )
    def test_matches_reference_on_multi_group_data(self, price_series_df, rolling_windows, lag_days):
        new_result = add_temporal_features(price_series_df, rolling_windows, lag_days)
        old_result = _reference_add_temporal_features(price_series_df, rolling_windows, lag_days)

        new_result = new_result.sort_values(["canonical_name", "supermarket", "date"]).reset_index(drop=True)
        old_result = old_result.sort_values(["canonical_name", "supermarket", "date"]).reset_index(drop=True)

        new_cols = [c for c in new_result.columns if c.startswith(("price_rol_", "price_lag_", "price_diff_"))]
        for col in new_cols:
            np.testing.assert_allclose(
                new_result[col].to_numpy(dtype=float),
                old_result[col].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Mismatch in column {col!r}",
            )

    def test_matches_reference_with_date_gaps(self):
        """Row-count (not calendar-day) rolling windows must be preserved exactly
        even when a product-store group has missing days -- a real, common
        pattern in the raw data (a product isn't scraped/sold every day)."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11", "2024-01-25"])
        df = pd.DataFrame(
            {
                "canonical_name": ["milk"] * 5,
                "supermarket": ["Tesco"] * 5,
                "date": dates,
                "prices": [1.0, 1.1, 1.5, 1.4, 2.0],
            }
        )
        new_result = add_temporal_features(df, rolling_windows=[3], lag_days=[1])
        old_result = _reference_add_temporal_features(df, rolling_windows=[3], lag_days=[1])

        for col in ["price_rol_mean_3d", "price_rol_std_3d", "price_lag_1d", "price_diff_1d"]:
            np.testing.assert_allclose(
                new_result[col].to_numpy(dtype=float),
                old_result[col].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"Mismatch in column {col!r}",
            )

    def test_matches_reference_on_single_row_group(self):
        """A group with exactly one observation: rolling stats and diff/lag
        should collapse to (value, NaN, NaN, ...) without erroring."""
        df = pd.DataFrame(
            {
                "canonical_name": ["rare_item"],
                "supermarket": ["Aldi"],
                "date": pd.to_datetime(["2024-01-01"]),
                "prices": [3.5],
            }
        )
        new_result = add_temporal_features(df, rolling_windows=[7], lag_days=[1])
        old_result = _reference_add_temporal_features(df, rolling_windows=[7], lag_days=[1])

        assert new_result["price_rol_mean_7d"].iloc[0] == pytest.approx(3.5)
        assert old_result["price_rol_mean_7d"].iloc[0] == pytest.approx(3.5)
        assert pd.isna(new_result["price_lag_1d"].iloc[0])
        assert pd.isna(new_result["price_diff_1d"].iloc[0])

    def test_output_row_order_is_sorted_by_group_then_date(self, price_series_df):
        """The function's contract is to return rows sorted by
        [canonical_name, supermarket, date] -- rolling/lag values are only
        meaningful in that order, so callers (and downstream stages) rely
        on it, not just on column presence."""
        shuffled = price_series_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        result = add_temporal_features(shuffled, rolling_windows=[7], lag_days=[1])
        expected_order = result.sort_values(["canonical_name", "supermarket", "date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True)[["canonical_name", "supermarket", "date"]],
            expected_order[["canonical_name", "supermarket", "date"]],
        )


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
