"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small DataFrame matching the raw ingestion schema."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "product_name": ["Tesco Bananas 5pk", "ASDA Whole Milk 1L", "Aldi Bread 800g"],
            "prices": [1.50, 1.20, 0.85],
            "supermarket": ["Tesco", "ASDA", "Aldi"],
        }
    )


@pytest.fixture
def sample_canonical_df() -> pd.DataFrame:
    """A small DataFrame matching the canonical products schema."""
    return pd.DataFrame(
        {
            "canonical_name": ["bananas", "whole milk", "bread"],
            "supermarket": ["Tesco", "ASDA", "Aldi"],
            "prices": [1.50, 1.20, 0.85],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "own_brand": [False, True, True],
        }
    )


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """A small DataFrame matching the feature-engineered data schema."""
    return pd.DataFrame(
        {
            "canonical_name": ["bananas", "whole milk", "bread"] * 2,
            "supermarket": ["Tesco", "ASDA", "Aldi"] * 2,
            "prices": [1.50, 1.20, 0.85, 1.52, 1.19, 0.86],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "own_brand": [False, True, True, False, True, True],
            "price_lag_1d": [None, None, None, 1.50, 1.20, 0.85],
            "price_rol_mean_7d": [1.50, 1.20, 0.85, 1.51, 1.195, 0.855],
            "price_rol_max_7d": [1.50, 1.20, 0.85, 1.52, 1.20, 0.86],
            "price_rol_min_7d": [1.50, 1.20, 0.85, 1.50, 1.19, 0.85],
            "price_diff_1d": [None, None, None, 0.02, -0.01, 0.01],
            "price_vs_market_avg": [-0.01, 0.0, 0.01, -0.02, 0.0, 0.02],
            "day_of_week_sin": [0.0, 0.78, 0.43] * 2,
            "day_of_week_cos": [1.0, 0.62, 0.90] * 2,
        }
    )
