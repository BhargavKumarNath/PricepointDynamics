"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small DataFrame matching the raw ingestion schema."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "product_name": ["Tesco Bananas 5pk", "ASDA Whole Milk 1L", "Aldi Bread 800g"],
        "prices": [1.50, 1.20, 0.85],
        "supermarket": ["Tesco", "ASDA", "Aldi"],
    })


@pytest.fixture
def sample_canonical_df() -> pd.DataFrame:
    """A small DataFrame matching the canonical products schema."""
    return pd.DataFrame({
        "canonical_name": ["bananas", "whole milk", "bread"],
        "supermarket": ["Tesco", "ASDA", "Aldi"],
        "prices": [1.50, 1.20, 0.85],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "own_brand": [False, True, True],
    })
