"""Tests for pricepoint/memory_utils.py.

Added alongside the memory optimizations themselves (project_v2.md Phase 1
Progress Log, 2026-07-02) after real pipeline runs against the full 9.5M-row
dataset repeatedly froze a 15 GB development machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory


class TestDowncastDtypes:
    def test_float64_downcast_to_float32(self):
        df = pd.DataFrame({"x": np.array([1.5, 2.5, 3.5], dtype="float64")})
        downcast_dtypes(df)
        assert df["x"].dtype == np.float32

    def test_int64_downcast_to_smaller_int(self):
        df = pd.DataFrame({"x": np.array([1, 2, 3], dtype="int64")})
        downcast_dtypes(df)
        assert df["x"].dtype.itemsize < 8

    def test_low_cardinality_object_column_becomes_category(self):
        df = pd.DataFrame({"supermarket": ["Tesco", "ASDA"] * 500})
        downcast_dtypes(df)
        assert isinstance(df["supermarket"].dtype, pd.CategoricalDtype)

    def test_high_cardinality_object_column_stays_object(self):
        # Every value unique -> converting to category would not save
        # memory and could even hurt some downstream operations.
        df = pd.DataFrame({"id": [f"item_{i}" for i in range(1000)]})
        downcast_dtypes(df, max_category_ratio=0.5)
        assert not isinstance(df["id"].dtype, pd.CategoricalDtype)

    def test_values_are_preserved(self):
        df = pd.DataFrame(
            {
                "price": [1.99, 2.49, np.nan],
                "supermarket": ["Tesco", "ASDA", "Tesco"],
            }
        )
        original_prices = df["price"].tolist()
        downcast_dtypes(df)
        # NaN != NaN, so compare null positions separately. float64 -> float32
        # is a deliberate, lossy-but-immaterial precision reduction (penny
        # prices need nowhere near float64's ~15-17 significant digits), so
        # values are compared approximately, not bit-exactly.
        assert df["price"].isna().tolist() == [False, False, True]
        np.testing.assert_allclose(
            df["price"].dropna().to_numpy(dtype=float),
            [v for v in original_prices if not np.isnan(v)],
            rtol=1e-6,
        )
        assert df["supermarket"].tolist() == ["Tesco", "ASDA", "Tesco"]

    def test_empty_dataframe_does_not_raise(self):
        df = pd.DataFrame({"x": pd.array([], dtype="float64")})
        result = downcast_dtypes(df)
        assert len(result) == 0

    def test_returns_same_object_for_chaining(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = downcast_dtypes(df)
        assert result is df


class TestMemoryLoggingHelpers:
    def test_log_memory_does_not_raise(self):
        log_memory("test checkpoint")  # should not raise even without psutil

    def test_collect_garbage_does_not_raise(self):
        collect_garbage()
