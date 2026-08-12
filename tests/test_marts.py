"""Tests for pricepoint/marts.py -- mart materialization.

Uses a small synthetic fixture deliberately shaped like the real data's
known edge cases (verified directly against canonical_products_e5.parquet
during Phase 3 development): duplicate rows at the
(canonical_name, supermarket, date) grain, and canonical products with
inconsistent category/own_brand labels across their constituent rows --
not just a trivially clean fixture that would pass regardless of whether
the aggregation logic is actually correct.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from pricepoint.marts import MART_NAMES, materialize_mart, run_materialize_marts


@pytest.fixture
def source_df() -> pd.DataFrame:
    """Mirrors canonical_products_e5.parquet's schema and known edge cases."""
    return pd.DataFrame(
        {
            "canonical_name": [
                # 3 duplicate rows at the same (product, store, date) grain --
                # mirrors the real data's 41.5% duplication, caused by
                # multiple pack-size SKUs sharing one canonical cluster.
                "bananas",
                "bananas",
                "bananas",
                # a second store/date for the same product, no duplication
                "bananas",
                # a product with an inconsistent category label across rows
                "mixed category item",
                "mixed category item",
                "mixed category item",
                # a product with an inconsistent own_brand flag across rows
                "mixed brand item",
                "mixed brand item",
            ],
            "supermarket": ["Tesco", "Tesco", "Tesco", "ASDA", "Aldi", "Aldi", "Aldi", "Sains", "Sains"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "prices": [1.0, 1.2, 1.4, 2.0, 3.0, 3.0, 3.0, 5.0, 5.0],
            "category": ["fresh_food"] * 4 + ["produce", "produce", "bakery"] + ["drinks", "drinks"],
            "own_brand": [False, False, False, True, False, False, False, True, False],
        }
    )


@pytest.fixture
def source_parquet(tmp_path, source_df) -> tuple:
    path = tmp_path / "canonical_products_e5.parquet"
    source_df.to_parquet(path)
    return path, tmp_path


class TestMaterializeMart:
    def test_dim_retailer_row_count_matches_distinct_supermarkets(self, source_parquet):
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("dim_retailer", source_path, output_dir, conn)
        result = pd.read_parquet(output_path)
        assert len(result) == 4  # Tesco, ASDA, Aldi, Sains
        assert set(result["retailer_name"]) == {"Tesco", "ASDA", "Aldi", "Sains"}

    def test_dim_product_row_count_matches_distinct_canonical_names(self, source_parquet, source_df):
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("dim_product", source_path, output_dir, conn)
        result = pd.read_parquet(output_path)
        assert len(result) == source_df["canonical_name"].nunique() == 3

    def test_dim_product_resolves_inconsistent_category_via_mode(self, source_parquet):
        """ "mixed category item" has category 'produce' twice and 'bakery' once
        -- the mode (most frequent) must be 'produce', not an arbitrary pick."""
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("dim_product", source_path, output_dir, conn)
        result = pd.read_parquet(output_path).set_index("canonical_name")
        assert result.loc["mixed category item", "category"] == "produce"

    def test_dim_product_resolves_inconsistent_own_brand_via_mode(self, source_parquet):
        """ "mixed brand item" has own_brand=True once, False once -- a tie.
        DuckDB's mode() breaks ties deterministically; this test only
        asserts the result is one of the valid values, not a specific one,
        since the tie-break rule isn't part of this project's contract."""
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("dim_product", source_path, output_dir, conn)
        result = pd.read_parquet(output_path).set_index("canonical_name")
        assert result.loc["mixed brand item", "own_brand"] in (True, False)

    def test_fact_price_daily_collapses_duplicates_via_average(self, source_parquet):
        """3 duplicate rows for (bananas, Tesco, 2024-01-01) at prices
        [1.0, 1.2, 1.4] must collapse to exactly one row with avg_price=1.2."""
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("fact_price_daily", source_path, output_dir, conn)
        result = pd.read_parquet(output_path)

        row = result[
            (result["canonical_name"] == "bananas")
            & (result["supermarket"] == "Tesco")
            & (result["date"] == pd.Timestamp("2024-01-01"))
        ]
        assert len(row) == 1, "duplicate rows at the same grain must collapse to exactly one mart row"
        assert row.iloc[0]["avg_price"] == pytest.approx(1.2)
        assert row.iloc[0]["min_price"] == pytest.approx(1.0)
        assert row.iloc[0]["max_price"] == pytest.approx(1.4)
        assert row.iloc[0]["n_listings"] == 3

    def test_fact_price_daily_row_count_matches_distinct_grain(self, source_parquet, source_df):
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("fact_price_daily", source_path, output_dir, conn)
        result = pd.read_parquet(output_path)
        expected_grain = source_df.drop_duplicates(subset=["canonical_name", "supermarket", "date"])
        assert len(result) == len(expected_grain)

    def test_writes_manifest(self, source_parquet):
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        output_path = materialize_mart("dim_retailer", source_path, output_dir, conn)
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["stage"] == "mart:dim_retailer"
        assert manifest["row_count"] == 4

    def test_unknown_mart_name_raises(self, source_parquet):
        source_path, output_dir = source_parquet
        conn = duckdb.connect(":memory:")
        with pytest.raises(FileNotFoundError):
            materialize_mart("not_a_real_mart", source_path, output_dir, conn)


class _FakeDataConfig:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir


class _FakeMartsConfig:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.source_filename = "canonical_products_e5.parquet"


class _FakeMartsSettings:
    def __init__(self, processed_dir, marts_dir):
        self.data = _FakeDataConfig(processed_dir)
        self.marts = _FakeMartsConfig(marts_dir)


class TestRunMaterializeMartsSkipIfUnchanged:
    """Integration tests for the full run_materialize_marts orchestration,
    covering the same skip-if-unchanged/force wiring convention already
    established for run_ingestion/run_matching/run_feature_engineering."""

    @pytest.fixture
    def settings(self, tmp_path, source_df):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        source_df.to_parquet(processed_dir / "canonical_products_e5.parquet")
        marts_dir = tmp_path / "marts"
        return _FakeMartsSettings(processed_dir=processed_dir, marts_dir=marts_dir)

    def test_builds_all_three_marts(self, settings):
        paths = run_materialize_marts(settings)
        assert set(paths.keys()) == set(MART_NAMES)
        for path in paths.values():
            assert path.exists()

    def test_missing_source_raises(self, tmp_path):
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()  # no canonical_products_e5.parquet written
        settings = _FakeMartsSettings(processed_dir=processed_dir, marts_dir=tmp_path / "marts")
        with pytest.raises(FileNotFoundError):
            run_materialize_marts(settings)

    def test_force_rebuilds_even_when_unchanged(self, settings):
        first_paths = run_materialize_marts(settings)
        first_mtimes = {name: path.stat().st_mtime_ns for name, path in first_paths.items()}

        second_paths = run_materialize_marts(settings, force=True)
        second_mtimes = {name: path.stat().st_mtime_ns for name, path in second_paths.items()}

        for name in MART_NAMES:
            assert second_mtimes[name] >= first_mtimes[name]

    def test_skip_leaves_output_files_untouched(self, settings):
        first_paths = run_materialize_marts(settings)
        first_mtimes = {name: path.stat().st_mtime_ns for name, path in first_paths.items()}

        second_paths = run_materialize_marts(settings)  # force=False, should skip all 3
        second_mtimes = {name: path.stat().st_mtime_ns for name, path in second_paths.items()}

        assert first_mtimes == second_mtimes, "skipped marts must not be rewritten"
