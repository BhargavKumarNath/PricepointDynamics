"""Tests for pricepoint/warehouse.py -- the DuckDB query wrapper over the marts.

Builds a small fixture set of mart Parquet files directly (bypassing
marts.py's SQL-driven build, which is tested separately in
test_marts.py) so these tests focus purely on Warehouse's query
correctness against a known, hand-constructed mart shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pricepoint.warehouse import MartNotFoundError, Warehouse


@pytest.fixture
def marts_dir(tmp_path):
    """A small, hand-built set of mart Parquet files."""
    dim_retailer = pd.DataFrame({"retailer_name": ["Tesco", "ASDA", "Aldi"]})
    dim_retailer.to_parquet(tmp_path / "dim_retailer.parquet")

    dim_product = pd.DataFrame(
        {
            "canonical_name": ["bananas", "banana bread", "milk", "cheddar cheese"],
            "category": ["fresh_food", "bakery", "fresh_food", "fresh_food"],
            "own_brand": [False, True, False, True],
            "n_retailers": [3, 2, 3, 1],
            "first_seen_date": pd.to_datetime(["2024-01-01"] * 4),
            "last_seen_date": pd.to_datetime(["2024-01-10"] * 4),
        }
    )
    dim_product.to_parquet(tmp_path / "dim_product.parquet")

    fact_price_daily = pd.DataFrame(
        {
            "canonical_name": [
                "bananas",
                "bananas",
                "bananas",
                "banana bread",
                "banana bread",
                "milk",
                "milk",
                "milk",
                "milk",  # milk/Tesco appears twice: an older and the latest date
                "cheddar cheese",
            ],
            "supermarket": ["Tesco", "ASDA", "Aldi", "Tesco", "ASDA", "Tesco", "Tesco", "ASDA", "Aldi", "Tesco"],
            "date": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-10",
                ]
            ),
            "avg_price": [0.99, 1.05, 0.85, 2.5, 2.6, 1.1, 1.2, 1.15, 1.1, 3.0],
            "min_price": [0.99, 1.05, 0.85, 2.5, 2.6, 1.1, 1.2, 1.15, 1.1, 3.0],
            "max_price": [0.99, 1.05, 0.85, 2.5, 2.6, 1.1, 1.2, 1.15, 1.1, 3.0],
            "n_listings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    fact_price_daily.to_parquet(tmp_path / "fact_price_daily.parquet")

    return tmp_path


class TestSearchProducts:
    def test_matches_substring_case_insensitive(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.search_products("BANANA")
        assert set(result["canonical_name"]) == {"bananas", "banana bread"}

    def test_respects_limit(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.search_products("a", limit=1)
        assert len(result) == 1

    def test_no_match_returns_empty(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.search_products("nonexistent_product_xyz")
        assert len(result) == 0

    def test_missing_mart_raises_clear_error(self, tmp_path):
        with Warehouse(tmp_path) as wh, pytest.raises(MartNotFoundError):
            wh.search_products("bananas")


class TestGetMarketOverview:
    def test_one_row_per_retailer(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_market_overview()
        assert set(result["supermarket"]) == {"Tesco", "ASDA", "Aldi"}
        assert len(result) == 3

    def test_portfolio_size_counts_distinct_products(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_market_overview()
        tesco = result[result["supermarket"] == "Tesco"].iloc[0]
        # Tesco appears in fact_price_daily for: bananas, banana bread, milk, cheddar cheese
        assert tesco["portfolio_size"] == 4

    def test_own_brand_pct_computed_correctly(self, marts_dir):
        """own_brand_pct is deliberately row-weighted (one fact_price_daily
        row per date), matching the existing dashboard's
        df.groupby('supermarket')['own_brand'].mean() over full row-level
        data -- a product tracked across more dates counts more, same as
        today. Tesco's 5 fact rows: bananas(False), banana bread(True),
        milk@01-09(False), milk@01-10(False), cheddar cheese(True) -> 2/5."""
        with Warehouse(marts_dir) as wh:
            result = wh.get_market_overview()
        tesco = result[result["supermarket"] == "Tesco"].iloc[0]
        assert tesco["own_brand_pct"] == pytest.approx(40.0)

    def test_price_columns_present_and_ordered(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_market_overview()
        for col in ("min_price", "price_p25", "price_median", "price_p75", "max_price"):
            assert col in result.columns
        row = result.iloc[0]
        assert row["min_price"] <= row["price_p25"] <= row["price_median"] <= row["price_p75"] <= row["max_price"]


class TestGetBasketCost:
    def test_returns_cost_and_coverage_per_retailer(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_basket_cost(["bananas", "milk"])
        # Tesco has both bananas (0.99, latest 2024-01-10) and milk (1.2, latest 2024-01-10)
        tesco = result[result["supermarket"] == "Tesco"].iloc[0]
        assert tesco["basket_cost"] == pytest.approx(0.99 + 1.2)
        assert tesco["items_found"] == 2

    def test_only_uses_latest_date(self, marts_dir):
        """milk's Tesco price on 2024-01-09 (1.1) must not be included --
        only the latest date (2024-01-10, avg_price=1.2) should count."""
        with Warehouse(marts_dir) as wh:
            result = wh.get_basket_cost(["milk"])
        tesco = result[result["supermarket"] == "Tesco"].iloc[0]
        assert tesco["basket_cost"] == pytest.approx(1.2)

    def test_partial_coverage_not_zero_filled(self, marts_dir):
        """cheddar cheese is only stocked by Tesco (n_retailers=1) --
        ASDA/Aldi must be silently excluded from the result, not
        zero-filled, per the "never zero-fill silently" principle."""
        with Warehouse(marts_dir) as wh:
            result = wh.get_basket_cost(["cheddar cheese"])
        assert set(result["supermarket"]) == {"Tesco"}

    def test_empty_basket_returns_empty_result(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_basket_cost([])
        assert len(result) == 0
        assert list(result.columns) == ["supermarket", "basket_cost", "items_found"]

    def test_unknown_product_returns_empty_result(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            result = wh.get_basket_cost(["not_a_real_product"])
        assert len(result) == 0


class TestWarehouseLifecycle:
    def test_context_manager_closes_connection(self, marts_dir):
        with Warehouse(marts_dir) as wh:
            wh.search_products("bananas")
        # Connection should be closed; further use should fail loudly,
        # not silently return stale/empty results.
        with pytest.raises(Exception):  # noqa: B017 -- duckdb's own exception type, not asserting which
            wh._conn.execute("SELECT 1").fetchall()
