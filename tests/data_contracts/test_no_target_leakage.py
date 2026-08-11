"""Regression test guarding against target leakage in competitive features.

Why this test exists
---------------------
``pricepoint/feature_engineering.py::add_competitive_features`` used to compute
``market_avg_price`` (and, by extension, ``price_vs_market_avg`` and
``price_rank``) from a same-day cross-retailer aggregate that included the
row's own price. Since the model's target *is* that same row's price, the
row's own value was leaking into a feature meant to describe the market
*around* it. See ``ENGINEERING_AUDIT_REPORT.md`` §1.5 and ``project_v2.md``
ADR-0006 for the full write-up.

The fixture and expectations below are computed independently of
``add_competitive_features``'s implementation (a plain hand-rolled
leave-one-out loop), so this test is a genuine external check, not a mirror
of the implementation. It is written and run against the *pre-fix* code
first to prove it actually fails there, before the fix is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricepoint.feature_engineering import add_competitive_features


@pytest.fixture
def competitive_fixture() -> pd.DataFrame:
    """Three canonical-product groups exercising the leakage-relevant edge cases.

    - "bananas" (3 retailers, distinct prices): the case that actually
      distinguishes leave-one-out statistics from self-inclusive ones.
    - "milk" (1 retailer): the n == 1 edge case -- there is no "market" to
      compare against, so the market stats must be NaN, not a vacuous
      self-comparison.
    - "bread" (2 retailers, tied price): exercises dense-rank tie handling.
    """
    rows = [
        {"canonical_name": "bananas", "supermarket": "Tesco", "date": "2024-01-01", "prices": 1.00},
        {"canonical_name": "bananas", "supermarket": "ASDA", "date": "2024-01-01", "prices": 1.20},
        {"canonical_name": "bananas", "supermarket": "Aldi", "date": "2024-01-01", "prices": 1.50},
        {"canonical_name": "milk", "supermarket": "Tesco", "date": "2024-01-01", "prices": 2.00},
        {"canonical_name": "bread", "supermarket": "Tesco", "date": "2024-01-01", "prices": 1.00},
        {"canonical_name": "bread", "supermarket": "ASDA", "date": "2024-01-01", "prices": 1.00},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _hand_computed_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """Independent (non-implementation) leave-one-out expectations.

    For each row, excludes that row from its (canonical_name, date) group
    before computing the market average and the rank -- this is the
    definition of "leave-one-out", not an optimisation of it.
    """
    market_avg_vs_price: list[float] = []
    price_rank: list[float] = []
    is_cheapest: list[int] = []

    for idx, row in df.iterrows():
        group = df[(df["canonical_name"] == row["canonical_name"]) & (df["date"] == row["date"])]
        others = group.drop(index=idx)

        if len(others) == 0:
            market_avg_vs_price.append(np.nan)
            price_rank.append(np.nan)
            is_cheapest.append(0)
            continue

        avg_of_others = others["prices"].mean()
        market_avg_vs_price.append(row["prices"] - avg_of_others)

        rank = int((others["prices"] < row["prices"]).sum()) + 1
        price_rank.append(rank)
        is_cheapest.append(1 if rank == 1 else 0)

    expected = df.copy()
    expected["expected_price_vs_market_avg"] = market_avg_vs_price
    expected["expected_price_rank"] = price_rank
    expected["expected_is_cheapest_in_market"] = is_cheapest
    return expected


class TestNoTargetLeakageInCompetitiveFeatures:
    """Guards ADR-0006: competitive features must be leave-one-out, never self-inclusive."""

    def test_price_vs_market_avg_matches_leave_one_out(self, competitive_fixture):
        expected = _hand_computed_leave_one_out(competitive_fixture)
        result = add_competitive_features(competitive_fixture)

        np.testing.assert_allclose(
            result["price_vs_market_avg"].to_numpy(dtype=float),
            expected["expected_price_vs_market_avg"].to_numpy(dtype=float),
            atol=1e-9,
            equal_nan=True,
        )

    def test_price_vs_market_avg_does_not_match_self_inclusive_average(self, competitive_fixture):
        """This is the assertion that would have caught the original bug.

        For the "bananas" group (three distinct prices), a self-inclusive
        mean gives a different -- wrong -- answer than the leave-one-out
        mean. If this assertion ever starts failing, ``add_competitive_features``
        has regressed back to including each row's own price in its own
        "market average".
        """
        result = add_competitive_features(competitive_fixture)
        bananas = result[result["canonical_name"] == "bananas"].sort_values("supermarket")

        self_inclusive_avg = competitive_fixture.loc[
            competitive_fixture["canonical_name"] == "bananas", "prices"
        ].mean()  # (1.00 + 1.20 + 1.50) / 3 == 1.2333...
        self_inclusive_delta = bananas["prices"] - self_inclusive_avg

        assert not np.allclose(
            bananas["price_vs_market_avg"].to_numpy(dtype=float),
            self_inclusive_delta.to_numpy(dtype=float),
        )

    def test_single_retailer_product_yields_nan_market_stats(self, competitive_fixture):
        result = add_competitive_features(competitive_fixture)
        milk = result[result["canonical_name"] == "milk"].iloc[0]

        assert pd.isna(milk["price_vs_market_avg"])
        assert pd.isna(milk["price_rank"])
        assert milk["is_cheapest_in_market"] == 0

    def test_price_rank_matches_leave_one_out(self, competitive_fixture):
        expected = _hand_computed_leave_one_out(competitive_fixture)
        result = add_competitive_features(competitive_fixture)

        np.testing.assert_allclose(
            result["price_rank"].to_numpy(dtype=float),
            expected["expected_price_rank"].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_is_cheapest_flag_matches_leave_one_out(self, competitive_fixture):
        expected = _hand_computed_leave_one_out(competitive_fixture)
        result = add_competitive_features(competitive_fixture)

        np.testing.assert_array_equal(
            result["is_cheapest_in_market"].to_numpy(dtype=int),
            expected["expected_is_cheapest_in_market"].to_numpy(dtype=int),
        )

    def test_market_avg_price_is_not_persisted_as_a_raw_feature(self, competitive_fixture):
        """A raw market-average column co-present with ``price_vs_market_avg``
        would let the target be reconstructed *exactly* via
        ``price = market_avg_price + price_vs_market_avg`` -- regardless of
        how market_avg_price itself is computed, since price_vs_market_avg is
        defined as price minus it. This is a structurally separate leak from
        self-inclusion (see the ADR-0006 addendum in ``project_v2.md``), so
        only the derived delta is ever exposed as a column.
        """
        result = add_competitive_features(competitive_fixture)
        assert "market_avg_price" not in result.columns
