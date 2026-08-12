"""Tests for pricepoint/market_analysis.py::compute_price_leadership.

Covers the Phase 2 vectorization of what was previously a
leader x follower x product x lag nested Python loop calling
pandas.Series.corr() individually. Includes:
1. A known-ground-truth test: a synthetic dataset where one store's
   prices are an exact N-day-delayed copy of another's, so the correct
   leader/follower/lag answer is known in advance.
2. A tolerance-based regression test comparing the vectorized
   implementation against the pre-Phase-2 nested-loop reference on a
   shared random fixture, per project_refactor.md Phase 2's testing
   requirement.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pricepoint.market_analysis import compute_price_leadership


class _FakeMarketDynamicsConfig:
    def __init__(self, sample_size=50, max_lag_days=7, min_correlation=0.15, min_stores_for_common=3):
        self.sample_size = sample_size
        self.max_lag_days = max_lag_days
        self.min_correlation = min_correlation
        self.min_stores_for_common = min_stores_for_common


class _FakeSettings:
    def __init__(self, **kwargs):
        self.market_dynamics = _FakeMarketDynamicsConfig(**kwargs)


def _old_compute_price_leadership(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Pre-Phase-2 nested-loop reference implementation.

    Kept only in this test module as the "old" side of an old-vs-new
    regression comparison for the vectorization in market_analysis.py --
    not used in production code.

    Uses the same MIN_VARIANCE_FOR_CORRELATION-based zero-variance check as
    the production function (rather than the original literal `var() ==
    0`), so this comparison isolates the vectorization itself: the
    epsilon fix is a separate, deliberate behaviour change (see
    market_analysis.py's `_MIN_VARIANCE_FOR_CORRELATION` for why an exact
    `== 0` check does not reliably catch degenerate near-constant price
    series on real data), not something this regression test is meant to
    catch. See TestDegenerateVariance below for a test of that fix
    specifically.
    """
    cfg = settings.market_dynamics
    min_variance = 1e-6

    product_counts = df.groupby("canonical_name", observed=True)["supermarket"].nunique()
    common = product_counts[product_counts >= cfg.min_stores_for_common].index
    if len(common) == 0:
        return pd.DataFrame(columns=["leader", "follower", "median_lag_days", "n_products_analyzed"])

    sampled = np.random.choice(common, min(cfg.sample_size, len(common)), replace=False)

    pivot = (
        df[df["canonical_name"].isin(sampled)]
        .pivot_table(index="date", columns=["supermarket", "canonical_name"], values="prices")
        .ffill()
    )

    supermarkets = df["supermarket"].unique()
    results: list[dict] = []

    for leader in supermarkets:
        for follower in supermarkets:
            if leader == follower:
                continue
            lags: list[int] = []
            for product in sampled:
                try:
                    if (leader, product) not in pivot.columns or (follower, product) not in pivot.columns:
                        continue
                    s1 = pivot[leader][product]
                    s2 = pivot[follower][product]
                    if s1.isnull().any() or s2.isnull().any() or s1.var() <= min_variance or s2.var() <= min_variance:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corrs = [s1.corr(s2.shift(lag)) for lag in range(-cfg.max_lag_days, cfg.max_lag_days + 1)]
                    if np.nanmax(np.abs(corrs)) > cfg.min_correlation:
                        lag_val = np.arange(-cfg.max_lag_days, cfg.max_lag_days + 1)[np.nanargmax(np.abs(corrs))]
                        lags.append(int(lag_val))
                except (KeyError, ValueError):
                    continue
            if lags:
                results.append(
                    {
                        "leader": leader,
                        "follower": follower,
                        "median_lag_days": float(np.median(lags)),
                        "n_products_analyzed": len(lags),
                    }
                )

    result_df = pd.DataFrame(results, columns=["leader", "follower", "median_lag_days", "n_products_analyzed"])
    result_df = result_df[result_df["median_lag_days"] != 0].copy()
    return result_df


def _make_random_market_df(n_products=30, n_stores=4, n_days=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stores = [f"Store{i}" for i in range(n_stores)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    rows = []
    for p in range(n_products):
        base = rng.uniform(1.0, 10.0)
        base_series = base + np.cumsum(rng.normal(0, 0.05, n_days))
        for store in stores:
            noise = rng.normal(0, 0.1, n_days)
            prices = np.clip(base_series + noise, 0.01, None)
            for d, price in zip(dates, prices, strict=True):
                rows.append(
                    {
                        "canonical_name": f"product_{p}",
                        "supermarket": store,
                        "date": d,
                        "prices": round(float(price), 4),
                    }
                )
    return pd.DataFrame(rows)


class TestComputePriceLeadershipKnownGroundTruth:
    """A dataset with a hand-constructed lag relationship: 'Follower' copies
    'Leader's prices exactly, delayed by 3 days. The function should
    recover leader=Leader, follower=Follower, median_lag_days=-3 (i.e.
    Follower's price at time t best matches Leader's price at time t-3,
    which is how compute_price_leadership defines lag sign via
    s1.corr(s2.shift(lag)) with s1=leader)."""

    @pytest.fixture
    def lagged_df(self):
        rng = np.random.default_rng(42)
        n_days = 80
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        n_products = 20
        rows = []
        for p in range(n_products):
            base_prices = np.clip(5.0 + np.cumsum(rng.normal(0, 0.3, n_days)), 0.5, None)
            leader_prices = base_prices
            # Follower's price at day t equals Leader's price at day t-3
            follower_prices = np.roll(base_prices, 3)
            follower_prices[:3] = base_prices[0]  # pad the undefined lead-in

            # A third, uncorrelated store to make sure it's NOT flagged as a pair
            noise_prices = np.clip(5.0 + np.cumsum(rng.normal(0, 0.3, n_days)), 0.5, None)

            for d, lp, fp, np_ in zip(dates, leader_prices, follower_prices, noise_prices, strict=True):
                rows.append({"canonical_name": f"prod_{p}", "supermarket": "Leader", "date": d, "prices": round(lp, 4)})
                rows.append(
                    {"canonical_name": f"prod_{p}", "supermarket": "Follower", "date": d, "prices": round(fp, 4)}
                )
                rows.append(
                    {"canonical_name": f"prod_{p}", "supermarket": "NoiseStore", "date": d, "prices": round(np_, 4)}
                )
        return pd.DataFrame(rows)

    def test_recovers_known_lag_relationship(self, lagged_df):
        settings = _FakeSettings(sample_size=20, max_lag_days=7, min_correlation=0.5, min_stores_for_common=2)
        result = compute_price_leadership(lagged_df, settings)

        pair = result[(result["leader"] == "Leader") & (result["follower"] == "Follower")]
        assert len(pair) == 1, "Expected exactly one Leader->Follower row"
        assert pair.iloc[0]["median_lag_days"] == pytest.approx(-3.0)
        assert pair.iloc[0]["n_products_analyzed"] > 0

    def test_output_schema(self, lagged_df):
        settings = _FakeSettings(sample_size=20, max_lag_days=7, min_correlation=0.5, min_stores_for_common=2)
        result = compute_price_leadership(lagged_df, settings)
        assert list(result.columns) == ["leader", "follower", "median_lag_days", "n_products_analyzed"]

    def test_zero_lag_pairs_excluded(self, lagged_df):
        """A store correlated with itself at lag 0 should never appear
        (and isn't, since leader==follower is skipped) -- also verifies no
        spurious median_lag_days == 0 rows leak through for any pair."""
        settings = _FakeSettings(sample_size=20, max_lag_days=7, min_correlation=0.5, min_stores_for_common=2)
        result = compute_price_leadership(lagged_df, settings)
        assert (result["median_lag_days"] != 0).all()


class TestDegenerateVariance:
    """Regression tests for the _MIN_VARIANCE_FOR_CORRELATION fix.

    Discovered while vectorizing: on real data, a bit-exact
    `pandas.Series.var() == 0` check unreliably identifies constant-price
    products -- variance computed over ~90+ repeats of a
    non-power-of-2-representable float (e.g. 1.95) accumulates
    floating-point rounding error into a spuriously nonzero value like
    7.98e-31, so `== 0` (equivalently `> 0` on the complement) silently
    lets these degenerate products through. Every lag's correlation is
    then ~1.0 (an arbitrary tie), and which lag "wins" becomes sensitive
    to sub-1e-15 differences between computation paths.
    """

    def test_near_constant_price_product_is_excluded(self):
        """A product whose price is bit-identical to itself ~90+ times
        (mirroring the real-data pattern that exposed this) must not
        produce a leadership signal, regardless of tiny floating-point
        residue in its computed variance."""
        n_days = 92
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        rows = []
        for store in ["StoreA", "StoreB", "StoreC"]:
            for d in dates:
                rows.append({"canonical_name": "flat_price_item", "supermarket": store, "date": d, "prices": 1.95})
        df = pd.DataFrame(rows)

        # Sanity-check the premise: pandas' own var() is not exactly 0 here.
        flat_series = df[df["supermarket"] == "StoreA"]["prices"]
        assert flat_series.var() != 0.0
        assert flat_series.var() < 1e-6

        settings = _FakeSettings(sample_size=5, max_lag_days=7, min_correlation=0.0, min_stores_for_common=2)
        result = compute_price_leadership(df, settings)
        assert len(result) == 0, "A genuinely constant-price product must not produce any leadership pair"

    def test_does_not_suppress_genuinely_small_but_real_variation(self):
        """A product with a single 1-penny price change (the smallest
        real-world price movement) must still be analysed -- the fix
        must not be so aggressive it swallows real, small signals.
        StoreB's price change is offset by 2 days from StoreA's so there
        is a genuine, non-zero best lag to detect (two series that change
        on the exact same day have a true best lag of 0, which the
        function correctly excludes as "no leadership relationship" --
        that exclusion is not what this test is checking)."""
        n_days = 92
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        rows = []
        for store, prices in [
            ("StoreA", [1.95] * 46 + [1.96] * 46),
            ("StoreB", [1.95] * 48 + [1.96] * 44),  # same jump, 2 days later
        ]:
            for d, p in zip(dates, prices, strict=True):
                rows.append({"canonical_name": "small_change_item", "supermarket": store, "date": d, "prices": p})
        df = pd.DataFrame(rows)

        real_variance = df[df["supermarket"] == "StoreA"]["prices"].var()
        assert real_variance > 1e-6, "This test's premise (a real, above-threshold variance) must hold"

        settings = _FakeSettings(sample_size=5, max_lag_days=3, min_correlation=0.0, min_stores_for_common=2)
        result = compute_price_leadership(df, settings)
        assert len(result) >= 1, "A product with genuine (if small) price variation must still be analysed"


class TestComputePriceLeadershipEdgeCases:
    def test_no_common_products_returns_empty_with_correct_schema(self):
        df = pd.DataFrame(
            {
                "canonical_name": ["a", "b"],
                "supermarket": ["Tesco", "ASDA"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "prices": [1.0, 2.0],
            }
        )
        settings = _FakeSettings(min_stores_for_common=3)
        result = compute_price_leadership(df, settings)
        assert list(result.columns) == ["leader", "follower", "median_lag_days", "n_products_analyzed"]
        assert len(result) == 0

    def test_no_qualifying_pairs_returns_empty_with_correct_schema(self):
        """Uncorrelated random noise across stores: no pair should clear
        min_correlation, exercising the previously-buggy all-empty path
        (pd.DataFrame([]) has zero columns; fixed to always specify them)."""
        df = _make_random_market_df(n_products=5, n_stores=3, n_days=20, seed=1)
        settings = _FakeSettings(sample_size=5, max_lag_days=2, min_correlation=0.999, min_stores_for_common=2)
        result = compute_price_leadership(df, settings)
        assert list(result.columns) == ["leader", "follower", "median_lag_days", "n_products_analyzed"]
        assert len(result) == 0
        # Must not raise KeyError on result["median_lag_days"] when empty.


class TestComputePriceLeadershipRegression:
    """Tolerance-based old-vs-new comparison on shared random fixtures."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference_implementation(self, seed):
        df = _make_random_market_df(n_products=25, n_stores=4, n_days=50, seed=seed)
        settings = _FakeSettings(sample_size=25, max_lag_days=5, min_correlation=0.1, min_stores_for_common=3)

        np.random.seed(123)
        new_result = compute_price_leadership(df, settings)
        np.random.seed(123)
        old_result = _old_compute_price_leadership(df, settings)

        new_sorted = new_result.sort_values(["leader", "follower"]).reset_index(drop=True)
        old_sorted = old_result.sort_values(["leader", "follower"]).reset_index(drop=True)

        assert new_sorted["leader"].tolist() == old_sorted["leader"].tolist()
        assert new_sorted["follower"].tolist() == old_sorted["follower"].tolist()
        assert new_sorted["n_products_analyzed"].tolist() == old_sorted["n_products_analyzed"].tolist()
        np.testing.assert_allclose(
            new_sorted["median_lag_days"].to_numpy(),
            old_sorted["median_lag_days"].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
        )
