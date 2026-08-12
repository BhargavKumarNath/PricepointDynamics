"""Tests for the product matching module."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

import pricepoint.product_matching as product_matching_module
from pricepoint.product_matching import cluster_by_similarity, normalise_product_name, run_matching


class TestNormaliseProductName:
    """Tests for normalise_product_name."""

    # ---- Basic normalisation ----

    def test_lowercase(self):
        assert normalise_product_name("HELLO WORLD") == "hello world"

    def test_strip_units_grams(self):
        assert normalise_product_name("Chicken Breast 500g") == "chicken breast"

    def test_strip_units_kg(self):
        assert normalise_product_name("Potatoes 2.5kg") == "potatoes"

    def test_strip_units_litres(self):
        assert normalise_product_name("Milk 1l") == "milk"

    def test_strip_units_ml(self):
        assert normalise_product_name("Orange Juice 500ml") == "orange juice"

    def test_strip_units_pack(self):
        assert normalise_product_name("Bananas 5pack") == "bananas"

    def test_strip_units_x_multiplier(self):
        assert normalise_product_name("Crisps 6x25g") == "crisps"

    # ---- Brand removal ----

    def test_remove_tesco(self):
        assert normalise_product_name("Tesco Finest Bananas") == "finest bananas"

    def test_remove_asda(self):
        assert normalise_product_name("ASDA Whole Milk") == "whole milk"

    def test_remove_aldi(self):
        assert normalise_product_name("Aldi Nature's Pick Bananas") == "natures pick bananas"

    def test_remove_morrisons(self):
        assert normalise_product_name("Morrisons The Best Bread") == "the best bread"

    def test_remove_sainsburys(self):
        # The function removes "saintsburys" (original spelling)
        assert normalise_product_name("Saintsburys SO Organic Milk") == "so organic milk"

    # ---- Punctuation ----

    def test_remove_punctuation(self):
        result = normalise_product_name("Ben & Jerry's Ice Cream")
        assert "&" not in result
        assert "'" not in result

    # ---- Whitespace ----

    def test_collapse_whitespace(self):
        assert normalise_product_name("  too   many   spaces  ") == "too many spaces"

    # ---- Edge cases ----

    def test_none_input(self):
        assert normalise_product_name(None) == ""

    def test_empty_string(self):
        assert normalise_product_name("") == ""

    def test_numeric_input(self):
        assert normalise_product_name(12345) == ""

    def test_only_units(self):
        # After removing "500g", only whitespace remains
        assert normalise_product_name("500g") == ""

    # ---- Regression tests ----

    def test_real_product_tesco_bananas(self):
        result = normalise_product_name("Tesco Finest Bananas 5pk - 1.2kg")
        assert "tesco" not in result
        assert "5pk" not in result
        assert "bananas" in result

    def test_real_product_walkers_crisps(self):
        result = normalise_product_name("Walkers Meaty Variety Crisps 12x25g")
        assert "12x25g" not in result
        assert "walkers" in result  # Walkers is NOT a retailer brand


def _unit_vector(angle_degrees: float) -> list[float]:
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


class TestClusterBySimilarity:
    """Tests for the graph-based (connected components) matching that
    replaced greedy top-1 nearest-neighbour matching — see ADR-0007 in
    project_v2.md and ENGINEERING_AUDIT_REPORT.md §1.6.
    """

    def test_transitive_chain_forms_one_cluster(self):
        """A <-> B <-> C are each pairwise above threshold via adjacent hops
        (20 degrees apart), but A and C alone are 40 degrees apart — below
        threshold and NOT each other's nearest neighbour. The old greedy
        top-1 algorithm could miss this chain entirely: whichever of A/C
        reached B first in iteration order "claimed" it via
        ``neighbour_name not in canonical_map``, leaving the other stranded
        as its own singleton cluster. Connected components must not.
        """
        names = ["product_a", "product_b", "product_c", "product_d"]
        embeddings = np.array(
            [
                _unit_vector(0),
                _unit_vector(20),  # 20 degrees from A: cos ~= 0.94 (>= threshold)
                _unit_vector(40),  # 40 degrees from A: cos ~= 0.77 (< threshold vs A directly)
                _unit_vector(90),  # orthogonal to A: an unrelated product
            ],
            dtype=np.float32,
        )

        canonical_map = cluster_by_similarity(embeddings, names, threshold=0.85)

        # A, B, and C must all land in the same cluster despite A-C alone
        # being below threshold, because B bridges them.
        assert canonical_map["product_a"] == canonical_map["product_b"]
        assert canonical_map["product_b"] == canonical_map["product_c"]
        # The unrelated product must not be swept into that cluster.
        assert canonical_map["product_d"] != canonical_map["product_a"]

    def test_canonical_name_is_lexicographically_smallest(self):
        names = ["zzz_product", "aaa_product", "mmm_product"]
        embeddings = np.array([_unit_vector(0), _unit_vector(1), _unit_vector(2)], dtype=np.float32)

        canonical_map = cluster_by_similarity(embeddings, names, threshold=0.85)

        assert canonical_map["zzz_product"] == "aaa_product"
        assert canonical_map["aaa_product"] == "aaa_product"
        assert canonical_map["mmm_product"] == "aaa_product"

    def test_clustering_is_order_independent(self):
        """The old greedy top-1 algorithm's output depended on the order
        items were iterated in; connected-component clustering must not.
        """
        names_forward = ["product_a", "product_b", "product_c"]
        embeddings_forward = np.array([_unit_vector(0), _unit_vector(20), _unit_vector(40)], dtype=np.float32)
        names_reversed = list(reversed(names_forward))
        embeddings_reversed = np.array([_unit_vector(40), _unit_vector(20), _unit_vector(0)], dtype=np.float32)

        forward_map = cluster_by_similarity(embeddings_forward, names_forward, threshold=0.85)
        reversed_map = cluster_by_similarity(embeddings_reversed, names_reversed, threshold=0.85)

        for name in names_forward:
            assert forward_map[name] == reversed_map[name]

    def test_unrelated_products_stay_in_separate_clusters(self):
        names = ["bananas", "milk"]
        embeddings = np.array([_unit_vector(0), _unit_vector(90)], dtype=np.float32)

        canonical_map = cluster_by_similarity(embeddings, names, threshold=0.85)

        assert canonical_map["bananas"] != canonical_map["milk"]

    def test_empty_input_returns_empty_mapping(self):
        result = cluster_by_similarity(np.empty((0, 2), dtype=np.float32), [], threshold=0.85)
        assert result == {}

    def test_hub_node_degree_is_bounded_not_all_absorbed(self):
        """Regression test for the real failure found in Phase 1 (project_v2.md,
        Progress Log 2026-07-02): with unbounded ``range_search``-based
        connected components, a single "hub" name that is independently
        similar to many otherwise-unrelated names pulled ALL of them into
        one giant cluster on the real 114K-product corpus (96% of all
        products collapsed into a single cluster at threshold=0.85, and 60%+
        still did at 0.90). ``mutual_k`` bounds each node's degree so a hub
        can only ever mutually-connect to at most ~``mutual_k`` others.

        Construction: one "hub" vector plus 20 "petal" vectors, each built
        from a shared component along the hub's axis (giving every petal
        the SAME similarity to the hub) plus a component along its own
        unique orthogonal axis (giving every pair of petals a LOWER,
        equal similarity to each other than to the hub). This is exactly
        the star/hub topology that caused the real bug: many points each
        independently close to a hub, but not close to each other.
        """
        n_petals = 20
        dim = n_petals + 1
        alpha = 0.95  # similarity of every petal to the hub
        threshold = 0.93  # chosen so alpha clears it but alpha**2 (petal-petal sim) does not

        hub = np.zeros(dim, dtype=np.float32)
        hub[0] = 1.0

        petals = np.zeros((n_petals, dim), dtype=np.float32)
        for i in range(n_petals):
            petals[i, 0] = alpha
            petals[i, i + 1] = np.sqrt(1 - alpha**2)

        embeddings = np.vstack([hub[None, :], petals]).astype(np.float32)
        names = ["hub"] + [f"petal_{i}" for i in range(n_petals)]

        canonical_map = cluster_by_similarity(embeddings, names, threshold=threshold, mutual_k=5)

        cluster_sizes: dict[str, int] = {}
        for canonical in canonical_map.values():
            cluster_sizes[canonical] = cluster_sizes.get(canonical, 0) + 1
        hub_cluster_size = cluster_sizes[canonical_map["hub"]]

        # Unbounded connected components would put all 21 names (hub + 20
        # petals) in one cluster, since every petal is independently above
        # threshold with the hub. Bounded mutual-k must not do that.
        assert hub_cluster_size < n_petals + 1, (
            f"hub cluster absorbed {hub_cluster_size} of {n_petals + 1} names -- "
            "degree bounding is not preventing hub explosion"
        )
        # Generous slack for FAISS's arbitrary tie-breaking among the
        # (tied) petal-to-hub similarities -- the invariant under test is
        # "bounded", not an exact count.
        assert hub_cluster_size <= 5 + 3


class _FakeDataConfig:
    def __init__(self, interim_dir, processed_dir):
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir


class _FakeMatchingConfig:
    def __init__(self, output_filename="canonical_products_e5.parquet"):
        self.output_filename = output_filename


class _FakeMatchingSettings:
    def __init__(self, interim_dir, processed_dir):
        self.data = _FakeDataConfig(interim_dir, processed_dir)
        self.matching = _FakeMatchingConfig()


def _fake_matched_output(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Stand-in for find_canonical_matches: skips SBERT/FAISS entirely,
    returning a small CANONICAL_PRODUCTS_SCHEMA-conformant frame so
    run_matching's own validate-and-write logic can be exercised for
    real."""
    return pd.DataFrame(
        {
            "canonical_name": ["bananas", "milk"],
            "supermarket": ["Tesco", "ASDA"],
            "prices": [1.0, 1.2],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "own_brand": [False, True],
        }
    )


class TestRunMatchingSkipIfUnchanged:
    """Integration tests for run_matching's manifest-based skip-if-unchanged
    wiring (product_matching.py previously never wrote a manifest at all,
    so this behaviour could not have worked before this fix). The
    expensive SBERT/FAISS step (find_canonical_matches) is monkeypatched
    out -- these tests are about the skip/force/manifest wiring around it,
    not the matching algorithm itself (covered by TestClusterBySimilarity)."""

    @pytest.fixture(autouse=True)
    def _stub_matching(self, monkeypatch):
        monkeypatch.setattr(product_matching_module, "find_canonical_matches", _fake_matched_output)

    @pytest.fixture
    def settings(self, tmp_path):
        interim_dir = tmp_path / "interim"
        interim_dir.mkdir()
        pd.DataFrame({"product_name": ["a", "b"]}).to_parquet(interim_dir / "cleaned_supermarket_data.parquet")
        processed_dir = tmp_path / "processed"
        return _FakeMatchingSettings(interim_dir=interim_dir, processed_dir=processed_dir)

    def test_writes_manifest_on_first_run(self, settings):
        """product_matching.py previously wrote no manifest at all --
        has_sources_changed() could never find one to compare against."""
        output_path = run_matching(settings)
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["stage"] == "product_matching"

    def test_second_run_skips_when_unchanged(self, settings, monkeypatch):
        first_path = run_matching(settings)
        assert first_path.exists()

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("find_canonical_matches was called on an unchanged re-run; skip logic did not trigger")

        monkeypatch.setattr(product_matching_module, "find_canonical_matches", _fail_if_called)

        second_path = run_matching(settings)
        assert second_path == first_path

    def test_force_reruns_even_when_unchanged(self, settings, monkeypatch):
        run_matching(settings)

        calls = []
        monkeypatch.setattr(
            product_matching_module,
            "find_canonical_matches",
            lambda *a, **kw: (calls.append(1), _fake_matched_output(*a, **kw))[1],
        )

        run_matching(settings, force=True)
        assert len(calls) == 1, "force=True must re-run matching even when the interim input is unchanged"

    def test_reruns_when_interim_data_modified(self, settings):
        first_path = run_matching(settings)
        first_manifest = json.loads(first_path.with_suffix(first_path.suffix + ".manifest.json").read_text())

        time.sleep(0.01)
        interim_path = settings.data.interim_dir / "cleaned_supermarket_data.parquet"
        pd.DataFrame({"product_name": ["a", "b", "c"]}).to_parquet(interim_path)

        second_path = run_matching(settings)
        second_manifest = json.loads(second_path.with_suffix(second_path.suffix + ".manifest.json").read_text())
        assert second_manifest["generated_at"] != first_manifest["generated_at"]
