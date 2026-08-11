"""Tests for the product matching module."""

from __future__ import annotations

import math

import numpy as np

from pricepoint.product_matching import cluster_by_similarity, normalise_product_name


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
        embeddings = np.array(
            [_unit_vector(0), _unit_vector(1), _unit_vector(2)], dtype=np.float32
        )

        canonical_map = cluster_by_similarity(embeddings, names, threshold=0.85)

        assert canonical_map["zzz_product"] == "aaa_product"
        assert canonical_map["aaa_product"] == "aaa_product"
        assert canonical_map["mmm_product"] == "aaa_product"

    def test_clustering_is_order_independent(self):
        """The old greedy top-1 algorithm's output depended on the order
        items were iterated in; connected-component clustering must not.
        """
        names_forward = ["product_a", "product_b", "product_c"]
        embeddings_forward = np.array(
            [_unit_vector(0), _unit_vector(20), _unit_vector(40)], dtype=np.float32
        )
        names_reversed = list(reversed(names_forward))
        embeddings_reversed = np.array(
            [_unit_vector(40), _unit_vector(20), _unit_vector(0)], dtype=np.float32
        )

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
