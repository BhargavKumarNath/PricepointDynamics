# ADR-0007: Bounded-degree mutual-nearest-neighbour clustering for product matching

**Status:** Accepted
**Date:** pre-existing decision, formalized here

## Context

Matching retailer-specific product names (e.g. "Tesco Finest Bananas
5pk") to a canonical identity requires clustering ~114K unique normalised
names by Sentence-BERT embedding similarity. Two prior approaches both
failed:

1. **Greedy top-1 nearest-neighbour matching** — non-transitive and
   order-dependent (a bug found and fixed with a real regression test).
2. **FAISS `range_search` connected components** — an edge between
   *every* pair of names with cosine similarity ≥ threshold. This fixes
   the transitivity bug, but gives each node **unbounded degree**: on the
   real 114K-name corpus, a handful of generic short phrases (e.g. "0 fat
   greek style") acted as hub nodes that chained together completely
   unrelated products. At the original 0.85 threshold, 96% of all names
   collapsed into a single cluster; raising the threshold to 0.90 still
   left 60%+ of the corpus in one giant cluster. This is the well-known
   "chaining" failure mode of single-linkage/connected-components
   clustering, and it gets *worse*, not better, as the corpus grows — a
   threshold that looks safe on a small sample can still fail at full
   scale.

## Decision

Cluster via connected components over a **bounded-degree mutual-nearest-
neighbour graph** (`cluster_by_similarity`, backed by a union-find with
path compression): only each name's `mutual_k` nearest neighbours are
ever considered as candidate edges, and an edge is kept only if the
relationship is **mutual** (A is one of B's `mutual_k` nearest neighbours
*and* B is one of A's). This bounds how many other names any single
"hub" phrase can ever pull into its cluster, while still fixing the
original transitivity bug — genuine A–B–C chains still connect, since
each hop is a real mutual nearest-neighbour pair.

`threshold=0.95, mutual_k=5` was empirically validated against the full
real dataset to avoid chaining while reproducing a canonical-product
count (~68,600) close to the project's historical headline figure
(67,341).

The canonical name for each cluster is its lexicographically smallest
member, chosen deterministically rather than by iteration order.

## Consequences

- Guarded by a regression test targeting exactly the chaining failure
  mode (a bounded-degree hub-node clustering test in
  `tests/test_product_matching.py`), not just a happy-path clustering
  check.
- `mutual_k` directly bounds per-node degree, which is what actually
  prevents chaining — the similarity threshold alone does not, at corpus
  sizes like this one.
- `(threshold, mutual_k) = (0.95, 5)` are the real, unchanged production
  values reused in `tests/test_pipeline_integration.py`'s fixture
  clustering test, rather than a separately-invented test configuration —
  keeping the integration test honest about what production actually
  runs.

## Alternatives considered

- **Greedy top-1 matching (original)** — rejected; non-transitive and
  order-dependent.
- **Unbounded connected components (`range_search`)** — rejected; fixes
  transitivity but causes catastrophic chaining at this corpus's scale.
- **A global clustering algorithm (e.g. HDBSCAN) over the full embedding
  space** — not adopted; the bounded-degree mutual-kNN approach is
  cheaper (FAISS `IndexFlatIP` + a `k`-nearest-neighbour search, not a
  full pairwise distance computation) and its failure mode (chaining) was
  already understood and directly addressed, rather than trading it for a
  different algorithm's own tuning surface.
