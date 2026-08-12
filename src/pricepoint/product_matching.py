"""Semantic product matching pipeline.

Provides text normalization, Sentence-BERT embedding generation,
FAISS similarity search, and canonical product assignment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pricepoint.config import Settings
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory
from pricepoint.schemas import CANONICAL_PRODUCTS_SCHEMA

logger = logging.getLogger(__name__)


# Text normalisation (migrated from src/data_processing.py)
_BRANDS_TO_REMOVE = frozenset(["tesco", "asda", "sainsburys", "saintsburys", "morrisons", "aldi"])

_UNIT_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(kg|g|ml|l|m|pack|pk|x\d+(\.\d+)?\s?(kg|g|ml|l)?|x)\b",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalise_product_name(name: str | None) -> str:
    """Aggressively normalise a product name for matching.

    Steps:
    1. Lowercase
    2. Strip units (e.g., ``500g``, ``1.5l``, ``6x``)
    3. Remove punctuation
    4. Remove known retailer brand names
    5. Collapse whitespace

    Parameters
    ----------
    name : str or None
        Raw product name.

    Returns
    -------
    str
        Normalised product name (empty string if input is invalid).
    """
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = _UNIT_PATTERN.sub("", name)
    name = _PUNCTUATION_PATTERN.sub("", name)
    for brand in _BRANDS_TO_REMOVE:
        name = name.replace(brand, "")
    name = _WHITESPACE_PATTERN.sub(" ", name).strip()
    return name


# Embedding & matching pipeline
def generate_embeddings(
    product_names: pd.Series,
    model_name: str = "intfloat/e5-large",
    batch_size: int = 256,
) -> np.ndarray:
    """Generate Sentence-BERT embeddings for product names.

    Parameters
    ----------
    product_names : pd.Series
        Series of product name strings.
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(n_products, embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    names = product_names.tolist()
    logger.info("Encoding %s product names …", f"{len(names):,}")
    embeddings = model.encode(names, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    logger.info("Embeddings generated. Shape: %s", embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS inner-product index from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Normalised embedding matrix.

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index ready for search.
    """
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32, copy=False))
    logger.info("FAISS index built with %s vectors (dim=%s).", f"{index.ntotal:,}", dim)
    return index


class _UnionFind:
    """Minimal union-find (disjoint-set) with path compression + union by rank.

    Used to turn a set of pairwise "similar enough" edges into connected
    components, i.e. transitively-closed clusters, in near-linear time.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1


def cluster_by_similarity(
    embeddings: np.ndarray,
    names: list[str],
    threshold: float,
    mutual_k: int = 5,
) -> dict[str, str]:
    """Cluster names by embedding similarity via connected components over a
    bounded-degree mutual-nearest-neighbour graph.

    An early version of this function built the similarity graph from FAISS
    ``range_search`` directly: an edge between *every* pair of names with
    cosine similarity >= ``threshold``. That fixes the original bug (the old
    greedy top-1 algorithm was non-transitive and order-dependent — see
    ENGINEERING_AUDIT_REPORT.md §1.6 / ADR-0007), but introduces a worse one:
    ``range_search`` gives each node *unbounded* degree, and on the real
    114K-name corpus this let a handful of generic short phrases ("0 fat
    greek style", …) act as hubs that chained together completely unrelated
    products — 96% of all names collapsed into a single cluster at the
    original 0.85 threshold, and even raising the threshold to 0.90 still
    left 60%+ of the corpus in one giant cluster (verified empirically,
    Phase 1 Progress Log 2026-07-02). This is a well-known failure mode of
    single-linkage/connected-components clustering ("chaining"), and it gets
    *worse*, not better, as the corpus grows — a threshold that looks safe
    on a small sample can still fail at full scale.

    The fix is to bound each node's degree: only the ``mutual_k`` nearest
    neighbours of a name are ever considered as candidate edges, and an edge
    is only kept if the relationship is *mutual* (A is one of B's
    ``mutual_k`` nearest neighbours AND B is one of A's). This caps how many
    other names any single "hub" phrase can ever pull into its cluster,
    while still fixing the original transitivity bug (A-B-C chains still
    connect when each hop is a genuine mutual nearest-neighbour pair).
    ``threshold`` + ``mutual_k`` = (0.95, 5) was empirically validated
    against the full real dataset to avoid chaining while reproducing a
    canonical-product count (~68,600) close to the project's historical
    headline figure (67,341) — see the Phase 1 Progress Log for the full
    investigation and the values that were tried and rejected.

    The canonical name for each cluster is its lexicographically smallest
    member, chosen deterministically rather than by iteration order.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalised embedding matrix, shape ``(n_names, dim)``, aligned
        1:1 with ``names``.
    names : list[str]
        Names corresponding to each embedding row.
    threshold : float
        Minimum cosine similarity for two names to be linked.
    mutual_k : int
        How many nearest neighbours of each name are considered as
        candidate edges. Bounds the maximum degree of any single node,
        which is what actually prevents chaining (the threshold alone does
        not, at corpus sizes like this one).

    Returns
    -------
    dict[str, str]
        Mapping from every input name to its cluster's canonical name.
    """
    n = len(names)
    if n == 0:
        return {}

    index = build_faiss_index(embeddings)
    k = min(mutual_k + 1, n)  # +1 for self; can't request more neighbours than exist
    distances, indices = index.search(embeddings.astype(np.float32, copy=False), k)

    neighbour_sets: list[set[int]] = [set(indices[i][1:].tolist()) for i in range(n)]

    uf = _UnionFind(n)
    for i in range(n):
        for pos in range(1, k):
            j = int(indices[i][pos])
            sim = distances[i][pos]
            if i == j or sim < threshold:
                continue
            if i in neighbour_sets[j]:  # mutual: i is also among j's nearest neighbours
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    canonical_map: dict[str, str] = {}
    for members in clusters.values():
        member_names = [names[m] for m in members]
        canonical = min(member_names)
        for name in member_names:
            canonical_map[name] = canonical

    return canonical_map


def find_canonical_matches(
    df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Run the full product matching pipeline.

    1. Normalise product names
    2. Generate embeddings
    3. Build a bounded-degree mutual-nearest-neighbour similarity graph and
       cluster it into connected components (see :func:`cluster_by_similarity`)
    4. Assign canonical product IDs (lexicographically smallest per cluster)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned supermarket data with a ``product_name`` column.
    settings : Settings
        Application settings.

    Returns
    -------
    pd.DataFrame
        Data with ``canonical_name`` column added.
    """
    df = df.copy()
    df["normalised_name"] = df["product_name"].apply(normalise_product_name)

    unique_names = df["normalised_name"].drop_duplicates().reset_index(drop=True)
    logger.info("Unique normalised names: %s", f"{len(unique_names):,}")
    log_memory("before embedding generation")

    embeddings = generate_embeddings(unique_names, model_name=settings.matching.model_name)
    log_memory("after embedding generation")

    threshold = settings.matching.similarity_threshold
    mutual_k = settings.matching.mutual_neighbors_k
    logger.info("Clustering by similarity (threshold=%.2f, mutual_k=%s) …", threshold, mutual_k)
    canonical_map = cluster_by_similarity(embeddings, unique_names.tolist(), threshold, mutual_k=mutual_k)
    # embeddings (~114K x 1024 float32, ~467 MB on the full dataset) and the
    # FAISS index built from them inside cluster_by_similarity are no longer
    # needed once canonical_map exists -- free them before building the
    # (also large) output DataFrame rather than holding both at once. Added
    # after real runs against the full dataset repeatedly froze a 15 GB
    # development machine -- see project_v2.md Phase 1 Progress Log.
    del embeddings
    collect_garbage()
    log_memory("after releasing embeddings")

    df["canonical_name"] = df["normalised_name"].map(canonical_map)
    matched = df["canonical_name"].notna().sum()
    logger.info("Canonical matching complete. %s / %s matched.", f"{matched:,}", f"{len(df):,}")
    downcast_dtypes(df)

    return df


def run_matching(settings: Settings) -> Path:
    """Execute the full product matching pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output canonical products Parquet file.
    """
    interim_path = settings.data.interim_dir / "cleaned_supermarket_data.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(f"Interim data not found at {interim_path}. Run ingestion first.")

    logger.info("Loading interim data from %s …", interim_path)
    df = pd.read_parquet(interim_path, engine="pyarrow")
    downcast_dtypes(df)

    df = find_canonical_matches(df, settings)

    # Validate against the canonical products schema before persisting
    logger.info("Validating against CANONICAL_PRODUCTS_SCHEMA …")
    df = CANONICAL_PRODUCTS_SCHEMA.validate(df, lazy=False)
    logger.info("Validation passed. ✓")

    output_dir = settings.data.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.matching.output_filename

    logger.info("Writing canonical products to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Product matching complete. Output: %s", output_path)

    del df
    collect_garbage()
    return output_path
