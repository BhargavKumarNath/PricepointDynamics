"""Memory-usage utilities.

Added after real pipeline runs against the full ~9.5M-row dataset repeatedly
froze the development machine (15 GB total RAM) -- see project_v2.md Phase 1
Progress Log, 2026-07-02, for the investigation. Two complementary tools:

1. :func:`downcast_dtypes` reduces a DataFrame's memory footprint (float64 ->
   float32, int64 -> smallest safe integer type, low-cardinality object/str
   columns -> ``category``), applied at pipeline stage boundaries so the
   reduction compounds through the rest of the pipeline rather than being
   undone by the next stage reloading a full-precision Parquet file.
2. :func:`log_memory` logs current process RSS (if ``psutil`` is available)
   at pipeline stage boundaries, so memory pressure is visible in logs
   rather than discovered via a frozen machine.
"""

from __future__ import annotations

import gc
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def downcast_dtypes(df: pd.DataFrame, max_category_ratio: float = 0.5) -> pd.DataFrame:
    """Reduce a DataFrame's memory footprint by downcasting dtypes in place.

    - float64 -> float32: halves memory for numeric columns. This project's
      prices only ever need penny-level precision, nowhere near float64's
      ~15-17 significant digits.
    - int64 -> the smallest integer dtype that safely holds the column's
      actual range.
    - object/str columns -> ``category``, but only when fewer than
      ``max_category_ratio`` of values are unique (e.g. ``supermarket`` has
      5 distinct values across 9.5M rows -- category dtype stores each
      value once instead of once per row. ``canonical_name`` has ~69K
      distinct values across 9.5M rows, a ~0.7% ratio, and benefits hugely
      too, even though the absolute unique count is large -- a ratio
      threshold catches this where a fixed unique-count threshold would
      not).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to downcast. Modified in place (mutating columns) and
        also returned, so this can be used inline in an assignment chain.
    max_category_ratio : float
        Convert an object/str column to category if
        ``n_unique / len(df) < max_category_ratio``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, with dtypes downcast.
    """
    if len(df) == 0:
        return df

    before_mb = df.memory_usage(deep=True).sum() / 1e6

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["object", "str"]).columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique / len(df) < max_category_ratio:
            df[col] = df[col].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    reduction_pct = 100 * (1 - after_mb / before_mb) if before_mb else 0
    logger.info(
        "Downcast reduced memory footprint: %.0f MB -> %.0f MB (%.0f%% smaller)",
        before_mb,
        after_mb,
        reduction_pct,
    )
    return df


def log_memory(label: str) -> None:
    """Log current process RSS memory, if ``psutil`` is available.

    Never raises -- this is an observability aid, not something that should
    ever break a pipeline run.
    """
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 1e6
        logger.info("[memory] %s: process RSS = %.0f MB", label, rss_mb)
    except Exception:  # noqa: BLE001
        logger.debug("Could not read process memory for %r (psutil unavailable?).", label)


def collect_garbage() -> None:
    """Force an immediate garbage-collection pass.

    Python's refcounting GC usually reclaims memory promptly on its own, but
    pandas operations (rolling windows, groupby.transform, get_dummies,
    concat) routinely create multiple full-size intermediate copies that can
    otherwise linger until the next collection cycle. Callers should ``del``
    their own large local variables (e.g. ``del df``) *before* calling this
    -- deleting a variable inside this function only removes this
    function's own reference, not the caller's, so it would not actually
    free anything on its own.
    """
    gc.collect()
