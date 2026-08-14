# ADR-0002: Polars for the pipeline's data-processing hot paths

**Status:** Accepted
**Date:** 2026-08-12

## Context

Two stages dominated pipeline wall-clock time on the real 791MB/9.5M-row
dataset:

1. **Feature engineering's rolling/lag statistics**
   (`add_temporal_features`) used
   `groupby(...).transform(lambda x: x.rolling(window).mean())`, run once
   per statistic (mean/std/max/min) per window size (4 stats × 3 windows
   = 12 full passes, plus lag/diff columns) — a Python-lambda-per-group
   pattern with no vectorized fast path. Identified in project_refactor.md
   §1 as "the single most concrete, benchmarkable target."
2. **CSV ingestion** (`load_raw_csvs`/`clean_raw_data`) used
   `pd.read_csv` × 5 + `pd.concat` + per-column pandas string cleaning
   over the full raw dataset.

The distributed-processing alternative (Spark/Dask) was explicitly
rejected first (project_refactor.md §1): after dtype downcasting, the
full dataset fits in ~1.2GB RAM — this was never a data-volume problem,
it was a single-machine vectorization and memory-discipline problem.

## Decision

Rewrite both hot paths on **Polars**, keeping the public function
signatures' *pandas-facing contract* unchanged where existing callers
depend on it (feature engineering's public API is still pandas in,
pandas out — the conversion to pandas happens once, at the end of
`add_temporal_features`, and again at the ingestion/Pandera-validation
boundary, consistent with the project's "convert to pandas only at the
sklearn/LightGBM/Pandera boundary" principle).

`add_temporal_features` now builds all rolling/lag/diff expressions as a
single Polars `.select()` over `.over()`-grouped expressions — one
multi-threaded, Rust-implemented pass instead of 12+ Python-level ones.

## Consequences

Measured on the real dataset (`docs/benchmarks.md`, Phase 2):

- Feature engineering rolling/lag, 10% sample: 51.42s → 0.34s (**151.7x**).
- Feature engineering rolling/lag, full 9.5M rows (Polars measured
  directly; pandas linearly extrapolated from the 10% sample, flagged as
  such, not presented as measured): ~514s → 6.32s (**~81x**).
- CSV ingestion (load + clean), full dataset: 14.77s → 4.98s (**3.0x**).
- Both rewrites were verified numerically identical to the pre-rewrite
  pandas implementation before being trusted (row-for-row output
  comparison across normal, date-gap, single-row-group, and
  oversized-window fixtures for feature engineering; row-for-row
  comparison against real raw CSV data for ingestion).
- No new dependency category introduced — Polars was already an implicit
  possibility given pandas' own ecosystem direction, and it directly
  replaces (not augments) the pandas code paths it touches.

## Alternatives considered

- **Spark / Dask** — rejected outright (project_refactor.md §1): this is
  not big-data-in-the-distributed-systems-sense; would add operational
  complexity that solves a problem this project doesn't have.
- **Numba/Cython-accelerated pandas** — not evaluated seriously; Polars'
  rolling/`.over()` expressions solve the exact shape of problem
  (grouped rolling windows) without hand-writing custom kernels.
