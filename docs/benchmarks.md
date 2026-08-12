# Phase 2 Performance Benchmarks

Measured on this project's real data (5 raw retailer CSVs, 791MB, 9.5M rows;
the same `canonical_products_e5.parquet` used by the rest of the pipeline)
-- not synthetic data, per project_refactor.md §15's benchmarking strategy.

**Machine:** 16 CPUs, 8.0GB RAM

Regenerate with: `uv run python scripts/benchmark_phase2.py` (or `make benchmark`).

## Results

| Benchmark | OLD (s) | NEW (s) | Speedup | OLD peak RSS | NEW peak RSS | Notes |
|---|---|---|---|---|---|---|
| CSV Ingestion (load + clean) | 14.77 | 4.98 | 3.0x | 3640MB | 2523MB | Full dataset: 5 files, 791MB, 9.5M rows |
| Feature Engineering rolling/lag (10% sample) | 51.42 | 0.34 | 151.7x | 43MB | 151MB | 10% sample: 941,336 rows (fast-iteration scale per project_refactor.md §15) |
| Feature Engineering rolling/lag (full 9.5M rows, NEW only -- OLD extrapolated) | 514.15 | 6.32 | 81.4x | N/A (not measured) | 4355MB | Full 9.5M rows. OLD time is a x10 linear extrapolation of the measured 10% sample (not run directly at full scale); OLD memory is not reported here at all -- see docs/benchmarks.md notes. |
| Price Leadership Cross-Correlation | 15.47 | 2.01 | 7.7x | 275MB | 280MB | sample_size=1000, max_lag_days=7, real 9.5M-row dataset. Numerical equivalence verified separately in tests/test_market_analysis.py. |

## Methodology notes

- **CSV ingestion**: pandas `read_csv`+`concat`+cleaning vs. Polars-backed `load_raw_csvs`+`clean_raw_data` (src/pricepoint/data_ingestion.py). Run on the full raw dataset.
- **Feature engineering**: pandas `groupby().transform(lambda x: x.rolling(...))` vs. Polars `.over()` rolling expressions (src/pricepoint/feature_engineering.py::add_temporal_features). The OLD implementation was only run on a 10% sample (by canonical_name group, preserving group structure) -- at full scale it is measured at roughly 8+ minutes, disproportionate to a single benchmark run; project_refactor.md §15 explicitly allows a 10% sample for "fast iteration during dev". The full-scale OLD row in this table is a **linear extrapolation** (10% measured time x10), not a directly measured number -- flagged as such, not presented as measured.
- **Price leadership**: nested Python loop calling `pandas.Series.corr()` once per (leader, follower, product, lag) vs. vectorized numpy correlation (src/pricepoint/market_analysis.py::compute_price_leadership). Both run at the real `config.yaml` default `sample_size=1000`. Both implementations use the identical numpy random seed (42) and the same near-zero-variance threshold fix (see market_analysis.py `_MIN_VARIANCE_FOR_CORRELATION`), so their outputs are directly comparable -- verified numerically identical (same leader/follower pairs, same median lags, same product counts) on this run.
- **Peak RSS**: each OLD/NEW call runs in its own forked subprocess, and reports that process's own OS-level peak-RSS high-water mark (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) from just before to just after the call. This -- not a same-process before/after `psutil` snapshot, and not a background sampler thread -- is what this script actually uses; both alternatives were tried first and produced systematically wrong (near-zero) numbers for pandas/Python-loop-heavy operations. Isolating each call in its own subprocess also means later benchmarks are never confounded by memory a prior benchmark already allocated in the same process.
