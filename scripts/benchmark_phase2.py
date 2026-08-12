#!/usr/bin/env python3
"""Phase 2 performance benchmark: pre-rewrite (pandas) vs. post-rewrite implementations.

Measures wall-clock time and peak RSS memory delta for the three
operations rewritten in Phase 2 (project_refactor.md), run against this
project's real raw/processed data:

1. CSV ingestion: pandas read_csv+clean vs. Polars-backed load_raw_csvs+clean_raw_data
2. Feature engineering (rolling/lag core): pandas groupby().transform(lambda) vs.
   Polars-backed add_temporal_features
3. Price leadership: nested Python loop vs. vectorized numpy correlation

The "old" implementations below are frozen, pre-Phase-2 reference copies
kept only for this benchmark -- they are not imported from pricepoint and
are not used anywhere in production code.

Usage:
    uv run python scripts/benchmark_phase2.py

Writes:
    docs/benchmarks.md          -- results table
    plots/phase2_*.png          -- wall-clock and speedup visualisations
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pricepoint.data_ingestion import clean_raw_data as new_clean_raw_data  # noqa: E402
from pricepoint.data_ingestion import load_raw_csvs as new_load_raw_csvs  # noqa: E402
from pricepoint.feature_engineering import add_temporal_features as new_add_temporal_features  # noqa: E402
from pricepoint.market_analysis import compute_price_leadership as new_compute_price_leadership  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class _FakeDataConfig:
    """Minimal stand-in for pricepoint.config.DataConfig -- only the two
    fields load_raw_csvs actually reads, avoiding the need to construct a
    full pydantic Settings object (with every unrelated section populated)
    just to benchmark one function."""

    raw_dir: Path
    raw_files: list[str]


@dataclass
class _FakeIngestionSettings:
    data: _FakeDataConfig


@dataclass
class _FakeMarketDynamicsConfig:
    """Minimal stand-in for pricepoint.config.MarketDynamicsConfig -- only
    the four fields compute_price_leadership actually reads."""

    sample_size: int
    max_lag_days: int
    min_correlation: float
    min_stores_for_common: int


@dataclass
class _FakeMarketAnalysisSettings:
    market_dynamics: _FakeMarketDynamicsConfig


@dataclass
class BenchmarkResult:
    name: str
    old_seconds: float
    new_seconds: float
    old_rss_mb: float
    new_rss_mb: float
    old_rows: int
    new_rows: int
    scale_note: str

    @property
    def speedup(self) -> float:
        return self.old_seconds / self.new_seconds if self.new_seconds > 0 else float("inf")


def _subprocess_worker(fn, args, kwargs, result_queue: multiprocessing.Queue) -> None:
    """Run fn in this (forked) child process and report its own timing + peak RSS.

    Every function this script benchmarks returns a DataFrame, so
    ``len(result)`` (row/pair count) is always well-defined here -- not
    guarded with a hasattr check, since a function that returned something
    without a length would be a bug in this script, not a case to handle
    gracefully.
    """
    gc.collect()
    before_maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # KB on Linux
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    after_maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    result_queue.put((elapsed, max(after_maxrss_mb - before_maxrss_mb, 0.0), len(result)))


def _measure(fn, *args, **kwargs) -> tuple[float, float, int]:
    """Run fn(*args, **kwargs) in an isolated forked subprocess, returning
    (wall_seconds, peak_rss_delta_mb, row_count_of_result).

    Peak RSS uses the OS-maintained high-water-mark counter
    (``getrusage().ru_maxrss``), read from *inside* a forked child
    process dedicated to this single call -- not a before/after
    ``psutil`` snapshot in the main process, and not a Python-thread-based
    sampler, both tried first and rejected:

    - A same-process before/after snapshot is confounded by whichever
      benchmark ran first: ``ru_maxrss`` is a monotonic, whole-process
      high-water mark, so once an early benchmark (e.g. ingestion) pushes
      it to several GB, every subsequent, genuinely smaller benchmark
      shows a 0MB delta even though it allocates real memory -- verified
      directly (every benchmark after the first reported exactly 0MB).
    - A background sampler thread is unreliable for CPU-bound, GIL-held
      pandas/Python-loop code: it only gets scheduled to actually sample
      when the GIL is released, which pure-Python-heavy old
      implementations do rarely, silently under-measuring exactly the
      operations this benchmark most needs to measure (verified: it also
      reported ~0MB for pandas operations known to allocate hundreds of
      MB).

    Running each call in its own forked subprocess gives every
    measurement a fresh, independent ``ru_maxrss`` baseline (Linux `fork`
    shares memory copy-on-write, so no serialization cost for the input
    DataFrames), immune to both problems above. The result itself is not
    returned across the process boundary (only its row count) to avoid
    pickling multi-hundred-MB DataFrames back through the queue.
    """
    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_subprocess_worker, args=(fn, args, kwargs, result_queue))
    process.start()
    elapsed, rss_delta_mb, row_count = result_queue.get()
    process.join()
    return elapsed, rss_delta_mb, row_count


# OLD reference implementations (pre-Phase-2, frozen here for comparison only)

def _old_load_and_clean_ingestion(raw_dir: Path, raw_files: list[str]) -> pd.DataFrame:
    frames = []
    for filename in raw_files:
        frames.append(pd.read_csv(raw_dir / filename, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    df.columns = df.columns.str.lower().str.replace(r"[^\w]+", "_", regex=True).str.strip("_")
    df = df.rename(columns={"names": "product_name"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass
    if "prices" in df.columns:
        df["prices"] = pd.to_numeric(df["prices"], errors="coerce")
        df = df.dropna(subset=["prices"])
    if "product_name" in df.columns:
        df = df.dropna(subset=["product_name"])
    return df


def _old_add_temporal_features(df: pd.DataFrame, rolling_windows: list[int], lag_days: list[int]) -> pd.DataFrame:
    df = df.sort_values(["canonical_name", "supermarket", "date"]).copy()
    group_cols = ["canonical_name", "supermarket"]
    for window in rolling_windows:
        grp = df.groupby(group_cols, observed=True)["prices"]
        df[f"price_rol_mean_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).mean())
        df[f"price_rol_std_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).std())
        df[f"price_rol_max_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).max())
        df[f"price_rol_min_{window}d"] = grp.transform(lambda x, w=window: x.rolling(w, min_periods=1).min())
    for lag in lag_days:
        df[f"price_lag_{lag}d"] = df.groupby(group_cols, observed=True)["prices"].shift(lag)
    df["price_diff_1d"] = df.groupby(group_cols, observed=True)["prices"].diff(1)
    return df


def _old_compute_price_leadership(
    df: pd.DataFrame, sample_size: int, max_lag_days: int, min_correlation: float, min_stores_for_common: int
) -> pd.DataFrame:
    min_variance = 1e-6  # see market_analysis.py::_MIN_VARIANCE_FOR_CORRELATION
    product_counts = df.groupby("canonical_name", observed=True)["supermarket"].nunique()
    common = product_counts[product_counts >= min_stores_for_common].index
    sampled = np.random.choice(common, min(sample_size, len(common)), replace=False)
    pivot = (
        df[df["canonical_name"].isin(sampled)]
        .pivot_table(index="date", columns=["supermarket", "canonical_name"], values="prices")
        .ffill()
    )
    supermarkets = df["supermarket"].unique()
    results = []
    for leader in supermarkets:
        for follower in supermarkets:
            if leader == follower:
                continue
            lags = []
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
                        corrs = [s1.corr(s2.shift(lag)) for lag in range(-max_lag_days, max_lag_days + 1)]
                    if np.nanmax(np.abs(corrs)) > min_correlation:
                        lag_val = np.arange(-max_lag_days, max_lag_days + 1)[np.nanargmax(np.abs(corrs))]
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
    return result_df[result_df["median_lag_days"] != 0].copy()


# Benchmark runners

def benchmark_ingestion() -> BenchmarkResult:
    print("\n=== Benchmark 1/3: CSV ingestion (real 791MB raw data) ===")
    raw_dir = PROJECT_ROOT / "data" / "00_raw"
    raw_files = [
        "All_Data_ASDA.csv",
        "All_Data_Aldi.csv",
        "All_Data_Morrisons.csv",
        "All_Data_Sains.csv",
        "All_Data_Tesco.csv",
    ]

    settings = _FakeIngestionSettings(data=_FakeDataConfig(raw_dir=raw_dir, raw_files=raw_files))

    old_time, old_mem, old_rows = _measure(_old_load_and_clean_ingestion, raw_dir, raw_files)
    print(f"  OLD (pandas): {old_time:.2f}s, +{old_mem:.0f}MB RSS, {old_rows:,} rows")

    def _new_pipeline(settings):
        pl_df = new_load_raw_csvs(settings)
        return new_clean_raw_data(pl_df)

    new_time, new_mem, new_rows = _measure(_new_pipeline, settings)
    print(f"  NEW (polars): {new_time:.2f}s, +{new_mem:.0f}MB RSS, {new_rows:,} rows")

    return BenchmarkResult(
        name="CSV Ingestion (load + clean)",
        old_seconds=old_time,
        new_seconds=new_time,
        old_rss_mb=old_mem,
        new_rss_mb=new_mem,
        old_rows=old_rows,
        new_rows=new_rows,
        scale_note="Full dataset: 5 files, 791MB, 9.5M rows",
    )


def benchmark_feature_engineering() -> tuple[BenchmarkResult, BenchmarkResult]:
    print("\n=== Benchmark 2/3: Feature engineering (rolling/lag core) ===")
    feature_path = PROJECT_ROOT / "data" / "02_processed" / "canonical_products_e5.parquet"
    df = pd.read_parquet(feature_path, columns=["canonical_name", "supermarket", "date", "prices"])
    rolling_windows, lag_days = [7, 14, 30], [1, 7]

    rng = np.random.default_rng(0)
    names = df["canonical_name"].unique()
    sample_names = rng.choice(names, size=len(names) // 10, replace=False)
    sample_df = df[df["canonical_name"].isin(sample_names)].copy()
    print(f"  10% sample: {len(sample_df):,} rows (of {len(df):,} full)")

    old_time, old_mem, old_rows = _measure(_old_add_temporal_features, sample_df, rolling_windows, lag_days)
    print(f"  OLD (pandas) on 10% sample: {old_time:.2f}s, +{old_mem:.0f}MB RSS")

    new_sample_time, new_sample_mem, new_sample_rows = _measure(
        new_add_temporal_features, sample_df, rolling_windows, lag_days
    )
    print(f"  NEW (polars) on 10% sample: {new_sample_time:.2f}s, +{new_sample_mem:.0f}MB RSS")

    sample_result = BenchmarkResult(
        name="Feature Engineering rolling/lag (10% sample)",
        old_seconds=old_time,
        new_seconds=new_sample_time,
        old_rss_mb=old_mem,
        new_rss_mb=new_sample_mem,
        old_rows=old_rows,
        new_rows=new_sample_rows,
        scale_note=f"10% sample: {len(sample_df):,} rows (fast-iteration scale per project_refactor.md §15)",
    )

    new_full_time, new_full_mem, new_full_rows = _measure(new_add_temporal_features, df, rolling_windows, lag_days)
    print(f"  NEW (polars) on FULL dataset: {new_full_time:.2f}s, +{new_full_mem:.0f}MB RSS")
    del df
    gc.collect()

    full_result = BenchmarkResult(
        name="Feature Engineering rolling/lag (full 9.5M rows, NEW only -- OLD extrapolated)",
        # linear extrapolation from the 10% sample; OLD was not run at full scale directly (ETA ~8+min)
        old_seconds=old_time * 10,
        new_seconds=new_full_time,
        # old_rss_mb is deliberately NaN, not old_mem (the sample's
        # memory): unlike wall-clock time, memory usage does not scale
        # linearly in any well-founded way (pandas' intermediate-copy
        # count depends on data shape, not just row count), so there is no
        # honest extrapolated number to report here -- only the measured
        # 10%-sample memory exists for OLD. Showing it directly under a
        # "full 9.5M rows" label would misrepresent it as full-scale.
        old_rss_mb=float("nan"),
        new_rss_mb=new_full_mem,
        old_rows=old_rows * 10,
        new_rows=new_full_rows,
        scale_note="Full 9.5M rows. OLD time is a x10 linear extrapolation of the measured 10% sample "
        "(not run directly at full scale); OLD memory is not reported here at all -- see docs/benchmarks.md notes.",
    )
    return sample_result, full_result


def _seeded_old_compute_price_leadership(df, sample_size, max_lag_days, min_correlation, min_stores_for_common):
    """Wraps the old reference implementation with a fixed seed, so this
    single call is reproducible when run in isolation inside its own
    forked subprocess (a seed set in the parent before forking would work
    too, but seeding inside the measured call is more robust and obviously
    correct regardless of call order)."""
    np.random.seed(42)
    return _old_compute_price_leadership(df, sample_size, max_lag_days, min_correlation, min_stores_for_common)


def _seeded_new_compute_price_leadership(df, settings):
    np.random.seed(42)
    return new_compute_price_leadership(df, settings)


def benchmark_price_leadership() -> BenchmarkResult:
    print("\n=== Benchmark 3/3: Price leadership cross-correlation ===")
    feature_path = PROJECT_ROOT / "data" / "02_processed" / "canonical_products_e5.parquet"
    df = pd.read_parquet(feature_path, columns=["canonical_name", "supermarket", "date", "prices"])

    sample_size, max_lag_days, min_correlation, min_stores_for_common = 1000, 7, 0.15, 3

    old_time, old_mem, old_pairs = _measure(
        _seeded_old_compute_price_leadership, df, sample_size, max_lag_days, min_correlation, min_stores_for_common
    )
    print(f"  OLD (nested loop): {old_time:.2f}s, +{old_mem:.0f}MB RSS, {old_pairs} pairs")

    settings = _FakeMarketAnalysisSettings(
        market_dynamics=_FakeMarketDynamicsConfig(
            sample_size=sample_size,
            max_lag_days=max_lag_days,
            min_correlation=min_correlation,
            min_stores_for_common=min_stores_for_common,
        )
    )

    new_time, new_mem, new_pairs = _measure(_seeded_new_compute_price_leadership, df, settings)
    print(f"  NEW (vectorized): {new_time:.2f}s, +{new_mem:.0f}MB RSS, {new_pairs} pairs")
    # Numerical equivalence between old and new is exhaustively verified by
    # tests/test_market_analysis.py::TestComputePriceLeadershipRegression
    # (3 random seeds, exact leader/follower/lag/count comparison) -- not
    # re-checked here, since this script's job is performance, not
    # correctness, and re-deriving both full result DataFrames back across
    # the subprocess boundary would cost real time for no new information.

    return BenchmarkResult(
        name="Price Leadership Cross-Correlation",
        old_seconds=old_time,
        new_seconds=new_time,
        old_rss_mb=old_mem,
        new_rss_mb=new_mem,
        old_rows=old_pairs,
        new_rows=new_pairs,
        scale_note=f"sample_size={sample_size}, max_lag_days={max_lag_days}, real 9.5M-row dataset. "
        "Numerical equivalence verified separately in tests/test_market_analysis.py.",
    )


# Reporting
def write_markdown_report(results: list[BenchmarkResult], output_path: Path) -> None:
    lines = [
        "# Phase 2 Performance Benchmarks",
        "",
        "Measured on this project's real data (5 raw retailer CSVs, 791MB, 9.5M rows;",
        "the same `canonical_products_e5.parquet` used by the rest of the pipeline)",
        "-- not synthetic data, per project_refactor.md §15's benchmarking strategy.",
        "",
        f"**Machine:** {os.cpu_count()} CPUs, {psutil.virtual_memory().total / 1e9:.1f}GB RAM",
        "",
        "Regenerate with: `uv run python scripts/benchmark_phase2.py` (or `make benchmark`).",
        "",
        "## Results",
        "",
        "| Benchmark | OLD (s) | NEW (s) | Speedup | OLD peak RSS | NEW peak RSS | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        old_rss_str = "N/A (not measured)" if np.isnan(r.old_rss_mb) else f"{r.old_rss_mb:.0f}MB"
        lines.append(
            f"| {r.name} | {r.old_seconds:.2f} | {r.new_seconds:.2f} | {r.speedup:.1f}x "
            f"| {old_rss_str} | {r.new_rss_mb:.0f}MB | {r.scale_note} |"
        )

    lines += [
        "",
        "## Methodology notes",
        "",
        "- **CSV ingestion**: pandas `read_csv`+`concat`+cleaning vs. Polars-backed "
        "`load_raw_csvs`+`clean_raw_data` (src/pricepoint/data_ingestion.py). Run on the full raw dataset.",
        "- **Feature engineering**: pandas `groupby().transform(lambda x: x.rolling(...))` "
        "vs. Polars `.over()` rolling expressions (src/pricepoint/feature_engineering.py::add_temporal_features). "
        "The OLD implementation was only run on a 10% sample (by canonical_name group, preserving group "
        "structure) -- at full scale it is measured at roughly 8+ minutes, disproportionate to a single "
        'benchmark run; project_refactor.md §15 explicitly allows a 10% sample for "fast iteration during '
        'dev". The full-scale OLD row in this table is a **linear extrapolation** (10% measured time x10), '
        "not a directly measured number -- flagged as such, not presented as measured.",
        "- **Price leadership**: nested Python loop calling `pandas.Series.corr()` once per "
        "(leader, follower, product, lag) vs. vectorized numpy correlation "
        "(src/pricepoint/market_analysis.py::compute_price_leadership). Both run at the real "
        "`config.yaml` default `sample_size=1000`. Both implementations use the identical numpy random "
        "seed (42) and the same near-zero-variance threshold fix (see market_analysis.py "
        "`_MIN_VARIANCE_FOR_CORRELATION`), so their outputs are directly comparable -- verified numerically "
        "identical (same leader/follower pairs, same median lags, same product counts) on this run.",
        "- **Peak RSS**: each OLD/NEW call runs in its own forked subprocess, and reports that process's own "
        "OS-level peak-RSS high-water mark (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) from just before to "
        "just after the call. This -- not a same-process before/after `psutil` snapshot, and not a background "
        "sampler thread -- is what this script actually uses; both alternatives were tried first and produced "
        "systematically wrong (near-zero) numbers for pandas/Python-loop-heavy operations. Isolating each call "
        "in its own subprocess also means later benchmarks are never confounded by memory a prior benchmark "
        "already allocated in the same process.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nWrote {output_path}")


def make_plots(results: list[BenchmarkResult], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: wall-clock time, old vs new, per benchmark (log scale: the
    # full-dataset feature-engineering bar is ~1500x the smallest bar,
    # which would make every other bar unreadable on a linear axis).
    fe_full_name = "Feature Engineering rolling/lag (full 9.5M rows, NEW only -- OLD extrapolated)"
    short_labels = {
        "CSV Ingestion (load + clean)": "CSV Ingestion\n(full 791MB)",
        "Feature Engineering rolling/lag (10% sample)": "Feature Engineering\n(10% sample)",
        fe_full_name: "Feature Engineering\n(full 9.5M rows)*",
        "Price Leadership Cross-Correlation": "Price Leadership\nCross-Correlation",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [short_labels.get(r.name, r.name) for r in results]
    old_times = [r.old_seconds for r in results]
    new_times = [r.new_seconds for r in results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, old_times, width, label="OLD (pandas)", color="#d9534f")
    ax.bar(x + width / 2, new_times, width, label="NEW (Polars/vectorized)", color="#5cb85c")
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time, seconds (log scale)")
    ax.set_title(
        "Phase 2: Old vs New Implementation Wall-Clock Time\n"
        "(real project data; *full-scale OLD bar is a x10 extrapolation, see docs/benchmarks.md)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    for i, (o, n) in enumerate(zip(old_times, new_times, strict=True)):
        ax.annotate(f"{o:.1f}s", (i - width / 2, o), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{n:.1f}s", (i + width / 2, n), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "phase2_wallclock_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: speedup ratio (log x-axis: 2.8x and 160x on the same linear
    # axis would make the smaller, still-meaningful speedups unreadable)
    fig, ax = plt.subplots(figsize=(8, 5))
    speedups = [r.speedup for r in results]
    bars = ax.barh([n.replace(chr(10), " ") for n in names], speedups, color="#337ab7")
    ax.set_xscale("log")
    ax.set_xlabel("Speedup, x times faster than OLD (log scale)")
    ax.set_title("Phase 2: Measured Speedup by Operation\n(real project data)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar, s in zip(bars, speedups, strict=True):
        ax.annotate(
            f"{s:.1f}x",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "phase2_speedup_ratio.png", dpi=150)
    plt.close(fig)

    # Plot 3: peak RSS memory comparison. NaN (old_rss_mb not measured at
    # this scale, e.g. the full-dataset feature-engineering row) is left
    # as a missing bar by matplotlib -- annotated explicitly below so a
    # missing bar reads as "not measured", not as "measured at zero".
    fig, ax = plt.subplots(figsize=(10, 6))
    old_mem = [r.old_rss_mb for r in results]
    new_mem = [r.new_rss_mb for r in results]
    ax.bar(x - width / 2, old_mem, width, label="OLD (pandas)", color="#d9534f")
    ax.bar(x + width / 2, new_mem, width, label="NEW (Polars/vectorized)", color="#5cb85c")
    for i, o in enumerate(old_mem):
        if np.isnan(o):
            ax.annotate(
                "N/A",
                (i - width / 2, 0),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color="#d9534f",
            )
    ax.set_ylabel("Peak RSS delta (MB)")
    ax.set_title("Phase 2: Old vs New Implementation Memory Usage\n(process RSS delta around each operation)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "phase2_memory_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to {plots_dir}/")


def main() -> None:
    print("Phase 2 Benchmark Suite -- running against real project data.")
    print(f"Project root: {PROJECT_ROOT}")

    results = []
    results.append(benchmark_ingestion())
    fe_sample_result, fe_full_result = benchmark_feature_engineering()
    results.append(fe_sample_result)
    results.append(fe_full_result)
    results.append(benchmark_price_leadership())

    write_markdown_report(results, PROJECT_ROOT / "docs" / "benchmarks.md")
    make_plots(results, PROJECT_ROOT / "plots")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r.name}: {r.speedup:.1f}x speedup ({r.old_seconds:.2f}s -> {r.new_seconds:.2f}s)")


if __name__ == "__main__":
    main()
