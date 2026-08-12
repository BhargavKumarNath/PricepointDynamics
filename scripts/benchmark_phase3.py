#!/usr/bin/env python3
"""Phase 3 performance benchmark: pandas-in-Streamlit vs. DuckDB-marts.

Measures wall-clock time and peak RSS memory for the two aggregate-query
patterns the Phase 3 marts + warehouse.py replace, run against this
project's real materialized marts and the full real
canonical_products_e5.parquet (9.5M rows):

1. Market overview: portfolio size, own-brand %, price distribution per
   retailer -- mirrors dashboard/pages/01_market_overview.py's
   `df.groupby('supermarket')` aggregates computed after loading the full
   canonical dataset into pandas.
2. Basket cost: latest-date price + coverage per retailer for a basket of
   products -- mirrors dashboard/pages/02_Basket_analysis.py's
   pivot-table-and-sum pattern, using one of that page's own real basket
   definitions ("The Essentials"), not a synthetic one.

The "old" implementations below are frozen, pandas-only reference copies
of the dashboard's actual current logic, kept only for this benchmark --
they are not imported from pricepoint/dashboard and are not used anywhere
in production code.

Usage:
    uv run python scripts/benchmark_phase3.py

Writes:
    docs/benchmarks.md          -- appends a Phase 3 section (does not overwrite Phase 2's)
    plots/phase3_*.png          -- wall-clock and speedup visualisations
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import resource
import sys
import time
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

from pricepoint.warehouse import Warehouse  # noqa: E402

# The exact "Essentials" basket from dashboard/pages/02_Basket_analysis.py --
# a real basket definition already in this repo, not synthesized for this
# benchmark.
_ESSENTIALS_BASKET = [
    "finest goat cheese caramelised red onion ravioli",
    "gran luchito mexican crunchy jalapeño pineapple",
    "lancashire farm greek style fat free yogurt",
    "sma pro follow on baby milk liquid ready to feed",
    "mamia organic tomato wheels",
    "gallo risotto with tomato and basil",
    "bodrum bodrum crispy fried onion",
    "lindt classic recipe hazelnut milk chocolate bar",
    "no added sugar apple pear juice drink cartons",
    "sainsburys british fresh chicken breast pieces in a salt chilli breadcrumb coating",
    "president french brie cheese",
    "sainsburys cauliflower cheese",
    "port salut french creamy cheese slices 6x20g",
    "mutti baby roma tomatoes",
    "up go banana honey breakfast shake",
]


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

    See scripts/benchmark_phase2.py for why this subprocess-isolated
    measurement is used instead of a same-process before/after snapshot
    or a background sampler thread -- both were tried in Phase 2 and gave
    systematically wrong (near-zero) numbers.
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
    (wall_seconds, peak_rss_delta_mb, row_count_of_result)."""
    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_subprocess_worker, args=(fn, args, kwargs, result_queue))
    process.start()
    elapsed, rss_delta_mb, row_count = result_queue.get()
    process.join()
    return elapsed, rss_delta_mb, row_count


# OLD reference implementations (pandas-in-Streamlit, frozen here for comparison only)


def _old_market_overview(canonical_path: str) -> pd.DataFrame:
    """Mirrors dashboard/pages/01_market_overview.py's
    `df.groupby('supermarket')` aggregates, computed after loading the
    full canonical dataset into pandas (as `load_canonical_data()` does)."""
    df = pd.read_parquet(
        canonical_path,
        columns=["supermarket", "prices", "canonical_name", "own_brand"],
        engine="pyarrow",
    )
    portfolio_size = df.groupby("supermarket", observed=True)["canonical_name"].nunique()
    own_brand_pct = df.groupby("supermarket", observed=True)["own_brand"].mean() * 100
    price_stats = df.groupby("supermarket", observed=True)["prices"].agg(
        ["min", lambda s: s.quantile(0.25), "median", lambda s: s.quantile(0.75), "max"]
    )
    result = pd.DataFrame(
        {
            "portfolio_size": portfolio_size,
            "own_brand_pct": own_brand_pct,
            "min_price": price_stats.iloc[:, 0],
            "price_p25": price_stats.iloc[:, 1],
            "price_median": price_stats.iloc[:, 2],
            "price_p75": price_stats.iloc[:, 3],
            "max_price": price_stats.iloc[:, 4],
        }
    ).reset_index()
    return result


def _old_basket_cost(canonical_path: str, basket_items: list[str]) -> pd.DataFrame:
    """Mirrors dashboard/pages/02_Basket_analysis.py's
    pivot-table-and-sum pattern exactly (including its
    drop_duplicates(keep="first") behaviour on the latest date)."""
    df = pd.read_parquet(
        canonical_path,
        columns=["supermarket", "prices", "canonical_name", "date"],
        engine="pyarrow",
    )
    latest_date = pd.to_datetime(df["date"]).max()
    df_latest = df[df["date"] == latest_date].copy()
    df_latest = df_latest.drop_duplicates(subset=["canonical_name", "supermarket"])
    pivot = df_latest.pivot_table(index="canonical_name", columns="supermarket", values="prices", observed=True)

    basket_df = pivot[pivot.index.isin(basket_items)]
    items_found = basket_df.notna().sum()
    basket_cost = basket_df.sum()
    result = pd.DataFrame(
        {"supermarket": basket_cost.index, "basket_cost": basket_cost.values, "items_found": items_found.values}
    )
    return result[result["items_found"] > 0].reset_index(drop=True)


# NEW: DuckDB marts via Warehouse
def _new_market_overview(marts_dir: str) -> pd.DataFrame:
    with Warehouse(Path(marts_dir)) as wh:
        return wh.get_market_overview()


def _new_basket_cost(marts_dir: str, basket_items: list[str]) -> pd.DataFrame:
    with Warehouse(Path(marts_dir)) as wh:
        return wh.get_basket_cost(basket_items)


# Benchmark runners
def benchmark_market_overview(canonical_path: Path, marts_dir: Path) -> BenchmarkResult:
    print("\n=== Benchmark 1/2: Market overview aggregates ===")
    old_time, old_mem, old_rows = _measure(_old_market_overview, str(canonical_path))
    print(f"  OLD (pandas, full 9.5M-row canonical parquet): {old_time:.3f}s, +{old_mem:.0f}MB RSS")

    new_time, new_mem, new_rows = _measure(_new_market_overview, str(marts_dir))
    print(f"  NEW (DuckDB over marts): {new_time:.3f}s, +{new_mem:.0f}MB RSS")

    return BenchmarkResult(
        name="Market Overview Aggregates",
        old_seconds=old_time,
        new_seconds=new_time,
        old_rss_mb=old_mem,
        new_rss_mb=new_mem,
        old_rows=old_rows,
        new_rows=new_rows,
        scale_note="5 retailers; OLD reads the full 9.5M-row canonical_products_e5.parquet, "
        "NEW reads only the pre-aggregated fact_price_daily/dim_product marts",
    )


def benchmark_basket_cost(canonical_path: Path, marts_dir: Path) -> BenchmarkResult:
    print("\n=== Benchmark 2/2: Basket cost ('The Essentials', 15 real products) ===")
    old_time, old_mem, old_rows = _measure(_old_basket_cost, str(canonical_path), _ESSENTIALS_BASKET)
    print(f"  OLD (pandas pivot_table over full canonical parquet): {old_time:.3f}s, +{old_mem:.0f}MB RSS")

    new_time, new_mem, new_rows = _measure(_new_basket_cost, str(marts_dir), _ESSENTIALS_BASKET)
    print(f"  NEW (DuckDB over fact_price_daily mart): {new_time:.3f}s, +{new_mem:.0f}MB RSS")

    return BenchmarkResult(
        name="Basket Cost Lookup",
        old_seconds=old_time,
        new_seconds=new_time,
        old_rss_mb=old_mem,
        new_rss_mb=new_mem,
        old_rows=old_rows,
        new_rows=new_rows,
        scale_note="15-item real basket ('The Essentials', dashboard/pages/02_Basket_analysis.py); "
        "OLD pivots the full 9.5M-row canonical parquet, NEW queries only fact_price_daily",
    )


# Reporting
def append_markdown_report(results: list[BenchmarkResult], output_path: Path) -> None:
    lines = [
        "",
        "## Phase 3: DuckDB Marts vs. pandas-in-Streamlit",
        "",
        "Measured on this project's real data: the full 9.5M-row "
        "`canonical_products_e5.parquet` for the OLD (pandas) side, and the "
        "real materialized marts (`data/03_marts/`) for the NEW (DuckDB) side "
        "-- not synthetic data.",
        "",
        f"**Machine:** {os.cpu_count()} CPUs, {psutil.virtual_memory().total / 1e9:.1f}GB RAM",
        "",
        "Regenerate with: `uv run python scripts/benchmark_phase3.py` (or `make benchmark-phase3`).",
        "",
        "| Benchmark | OLD (s) | NEW (s) | Speedup | OLD peak RSS | NEW peak RSS | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.old_seconds:.3f} | {r.new_seconds:.3f} | {r.speedup:.1f}x "
            f"| {r.old_rss_mb:.0f}MB | {r.new_rss_mb:.0f}MB | {r.scale_note} |"
        )

    lines += [
        "",
        "### Methodology notes",
        "",
        "- **Market overview**: OLD reproduces "
        "`dashboard/pages/01_market_overview.py`'s exact aggregates "
        "(`df.groupby('supermarket')['canonical_name'].nunique()`, "
        "`['own_brand'].mean()`, and price quartiles) computed in pandas "
        "after loading the full canonical dataset. NEW is "
        "`Warehouse.get_market_overview()`, querying only the "
        "pre-aggregated `fact_price_daily`/`dim_product` marts via DuckDB.",
        "- **Basket cost**: OLD reproduces "
        "`dashboard/pages/02_Basket_analysis.py`'s pivot-table-and-sum "
        'pattern (including its `drop_duplicates(keep="first")` on the '
        "latest date) against the full canonical dataset, using that "
        'page\'s own real "The Essentials" basket (15 products). NEW is '
        "`Warehouse.get_basket_cost()`.",
        "- **Peak RSS**: each OLD/NEW call runs in its own forked "
        "subprocess, reporting that process's own OS-level peak-RSS "
        "high-water mark (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) -- "
        "see `scripts/benchmark_phase2.py`'s methodology notes for why "
        "this approach was adopted over same-process snapshots.",
        "- **Row counts** in the summary above count result rows (typically "
        "5, one per retailer), not input rows scanned -- both "
        "implementations answer the identical question over the identical "
        "underlying data, just via different code paths.",
        "",
    ]
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nAppended Phase 3 results to {output_path}")


def make_plots(results: list[BenchmarkResult], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]
    old_times = [r.old_seconds for r in results]
    new_times = [r.new_seconds for r in results]
    x = np.arange(len(names))
    width = 0.35

    # Plot 1: wall-clock time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, old_times, width, label="OLD (pandas, full parquet)", color="#d9534f")
    ax.bar(x + width / 2, new_times, width, label="NEW (DuckDB over marts)", color="#5cb85c")
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time, seconds (log scale)")
    ax.set_title("Phase 3: DuckDB Marts vs. pandas-in-Streamlit\n(real project data)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.legend()
    for i, (o, n) in enumerate(zip(old_times, new_times, strict=True)):
        ax.annotate(f"{o:.3f}s", (i - width / 2, o), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.annotate(f"{n:.3f}s", (i + width / 2, n), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "phase3_wallclock_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: speedup ratio
    fig, ax = plt.subplots(figsize=(8, 4))
    speedups = [r.speedup for r in results]
    bars = ax.barh(names, speedups, color="#337ab7")
    ax.set_xscale("log")
    ax.set_xlabel("Speedup, x times faster than OLD (log scale)")
    ax.set_title("Phase 3: Measured Speedup by Query\n(real project data)")
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
    fig.savefig(plots_dir / "phase3_speedup_ratio.png", dpi=150)
    plt.close(fig)

    # Plot 3: peak memory
    fig, ax = plt.subplots(figsize=(8, 6))
    old_mem = [r.old_rss_mb for r in results]
    new_mem = [r.new_rss_mb for r in results]
    ax.bar(x - width / 2, old_mem, width, label="OLD (pandas, full parquet)", color="#d9534f")
    ax.bar(x + width / 2, new_mem, width, label="NEW (DuckDB over marts)", color="#5cb85c")
    ax.set_ylabel("Peak RSS delta (MB)")
    ax.set_title("Phase 3: Memory Usage -- DuckDB Marts vs. pandas-in-Streamlit\n(real project data)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "phase3_memory_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to {plots_dir}/")


def main() -> None:
    print("Phase 3 Benchmark Suite -- running against real project data.")

    canonical_path = PROJECT_ROOT / "data" / "02_processed" / "canonical_products_e5.parquet"
    marts_dir = PROJECT_ROOT / "data" / "03_marts"

    if not canonical_path.exists():
        raise FileNotFoundError(f"{canonical_path} not found -- run `python run.py match` first.")
    if not (marts_dir / "fact_price_daily.parquet").exists():
        raise FileNotFoundError(f"Marts not found in {marts_dir} -- run `python run.py marts` first.")

    results = [
        benchmark_market_overview(canonical_path, marts_dir),
        benchmark_basket_cost(canonical_path, marts_dir),
    ]

    append_markdown_report(results, PROJECT_ROOT / "docs" / "benchmarks.md")
    make_plots(results, PROJECT_ROOT / "plots")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r.name}: {r.speedup:.1f}x speedup ({r.old_seconds:.3f}s -> {r.new_seconds:.3f}s)")


if __name__ == "__main__":
    main()
