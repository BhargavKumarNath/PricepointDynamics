"""Inference benchmarking utility.

Measures model prediction latency and reports percentile statistics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from pricepoint.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark statistics."""

    n_iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"Benchmark ({self.n_iterations} iterations):\n"
            f"  p50:  {self.p50_ms:.3f} ms\n"
            f"  p95:  {self.p95_ms:.3f} ms\n"
            f"  p99:  {self.p99_ms:.3f} ms\n"
            f"  mean: {self.mean_ms:.3f} ms\n"
            f"  min:  {self.min_ms:.3f} ms\n"
            f"  max:  {self.max_ms:.3f} ms"
        )


def benchmark_inference(
    model,
    sample_input: pd.DataFrame,
    n_iterations: int = 1000,
    warmup_iterations: int = 100,
) -> BenchmarkResult:
    """Measure single-row prediction latency.

    Parameters
    ----------
    model
        Trained model with a ``predict`` method.
    sample_input : pd.DataFrame
        A single-row DataFrame matching the model's feature schema.
    n_iterations : int
        Number of timed iterations.
    warmup_iterations : int
        Warmup iterations (not timed) to stabilise JIT/caches.

    Returns
    -------
    BenchmarkResult
        Percentile latency statistics.
    """
    logger.info(
        "Benchmarking inference: %s warmup + %s timed iterations …",
        warmup_iterations,
        n_iterations,
    )

    # Warmup
    for _ in range(warmup_iterations):
        model.predict(sample_input)

    # Timed runs
    latencies: list[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        model.predict(sample_input)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    arr = np.array(latencies)
    result = BenchmarkResult(
        n_iterations=n_iterations,
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        p99_ms=round(float(np.percentile(arr, 99)), 3),
        mean_ms=round(float(np.mean(arr)), 3),
        min_ms=round(float(np.min(arr)), 3),
        max_ms=round(float(np.max(arr)), 3),
    )

    logger.info("Benchmark complete:\n%s", result)
    return result


def run_benchmark(settings: Settings) -> BenchmarkResult:
    """Run a full inference benchmark using the saved model.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    BenchmarkResult
        Latency statistics.
    """
    model_path = settings.model.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    model = joblib.load(model_path)
    features = model.feature_name_
    sample = pd.DataFrame([{f: 1.0 for f in features}])

    return benchmark_inference(
        model,
        sample,
        n_iterations=settings.benchmarking.n_iterations,
        warmup_iterations=settings.benchmarking.warmup_iterations,
    )
