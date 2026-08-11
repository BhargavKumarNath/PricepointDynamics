#!/usr/bin/env python
"""PricePoint Dynamics — CLI Entry Point.

Usage:
    python run.py ingest       # Run data ingestion + validation
    python run.py match        # Run semantic product matching
    python run.py features     # Run feature engineering
    python run.py train        # Train LightGBM model
    python run.py anomaly      # Run anomaly detection
    python run.py precompute   # Precompute SHAP + market dynamics
    python run.py benchmark    # Run inference benchmark
    python run.py hhi          # Calculate HHI index
"""

from __future__ import annotations

import typer

from pricepoint.config import load_settings
from pricepoint.logging_config import setup_logging

app = typer.Typer(
    name="pricepoint",
    help="PricePoint Dynamics — UK Supermarket Competitive Intelligence Pipeline.",
    add_completion=False,
)


def _init() -> "Settings":
    """Load settings and configure logging."""
    settings = load_settings()
    setup_logging(settings)
    return settings


@app.command()
def ingest() -> None:
    """Run data ingestion: load raw CSVs → clean → validate → Parquet."""
    from pricepoint.data_ingestion import run_ingestion

    settings = _init()
    path = run_ingestion(settings)
    typer.echo(f"✓ Ingestion complete → {path}")


@app.command()
def match() -> None:
    """Run semantic product matching (Sentence-BERT + FAISS)."""
    from pricepoint.product_matching import run_matching

    settings = _init()
    path = run_matching(settings)
    typer.echo(f"✓ Product matching complete → {path}")


@app.command()
def features() -> None:
    """Run feature engineering (rolling stats, lags, competitive)."""
    from pricepoint.feature_engineering import run_feature_engineering

    settings = _init()
    path = run_feature_engineering(settings)
    typer.echo(f"✓ Feature engineering complete → {path}")


@app.command()
def train() -> None:
    """Train LightGBM price prediction model."""
    from pricepoint.training import run_training

    settings = _init()
    path = run_training(settings)
    typer.echo(f"✓ Training complete → {path}")


@app.command()
def anomaly() -> None:
    """Run Isolation Forest anomaly detection."""
    from pricepoint.anomaly import run_anomaly_detection

    settings = _init()
    path = run_anomaly_detection(settings)
    typer.echo(f"✓ Anomaly detection complete → {path}")


@app.command()
def precompute() -> None:
    """Pre-compute SHAP values + market dynamics for the dashboard."""
    from pricepoint.market_analysis import run_precompute

    settings = _init()
    run_precompute(settings)
    typer.echo("✓ All precomputation complete.")


@app.command()
def benchmark() -> None:
    """Benchmark model inference latency."""
    from pricepoint.benchmarking import run_benchmark

    settings = _init()
    result = run_benchmark(settings)
    typer.echo(str(result))


@app.command()
def hhi() -> None:
    """Calculate Herfindahl-Hirschman Index (market concentration)."""
    from pricepoint.market_analysis import run_hhi

    settings = _init()
    path = run_hhi(settings)
    typer.echo(f"✓ HHI calculation complete → {path}")


if __name__ == "__main__":
    app()
