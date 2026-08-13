#!/usr/bin/env python
"""PricePoint Dynamics — CLI Entry Point.

Usage:
    python run.py ingest       # Run data ingestion + validation
    python run.py match        # Run semantic product matching
    python run.py features     # Run feature engineering
    python run.py train        # Train LightGBM model
    python run.py anomaly      # Run anomaly detection
    python run.py marts        # Materialize the DuckDB analytics marts
    python run.py precompute   # Precompute SHAP + market dynamics
    python run.py web-artifacts # Export precomputed frontend artifacts
    python run.py benchmark    # Run inference benchmark
    python run.py hhi          # Calculate HHI index
"""

from __future__ import annotations

import typer

from pricepoint.config import Settings, load_settings
from pricepoint.logging_config import setup_logging

app = typer.Typer(
    name="pricepoint",
    help="PricePoint Dynamics — UK Supermarket Competitive Intelligence Pipeline.",
    add_completion=False,
)


def _init() -> Settings:
    """Load settings and configure logging."""
    settings = load_settings()
    setup_logging(settings)
    return settings


_FORCE_OPTION = typer.Option(False, "--force", "-f", help="Re-run even if inputs are unchanged since the last run.")


@app.command()
def ingest(force: bool = _FORCE_OPTION) -> None:
    """Run data ingestion: load raw CSVs → clean → validate → Parquet."""
    from pricepoint.data_ingestion import run_ingestion

    settings = _init()
    path = run_ingestion(settings, force=force)
    typer.echo(f"✓ Ingestion complete → {path}")


@app.command()
def match(force: bool = _FORCE_OPTION) -> None:
    """Run semantic product matching (Sentence-BERT + FAISS)."""
    from pricepoint.product_matching import run_matching

    settings = _init()
    path = run_matching(settings, force=force)
    typer.echo(f"✓ Product matching complete → {path}")


@app.command()
def features(force: bool = _FORCE_OPTION) -> None:
    """Run feature engineering (rolling stats, lags, competitive)."""
    from pricepoint.feature_engineering import run_feature_engineering

    settings = _init()
    path = run_feature_engineering(settings, force=force)
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
def marts(force: bool = _FORCE_OPTION) -> None:
    """Materialize the DuckDB analytics marts from 02_processed/."""
    from pricepoint.marts import run_materialize_marts

    settings = _init()
    paths = run_materialize_marts(settings, force=force)
    for name, path in paths.items():
        typer.echo(f"✓ Mart '{name}' → {path}")


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
def web_artifacts() -> None:
    """Export precomputed frontend artifacts (project_refactor.md §25.3)."""
    from pricepoint.web_artifacts import run_export_web_artifacts

    settings = _init()
    paths = run_export_web_artifacts(settings)
    for name, path in paths.items():
        typer.echo(f"✓ Web artifact '{name}' → {path}")


@app.command()
def hhi() -> None:
    """Calculate Herfindahl-Hirschman Index (market concentration)."""
    from pricepoint.market_analysis import run_hhi

    settings = _init()
    path = run_hhi(settings)
    typer.echo(f"✓ HHI calculation complete → {path}")


if __name__ == "__main__":
    app()
