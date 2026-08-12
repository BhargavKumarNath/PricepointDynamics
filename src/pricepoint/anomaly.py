"""Anomaly detection pipeline using Isolation Forest.

Identifies pricing irregularities such as scraping errors, algorithmic
A/B testing, and oscillation patterns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from pricepoint.config import Settings

logger = logging.getLogger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Isolation Forest anomaly detection on price data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data with numeric columns.
    contamination : float
        Expected proportion of anomalies.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Input data with an ``is_anomaly`` column appended.
    """
    logger.info(
        "Running Isolation Forest (contamination=%.3f) on %s rows …",
        contamination,
        f"{len(df):,}",
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    X = df[numeric_cols].dropna()

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso_forest.fit_predict(X)

    df = df.copy()
    df.loc[X.index, "is_anomaly"] = preds == -1
    df["is_anomaly"] = df["is_anomaly"].fillna(False).astype(bool)

    n_anomalies = df["is_anomaly"].sum()
    pct = n_anomalies / len(df) * 100
    logger.info("Anomalies detected: %s (%.2f%%)", f"{n_anomalies:,}", pct)

    return df


def run_anomaly_detection(settings: Settings) -> Path:
    """Execute the anomaly detection pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the output file with anomalies flagged.
    """
    feature_path = settings.data.processed_dir / settings.features.output_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found at {feature_path}. Run feature engineering first.")

    logger.info("Loading feature data from %s …", feature_path)
    df = pd.read_parquet(feature_path, engine="pyarrow")

    df = detect_anomalies(
        df,
        contamination=settings.anomaly.contamination,
        random_state=settings.anomaly.random_state,
    )

    output_path = settings.data.processed_dir / "anomalies_flagged.parquet"
    logger.info("Saving anomaly-flagged data to %s …", output_path)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info("Anomaly detection complete. Output: %s", output_path)

    return output_path
