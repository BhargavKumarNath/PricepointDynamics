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

    Uses a curated feature set that captures price-level anomalies:
    - prices: the target variable
    - price_diff_1d: day-over-day change (captures sudden jumps)
    - price_rol_std_7d: 7-day rolling volatility (captures erratic pricing)
    - price_vs_market_avg: deviation from market (captures competitive anomalies)

    This is more interpretable and robust than using all numeric features,
    which would include categorical embeddings, row indices, and other
    features unrelated to pricing anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data with the expected feature columns.
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

    # Curated feature set for price anomaly detection
    anomaly_features = ["prices", "price_diff_1d", "price_rol_std_7d", "price_vs_market_avg"]

    # Check that all features exist
    missing_features = [f for f in anomaly_features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Feature-engineered data missing required anomaly-detection columns: {missing_features}. "
            "Ensure feature engineering ran successfully before anomaly detection."
        )

    # Select only the curated features and drop rows with NaN in any of them
    X = df[anomaly_features].dropna()
    logger.info("Selected %d rows with complete anomaly feature values.", len(X))

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
