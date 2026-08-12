"""LightGBM model training pipeline.

Handles train/test splitting, model fitting, evaluation, and artifact
serialization.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pricepoint.config import Settings
from pricepoint.manifest import get_git_sha
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory

logger = logging.getLogger(__name__)


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare train/test splits using time-series strategy.

    The final week of data is held out as the test set to prevent
    data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    logger.info("Preparing train/test split (time-series strategy) …")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=7)

    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    logger.info("Train: %s rows, Test: %s rows", f"{len(train):,}", f"{len(test):,}")

    # Separate target
    target_col = "prices"
    drop_cols = [
        target_col,
        "date",
        "product_name",
        "canonical_name",
        "normalised_name",
        # prices_unit is a per-unit rescaling of THIS SAME ROW's `prices`
        # (e.g. £/kg derived from pack price / pack weight) -- within a
        # given (canonical_name, supermarket) group the ratio between the
        # two is close to constant (empirically true for ~82% of groups,
        # verified against real data in Phase 1), so it is same-row target
        # leakage, arguably more direct than the market-average leak this
        # phase already fixed (ADR-0006/0018): it doesn't even require
        # cross-referencing other rows, just a near-fixed per-row rescaling
        # of the value being predicted.
        "prices_unit",
    ]
    drop_cols = [c for c in drop_cols if c in train.columns]

    # One-hot encode categoricals. Include "str" explicitly alongside
    # "object": pandas >= 3.0 has a dedicated string dtype distinct from
    # "object", currently still caught by an "object" select_dtypes query
    # only via a deprecated backward-compat shim (a live Pandas4Warning on
    # this exact line was observed against pandas 3.0.5) -- listing it
    # explicitly avoids silently losing every string-typed categorical
    # column once that shim is removed upstream.
    cat_cols = train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    if cat_cols:
        train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
        test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

    y_train = train[target_col]
    y_test = test[target_col]

    # Select numeric AND boolean columns. One-hot dummies created by
    # pd.get_dummies() above (and the raw `own_brand` flag) are bool dtype
    # in modern pandas -- a numeric-only select_dtypes(["number"]) silently
    # drops them entirely, with no error, meaning every categorical feature
    # (supermarket, category, own_brand) would vanish from training without
    # a trace. Include bool explicitly and cast to int8 for a consistent
    # numeric dtype across the whole feature matrix.
    feature_dtypes = ["number", "bool"]
    X_train = train.drop(columns=drop_cols, errors="ignore").select_dtypes(include=feature_dtypes)
    X_test = test.drop(columns=drop_cols, errors="ignore").select_dtypes(include=feature_dtypes)

    for frame in (X_train, X_test):
        bool_cols = frame.select_dtypes(include=["bool"]).columns
        if len(bool_cols):
            frame[bool_cols] = frame[bool_cols].astype("int8")

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Drop NaN rows
    mask_train = X_train.notna().all(axis=1) & y_train.notna()
    mask_test = X_test.notna().all(axis=1) & y_test.notna()
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    # float64 -> float32 roughly halves the training matrix's memory
    # footprint; LightGBM handles float32 natively and the extra precision
    # is not meaningful for penny-level prices. Added after real training
    # runs on the full dataset repeatedly froze a 15 GB development
    # machine -- see project_v2.md Phase 1 Progress Log, 2026-07-02.
    downcast_dtypes(X_train)
    downcast_dtypes(X_test)

    logger.info("Final train: %s, test: %s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: dict,
):
    """Train a LightGBM regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    lgbm_params : dict
        LightGBM hyperparameters from config.

    Returns
    -------
    lightgbm.LGBMRegressor
        Fitted model.
    """
    import lightgbm as lgb

    logger.info("Training LightGBM with params: %s", lgbm_params)
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)
    logger.info("Training complete. ✓")
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model performance on the test set.

    Parameters
    ----------
    model : LGBMRegressor
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, and R² metrics.
    """
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def run_training(settings: Settings) -> Path:
    """Execute the full training pipeline.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    Path
        Path to the saved model artifact.
    """
    feature_path = settings.data.processed_dir / settings.features.output_filename
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found at {feature_path}. Run feature engineering first.")

    logger.info("Loading feature data from %s …", feature_path)
    df = pd.read_parquet(feature_path, engine="pyarrow")
    downcast_dtypes(df)
    log_memory("after loading feature data")

    X_train, y_train, X_test, y_test = prepare_training_data(df)
    # The full feature frame (9.5M rows x 34 columns) is no longer needed
    # once split into X/y -- release it before training, which itself needs
    # substantial memory to build 1000 trees over ~5M rows.
    del df
    collect_garbage()
    log_memory("before model training")

    model = train_model(X_train, y_train, settings.model.lgbm_params)
    log_memory("after model training")
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    output_dir = settings.model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.model.model_path

    logger.info("Saving model to %s …", model_path)
    joblib.dump(model, model_path)

    metrics_path = output_dir / "metrics.json"
    metrics_record = {
        **metrics,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "lgbm_params": settings.model.lgbm_params,
        "git_sha": get_git_sha(),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_record, fh, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)

    logger.info("Training pipeline complete. MAE=£%.2f, RMSE=£%.2f", metrics["MAE"], metrics["RMSE"])

    return model_path
