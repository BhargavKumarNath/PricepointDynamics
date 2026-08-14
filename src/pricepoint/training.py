"""LightGBM model training pipeline.

Handles train/test splitting, model fitting, evaluation, and artifact
serialization.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pricepoint.config import Settings
from pricepoint.manifest import get_git_sha
from pricepoint.memory_utils import collect_garbage, downcast_dtypes, log_memory
from pricepoint.run_reports import default_report_dir, write_run_report

logger = logging.getLogger(__name__)

# Columns from feature_engineered_data.parquet that are never real
# predictive features -- either the target itself, identifiers, or
# same-row target leakage. Shared between training-time
# (prepare_training_data) and serving-time (build_feature_vector) so both
# agree on exactly which raw columns are eligible to become model inputs
# -- the single most important place train/serve skew could otherwise
# creep in.
#
# prices_unit specifically: a per-unit rescaling of THIS SAME ROW's
# `prices` (e.g. £/kg derived from pack price / pack weight) -- within a
# given (canonical_name, supermarket) group the ratio between the two is
# close to constant (empirically true for ~82% of groups, verified
# against real data in Phase 1), so it is same-row target leakage,
# arguably more direct than the market-average leak already fixed
# (ADR-0006/0018): it doesn't even require cross-referencing other rows,
# just a near-fixed per-row rescaling of the value being predicted.
_NON_FEATURE_COLUMNS = [
    "prices",
    "date",
    "product_name",
    "canonical_name",
    "normalised_name",
    "prices_unit",
]


def _encode_categorical_features(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns, matching training's exact convention.

    "str" is listed alongside "object" for the same pandas >= 3.0
    forward-compatibility reason as elsewhere in this pipeline (a
    dedicated string dtype exists separately from "object" and is only
    still caught by an "object" query via a deprecated shim).
    """
    cat_cols = df.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def _select_model_feature_columns(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """Select numeric+bool columns (after dropping exclude_cols), casting bool -> int8.

    Bool must be selected explicitly alongside "number": a numeric-only
    select_dtypes(["number"]) silently drops every bool-dtype column (the
    raw own_brand flag, and every one-hot dummy pd.get_dummies() just
    created) with no error, meaning every categorical feature would
    vanish from the model input without a trace.
    """
    result = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=["number", "bool"])
    bool_cols = result.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        result[bool_cols] = result[bool_cols].astype("int8")
    return result


def build_feature_vector(
    row: pd.DataFrame,
    model_features: list[str],
    price_override: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a single-row model input from a real historical feature row.

    Applies the *exact same* categorical-encoding and column-selection
    transform as :func:`prepare_training_data` (via the shared
    ``_encode_categorical_features``/``_select_model_feature_columns``
    helpers above), so a served prediction can never silently diverge
    from what the model was actually trained on. This is the mechanism
    that makes an API-served prediction genuinely equivalent to
    ``model.predict()`` on identically-resolved input, not merely
    similar -- see ``tests/api/test_predict.py`` for the byte-for-byte
    comparison this enables.

    Unlike training, missing (NaN) feature values are **not** dropped: a
    single real historical row may legitimately have NaN in a rolling/lag
    column (e.g. a product with less than 30 days of history has no
    ``price_rol_mean_30d`` yet), and LightGBM handles NaN inputs natively
    at inference time (it learns a default split direction for missing
    values during training) -- there is no "just drop this row" option
    for a single prediction request. The caller is told which features
    were unresolved so the API can report that honestly rather than let a
    silent NaN pass through un-flagged (project_refactor.md §8.1: "if a
    feature can't be resolved, the response says so explicitly rather
    than guessing").

    Parameters
    ----------
    row : pd.DataFrame
        A single-row DataFrame with the raw feature-engineered columns
        for one (canonical_name, supermarket, date) observation, as
        looked up from real history (e.g. from
        ``feature_engineered_data.parquet``) -- never fabricated.
    model_features : list[str]
        The trained model's exact expected column list/order
        (``model.feature_name_``).
    price_override : float, optional
        If given, overrides ``price_lag_1d`` before encoding -- the one
        field genuinely user-controllable via `POST /v1/predict` (a
        "what if yesterday's price were different" scenario). Every
        other feature remains the real resolved historical value;
        nothing else is user-settable.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (single-row input ready for ``model.predict()``, names of the
        requested ``model_features`` that could not be resolved --
        genuinely NaN in the source row, not the expected zero-fill of an
        absent one-hot category).
    """
    row = row.copy()
    if price_override is not None and "price_lag_1d" in row.columns:
        row["price_lag_1d"] = price_override

    drop_cols = [c for c in _NON_FEATURE_COLUMNS if c in row.columns]
    encoded = _encode_categorical_features(row, exclude_cols=drop_cols)
    selected = _select_model_feature_columns(encoded, exclude_cols=drop_cols)

    input_vector = selected.reindex(columns=model_features, fill_value=0)
    unresolved = input_vector.columns[input_vector.isna().any(axis=0)].tolist()

    return input_vector, unresolved


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

    target_col = "prices"
    drop_cols = [c for c in _NON_FEATURE_COLUMNS if c in train.columns]

    train = _encode_categorical_features(train, exclude_cols=drop_cols)
    test = _encode_categorical_features(test, exclude_cols=drop_cols)

    y_train = train[target_col]
    y_test = test[target_col]

    X_train = _select_model_feature_columns(train, exclude_cols=drop_cols)
    X_test = _select_model_feature_columns(test, exclude_cols=drop_cols)

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

    start_time = time.perf_counter()
    logger.info("Loading feature data from %s …", feature_path)
    df = pd.read_parquet(feature_path, engine="pyarrow")
    rows_in = len(df)
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

    # df_out is X_train here, not the full loaded feature frame (already
    # released above) -- rows_out therefore means "rows actually used for
    # training" (post NaN-drop/time-split), a more useful signal than the
    # raw load count. Test-set size and the run's own metrics are folded
    # in via `extra` since they're already computed and directly relevant
    # to spotting a training regression, not just a row-count one.
    write_run_report(
        "training",
        default_report_dir(settings),
        rows_in=rows_in,
        df_out=X_train,
        duration_seconds=time.perf_counter() - start_time,
        extra={"test_rows": len(X_test), "mae": metrics["MAE"], "rmse": metrics["RMSE"]},
    )

    return model_path
