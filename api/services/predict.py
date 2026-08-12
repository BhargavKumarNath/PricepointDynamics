"""Service function backing api/routers/predict.py.

This is the piece that closes defect #6: the old
dashboard/pages/03_price_predictor.py fabricated 2 of the model's 42
features via arbitrary constant multipliers
(`price_rol_min_7d = price_rol_mean_7d * 0.9`) and zero-filled the
remaining ~37 via a blind `reindex(fill_value=0)`. Everything here
resolves from a real historical row instead -- see
`pricepoint.training.build_feature_vector` for the shared
train/serve-consistent encoding this relies on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMRegressor

from api.schemas.predict import PredictRequest, PredictResponse
from pricepoint.exceptions import ArtifactNotFoundError, DataValidationError
from pricepoint.training import build_feature_vector


def _resolve_feature_row(
    feature_data_path: Path, canonical_name: str, supermarket: str, as_of_date: date
) -> pd.DataFrame | None:
    """The most recent real historical row for (product, store) on or
    before `as_of_date`, via DuckDB predicate pushdown -- never loads the
    full 9.5M-row feature file into memory to answer one lookup."""
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute(
            """
            SELECT * FROM read_parquet(?)
            WHERE canonical_name = ? AND supermarket = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            [str(feature_data_path), canonical_name, supermarket, as_of_date],
        ).fetchdf()
    finally:
        conn.close()
    return row if not row.empty else None


def _product_store_exists(feature_data_path: Path, canonical_name: str, supermarket: str) -> bool:
    """Whether (product, store) has *any* history at all, regardless of
    date -- used to distinguish "unknown product/store" (404) from
    "known product/store, but nothing on or before this date" (422)."""
    conn = duckdb.connect(":memory:")
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE canonical_name = ? AND supermarket = ? LIMIT 1",
            [str(feature_data_path), canonical_name, supermarket],
        ).fetchone()
    finally:
        conn.close()
    return bool(count and count[0] > 0)


def predict_price(
    model: LGBMRegressor,
    feature_data_path: Path,
    date_bounds: tuple[date, date],
    request: PredictRequest,
) -> PredictResponse:
    """Resolve a real feature vector and predict a price.

    Every feature except an optional ``price_lag_1d`` override
    (project_refactor.md §8.1's "genuinely user-controllable" field, see
    Phase 4's Progress Log for the reasoning) comes from the product's
    actual last-known observation on or before the requested date --
    never fabricated, never zero-filled without saying so.

    Raises
    ------
    DataValidationError
        If the requested date falls outside the dataset's observed range.
    ArtifactNotFoundError
        If the (product, supermarket) pair has no history at all.
    """
    min_date, max_date = date_bounds
    if not (min_date <= request.date <= max_date):
        raise DataValidationError(
            f"date {request.date} is outside the dataset's observed range ({min_date} to {max_date}).",
            context={"min_date": str(min_date), "max_date": str(max_date), "requested_date": str(request.date)},
        )

    row = _resolve_feature_row(feature_data_path, request.canonical_name, request.supermarket, request.date)
    if row is None:
        if _product_store_exists(feature_data_path, request.canonical_name, request.supermarket):
            raise DataValidationError(
                f"{request.canonical_name!r} at {request.supermarket!r} has no observations "
                f"on or before {request.date} (its history starts later).",
                context={"canonical_name": request.canonical_name, "supermarket": request.supermarket},
            )
        raise ArtifactNotFoundError(
            f"No history found for {request.canonical_name!r} at {request.supermarket!r}.",
            context={"canonical_name": request.canonical_name, "supermarket": request.supermarket},
        )

    resolved_from_date = row.iloc[0]["date"]
    input_vector, unresolved = build_feature_vector(row, model.feature_name_, price_override=request.price_override)
    predicted_price = float(model.predict(input_vector)[0])

    return PredictResponse(
        canonical_name=request.canonical_name,
        supermarket=request.supermarket,
        requested_date=request.date,
        resolved_from_date=resolved_from_date,
        predicted_price=predicted_price,
        price_override_applied=request.price_override is not None,
        unresolved_features=unresolved,
    )
