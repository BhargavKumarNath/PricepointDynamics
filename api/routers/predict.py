"""POST /v1/predict."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from lightgbm import LGBMRegressor

from api.dependencies import get_date_bounds, get_feature_data_path, get_model, get_settings
from api.rate_limit import limiter
from api.schemas.predict import PredictRequest, PredictResponse
from api.services.predict import predict_price

router = APIRouter(prefix="/v1", tags=["predict"])


def _predict_rate_limit() -> str:
    """Resolved lazily per request (not once at import time) so tests can
    point `get_settings` at fixture config before this is read."""
    return get_settings().api.predict_rate_limit


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(_predict_rate_limit)
def predict(
    request: Request,  # noqa: ARG001 -- required by slowapi's limiter decorator
    body: PredictRequest,
    model: LGBMRegressor = Depends(get_model),
    feature_data_path: Path = Depends(get_feature_data_path),
    date_bounds: tuple[date, date] = Depends(get_date_bounds),
) -> PredictResponse:
    return predict_price(model, feature_data_path, date_bounds, body)
