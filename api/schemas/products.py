"""Schemas for GET /v1/products/{canonical_id}/history.

The one endpoint promoted back from §25.2's superseded §8.1 list --
UI_refactor.md's Price Predictor redesign needed a real historical price
chart, the named "arbitrary historical drill-down" trigger §25.2 already
anticipated for this specific endpoint.
"""

from __future__ import annotations

# Aliased for the same PEP 563 reason as api/schemas/predict.py: a field
# named `date` would otherwise clash with this same-named type under
# deferred annotation evaluation.
from datetime import date as _date

from pydantic import BaseModel, Field


class ProductHistoryPoint(BaseModel):
    date: _date
    supermarket: str
    avg_price: float
    min_price: float
    max_price: float
    n_listings: int = Field(..., description="Number of individual listings averaged into this day's price.")


class ProductHistoryResponse(BaseModel):
    canonical_name: str
    history: list[ProductHistoryPoint] = Field(
        default_factory=list,
        description="Daily price history, ordered by date. Empty history is reported as a 404, not an empty 200.",
    )
