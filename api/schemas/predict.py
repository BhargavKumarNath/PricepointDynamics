"""Schemas for POST /v1/predict.

See api/services/predict.py and project_refactor.md Phase 4's Progress
Log for the reasoning behind exactly which single field is
user-overridable here, and why every other feature is resolved from real
history rather than accepted from the client.
"""

from __future__ import annotations

# Aliased: a field named `date` on these models would otherwise clash
# with this same-named type annotation under PEP 563 deferred evaluation
# (`from __future__ import annotations` above) -- pydantic resolves
# annotations as strings against this module's namespace, where `date`
# the field and `date` the type are the same name.
from datetime import date as _date

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    canonical_name: str = Field(..., description="The canonical product name to predict a price for.")
    supermarket: str = Field(..., description="The retailer to predict for.")
    date: _date = Field(..., description="The date to predict for. Must fall within the dataset's observed range.")
    price_override: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional: substitute this value for the resolved 'yesterday's price' "
            "(price_lag_1d) feature -- a 'what if yesterday's price were different' "
            "scenario. Every other feature is always resolved from real history; "
            "this is the only user-controllable input."
        ),
    )


class PredictResponse(BaseModel):
    canonical_name: str
    supermarket: str
    requested_date: _date
    resolved_from_date: _date = Field(
        ..., description="The actual historical observation date the feature vector was resolved from."
    )
    predicted_price: float
    price_override_applied: bool
    unresolved_features: list[str] = Field(
        default_factory=list,
        description=(
            "Model features that had no real historical value for this product/store/date "
            "(e.g. no competitor data on that date) and were passed to the model as missing, "
            "not fabricated or zero-filled -- LightGBM handles missing values natively."
        ),
    )
