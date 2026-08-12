"""GET /health -- liveness/readiness probe."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()
