"""Schemas shared across routers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Structured error body for every non-2xx response (project_refactor.md §8.2).

    ``context`` carries machine-readable detail (e.g. the valid date
    range for a rejected `/v1/predict` request) so a client can act on
    the failure programmatically, not just display ``detail`` to a user.
    """

    detail: str
    context: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
