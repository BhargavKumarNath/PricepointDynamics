"""FastAPI app factory (project_refactor.md §25.2 -- the locked, minimal
API surface: `GET /health` and `POST /v1/predict` only. §8.1's original
9-endpoint list is superseded; the other 7 are replaced by precomputed
static/R2 artifacts for the dashboard's current scope, each with a named
promotion trigger back to a live endpoint if a specific future feature
needs it -- see §25.2).

Assembles the routers in `api/routers/` behind one app, with:
- CORS restricted to explicit configured origins, not `*` (§16).
- `slowapi` in-process rate limiting on `/v1/predict` (§16) -- no Redis,
  proportional to a public read-mostly analytics API's actual threat model.
- Domain exceptions (`pricepoint.exceptions.PricePointError` and its
  subclasses) mapped to HTTP status codes in one place (§8.2), so route
  handlers and services never need their own try/except -> HTTPException
  translation.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from api.dependencies import get_settings
from api.rate_limit import limiter
from api.routers import health, predict
from api.schemas.common import ErrorResponse
from pricepoint.exceptions import ArtifactNotFoundError, DataValidationError, PricePointError

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="PricePoint Dynamics API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_exception_handler(ArtifactNotFoundError, _artifact_not_found_handler)
    app.add_exception_handler(DataValidationError, _data_validation_handler)
    app.add_exception_handler(PricePointError, _pricepoint_error_handler)

    app.include_router(health.router)
    app.include_router(predict.router)

    return app


def _rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RateLimitExceeded)
    response = JSONResponse(
        status_code=429,
        content=ErrorResponse(detail=f"Rate limit exceeded: {exc.detail}", context={}).model_dump(),
    )
    return limiter._inject_headers(response, request.state.view_rate_limit)


def _artifact_not_found_handler(request: Request, exc: Exception) -> Response:  # noqa: ARG001
    assert isinstance(exc, ArtifactNotFoundError)
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(detail=exc.message, context=exc.context).model_dump(),
    )


def _data_validation_handler(request: Request, exc: Exception) -> Response:  # noqa: ARG001
    assert isinstance(exc, DataValidationError)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=exc.message, context=exc.context).model_dump(),
    )


def _pricepoint_error_handler(request: Request, exc: Exception) -> Response:  # noqa: ARG001
    """Catch-all for any `PricePointError` subclass not handled more
    specifically above -- still a structured 500, never an unhandled
    stack-trace leak (project_refactor.md §16)."""
    assert isinstance(exc, PricePointError)
    logger.exception("Unhandled PricePointError", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail=exc.message, context=exc.context).model_dump(),
    )


app = create_app()
