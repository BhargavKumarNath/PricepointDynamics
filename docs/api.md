# API

The live, authoritative API reference is FastAPI's auto-generated
OpenAPI/Swagger UI at `/docs` (and ReDoc at `/redoc`) on a running
instance — this page is a curated narrative companion, not a duplicate
spec to keep hand-in-sync.

Base URL: `http://localhost:8000` locally (`uv run uvicorn api.main:app
--reload`); the deployed Cloud Run URL in production
(`NEXT_PUBLIC_API_BASE_URL` on the frontend side).

## Why only 3 endpoints

The API surface is deliberately narrow — see
[ADR-0005](adr/0005-static-first-zero-wait-frontend.md). 5 of 6 dashboard
pages need nothing but a static file fetch; only the Price Predictor's
"Predict" action is inherently unbounded and can't be precomputed. Every
endpoint originally scoped in `project_refactor.md` §8.1 beyond these 3
is replaced by a precomputed static/R2 artifact, each with a named,
concrete trigger to bring it back as a live endpoint if a specific future
feature needs it (§23) — one such promotion has already happened (the
history endpoint below).

## Endpoints

### `GET /health`

Liveness check. Also the target of a scheduled keep-alive ping
(`project_refactor.md` §25.5) to keep Cloud Run's request-processing-time
billing nearly free under continuous polling.

### `POST /v1/predict`

The one live-inference endpoint. Rate-limited (`slowapi`, default
`30/minute`, configurable via `settings.api.predict_rate_limit`).

**Request** (`PredictRequest`):

| Field | Type | Notes |
|---|---|---|
| `canonical_name` | string | Required. |
| `supermarket` | string | Required. |
| `date` | date | Required. Must fall within the dataset's observed range (422 otherwise). |
| `price_override` | float, optional | The **only** user-controllable model input — substituted into the resolved `price_lag_1d` ("yesterday's price") feature as a "what if" scenario. Every other feature is resolved from real history, never fabricated or accepted from the client. |

**Response** (`PredictResponse`): `predicted_price`, the actual
`resolved_from_date` the feature vector came from, whether an override
was applied, and `unresolved_features` — any model feature that had no
real historical value for this product/store/date (e.g. no competitor
listed that day) and was passed to the model as a genuine missing value
(LightGBM handles this natively), never zero-filled or fabricated. This
is the fix for the legacy dashboard's defect #6 (fabricated predictor
inputs) — see `project_refactor.md` §2, defect 6.

The feature vector is built by `training.py::build_feature_vector`, the
*same* function (same categorical encoding, same column selection) used
at training time — a served prediction is provably equivalent to a
direct `model.predict()` call on identically-resolved historical data.

### `GET /v1/products/{canonical_id}/history`

Optional `?supermarket=` query param to filter to one retailer; omitted
returns history across all retailers that sold the product. Backed by
`Warehouse.get_product_history` (DuckDB over `fact_price_daily`). An
unknown product, or a known product with no history at the requested
retailer, is a 404 — never an empty 200 masquerading as "no data."

Promoted from a static artifact to a live endpoint when the Price
Predictor page needed a real historical price chart alongside the model
prediction — see [ADR-0005](adr/0005-static-first-zero-wait-frontend.md)'s
"Consequences" section for the full promotion story.

## Error handling

All domain errors are `pricepoint.exceptions.PricePointError` subclasses,
mapped centrally in `api/main.py` (never a per-route `try/except`):

| Exception | HTTP status | When |
|---|---|---|
| `ArtifactNotFoundError` | 404 | Referenced product/artifact doesn't exist. |
| `DataValidationError` | 422 | Request is well-formed but fails a domain rule (e.g. date out of range). |
| `MartNotFoundError` | 404 | A required mart Parquet file is missing on disk. |
| `PricePointError` (bare) | 500 | Any other domain error — still a structured JSON body, never a bare stack trace. |
| `RateLimitExceeded` | 429 | `/v1/predict` rate limit hit. |

Every error response is `ErrorResponse {detail: str, context: dict}` —
`context` carries structured, client-actionable detail (e.g. the valid
date range), not just a human-readable message.

## CORS

Restricted to explicit configured origins (`settings.api.cors_origins`),
never `*` — verified by `tests/api/test_cors.py`, not just asserted in
config. Override via the `PRICEPOINT_API__CORS_ORIGINS` environment
variable (a JSON list) once a real frontend domain exists.
