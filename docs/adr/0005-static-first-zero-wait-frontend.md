# ADR-0005: Static-first, zero-wait frontend architecture

**Status:** Accepted (locked)
**Date:** 2026-08-12; one promotion 2026-08-13 (see Consequences)

## Context

The primary requirement driving this project's deployment shape: *a
recruiter opening the project URL must see and interact with the
dashboard without waiting on a sleeping backend.* A dedicated
page-by-page review (project_refactor.md §25.1) found that **5 of the 6
dashboard pages (Home, Market Overview, Basket Analysis, Model Insights,
Market Dynamics) need nothing but a static file fetch** — no FastAPI, no
live compute, no model inference. Only the Price Predictor's "Predict"
button is an inherently unbounded, user-chosen input that cannot be
precomputed.

This finding directly superseded two earlier, more conservative designs:
§8.1's original 9-endpoint API list, and §9/§10's open "Render or Cloud
Run" framing.

## Decision

- **Frontend:** Next.js, statically exported (SSG), for all 5
  non-Predictor pages. Static pages served from a CDN have no cold start
  — there is no compute in the request path at all for 5/6 of the
  product.
- **API surface:** narrowed to the genuine minimum —
  `POST /v1/predict` and `GET /health`. Every other endpoint originally
  proposed is replaced by a precomputed static (small JSON, committed to
  the frontend repo) or Parquet (Cloudflare R2, free egress) artifact,
  exported by a new pipeline step (`web_artifacts.py`).
- **Hosting:** Google Cloud Run for the API, chosen specifically for its
  request-processing-time billing model — a GitHub Actions keep-alive
  ping (`GET /health` every ~10 min) is nearly free under this model,
  where Render's free tier's wall-clock-hour billing would consume
  almost its entire monthly budget doing the same thing. This is the
  deciding factor, not raw sticker price (§25.5).
- **Endpoint promotion is trigger-gated, not speculative:** each replaced
  endpoint has a named, concrete trigger for being brought back as a live
  endpoint (e.g. "arbitrary historical drill-down" → promote
  `/v1/products/{id}/history`), per §23's governing rule that an
  interface is justified by a concrete, plausible second use, not
  spare-capacity planning.

## Consequences

- Zero-wait is achieved by *not having a server in the request path* for
  5/6 of the product — not by caching tricks layered on top of a
  server-backed design.
- API failure/cold-start only ever affects the Predictor page, which has
  a designed fallback and an honest "waking up, up to ~15s" state, never
  a spinner or crash on the other 5 pages (§25.6).
- **Promotion actually happened once:** `GET /v1/products/{canonical_id}/history`
  was promoted to a live endpoint (UI_refactor.md, 2026-08-13) when the
  Price Predictor redesign needed a real historical price chart — exactly
  the trigger this ADR named in advance. Low-cost because
  `Warehouse.get_product_history` (Phase 3) already existed and was
  already tested; only router/schema/service wiring was needed. This is
  evidence the trigger-gated design works as intended, not a reversal of
  the narrowing decision.
- Nothing in this architecture has been deployed to real Vercel/Cloud
  Run/R2 infrastructure as of Phase 7 — building, local verification, and
  a dormant deploy-workflow are as far as this environment can go without
  provisioning real accounts/credentials (see `docs/deployment.md` and
  `docs/cost_verification.md`).

## Alternatives considered

- **Server-rendered Next.js on every page** — rejected; would reintroduce
  a cold-start-prone compute path for pages that don't need one.
- **Render for the API** — considered, not chosen; simpler zero-friction
  setup, but its wall-clock-hour billing model is the wrong fit for a
  keep-alive-ping strategy (§25.5).
