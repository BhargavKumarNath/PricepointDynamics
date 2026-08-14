# PricePoint Dynamics

UK supermarket competitive intelligence: an end-to-end pipeline that
cleans and semantically matches 9.5M+ daily price records across 5
retailers (ASDA, Aldi, Morrisons, Sainsbury's, Tesco), engineers
temporal/competitive features, trains a LightGBM price predictor (MAE
£0.15), and serves the result through a statically-exported dashboard
plus one live prediction endpoint.

- **New here?** Start with [Architecture](architecture.md) for the
  system-wide picture, then [Data pipeline](data_pipeline.md) for
  stage-by-stage detail.
- **Running it locally?** [Development](development.md) has setup,
  the full pipeline command sequence, and how to run the tests.
- **Deploying it?** [Deployment](deployment.md) covers each piece
  (Vercel, Cloud Run, Cloudflare R2) and what's actually been
  provisioned versus designed-but-not-yet-deployed.
- **Curious about a specific decision?** [Architecture Decision
  Records](adr/README.md) — DuckDB over Postgres, Polars for the
  performance-critical stages, why an orchestrator was rejected, and the
  two target-leakage bugs found and fixed during this project's
  refactor.
- **Want the numbers?** [Benchmarks](benchmarks.md) for measured
  before/after performance, [Cost verification](cost_verification.md)
  for the free-tier claims and what's honestly unverifiable without a
  live deployment.

The source of truth for the full refactor plan, its reasoning, and a
running progress log is `project_refactor.md` at the repo root — this
site is the polished, reader-facing companion to that living working
document, not a replacement for it.
