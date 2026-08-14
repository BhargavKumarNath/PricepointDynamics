# Deployment

This describes how each piece is *designed* to deploy
([ADR-0005](adr/0005-static-first-zero-wait-frontend.md)) and what has
and hasn't actually been provisioned. Nothing in this project has been
deployed to real Vercel/Cloud Run/Cloudflare R2 infrastructure as of
Phase 7 — see `docs/cost_verification.md` for what that means for the
project's cost claims, and why: deploying requires real accounts and
billing-enabled credentials this development environment doesn't have
and shouldn't provision unilaterally.

## Frontend → Vercel

Static export (`next build`, `output: "export"`), deployed as a Vercel
project pointed at `frontend/`. No server-side rendering, no serverless
functions in the critical path — Vercel's function limits are largely
moot for this project.

**To deploy:** connect the GitHub repo to a new Vercel project, root
directory `frontend/`, and set `NEXT_PUBLIC_API_BASE_URL` (the Cloud Run
URL below) and optionally `NEXT_PUBLIC_PARQUET_BASE_URL` (the R2 bucket
URL) as Vercel environment variables — both are baked in at build time,
per Next.js's static-export env-var handling.

**Known limitation:** Vercel Hobby's ToS restricts free usage to
non-commercial/personal projects — a portfolio site is the standard
real-world use case this restriction anticipates, not a violation of it,
but it's worth stating plainly rather than glossing over.

## API → Google Cloud Run

Chosen over Render specifically for billing model, not raw price — see
[ADR-0005](adr/0005-static-first-zero-wait-frontend.md). Deploy as a
container built from the repo root (needs a `Dockerfile`, not yet
written — the API currently only has a documented local-run path via
`uvicorn`).

**Required for a real deployment:**
- A Google Cloud project with billing enabled (Cloud Run's free tier
  still requires a card on file, even though usage stays within the
  Always Free quota).
- The trained model artifact and materialized marts baked into the image
  or mounted at startup (`models/`, `data/03_marts/` are gitignored, not
  committed).
- `PRICEPOINT_API__CORS_ORIGINS` set to the real deployed frontend
  origin (a JSON list, e.g. `["https://your-project.vercel.app"]`) —
  never left at the `localhost` defaults in `config.yaml`.
- A scheduled keep-alive: `GET /health` every ~10 minutes via a GitHub
  Actions workflow, to keep Cloud Run warm under its
  request-processing-time billing model (nearly free under this model;
  see `docs/cost_verification.md` for why this specifically doesn't work
  out cheaply on Render's wall-clock-hour billing instead).

## Large artifacts → Cloudflare R2

`predictor_context.parquet` and `shap_explorer.parquet`
(`project_refactor.md` §25.3) are meant to live in a public-read R2
bucket (zero egress fees, unlike S3). Locally and in this repo, they are
committed as-is under `frontend/public/data-r2/` as a documented
stand-in — `NEXT_PUBLIC_PARQUET_BASE_URL` unset means the frontend reads
this local copy directly, no R2 account needed for local dev or a
from-source build.

**To actually use R2:** provision a bucket, upload the two Parquet files
(regenerate via `run.py web-artifacts` after any pipeline re-run), enable
public read, and set `NEXT_PUBLIC_PARQUET_BASE_URL` to the bucket's
public URL.

## CI/CD

`.github/workflows/ci.yml` gates every push/PR: lint+typecheck, unit
tests, API contract tests, the pipeline-integration test, and a frontend
build+e2e job — see `project_refactor.md` §13. `.github/workflows/docs.yml`
builds and deploys `docs/` to GitHub Pages on push to `main` (dormant
until GitHub Pages is enabled for this repo in Settings → Pages, source
"GitHub Actions").

## Failure behaviour

Designed, not aspirational (`project_refactor.md` §25.6): if the API is
fully down, only the Predictor page is affected (a designed fallback, not
a spinner or crash); a cold-starting API shows an honest "waking up"
message with a client-side timeout, not a silent hang; if R2 is
unavailable, a small committed fallback list keeps the Predictor's
product selector minimally functional.
