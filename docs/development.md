# Development

## Setup

```bash
git clone <repo-url>
cd PricePoint_Dynamics
uv sync --all-extras
```

[`uv`](https://docs.astral.sh/uv/) creates `.venv` and installs the
pipeline, API, and dev tooling (pytest, ruff, mypy, pre-commit) from the
locked `uv.lock`. Prefix commands with `uv run` (e.g.
`uv run python run.py ingest`), or activate the venv directly with
`source .venv/bin/activate`. A plain `pip install -e ".[dev]"` also works
if you'd rather not use `uv`.

Install the pre-commit git hook once (ruff, formatting, `detect-secrets`
run automatically on every commit):

```bash
uv run pre-commit install
```

For the frontend: `cd frontend && npm ci`.

## Environment variables

Copy `.env.example` → `.env` at the repo root for backend-side overrides
(Kaggle dataset slug, `PRICEPOINT_<SECTION>__<FIELD>` config overrides).
Copy `frontend/.env.example` → `frontend/.env.local` for frontend-side
overrides (API base URL, Parquet artifact base URL). Neither is required
for local development against default values.

## Getting real data

`config.yaml` assumes 5 raw CSVs already exist in `data/00_raw/`. To
acquire them:

```bash
uv run python scripts/download_raw_data.py
```

Requires a free Kaggle account + API token (`~/.kaggle/access_token` or
`KAGGLE_API_TOKEN`) — install the `kaggle` extra first:
`uv sync --extra kaggle`.

## Running the pipeline

```bash
uv run python run.py ingest      # clean + validate raw CSVs
uv run python run.py match       # Sentence-BERT + FAISS product matching
uv run python run.py features    # rolling/lag/competitive features
uv run python run.py train       # train LightGBM, write metrics.json
uv run python run.py marts       # materialize DuckDB marts
uv run python run.py precompute  # SHAP + market dynamics
uv run python run.py hhi         # Herfindahl-Hirschman Index
uv run python run.py anomaly     # Isolation Forest anomaly detection
uv run python run.py web-artifacts  # export frontend JSON/Parquet artifacts
```

Every stage is idempotent — re-running with unchanged inputs is a no-op
(logged as "skipping"), unless `--force`/`-f` is passed. See
`docs/data_pipeline.md` for what each stage actually does.

## Running the API locally

```bash
uv run uvicorn api.main:app --reload
```

Requires a trained model (`run.py train`) and materialized marts
(`run.py marts`) to serve real requests; `/docs` works regardless.

## Running the frontend locally

```bash
cd frontend
npm run dev
```

Reads the committed static artifacts directly; no backend required
unless you're exercising the Price Predictor page's live calls (start the
API above first, or rely on the mocked Playwright tests instead of
manual clicking).

## Tests

```bash
make test              # full suite: uv run pytest tests/ -v
uv run pytest tests/test_pipeline_integration.py -v   # the one full ingest->train integration test
uv run pytest tests/api -v                             # API contract tests (self-contained, no real data needed)
cd frontend && npm run test:e2e                        # Playwright, against the static build + mocked Predictor API
```

`tests/api/` and `tests/test_pipeline_integration.py` are both
self-contained: neither touches `data/`/`models/` (gitignored, absent in
a fresh checkout) — they build small synthetic/fixture data instead. See
`project_refactor.md` §11 for the full testing strategy and §13 for how
each test group maps onto a CI job.

## Lint, format, type-check

```bash
make lint       # ruff check + ruff format --check
make format     # ruff format + ruff check --fix
make typecheck  # mypy src/pricepoint (CI additionally checks api/)
```

## Benchmarks

```bash
make benchmark          # Phase 2: ingestion + feature engineering + price leadership
make benchmark-phase3   # Phase 3: DuckDB marts vs. pandas-in-Streamlit
```

Regenerates `docs/benchmarks.md` and the plots under `plots/` — see
`project_refactor.md` §15 for methodology.

## Docs site

```bash
uv sync --extra docs   # installs mkdocs-material
mkdocs serve           # live preview at http://localhost:8000
mkdocs build --strict  # what CI runs before deploying to GitHub Pages
```
