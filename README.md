# PricePoint Dynamics: UK Supermarket Competitive Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/BhargavKumarNath/PricepointDynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/BhargavKumarNath/PricepointDynamics/actions)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js-black.svg)](https://nextjs.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-9cf.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)](https://shap.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-blue.svg)](https://faiss.ai/)
[![Sentence-BERT](https://img.shields.io/badge/NLP-Sentence--BERT-green.svg)](https://www.sbert.net/)
[![DuckDB](https://img.shields.io/badge/Analytics-DuckDB-FFF000.svg)](https://duckdb.org/)
[![Polars](https://img.shields.io/badge/Data-Polars-CD792C.svg)](https://pola.rs/)
[![Pandera](https://img.shields.io/badge/Validation-Pandera-blueviolet.svg)](https://pandera.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Testing-Pytest-red.svg)](https://pytest.org/)
[![Data: 9.5M rows](https://img.shields.io/badge/Data-9.5M%20rows-informational.svg)](#)

## 📖 Executive Summary

PricePoint Dynamics is an end-to-end machine learning pipeline and
analytics engine that decodes the pricing strategies of the "Big 5" UK
supermarkets: **Tesco, Sainsbury's, ASDA, Morrisons, and Aldi**, from
9,529,216 daily price records (791MB raw CSV).

It solves the "product matching problem" across competing retailers using
Sentence-BERT + FAISS semantic clustering (68,596 canonical products
identified), trains a LightGBM regressor to forecast daily prices
(**MAE: £0.15**), and uses SHAP to explain what drives each prediction.

Originally prototyped in notebooks, this repository has been fully
refactored into a **production-shaped system**: a typed, tested,
`src/`-layout Python package; a narrow FastAPI service for the one thing
that genuinely needs live compute; a statically-exported Next.js
dashboard for everything that doesn't; DuckDB for analytics; and a CI
pipeline that gates every layer. `project_refactor.md` is the living plan
and progress log behind this refactor; `docs/` (also published as a
browsable site, see below) is the polished, reader-facing write-up of the
same system.

**📚 [Full documentation](docs/index.md)** — architecture, API reference,
development guide, deployment guide, and 8 Architecture Decision Records
covering the real bugs found and fixed along the way (target leakage,
non-transitive product matching, and more).

---

## 🏗️ System Architecture

The system is three layers, each independently deployable:

1. **The pipeline** (`src/pricepoint/`, CLI: `run.py`) — ingests, cleans,
   validates, matches, feature-engineers, trains, and precomputes
   analytics over the full 9.5M-row dataset.
2. **The API** (`api/`) — a deliberately narrow FastAPI service:
   `GET /health`, `POST /v1/predict` (the one live-inference call), and
   `GET /v1/products/{id}/history`. Everything else the dashboard needs
   is a precomputed static artifact, not a live endpoint — see
   [ADR-0005](docs/adr/0005-static-first-zero-wait-frontend.md) for why.
3. **The frontend** (`frontend/`) — a statically-exported Next.js
   dashboard. 5 of 6 pages read committed JSON/Parquet artifacts at
   build time with zero backend calls; only the Price Predictor page
   calls the API, and only on explicit user action.

A legacy Streamlit app (`dashboard/`) from before this refactor is
retained for reference but is no longer the primary serving layer —
see `docs/architecture.md` for the full picture and a system diagram.

---

## 💻 Running It Locally

### 1. Setup

```bash
git clone https://github.com/BhargavKumarNath/PricepointDynamics.git
cd PricepointDynamics
uv sync --all-extras
```

Dependencies are managed via [`uv`](https://docs.astral.sh/uv/) and a
single `pyproject.toml` (no separate `requirements.txt` files, no
drifting version pins). `uv sync` installs the pipeline, API, dashboard,
and dev tooling (pytest, ruff, mypy, pre-commit) from the locked
`uv.lock`. Prefix commands with `uv run`, or activate the venv directly
with `source .venv/bin/activate`.

Install the pre-commit hook once: `uv run pre-commit install`.

### 2. Get the data (optional — needed for a real pipeline run)

```bash
uv sync --extra kaggle
uv run python scripts/download_raw_data.py
```

Requires a free Kaggle account + API token. See `docs/development.md`.

### 3. Run the pipeline

```bash
uv run python run.py ingest        # clean + validate
uv run python run.py match         # Sentence-BERT + FAISS product matching
uv run python run.py features      # rolling/lag/competitive features
uv run python run.py train         # train LightGBM, write metrics.json
uv run python run.py marts         # materialize DuckDB analytics marts
uv run python run.py precompute    # SHAP + market dynamics
uv run python run.py hhi           # Herfindahl-Hirschman Index
uv run python run.py anomaly       # Isolation Forest anomaly detection
uv run python run.py web-artifacts # export frontend JSON/Parquet artifacts
```

Every stage is idempotent — re-running with unchanged inputs is a no-op.
Full detail in `docs/data_pipeline.md`.

### 4. Run the API

```bash
uv run uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

### 5. Run the frontend

```bash
cd frontend
npm ci
npm run dev
```

### (Legacy) Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Kept for reference; not actively developed against — see
`docs/architecture.md` for why the frontend/API pair superseded it.

---

## 🔬 Technical Highlights

### Rigorous software engineering

- **Config-driven:** zero hardcoded paths or hyperparameters — everything
  routes through `config.yaml` via `pydantic-settings`, with
  `PRICEPOINT_<SECTION>__<FIELD>` environment-variable overrides for
  deployed environments.
- **Data contracts:** Pandera schemas enforced at every pipeline
  boundary (raw, canonical, feature-engineered) — a schema violation
  fails the run loudly, not silently.
- **Manifests + run reports:** every stage output gets a
  `<artifact>.manifest.json` (what produced it, schema hash, skip-if-
  unchanged tracking) and a timestamped
  `data/_run_reports/<stage>_<timestamp>.json` (rows in/out, null
  counts, duration) — real observability, not a dashboard tool
  disproportionate to a solo-maintained pipeline.
- **CI that actually gates every layer:** 5 GitHub Actions jobs —
  lint/type-check (`ruff`, `mypy`, covering both `src/pricepoint` and
  `api/`), unit tests, API contract tests, a full
  ingest→match→features→train pipeline-integration test against fixture
  data, and a frontend build + type-check + Playwright e2e job — plus
  Dependabot on pip/npm/GitHub Actions.
- **195 tests** (`uv run pytest tests/ --collect-only -q`), run on every
  push/PR — see `docs/development.md` for how to run each group locally.

### The "apples-to-apples" problem (NLP)

Simple fuzzy matching fails across 100K+ product name variants. This
project uses Sentence-BERT to extract semantic embeddings and FAISS for
fast similarity search — but naive similarity-threshold clustering
**chains unrelated products together** through generic hub phrases at
this corpus's scale (a real failure mode found and fixed:
[ADR-0007](docs/adr/0007-bounded-degree-mutual-k-clustering.md)). The fix
is a bounded-degree mutual-nearest-neighbour graph, which caps how many
products any single "hub" name can pull into its cluster while still
resolving genuine multi-hop chains — expanding the match rate from
~3,000 to 68,596 canonical cross-retailer products.

### Predictive modeling & interpretability

- **Algorithm:** LightGBM, optimized for MAE (robust to raw-scraping
  outliers, unlike MSE).
- **No target leakage, by construction:** competitive features
  (same-day market average, price rank) are computed *leave-one-out* —
  a row's own price never contributes to its own "market" comparison
  ([ADR-0006](docs/adr/0006-no-target-leakage-leave-one-out.md)), and a
  same-row derived rescaling column (`prices_unit`) is explicitly
  excluded from model input for the same reason
  ([ADR-0018](docs/adr/0018-exclude-same-row-derived-features.md)).
- **Train/serve consistency, by construction:** the API's
  `POST /v1/predict` builds its feature vector via the exact same
  encoding function training uses, not a hand-maintained second copy —
  verified byte-identical against a direct `model.predict()` call.
- **Performance:** MAE **£0.1487**, RMSE £1.2589, R² 0.9652 on 5,151,151
  train / 444,314 test rows, 42 features (`models/metrics.json` is the
  live source of truth this number is quoted from).
- **Explainability:** precomputed SHAP matrices, browsable client-side in
  the frontend's Model Insights page with no live model call needed —
  a prediction's exact value is reconstructed from
  `base_value + sum(shap_values)`.

### Performance work, measured not assumed

- Feature-engineering rolling/lag statistics rewritten on Polars: **~150x**
  speedup on a 10% sample versus the original pandas
  `groupby().transform(lambda ...)` pattern
  ([ADR-0002](docs/adr/0002-polars-for-feature-engineering.md)).
- CSV ingestion rewritten on Polars: **3x** speedup on the full 791MB raw
  dataset.
- Market-overview and basket-cost queries moved from pandas-in-Streamlit
  to DuckDB over Parquet marts: **4.6x** / **15.7x** speedup
  ([ADR-0001](docs/adr/0001-duckdb-not-postgres.md)).
- Full methodology and regenerable numbers in `docs/benchmarks.md`
  (`make benchmark`).

---

## 📊 Key Findings & Market Insights

1. **The budget anchor:** Aldi consistently defines the absolute price
   floor — SHAP analysis shows the `supermarket=Aldi` feature
   systematically depresses price predictions across categories.
2. **The mainstream lockstep:** Tesco and Sainsbury's price in
   near-parallel; cross-correlation analysis shows Tesco leading, with
   Sainsbury's typically following with a measurable lag.
3. **Predictors of price:** short-horizon rolling price statistics are
   the strongest predictors, but deviation from the daily market average
   is consistently a top-tier predictor — retailers algorithmically
   react to competitor signals, not just their own price history.

---

## 📂 Project Structure

```text
PricepointDynamics/
├── .github/workflows/       # CI (5 jobs) + docs-site deploy
├── api/                      # FastAPI service (routers/schemas/services)
├── configs/                  # baskets.yaml and other non-pipeline config
├── dashboard/                 # Legacy Streamlit app (reference only)
├── docs/                      # Published documentation (docs/index.md, adr/)
├── frontend/                  # Next.js static-export dashboard
├── notebooks/                 # Original exploratory analysis
├── scripts/                   # One-off utilities (data acquisition, benchmarks)
├── sql/marts/                 # Parameterized DuckDB mart-materialization SQL
├── src/pricepoint/            # Core pipeline package (installable, src/-layout)
│   ├── config.py               # pydantic-settings configuration
│   ├── schemas.py               # Pandera data contracts
│   ├── data_ingestion.py        # Clean + validate
│   ├── product_matching.py      # Sentence-BERT + FAISS
│   ├── feature_engineering.py   # Temporal/competitive/cyclical features
│   ├── training.py              # LightGBM train + serve-shared encoding
│   ├── warehouse.py             # DuckDB query layer over the marts
│   ├── manifest.py              # Artifact provenance sidecars
│   ├── run_reports.py           # Per-run observability reports
│   └── market_analysis.py, anomaly.py, marts.py, web_artifacts.py, ...
├── tests/                     # pytest — unit, API contract, pipeline-integration
├── run.py                     # Typer CLI orchestrator
├── config.yaml                 # Central pipeline configuration
├── mkdocs.yml                  # Documentation site config
├── pyproject.toml              # Single dependency source of truth (uv)
└── uv.lock                     # Locked dependency versions
```

---

*Author: Bhargav Kumar Nath*
*Data Science | ML Engineering | Strategy*
