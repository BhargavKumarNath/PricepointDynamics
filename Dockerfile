# PricePoint Dynamics -- API serving image (project_refactor.md §25.5,
# docs/deployment.md, ADR-0005).
#
# Deliberately excludes the `pipeline`/`dev`/`docs`/`kaggle` extras
# (pyproject.toml): api/'s import graph (config, schemas, warehouse,
# training's train/serve-shared helpers) never touches
# sentence-transformers/faiss/shap/streamlit -- installing them here would
# add ~1-2GB and materially slow cold starts for code paths a live
# request never executes.
#
# `models/`, `data/03_marts/`, and `data/02_processed/feature_engineered_
# data.parquet` are gitignored (never committed) and must already exist on
# disk wherever this image is built from -- run the pipeline locally
# first (`python run.py train && python run.py marts`), or restore them
# from wherever they're archived. See docs/deployment.md.

FROM python:3.11-slim AS base

# Static uv binary -- no separate pip-install-uv step or build toolchain
# needed in the final image.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# LightGBM's compiled core links against libgomp (GNU OpenMP) for
# multi-threading -- python:3.11-slim doesn't include it, so importing
# lightgbm fails at runtime with "libgomp.so.1: cannot open shared object
# file" even though the build itself succeeds (the .so dependency is only
# resolved when the extension is actually loaded, not at pip/uv install
# time). scikit-learn's compiled extensions need it too.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own layer, so an app-code-only change
# doesn't invalidate the (slow) dependency-install layer. uv's
# download-wheel cache is deleted at the end of this same RUN step so it
# never becomes part of the image layer -- without this, `uv sync` left a
# ~730MB /root/.cache/uv behind on top of the ~860MB actually-installed
# .venv, nearly tripling the image for pure waste (measured on this
# project's real dependency set). A `--mount=type=cache` BuildKit cache
# mount would be the more elegant way to do this, but Cloud Run's
# `gcloud run deploy --source .` builds via the classic (non-BuildKit)
# `gcr.io/cloud-builders/docker` builder, which rejects `--mount` outright
# -- this `rm -rf` approach works identically on every builder.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    && rm -rf /root/.cache/uv

# Application code and the artifacts a live request actually needs.
# .dockerignore keeps the rest of data/ (raw/interim/full-processed,
# ~1GB+) and everything dev-only (tests/, docs/, frontend/, notebooks/,
# dashboard/) out of the build context entirely.
COPY README.md config.yaml ./
COPY src/pricepoint ./src/pricepoint
COPY api ./api
COPY models ./models
COPY data/03_marts ./data/03_marts
# The predict endpoint resolves real historical feature vectors from this
# file via DuckDB predicate pushdown (never the full row set into memory)
# -- see api/dependencies.py::get_feature_data_path. Only this one file
# from data/02_processed/ is needed; the other two (canonical_products_e5,
# anomalies_flagged) are pipeline-internal, not read by any live route.
COPY data/02_processed/feature_engineered_data.parquet ./data/02_processed/feature_engineered_data.parquet

RUN uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv

ENV PATH="/app/.venv/bin:${PATH}"

# Cloud Run injects $PORT and requires the container to listen on it --
# never hardcode 8000 here (docs/deployment.md / §25.5). 8080 is Cloud
# Run's own default, used as the fallback for `docker run` outside Cloud
# Run.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
