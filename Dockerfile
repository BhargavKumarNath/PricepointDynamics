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
# `models/` and `data/03_marts/` are gitignored (never committed) and must
# already exist on disk wherever this image is built from -- run the
# pipeline locally first (`python run.py train && python run.py marts`),
# or restore them from wherever they're archived. See docs/deployment.md.

FROM python:3.11-slim AS base

# Static uv binary -- no separate pip-install-uv step or build toolchain
# needed in the final image.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer, so an app-code-only change
# doesn't invalidate the (slow) dependency-install layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code and the artifacts a live request actually needs.
# .dockerignore keeps the rest of data/ (raw/interim/full-processed,
# ~1GB+) and everything dev-only (tests/, docs/, frontend/, notebooks/,
# dashboard/) out of the build context entirely.
COPY README.md config.yaml ./
COPY src/pricepoint ./src/pricepoint
COPY api ./api
COPY models ./models
COPY data/03_marts ./data/03_marts

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Cloud Run injects $PORT and requires the container to listen on it --
# never hardcode 8000 here (docs/deployment.md / §25.5). 8080 is Cloud
# Run's own default, used as the fallback for `docker run` outside Cloud
# Run.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
