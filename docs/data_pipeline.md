# Data Pipeline

Stage-by-stage detail for `src/pricepoint/`'s pipeline. For the
system-wide picture, see `docs/architecture.md`.

## Data layers

| Layer | Location | Format | Produced by |
|---|---|---|---|
| Raw | `data/00_raw/` | 5 CSVs, one per retailer | `scripts/download_raw_data.py` (Kaggle) |
| Interim | `data/01_interim/` | `cleaned_supermarket_data.parquet` | `run.py ingest` |
| Processed | `data/02_processed/` | `canonical_products_e5.parquet`, `feature_engineered_data.parquet` | `run.py match`, `run.py features` |
| Marts | `data/03_marts/` | `dim_product.parquet`, `dim_retailer.parquet`, `fact_price_daily.parquet` | `run.py marts` |
| Model | `models/` | `price_predictor_lgbm.joblib`, `metrics.json` | `run.py train` |
| Precomputed | `shap_precomputed/`, `market_dynamics_precomputed/` | Parquet/JSON | `run.py precompute` |
| Web artifacts | `frontend/public/data/`, `frontend/public/data-r2/` | JSON (committed), Parquet (R2-bound stand-in) | `run.py web-artifacts` |
| Run reports | `data/_run_reports/` | one timestamped JSON per stage run | every `run_*` stage function |

All `data/`-rooted paths are gitignored; the only reproducible way to get
from a fresh clone to a working pipeline is running the stages below in
order (or letting the skip-if-unchanged manifests short-circuit stages
that don't need to re-run).

## Stages

### 1. Ingestion (`data_ingestion.py`, `run.py ingest`)

`load_raw_csvs` reads all 5 raw CSVs (Polars, full-file schema scan —
not sample-based, to avoid mid-file dtype misdetection) and concatenates
them. `clean_raw_data` normalises column names, coerces `date` from
`YYYYMMDD`, strips whitespace from string columns, coerces `prices` to
float and drops **null** prices only (a **zero** price is a valid,
retained value — only `NaN`/unparseable prices are dropped), and drops
rows with a missing `product_name`. The result is validated against
`RAW_DATA_SCHEMA` (Pandera) before being written to
`cleaned_supermarket_data.parquet`.

### 2. Product matching (`product_matching.py`, `run.py match`)

`normalise_product_name` lowercases, strips units/pack-size tokens,
removes known retailer brand names, and collapses whitespace.
`generate_embeddings` encodes each *unique* normalised name via
Sentence-BERT (`intfloat/e5-large`). `cluster_by_similarity` groups
embeddings into canonical products via a bounded-degree
mutual-nearest-neighbour graph — see
[ADR-0007](adr/0007-bounded-degree-mutual-k-clustering.md) for why a
naive similarity-threshold connected-components approach fails at this
corpus's scale. Result validated against `CANONICAL_PRODUCTS_SCHEMA`.

### 3. Feature engineering (`feature_engineering.py`, `run.py features`)

Three feature families, computed per `(canonical_name, supermarket)`
group unless noted:

- **Temporal**: rolling mean/std/max/min over `[7, 14, 30]`-row windows,
  1- and 7-day lags, and 1-day diff — computed via Polars `.over()`
  expressions (see
  [ADR-0002](adr/0002-polars-for-feature-engineering.md)).
- **Competitive**: same-day, cross-retailer leave-one-out comparisons
  (`price_vs_market_avg`, `price_rank`, `is_cheapest_in_market`) — see
  [ADR-0006](adr/0006-no-target-leakage-leave-one-out.md) for why these
  must exclude the row's own price.
- **Cyclical**: sin/cos encodings of day-of-week, day-of-month, and
  week-of-year.

Result validated against `FEATURE_DATA_SCHEMA`.

### 4. Training (`training.py`, `run.py train`)

Time-series split (final 7 days held out as test). `prepare_training_data`
and the serving-time `build_feature_vector` share the same categorical
encoding and column-selection helpers, so a served prediction is
provably built the same way a training row was, not just similarly.
`_NON_FEATURE_COLUMNS` excludes the target, identifiers, and same-row
derived leakage (`prices_unit` — see
[ADR-0018](adr/0018-exclude-same-row-derived-features.md)). Trains a
LightGBM regressor (`objective=mae`), writes the model artifact and a
`metrics.json` (MAE, RMSE, R², row/feature counts, git SHA, hyperparameters).

Current production model: **MAE £0.1487, RMSE £1.2589, R² 0.9652** on
5,151,151 train / 444,314 test rows, 42 features (`models/metrics.json`
is the source of truth — this number is quoted here for context and will
drift from the file as the model is retrained; always trust the file).

### 5. Marts (`marts.py`, `run.py marts`)

Materializes `sql/marts/*.sql` (DuckDB, parameterized) into 3 Parquet
mart tables: `dim_product`, `dim_retailer`, `fact_price_daily`
(collapses duplicate `(canonical_name, supermarket, date)` rows via
`AVG()`, since ~41.5% of canonical rows are duplicates at that grain from
multiple pack-size SKUs sharing one canonical cluster).

`src/pricepoint/warehouse.py` (`Warehouse` class) is the one place that
queries the marts — every dashboard/API aggregate goes through it, via
parameterized DuckDB SQL embedded in Python (not extracted to separate
`.sql` query files, a deviation from `project_refactor.md` §7's original
framing; still safe against injection via DuckDB's own `?` parameter
binding, just less separated than originally planned).

### 6. Precompute (`market_analysis.py`, `run.py precompute`, `run.py hhi`, `run.py anomaly`)

SHAP value matrices (fixed 8K-sample), the Herfindahl-Hirschman Index,
cross-correlation price leadership/lag, and Isolation Forest anomaly
detection — all expensive-to-compute, cheap-to-serve-once-precomputed
metrics.

### 7. Web artifacts (`web_artifacts.py`, `run.py web-artifacts`)

Exports the small, page-shaped JSON summaries and the two larger Parquet
artifacts (`predictor_context.parquet`, `shap_explorer.parquet`) the
frontend reads — see `project_refactor.md` §25.3 for the full artifact
table and reasoning.

## Manifests and run reports

Two distinct, complementary JSON artifacts accompany every stage's
output:

- **`<output>.manifest.json`** (`manifest.py`) — describes the *current*
  state of an output artifact (row count, schema hash, source files, git
  SHA). Overwritten every run. Drives skip-if-unchanged: `run_*(force=False)`
  compares current source files against the manifest and skips
  reprocessing if nothing changed.
- **`data/_run_reports/<stage>_<timestamp>.json`** (`run_reports.py`) —
  a historical, append-only record of one stage *execution* (rows
  in/out, row delta, per-column null counts, duration, schema hash). A
  stage that unexpectedly loses more than 20% of its input rows logs a
  warning rather than passing through silently.

See [ADR-0004](adr/0004-manifest-not-dvc.md) for why this lightweight
approach was chosen over a data-versioning tool like DVC.
