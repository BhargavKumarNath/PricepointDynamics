# Architecture Decision Records

Short, high-signal records of the major decisions made during this
project's refactor — the reasoning behind them, not just the outcome.
Numbering follows the ADR numbers already referenced from code comments
and tests across the codebase (e.g. `ADR-0006`, `ADR-0007`); gaps in the
sequence (0008–0017) are reserved for decisions from the project's
pre-refactor history that predate this `docs/adr/` directory and are not
reconstructed here.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-duckdb-not-postgres.md) | DuckDB as the sole analytical query engine, not PostgreSQL | Accepted |
| [0002](0002-polars-for-feature-engineering.md) | Polars for the pipeline's data-processing hot paths | Accepted |
| [0003](0003-reject-dagster.md) | Reject an orchestrator (Dagster/Airflow) | Accepted |
| [0004](0004-manifest-not-dvc.md) | Lightweight JSON manifest sidecars, not DVC/lakeFS | Accepted |
| [0005](0005-static-first-zero-wait-frontend.md) | Static-first, zero-wait frontend architecture | Accepted (locked) |
| [0006](0006-no-target-leakage-leave-one-out.md) | Leave-one-out competitive features (no same-row target leakage) | Accepted |
| [0007](0007-bounded-degree-mutual-k-clustering.md) | Bounded-degree mutual-nearest-neighbour clustering for product matching | Accepted |
| [0018](0018-exclude-same-row-derived-features.md) | Exclude same-row derived-rescaling columns from model features | Accepted |

See `project_refactor.md` §22 (Decisions log) for the index maintained
alongside the living refactor plan, and §1 for the full "explicit
pushback" table these ADRs are largely drawn from.
