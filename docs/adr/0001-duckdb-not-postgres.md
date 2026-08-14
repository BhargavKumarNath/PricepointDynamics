# ADR-0001: DuckDB as the sole analytical query engine, not PostgreSQL

**Status:** Accepted
**Date:** 2026-08-12

## Context

The analytics layer (market overview stats, basket cost comparison, price
dispersion, HHI, price leadership) needs a query engine over the
pipeline's Parquet marts. The obvious "production" default is a hosted
relational database, but every one of these queries is a **read-only
analytical aggregation over historically-static data** — there are no
concurrent writers, no transactional/mutable state, and no relational
integrity constraints across the analytical data.

## Decision

Use **DuckDB**, embedded directly against the mart Parquet files
(`data/03_marts/*.parquet`), as the only query engine for the core data.
No PostgreSQL, no other hosted database.

`03_marts/` stays Parquet, not a `.duckdb` file — DuckDB is the query
interface on top of the existing file-based storage, not a second copy of
the data in a different format that needs to be kept in sync.

## Consequences

- Zero hosting cost, zero operational surface (no server to provision,
  patch, or pay for) — a genuine free-tier win, not a qualified one.
- Query performance is bounded by Parquet + DuckDB's own predicate/column
  pushdown, not by network round-trips to a database server. Measured:
  market overview 2.77s → 0.60s, basket cost 1.31s → 0.08s versus the
  pandas-in-Streamlit baseline (`docs/benchmarks.md`, Phase 3).
- Postgres (or SQLite for a simpler single-instance case) becomes
  justified the moment a feature needs **mutable, concurrent, or
  relational state** — e.g. a "save this basket" user feature,
  prediction-request logging, or rate-limit counters that must survive a
  restart. None of these exist today; do not provision speculatively.
- `sql/marts/*.sql` and `sql/queries/*.sql` are version-controlled,
  parameterized files, never raw strings interpolated in Python — closes
  off SQL injection by construction, not by discipline alone (project_refactor.md §16).

## Alternatives considered

- **PostgreSQL** — rejected. Would mean provisioning and paying for (or
  fighting the free-tier limits of) a hosted database server to serve
  read-only aggregates DuckDB already serves for free, embedded. Named
  explicitly in project_refactor.md §1 as "the single most avoidable
  'because production systems have one' decision in this redesign."
- **SQLite** — rejected for the same reason Postgres is, minus the
  hosting cost: still solves a mutable-state problem this project doesn't
  have yet.
