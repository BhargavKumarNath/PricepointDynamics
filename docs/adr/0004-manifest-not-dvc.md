# ADR-0004: Lightweight JSON manifest sidecars, not DVC/lakeFS

**Status:** Accepted
**Date:** 2026-08-12 (pre-existing decision, formalized here)

## Context

Every pipeline stage writes an artifact (`.parquet`) that later stages
and consumers need to trust: "what produced this file, is it stale, can
I skip re-processing it." Data-versioning tools like DVC or lakeFS solve
this at the cost of a second tool, a second mental model, and (for
lakeFS) a hosted service.

## Decision

Each Gold/Silver-layer write is accompanied by a
`<artifact>.manifest.json` sidecar (`manifest.py::write_manifest`)
recording row count, a schema fingerprint (`schema_hash`), the source
file(s) it was derived from, and the git commit of the code that produced
it. `has_sources_changed()` compares a manifest against current source
files' row counts/mtimes to drive skip-if-unchanged behaviour in every
`run_*` stage function.

## Consequences

- Single-contributor, no-concurrent-branches-touching-data project: a
  JSON sidecar is enough traceability. "What is this file, and can I
  trust it" is answerable by reading one small file, without re-running
  the pipeline or standing up a second service.
- Extended in Phase 7 (`run_reports.py`) with a complementary,
  *historical* (not overwritten) per-run JSON report — rows in/out, null
  counts, duration — sharing the same `schema_hash` computation
  (`manifest.schema_hash`, promoted to a public function specifically so
  both modules could reuse it without duplicating the hashing logic).
- If this project ever gains a second contributor or a second machine
  producing data concurrently, this decision should be revisited — DVC's
  real value (conflict-free data versioning across branches/machines)
  only appears once there's more than one producer to reconcile.

## Alternatives considered

- **DVC** — rejected; solves branch/remote data versioning this project
  doesn't have (one contributor, one machine, data is gitignored and
  regenerated from source CSVs, not versioned as an asset).
- **lakeFS** — rejected for the same reason, plus it requires a hosted
  service, which conflicts with the project's zero-cost constraint (§18)
  for no corresponding benefit at this scale.
