# ADR-0003: Reject an orchestrator (Dagster/Airflow)

**Status:** Accepted
**Date:** 2026-08-12

## Context

The pipeline is 5 sequential CLI stages (`ingest → match → features →
train → precompute`), each already idempotent and skip-if-unchanged
(`manifest.py`'s `has_sources_changed()` gating), run roughly monthly by
a single maintainer. Orchestrators like Dagster or Airflow are the
default reach for "a pipeline with stages" in many production contexts.

## Decision

**Reject** a dedicated orchestrator. Use a `Makefile` target plus an
optional scheduled GitHub Actions workflow instead.

## Consequences

- An orchestrator's real value — scheduling, retries, a lineage UI,
  backfills — is genuine but disproportionate to a solo-maintained,
  low-cadence batch job with no cross-team scheduling contention.
- The `Makefile` + scheduled-workflow combination gives, in
  project_refactor.md §1's own framing, "90% of the practical benefit at
  0% of the operational cost": no new service to run, no new UI to learn,
  no new dependency to keep patched.
- This project's own manifest system already provides the one piece of
  "lineage" that actually gets used day to day — "what produced this
  file, and can I trust it" — without needing an orchestrator's UI to
  answer that question (see ADR-0004).
- If the project ever gains a second contributor, multiple pipelines with
  real cross-dependencies, or a scheduling cadence tighter than monthly,
  this decision should be revisited — the trigger is concrete
  organizational/scheduling complexity, not "orchestrators are what
  production pipelines use."

## Alternatives considered

- **Dagster** — rejected; asset-based lineage and a UI are real features,
  but this pipeline's actual "lineage" question is already answered by
  manifest sidecars, and Dagster's own deployment/hosting footprint isn't
  free at this project's scale in the way DuckDB or GitHub Actions are.
- **Airflow** — rejected for the same reason, with additional operational
  weight (a scheduler + webserver + metadata database) that has no
  free-tier home matching this project's zero-cost constraint (§18).
