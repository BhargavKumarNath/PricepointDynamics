# ADR-0018: Exclude same-row derived-rescaling columns from model features

**Status:** Accepted
**Date:** pre-existing decision, formalized here

## Context

`prices_unit` (a per-unit rescaling of the *same row's* `prices` — e.g.
£/kg derived from pack price ÷ pack weight) was a candidate model
feature. Within a given `(canonical_name, supermarket)` group, the ratio
between `prices` and `prices_unit` is close to constant (empirically true
for ~82% of groups, verified against real data in Phase 1). That makes
`prices_unit` **same-row target leakage**: a near-fixed per-row rescaling
of the exact value being predicted — arguably more direct than the
same-day market-average leak ADR-0006 fixes, since it doesn't even
require cross-referencing other rows to reconstruct the target.

## Decision

`prices_unit` is listed in `training.py::_NON_FEATURE_COLUMNS` — the
single shared list of columns excluded from model input, used by *both*
`prepare_training_data` (training-time) and `build_feature_vector`
(serving-time), so both code paths provably agree on exactly which raw
columns are eligible to become model inputs.

## Consequences

- Because `_NON_FEATURE_COLUMNS` is the one shared source of truth
  between training and serving, this exclusion can't silently drift
  between the two — the alternative (two separately-maintained exclusion
  lists) is exactly the kind of train/serve skew this project's API
  design (§8.3) was built to avoid.
- `prices`, `date`, `product_name`, `canonical_name`, `normalised_name`,
  and `prices_unit` are all excluded together for the same underlying
  reason (target itself, identifier, or same-row leakage) — this ADR
  documents `prices_unit` specifically because it is the one entry in
  that list whose leakage mechanism is a *derived rescaling*, not an
  identifier or the target itself, and is easy to miss on a future
  feature-list edit if the reasoning isn't written down.

## Alternatives considered

- **Include `prices_unit` as a feature** — rejected; would have let the
  model achieve an artificially low training error by recovering the
  target from a near-fixed rescaling of itself, without that
  generalizing to genuinely predictive signal.
- **Include `prices_unit` but only as an engineered ratio feature
  (e.g. a fixed per-group conversion factor)** — not pursued; the ratio
  itself is derived from the same leaking relationship, so it does not
  avoid the underlying problem, only obscures it.
