# ADR-0006: Leave-one-out competitive features (no same-row target leakage)

**Status:** Accepted
**Date:** pre-existing decision, formalized here

## Context

`add_competitive_features` computes cross-retailer competitive context
(same-day market average, price rank, "is cheapest") for each row. An
earlier version computed the same-day market average from a same-day
cross-retailer aggregate that **included the row's own price**. Since the
model's training target *is* that same row's `prices` value, the row's
own value was leaking into a feature meant to describe the market
*around* it — the model could partially recover its own target from its
own input.

## Decision

All competitive statistics are computed **leave-one-out**: a row's own
price is explicitly excluded from the same-day, same-product aggregate
used to describe "the market" around it —
`market_avg_others = (group_sum - own_price) / (group_count - 1)`.

Only the *derived* comparison (`price_vs_market_avg = prices -
market_avg_others`) is persisted, never the raw market-average aggregate
itself. This is deliberate and separate from the leave-one-out fix:
persisting both a raw aggregate and `price - that aggregate` as
simultaneous features would let the target be reconstructed exactly via
simple addition, regardless of how the aggregate itself was computed.
Exposing only the delta closes that door structurally, rather than
relying on downstream feature selection to catch it.

Products sold by only one retailer on a given day have no "market" to
compare against — their competitive features are `NaN` (`price_rank`,
`price_vs_market_avg`), not a vacuous self-comparison like "rank 1 of 1".

## Consequences

- Guarded by a regression test
  (`tests/data_contracts/test_no_target_leakage.py`) whose expected
  values are computed independently of the implementation (a plain
  hand-rolled leave-one-out loop, not a mirror of the production code) —
  written and run against the pre-fix code first to prove it actually
  failed there, before the fix was applied.
- Single-retailer products contribute zero rows to model training (their
  `price_vs_market_avg` is always `NaN`, and `prepare_training_data`
  drops any row with a `NaN` feature) — a real, expected consequence
  verified directly in `tests/test_pipeline_integration.py`'s fixture
  (the "Finest Chocolate Digestives" singleton product), not a
  theoretical edge case.
- See also ADR-0018, which closes a related but structurally different
  leakage path (a same-row *derived* rescaling of the target, not a
  cross-row aggregate).

## Alternatives considered

- **Self-inclusive market average (the original bug)** — rejected; this
  is the leakage this ADR fixes, not a design option.
- **Dropping single-retailer products from the dataset entirely** —
  rejected; `NaN`-and-drop-at-training-time preserves the row in the
  interim/canonical/feature Parquet files for every other downstream use
  (marts, warehouse queries, the dashboard) that has no leakage concern,
  rather than silently deleting real data earlier in the pipeline than
  necessary.
