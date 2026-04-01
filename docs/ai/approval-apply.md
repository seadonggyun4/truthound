# Approval and Apply Semantics

Truthound AI separates suggestion from mutation.

## Lifecycle States

Proposal lifecycle is explicit:

- `pending`
- `approved`
- `rejected`
- `applied`

`applied` is terminal. Repeated transitions to the same state are treated as
no-op operations rather than noisy duplicate events.

## Why Approval Exists

The system is designed around human review:

- proposals are reviewable before mutation
- analyses are read-only operational aids
- mutation only happens through explicit apply

This is why `apply_proposal(...)` only accepts proposals already marked
`approved`.

## Apply Target

Applied suites are not stored under `.truthound/ai/`. They become part of the
active validation configuration surface:

- `.truthound/suites/index.json`
- `.truthound/suites/<source_key_hash>.json`

That separation matters because applied suites are runtime state, not just AI
review artifacts.

## Runtime Merge Policy

When an applied suite is merged into the deterministic auto-suite:

- exact duplicates are skipped
- same validator/columns with different params favor the applied check
- explicit `validators=` on `th.check(...)` bypasses applied-suite merging

This keeps the human-approved path explicit and predictable.
