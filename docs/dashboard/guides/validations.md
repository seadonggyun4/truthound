# Validations

Validation execution is built on top of `truthound.check`.

## Canonical run contract

Dashboard validation views are backed by these Truthound 3.0 fields:

- `run_id`
- `run_time`
- `checks`
- `issues`
- `execution_issues`
- `row_count`
- `column_count`
- `metadata`

## What the dashboard does

- Stores run summaries for history views
- Preserves issue severity and issue detail
- Exposes rule and validator configuration without reintroducing legacy
  “phase” semantics
