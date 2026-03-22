---
title: dbt Warehouse Runtime Behavior
---

# dbt Warehouse Runtime Behavior

The dbt adapter validates compiled relations, not local DataFrames or file paths. That is the defining runtime difference between dbt and the Python orchestration adapters in this repository.

## Who This Is For

- analytics engineers deciding whether dbt is the right host
- teams debugging adapter dispatch or warehouse-specific SQL
- operators comparing dbt behavior with Airflow, Dagster, or Prefect

## When To Use It

Use this page when:

- a team expects zero-config local file behavior inside dbt
- warehouse-specific SQL differences appear across adapters
- you need to understand what the dbt adapter actually validates

## Prerequisites

- a dbt project using the Truthound package
- a working profile and target
- familiarity with dbt models, sources, and `adapter.dispatch`

## Minimal Quickstart

The validation target is always a relation that dbt can compile:

```yaml
models:
  - name: test_orders_model
    tests:
      - truthound.truthound_check:
          rules:
            - column: order_id
              check: not_null
            - column: total_amount
              check: non_negative
```

For ad hoc execution, use `run-operation` against a compiled relation:

```bash
dbt run-operation run_truthound_check --args '{
  "model_name": "test_model_valid",
  "rules": [{"column": "id", "check": "not_null"}],
  "options": {"limit": 50}
}'
```

## Production Pattern

Key runtime differences:

| Topic | dbt Adapter Behavior |
|------|----------------------|
| data target | compiled model or source relation |
| source resolution | no local DataFrame or file-path execution |
| execution engine | warehouse SQL via dbt and adapter dispatch |
| result transport | dbt test or `run-operation` output |

Recommended checklist:

- make sure the target relation exists before quality execution
- keep warehouse-specific assumptions in dbt, not in the rule vocabulary
- use the first-party execution suite to validate behavior on supported targets

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| local path examples from Python adapters do not work | dbt does not validate local files at runtime | move those checks to a Python host adapter |
| compiled SQL fails on one warehouse | adapter-dispatched SQL differs by backend | verify compile parity and adapter support |
| result counts differ from Python execution | the warehouse relation and local sample are not equivalent | compare the actual compiled relation |

## Related Pages

- [dbt Overview](index.md)
- [Adapter Behavior](adapter-behavior.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
