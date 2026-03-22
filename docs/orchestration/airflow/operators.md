---
title: Airflow Operators
---

# Airflow Operators

Airflow operators are the main execution boundary for Truthound in DAGs. They keep scheduling, retries, task IDs, and XCom behavior inside Airflow while delegating source normalization, engine resolution, and result semantics to the shared runtime.

## DataQualityCheckOperator

Use `DataQualityCheckOperator` when a task should validate a dataset and either continue, warn, or fail based on the configured policy.

```python
from truthound_airflow import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="quality_check",
    data_path="/opt/airflow/data/users.parquet",
    fail_on_error=True,
    rules=[
        {"column": "user_id", "type": "not_null"},
        {"column": "email", "type": "unique"},
    ],
)
```

Typical uses:

- gate downstream tasks
- publish structured validation results to XCom
- wrap a connection-backed source in a first-class Airflow task
- run Truthound auto-schema validation for a fast smoke check

## DataQualityProfileOperator

Use `DataQualityProfileOperator` when the DAG needs descriptive profile output instead of strict pass/fail validation.

```python
from truthound_airflow import DataQualityProfileOperator

profile = DataQualityProfileOperator(
    task_id="profile_users",
    data_path="/opt/airflow/data/users.parquet",
)
```

## DataQualityLearnOperator

Use `DataQualityLearnOperator` when you need to infer a candidate rule set from baseline data.

```python
from truthound_airflow import DataQualityLearnOperator

learn = DataQualityLearnOperator(
    task_id="learn_users",
    data_path="/opt/airflow/data/baseline_users.parquet",
)
```

Common pattern:

- baseline learn task
- reviewed rule set
- later `DataQualityCheckOperator` using the reviewed rules

## DataQualityStreamOperator

Use `DataQualityStreamOperator` when the source is large or incremental enough that bounded-memory execution matters.

```python
from truthound_airflow import DataQualityStreamOperator

stream_check = DataQualityStreamOperator(
    task_id="stream_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

Streaming support still depends on the selected engine capability. Use preflight when you need to validate that path explicitly.

## Truthound-Specific Variants

`TruthoundCheckOperator`, `TruthoundProfileOperator`, and `TruthoundLearnOperator` exist for teams that want the Truthound-first path to be explicit in DAG code. Use them when that clarity is more valuable than host-neutral operator naming.

## XCom Contract

Operators publish structured results that should be treated as shared runtime payloads with Airflow metadata around them. Downstream tasks should read documented result fields rather than inventing their own parser assumptions.

## Configuration Guidance

Make behavior explicit when:

- the source needs an Airflow connection
- you want `warning_threshold` rather than a hard fail
- parallel Truthound execution matters
- you need to tune timeout behavior for long-running validation

## Recommended Usage Pattern

1. Use local paths first when onboarding.
2. Move to hooks or connection-backed sources once the DAG path is proven.
3. Push alerting and SLA logic into callbacks or downstream tasks rather than hand-parsing operator internals.

## Related Reading

- [Hooks](hooks.md)
- [Sensors and Triggers](sensors.md)
- [SLA and Callbacks](sla.md)
- [Recipes](recipes.md)
