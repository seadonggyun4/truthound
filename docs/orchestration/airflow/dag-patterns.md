---
title: Airflow DAG Patterns
---

# Airflow DAG Patterns

Truthound works best in Airflow when the validation boundary is obvious in the DAG itself: validate before publish, validate before fan-out, or validate after a materialization step. This page documents the patterns that map cleanly to the exported Airflow operators, sensors, and callbacks in this repository.

## Who This Is For

- DAG authors deciding where quality checks belong
- platform teams turning one-off checks into repeatable DAG templates
- operators standardizing failure and escalation behavior

## When To Use It

Use this page when a team asks:

- should the quality task run before or after transformation?
- should a check fail hard, warn, or gate a downstream task?
- how should streaming, profiling, or learning steps fit into the DAG?

## Prerequisites

- a working Airflow deployment
- familiarity with `DataQualityCheckOperator` and `DataQualitySensor`
- a supported source shape such as local path, URI, or SQL relation

## Minimal Quickstart

The most common pattern is validate-after-load, then gate the publish step:

```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from truthound_airflow import DataQualityCheckOperator

with DAG("validated_publish", schedule="@daily", catchup=False) as dag:
    load_curated = EmptyOperator(task_id="load_curated")

    validate_curated = DataQualityCheckOperator(
        task_id="validate_curated",
        data_path="/opt/airflow/data/curated/users.parquet",
        rules=[
            {"column": "id", "check": "not_null"},
            {"column": "email", "check": "unique"},
        ],
    )

    publish = EmptyOperator(task_id="publish")

    load_curated >> validate_curated >> publish
```

A second common pattern is quality gating before a wider fan-out:

```python
from truthound_airflow import DataQualitySensor

wait_for_quality = DataQualitySensor(
    task_id="wait_for_quality",
    data_path="/opt/airflow/data/curated/users.parquet",
    rules=[{"column": "id", "check": "not_null"}],
)
```

## Production Pattern

Use these host-native shapes as the default decision table:

| Pattern | Use It When | Recommended Surface |
|--------|-------------|---------------------|
| validate-after-load | the upstream step already materialized data | `DataQualityCheckOperator` |
| gate-before-publish | a downstream task must not run unless a threshold passes | `DataQualitySensor` or `DeferrableDataQualitySensor` |
| profile-before-tuning | you are learning about a new dataset or migration | `DataQualityProfileOperator` |
| learn-then-freeze | you want baseline rules from stable data | `DataQualityLearnOperator` |
| bounded-memory stream validation | the input is large or continuous | `DataQualityStreamOperator` |

Recommended lifecycle:

1. materialize or fetch the source
2. run preflight-capable Truthound operator or sensor
3. emit shared result payloads to XCom and logs
4. branch, fail, or publish based on the result contract

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| DAG fails too early during migrations | hard-fail operator used for data that still has known issues | use warning-mode rollout or isolate the migration branch |
| task blocks too long | a sensor is doing active waiting with an aggressive interval | prefer the deferrable sensor for long waits |
| duplicated logic across DAGs | rules are embedded ad hoc in many tasks | move shared rule sets into imported Python modules |
| one bad dataset stops unrelated branches | validation is placed too high in the graph | move checks closer to the branch that owns the data |

## Related Pages

- [Airflow Overview](index.md)
- [Operators](operators.md)
- [Sensors and Triggers](sensors.md)
- [Recipes](recipes.md)
- [Production Readiness](../production-readiness.md)
