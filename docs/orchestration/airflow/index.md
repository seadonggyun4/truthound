---
title: Airflow
---

# Airflow

Airflow is the best fit when Truthound validation needs to live inside scheduled DAGs, gate downstream tasks, or reuse existing Airflow connection and alerting patterns.

The Airflow package follows provider-style boundaries: operators, sensors, hooks, and SLA callbacks stay Airflow-native, while engine resolution, source normalization, and preflight behavior stay in the shared runtime.

## Why Teams Choose Airflow

- you already operate DAG-based pipelines
- you want quality checks to behave like first-class Airflow tasks
- you need sensor-based gating before downstream work continues
- you want connection-backed SQL sources but still keep local-path onboarding simple

## Quickstart

Install the supported Airflow surface:

```bash
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
```

Then create a basic operator:

```python
from airflow import DAG
from truthound_airflow import DataQualityCheckOperator

with DAG("quality_pipeline", schedule="@daily", catchup=False) as dag:
    check_users = DataQualityCheckOperator(
        task_id="check_users",
        data_path="/opt/airflow/data/users.parquet",
        rules=[
            {"column": "user_id", "type": "not_null"},
            {"column": "email", "type": "unique"},
        ],
    )
```

## What Zero-Config Covers

- local file paths do not need an Airflow connection
- Truthound remains the default engine
- omitted persistence stays ephemeral
- shared preflight still runs before execution

What it does **not** cover:

- SQL execution without an Airflow connection
- secret discovery outside standard Airflow patterns
- persistent Truthound workspace behavior unless you configure it explicitly

## Primary Components

| Component | Use It For |
|-----------|------------|
| `DataQualityCheckOperator` | row-level validation and pass/fail gating |
| `DataQualityProfileOperator` | profiling and shape inspection |
| `DataQualityLearnOperator` | learning rules from baseline data |
| `DataQualityStreamOperator` | bounded-memory streaming checks |
| `DataQualitySensor` / `DeferrableDataQualitySensor` | waiting for a quality threshold before continuing |
| `DataQualityHook` / `TruthoundHook` | source loading and connection-aware execution |
| SLA callbacks and monitors | warning, escalation, and alert routing |

## Operational Notes

- Use operators for execution and hooks for source or connection concerns.
- Prefer deferrable or reschedule-friendly patterns for long waits.
- Keep XCom payload handling on the shared result contract instead of ad hoc serialization.
- Treat local files as the onboarding path and connections as the production path.

## Read Next

- [Install and Compatibility](install-compatibility.md)
- [Operators](operators.md)
- [Hooks](hooks.md)
- [Sensors and Triggers](sensors.md)
- [SLA and Callbacks](sla.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
