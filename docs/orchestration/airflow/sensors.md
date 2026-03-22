---
title: Airflow Sensors
---

# Airflow Sensors And Triggers

Sensors are for waiting on a quality condition before a downstream task is allowed to continue. In this integration, they stay Airflow-native while leaning on the shared runtime for validation semantics.

## DataQualitySensor

`DataQualitySensor` re-checks a source until the quality threshold is met or the timeout expires.

```python
from truthound_airflow import DataQualitySensor

sensor = DataQualitySensor(
    task_id="wait_for_quality",
    data_path="/opt/airflow/data/upstream_users.parquet",
    quality_threshold=0.95,
    poke_interval=60,
    timeout=3600,
    mode="poke",
)
```

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_id` | str | - | Task ID |
| `data_path` / source input | str | - | path, URI, or other supported source |
| `quality_threshold` | float | 0.95 | pass rate threshold |
| `poke_interval` | int | 60 | polling interval |
| `timeout` | int | 3600 | total wait time |
| `mode` | str | `"poke"` | `poke` or `reschedule` |
| `engine_name` | str | `"truthound"` | explicit engine override |

## DeferrableDataQualitySensor

Use `DeferrableDataQualitySensor` when long waits would otherwise waste worker slots.

## TruthoundSensor

Use `TruthoundSensor` when you want the Truthound-first engine choice to be explicit.

## Triggers

The Airflow package also includes trigger support so deferrable sensor behavior can stay aligned with the same first-party quality semantics. Treat triggers as the deferrable execution mechanism and sensors as the DAG authoring surface.

## Usage Example

```python
from airflow import DAG
from truthound_airflow import DataQualitySensor, DataQualityCheckOperator
from datetime import datetime

with DAG(
    dag_id="quality_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
) as dag:
    wait_for_quality = DataQualitySensor(
        task_id="wait_for_quality",
        data_path="/opt/airflow/data/upstream_users.parquet",
        quality_threshold=0.99,
        poke_interval=300,
        timeout=7200,
    )

    process_data = DataQualityCheckOperator(
        task_id="process_data",
        data_path="/opt/airflow/data/upstream_users.parquet",
        rules=[{"column": "user_id", "type": "not_null"}],
    )

    wait_for_quality >> process_data
```

## SensorConfig

Use `SensorConfig` when you want reusable defaults across several DAGs or sensors.

## Sensor Behavior

1. Airflow calls the sensor on the configured schedule.
2. The source is normalized through the shared runtime.
3. Validation executes against the configured threshold.
4. The sensor returns success only when the threshold is satisfied.
5. Timeout or repeated failure stays visible to the DAG like any other Airflow sensor failure.

## Modes

`reschedule` mode is recommended for long wait times and worker efficiency.

## Related Reading

- [Operators](operators.md)
- [SLA and Callbacks](sla.md)
- [Recipes](recipes.md)
