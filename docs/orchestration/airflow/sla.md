---
title: Airflow SLA Monitoring
---

# Airflow SLA And Callbacks

Airflow is a natural place to attach SLA and alerting behavior to quality checks. The first-party package keeps the metric definitions and violation semantics shared, while leaving callback wiring and DAG behavior inside Airflow.

## SLAConfig

Use `SLAConfig` to define acceptable latency and failure envelopes for quality tasks or quality-gated DAG steps.

```python
from truthound_airflow import SLAConfig

config = SLAConfig(
    max_failure_rate=0.01,
    max_execution_time_seconds=300.0,
)
```

Typical questions this answers:

- when should a quality task be considered degraded?
- when should a callback warn versus fail loudly?
- how do we route high-severity quality regressions?

## SLAMetrics

`SLAMetrics` is the shared metric payload that the monitor evaluates. It keeps the same core logic available across Airflow and other hosts.

## SLAViolation

`SLAViolation` and `SLAViolationType` are the structured outputs that callbacks and monitors consume.

## SLAMonitor

`SLAMonitor` evaluates recorded results against the configured SLA contract.

```python
from truthound_airflow import SLAMonitor, SLAConfig

monitor = SLAMonitor(
    config=SLAConfig(
        max_failure_rate=0.05,
        max_execution_time_seconds=600.0,
    )
)
```

## Callback Types

The Airflow package exposes several callback helpers:

- `DataQualitySLACallback`
- `QualityAlertCallback`
- `CallbackChain`

Use them to map structured violation information into Airflow-native alerting behavior.

## Registry Support

`SLARegistry` is useful when you want several DAGs to share named SLA definitions instead of re-declaring them inline.

## Recommended Airflow Pattern

```python
from truthound_airflow import DataQualityCheckOperator, DataQualitySLACallback, SLAConfig

sla_callback = DataQualitySLACallback(
    sla_config=SLAConfig(max_failure_rate=0.02, max_execution_time_seconds=300),
)

check = DataQualityCheckOperator(task_id="quality_check", data_path="users.parquet")
```

The important design rule is to keep rule execution in operators and policy escalation in callbacks or monitors.

## Related Reading

- [Operators](operators.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
