---
title: Airflow Observability and Alerting
---

# Airflow Observability and Alerting

Airflow is often the place where Truthound results become operational signals. The Airflow package ships host-native callback and monitoring surfaces so teams can route validation failures through the same alerting channels they already use for task and DAG health.

## Who This Is For

- on-call operators and platform teams
- DAG authors wiring warning and failure signals into callbacks
- teams standardizing metrics and alert payloads across pipelines

## When To Use It

Use this page when:

- a validation failure should send an operational notification
- warning-only checks still need visibility
- you want Airflow-native callbacks around shared Truthound results

## Prerequisites

- working Airflow callbacks or alerting conventions
- a supported Truthound Airflow operator
- understanding of the team's warning vs failure policy

## Minimal Quickstart

Attach a callback to a quality task:

```python
from airflow import DAG
from truthound_airflow import DataQualityCheckOperator, DataQualitySLACallback

with DAG("quality_alerts", schedule="@daily", catchup=False) as dag:
    validate_users = DataQualityCheckOperator(
        task_id="validate_users",
        data_path="/opt/airflow/data/users.parquet",
        rules=[{"column": "id", "check": "not_null"}],
        on_failure_callback=DataQualitySLACallback(),
    )
```

For composite policies, chain callbacks instead of duplicating logic in DAG code:

```python
from truthound_airflow import CallbackChain, DataQualitySLACallback, QualityAlertCallback

callback_chain = CallbackChain(
    callbacks=[DataQualitySLACallback(), QualityAlertCallback()]
)
```

## Production Pattern

A durable Airflow alerting design separates concerns:

| Concern | Recommended Surface |
|--------|----------------------|
| task-local failure routing | `on_failure_callback` |
| quality-specific alert policy | `DataQualitySLACallback` or `QualityAlertCallback` |
| multiple sinks | `CallbackChain` |
| raw counts and metadata | shared result payload in XCom/logs |

Recommended production checklist:

- define which checks fail hard and which warn
- route failure callbacks to the same destination your team already monitors
- keep alert text short and link back to the DAG run for detail
- emit metrics from the callback layer instead of scraping logs later

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| warnings are invisible | only hard failures trigger callback logic | add warning-aware callbacks or a metrics sink |
| duplicate notifications | both task callbacks and DAG-level callbacks emit the same alert | pick one owner for final alert emission |
| alert lacks dataset context | the callback reads only Airflow task state | enrich it from the shared result payload |
| operator succeeds but on-call still gets paged | warning/failure policy is mixed in callback code | make severity mapping explicit and documented |

## Related Pages

- [Airflow Overview](index.md)
- [SLA and Callbacks](sla.md)
- [Airflow XCom and Result Payloads](xcom-result-payloads.md)
- [Enterprise Notifications](../enterprise/notifications.md)
