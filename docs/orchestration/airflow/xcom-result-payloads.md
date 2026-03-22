---
title: Airflow XCom and Result Payloads
---

# Airflow XCom and Result Payloads

Truthound does not invent an Airflow-only result format. Airflow tasks push shared result payloads to XCom so the rest of the repository can preserve one serialization contract across hosts.

## Who This Is For

- DAG authors reading validation outputs in downstream tasks
- platform teams building Slack, PagerDuty, or audit reporting around XCom data
- operators debugging why a task passed but still emitted warnings

## When To Use It

Use this page when:

- a downstream Airflow task needs the check result
- you want to inspect `xcom_push_key`
- a cross-host reporting path expects a stable payload shape

## Prerequisites

- an Airflow operator configured with the default XCom behavior
- familiarity with the shared runtime result contract
- a downstream task that knows which XCom key to read

## Minimal Quickstart

The default XCom key for checks is `data_quality_result`:

```python
from airflow.decorators import task
from truthound_airflow import DataQualityCheckOperator

quality_task = DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "id", "check": "not_null"}],
)

@task
def summarize(result: dict):
    return {
        "status": result.get("status"),
        "failures": result.get("failure_count"),
    }

summarize(quality_task.output)
```

You can also set a custom key when an existing DAG standard already exists:

```python
DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "id", "check": "not_null"}],
    xcom_push_key="quality_gate_result",
)
```

## Production Pattern

Think of the Airflow result surface in two layers:

| Layer | Purpose | Owner |
|------|---------|-------|
| shared result payload | status, counts, metadata, normalized details | Truthound shared runtime |
| Airflow transport | where and how the payload is exposed to downstream tasks | Airflow operator/XCom |

Recommended practices:

- read only documented keys in downstream tasks
- keep custom XCom keys stable across DAG versions
- serialize alerting inputs from the shared payload rather than reconstructing them
- prefer small summaries in alerts and keep full detail in logs or artifacts

Typical consumers:

- branch operators deciding whether to continue
- alert callbacks turning failures into notifications
- audit tasks writing summaries to a warehouse or log sink

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| downstream task cannot find the result | XCom key mismatch | verify `xcom_push_key` on the operator |
| payload shape differs across hosts | a custom serializer was added outside the documented path | route through shared serialization instead |
| alerts miss details | downstream code only reads Airflow task state | read the serialized Truthound payload, not just success/failure |
| XCom gets too noisy | full detail is being pushed for every step | summarize for alerts and preserve raw details only where needed |

## Related Pages

- [Airflow Overview](index.md)
- [Operators](operators.md)
- [Shared Result Serialization](../common/result-serialization.md)
- [Airflow Observability and Alerting](observability-alerting.md)
