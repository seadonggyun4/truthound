---
title: Airflow Recipes
---

# Airflow Recipes

## Local File Smoke Check

Use this first when validating that the Airflow integration is installed correctly.

```python
from truthound_airflow import DataQualityCheckOperator

DataQualityCheckOperator(
    task_id="smoke_check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

## Gate A Downstream Task On Quality

Use a sensor when a downstream task must wait until an upstream dataset meets a pass-rate threshold.

## Reuse An Airflow Connection For SQL

Keep SQL access in hooks or explicit operator parameters and let Airflow manage credentials through its normal connection model.

## Profile Before Enforcing Rules

Run `DataQualityProfileOperator` first when adopting Truthound in an existing DAG and you do not want the first rollout to be fail-closed.

## Add SLA Alerts Without Rewriting Tasks

Attach callbacks and monitors instead of building custom alert logic into every operator body.

## Recommended Sequence For Production Rollout

1. local-path smoke check
2. connection-backed source check
3. XCom and result consumer validation
4. sensor or SLA gating
5. DAG-wide rollout
