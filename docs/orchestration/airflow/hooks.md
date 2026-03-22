---
title: Airflow Hooks
---

# Airflow Hooks

Hooks are the right place for source loading, connection handling, and reusable Airflow-side data access patterns. They are especially important when the execution path is not a pure local-file zero-config workflow.

## DataQualityHook

`DataQualityHook` is the general-purpose Airflow hook for first-party quality execution.

```python
from truthound_airflow import DataQualityHook

hook = DataQualityHook(connection_id="warehouse")
```

Use it when:

- the DAG already models source access through Airflow connections
- you want to reuse connection-aware logic across operators
- you need a hook boundary that can normalize sources before execution

## TruthoundHook

`TruthoundHook` is the explicit Truthound-first hook path for teams that want the engine choice to be visible in code.

## DataLoader And DataWriter

`DataLoader` and `DataWriter` are useful when you want source access or result persistence patterns to stay hook-level and reusable instead of being embedded into individual tasks.

## Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `connection_id` | str | Airflow connection ID for SQL or credential-backed sources |
| `engine_name` | str | explicit engine override when not using the default Truthound path |
| `timeout` | int | hook-level timeout for source access or execution |
| `retries` | int | hook-level retry policy |

## ConnectionConfig

Use `ConnectionConfig` when you want to centralize connection metadata instead of scattering connection details across DAG code.

## How Hooks Fit With Operators

```python
from truthound_airflow import DataQualityCheckOperator, DataQualityHook

hook = DataQualityHook(connection_id="warehouse")

check = DataQualityCheckOperator(
    task_id="quality_check",
    sql="select * from analytics.users",
    hook=hook,
    rules=[{"column": "id", "type": "not_null"}],
)
```

Use hooks when the source contract belongs to Airflow. Use the operator alone when a simple local-path or already-normalized source is enough.

## Connection Setup Guidance

- create or reuse a standard Airflow connection
- keep credentials in Airflow-managed storage
- let the hook mediate data access rather than embedding credentials into DAG code

## Related Reading

- [Airflow Overview](index.md)
- [Install and Compatibility](install-compatibility.md)
- [Troubleshooting](troubleshooting.md)
