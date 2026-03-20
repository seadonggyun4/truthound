---
title: Airflow
---

# Airflow

The Airflow package follows provider-style boundaries: operators and sensors stay Airflow-native, while engine resolution, source normalization, and preflight checks live in `common`.

## Install

```bash
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
```

## Zero-Config Path

```python
from truthound_airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
)
```

Notes:

- local file paths do not need an Airflow connection
- SQL sources still need a connection
- shared preflight runs before execution and fails fast on incompatible inputs

## Runtime Shape

- provider metadata stays in the Airflow package
- operator construction stays lazy
- result payloads are based on the shared orchestration wire format plus Airflow metadata

## Common Next Reads

- [Zero-Config](../zero-config.md)
- [Architecture](../architecture.md)
