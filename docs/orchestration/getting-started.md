---
title: Getting Started
---

# Getting Started

## Install Truthound 3.x With A Platform Package

```bash
pip install truthound-orchestration "truthound>=3.0,<4.0"
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
pip install truthound-orchestration[dagster] "truthound>=3.0,<4.0"
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
pip install truthound-orchestration[mage] "truthound>=3.0,<4.0"
pip install truthound-orchestration[kestra] "truthound>=3.0,<4.0"
```

`truthound-orchestration 3.x` supports `Truthound 3.x` only.

## Fastest First Run

```python
import polars as pl

from common.engines import TruthoundEngine

engine = TruthoundEngine()
data = pl.read_csv("data.csv")

result = engine.check(data)
print(result.status.name)
```

Why this works with no extra setup:

- omitted engine selection resolves to Truthound
- omitted rules on Truthound `check()` use auto-schema validation
- omitted context uses an ephemeral Truthound runtime with no persisted runs or docs

## Per-Platform Zero-Config Quickstarts

### Airflow

```python
from truthound_airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
)
```

### Dagster

```python
from dagster import Definitions, asset
from truthound_dagster.resources import DataQualityResource

@asset
def validated_users(data_quality: DataQualityResource):
    return data_quality.check(load_users())

defs = Definitions(resources={"data_quality": DataQualityResource()})
```

### Prefect

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task

@flow
async def validate_users(data):
    return await data_quality_check_task(data)
```

### dbt

```yaml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

```yaml
models:
  - name: stg_users
    tests:
      - truthound_check:
          rules:
            - column: user_id
              check: not_null
```

### Mage

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

transformer = CheckTransformer(config=CheckBlockConfig(auto_schema=True))
result = transformer.execute(dataframe)
```

### Kestra

```python
from truthound_kestra.scripts import check_quality_script

result = check_quality_script(data_uri="data/users.parquet")
```

## When You Need More Control

- [Architecture](architecture.md) explains the shared runtime boundary.
- [Compatibility](compatibility.md) covers supported versions and package lines.
- [Zero-Config](zero-config.md) documents discovery rules and persistence guarantees.
- [Advanced Engines](advanced-engines/index.md) covers Great Expectations, Pandera, and custom engines.
