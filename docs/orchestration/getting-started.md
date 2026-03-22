---
title: Getting Started
---

# Getting Started

This page gets you from install to a validated run as quickly as possible, then shows where to go next when you need more control.

## Prerequisites

- Python aligned with a [supported compatibility tuple](compatibility.md)
- `truthound>=3.0,<4.0`
- one host package or runtime surface:
  - Airflow
  - Dagster
  - Prefect
  - dbt
  - Mage
  - Kestra

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

Use the base package when you only need shared runtime helpers or engine contracts. Install a platform extra when you want the host-native integration boundary and compatibility guarantees for that host.

## Fastest First Run

If you want to sanity-check the Truthound-first path before choosing a scheduler, use the shared runtime directly:

```python
import polars as pl

from common.engines import TruthoundEngine

engine = TruthoundEngine()
data = pl.read_csv("data.csv")

result = engine.check(data)
print(result.status.name)
```

Why this works with no extra setup:

- omitted engine selection still resolves to Truthound
- omitted rules on the Truthound `check()` path enable auto-schema validation
- omitted runtime context keeps execution ephemeral and non-persistent by default
- the result payload uses the same semantics that platform adapters wrap later

## Per-Platform Zero-Config Quickstarts

### Airflow

```python
from truthound_airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

Use this path when the source is a local path or a connection-backed query you already manage inside Airflow.

### Dagster

```python
from dagster import Definitions, asset
from truthound_dagster.resources import DataQualityResource

@asset
def validated_users(data_quality: DataQualityResource):
    return data_quality.check(load_users(), rules=[{"column": "id", "type": "not_null"}])

defs = Definitions(resources={"data_quality": DataQualityResource()})
```

`DataQualityResource()` is the canonical Dagster zero-config entry point.

### Prefect

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task

@flow(name="validate-users")
async def validate_users(data):
    return await data_quality_check_task(
        data=data,
        rules=[{"column": "id", "type": "not_null"}],
    )
```

When you omit a saved block, Prefect uses an in-memory Truthound-backed configuration for that run.

### dbt

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=3.0.0,<4.0.0"
```

```yaml
# models/schema.yml
version: 2

models:
  - name: stg_users
    tests:
      - truthound.truthound_check:
          rules:
            - column: user_id
              check: not_null
```

Run `dbt deps` and then `dbt test`. The package-qualified form is the safest documented form for first-party integration projects.

### Mage

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

transformer = CheckTransformer(
    config=CheckBlockConfig(
        auto_schema=True,
        rules=[{"column": "id", "type": "not_null"}],
    )
)
result = transformer.execute(dataframe)
```

### Kestra

```python
from truthound_kestra.scripts import check_quality_script

result = check_quality_script(
    input_uri="data/users.parquet",
    rules=[{"column": "id", "type": "not_null"}],
)
```

## What To Expect From Zero-Config

The first run is intentionally conservative:

- default engine: Truthound
- default policy: `safe_auto`
- default runtime context: ephemeral
- default persistence: disabled
- default host discovery: read-only

That means you get a fast result without hidden workspace creation, generated docs, or stored run state unless you opt in.

## Where To Go Next

Choose the next page based on what you need:

- [Choose a Platform](choose-a-platform.md) if you are still comparing hosts
- [Architecture](architecture.md) if you want to understand runtime boundaries
- [Zero-Config](zero-config.md) for discovery rules and persistence guarantees
- [Compatibility](compatibility.md) for tested host and Python tuples
- [Troubleshooting](troubleshooting.md) if your first run fails at install, preflight, or serialization time

## When You Need More Control

Move from zero-config to explicit configuration when:

- the source needs credentials or a host-specific connection object
- you want persisted Prefect blocks or reusable Dagster resource config
- you need engine lifecycle tuning, batching, streaming, or advanced observability
- you are adopting Great Expectations or Pandera instead of the Truthound-first path
- you need repeatable CI behavior for dbt or workflow release gates

Those paths are all covered in the platform and shared runtime sections of this site.
