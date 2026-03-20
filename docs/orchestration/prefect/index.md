---
title: Prefect
---

# Prefect

Prefect integration uses Blocks as the persisted boundary and tasks or flows as the execution boundary.

## Install

```bash
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
```

## Zero-Config Path

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task

@flow
async def validate_users(data):
    return await data_quality_check_task(data)
```

If you omit a block, Prefect creates an in-memory Truthound-backed block for the run.

## Saved Block Path

```python
from truthound_prefect.blocks import DataQualityBlock

block = DataQualityBlock(engine_name="truthound")
```

Use saved blocks when you want reusable deployment configuration. Use the no-block path when you want the fastest first run.
