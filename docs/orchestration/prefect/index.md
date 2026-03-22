---
title: Prefect
---

# Prefect

Prefect is the best fit when your team wants Python-first workflows with an optional persisted configuration boundary. Blocks handle reusable configuration, while tasks and flows handle execution ergonomics.

## Why Teams Choose Prefect

- flows stay readable Python
- saved blocks make it easy to share configuration across deployments
- ephemeral paths still exist for fast local onboarding
- task helpers provide a smooth path from one-off checks to reusable flow factories

## Quickstart

Install the supported Prefect surface:

```bash
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
```

Then start with an ephemeral task-based flow:

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task

@flow
async def validate_users(data):
    return await data_quality_check_task(
        data,
        rules=[{"column": "id", "type": "not_null"}],
    )
```

If you omit a block, Prefect creates an in-memory Truthound-backed block for the run.

## Saved Block Path

Use a saved block when the configuration should outlive one run:

```python
from truthound_prefect.blocks import DataQualityBlock

block = DataQualityBlock(engine_name="truthound")
```

## Primary Surfaces

| Surface | Use It For |
|---------|------------|
| `DataQualityBlock` | persisted configuration and direct block-level execution |
| task helpers | ad hoc flow composition |
| flow decorators and factories | repeatable workflow patterns |
| SLA helpers | thresholds and alert routing |

## Read Next

- [Install and Compatibility](install-compatibility.md)
- [Blocks](blocks.md)
- [Tasks](tasks.md)
- [Flows](flows.md)
- [SLA and Hooks](sla.md)
- [Deployment Patterns](deployment-patterns.md)
- [Troubleshooting](troubleshooting.md)
