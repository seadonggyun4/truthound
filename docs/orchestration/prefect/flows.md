---
title: Prefect Flows
---

# Prefect Flows

Flows are where Prefect teams usually assemble the real user experience: load, validate, profile, learn, notify, and publish. The Truthound package gives you decorators, factories, and task helpers so those flows stay readable.

## Flow Configuration

The package exposes flow-oriented config types such as:

- `FlowConfig`
- `QualityFlowConfig`
- `PipelineFlowConfig`

## Basic Quality Validation Flow

```python
from prefect import flow, task
from truthound_prefect.tasks import data_quality_check_task

@task
def load_data():
    return ...

@flow(name="basic_quality_flow")
async def basic_quality_flow():
    data = load_data()
    result = await data_quality_check_task(
        data,
        rules=[{"column": "id", "type": "not_null"}],
    )
    return result
```

## Flow With Profiling

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task, data_quality_profile_task

@flow(name="full_quality_flow")
async def full_quality_flow():
    data = ...
    check_future = data_quality_check_task.submit(data, rules=[{"column": "id", "type": "not_null"}])
    profile_future = data_quality_profile_task.submit(data)
    return {"check": check_future.result(), "profile": profile_future.result()}
```

## Flow Decorators And Factories

Use these when you want reusable patterns instead of rebuilding the same orchestration shape repeatedly:

- `quality_checked_flow`
- `profiled_flow`
- `validated_flow`
- `create_quality_flow`
- `create_validation_flow`
- `create_pipeline_flow`

## Recommended Usage Pattern

- start with explicit task calls
- move to a saved block when configuration should be shared
- adopt flow decorators or factories when the same orchestration pattern repeats across teams or datasets

## Deployment Guidance

Keep deployment concerns separate from validation semantics:

- flows define orchestration
- blocks define reusable config
- Prefect deployments define schedule, work pool, and environment bindings

See [Deployment Patterns](deployment-patterns.md) for the operational details.

## Related Reading

- [Tasks](tasks.md)
- [SLA and Hooks](sla.md)
- [Deployment Patterns](deployment-patterns.md)
