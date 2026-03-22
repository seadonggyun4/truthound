---
title: Prefect Rollout Playbooks
---

# Prefect Rollout Playbooks

Most Prefect quality adoption happens in stages: ad hoc task, saved block, reusable flow, then deployment. This page turns that pattern into a deliberate rollout playbook.

## Who This Is For

- teams standardizing Prefect-based quality workflows
- operators moving from experimentation to production
- platform engineers defining rollout steps for new domains

## When To Use It

Use this page when:

- a successful prototype needs to become a shared pattern
- the team is moving from ephemeral blocks to saved blocks
- production rollout needs a checklist instead of ad hoc promotion

## Prerequisites

- at least one working Prefect quality flow or task
- a supported compatibility tuple
- a deployment target or work pool strategy

## Minimal Quickstart

Start with a task-only flow:

```python
from prefect import flow
from truthound_prefect import data_quality_check_task

@flow
async def validate_users(data):
    return await data_quality_check_task(
        data,
        rules=[{"column": "id", "check": "not_null"}],
    )
```

Promote to a reusable saved block when the configuration should be shared:

```python
from truthound_prefect import DataQualityBlock

block = DataQualityBlock(engine_name="truthound")
```

## Production Pattern

Recommended rollout ladder:

1. ephemeral task-based proof of value
2. saved `DataQualityBlock` for shared configuration
3. `create_quality_flow` or `quality_checked_flow` for standard flow composition
4. deployment into work pools and environment-specific automation

Checklist before promotion:

- block configuration is explicit and reproducible
- retry and cache policy are documented
- result artifacts and notifications are wired
- the flow is tested against representative input shapes

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| early flows are fast but production ones are confusing | configuration drift between ephemeral and saved paths | standardize on block-backed config before wide rollout |
| teams fork their own wrappers | reusable flow factories were never formalized | publish one blessed flow pattern |
| production incidents are hard to triage | artifacts and notifications were added late | wire them before general availability |

## Related Pages

- [Prefect Overview](index.md)
- [Blocks](blocks.md)
- [Flows](flows.md)
- [Deployment Patterns](deployment-patterns.md)
