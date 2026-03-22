---
title: Dagster Jobs and Schedules
---

# Dagster Jobs and Schedules

Truthound integrates cleanly into Dagster jobs when quality checks are modeled as explicit ops or resource-backed asset steps. The goal is to keep scheduling and automation Dagster-native while preserving the shared Truthound runtime contract underneath.

## Who This Is For

- Dagster operators designing scheduled validation jobs
- teams combining asset materialization and quality checks in one graph
- platform engineers standardizing reusable quality jobs

## When To Use It

Use this page when:

- a resource-backed check should run on a schedule
- you need a dedicated validation job separate from asset materialization
- you want to reuse `data_quality_check_op` across datasets

## Prerequisites

- `truthound-orchestration[dagster]` installed
- Dagster Definitions-based project layout
- familiarity with `DataQualityResource` and the prebuilt ops

## Minimal Quickstart

Create a small quality job with the exported op surface:

```python
from dagster import Definitions, job
from truthound_dagster import DataQualityResource, data_quality_check_op

@job(resource_defs={"data_quality": DataQualityResource()})
def users_quality_job():
    data_quality_check_op.alias("check_users")()

defs = Definitions(jobs=[users_quality_job])
```

For mixed jobs, keep materialization and validation as separate nodes:

```python
from dagster import job, op
from truthound_dagster import create_check_op

@op
def load_users():
    return read_users_dataframe()

validate_users = create_check_op(
    name="validate_users",
    rules=[{"column": "id", "check": "not_null"}],
)
```

## Production Pattern

Use one of these shapes:

| Pattern | Best Fit |
|--------|----------|
| asset-only schedule | quality logic naturally belongs to assets |
| dedicated validation job | operators want a separate operational surface |
| mixed pipeline job | materialization and validation must stay in one Dagster graph |

Recommended schedule policy:

- keep validation schedules aligned with materialization freshness windows
- separate “fast gate” checks from expensive profile or learn jobs
- name jobs after the dataset or domain, not the engine

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| schedule is green but asset quality is stale | validation job is not aligned with materialization cadence | reschedule to run after the producing asset |
| one validation graph becomes too large | too many unrelated checks live in one job | split by domain or ownership boundary |
| jobs are hard to debug | materialization and validation share the same op name patterns | use explicit aliases for quality steps |

## Related Pages

- [Dagster Overview](index.md)
- [Ops](ops.md)
- [Assets and Asset Checks](assets.md)
- [Dagster Partitions and Automation](partitions-automation.md)
