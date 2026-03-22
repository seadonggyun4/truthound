---
title: Prefect Retries, Caching, and Concurrency
---

# Prefect Retries, Caching, and Concurrency

Prefect is strongest when operational behavior is explicit in Python. Truthound leans into that by exposing task and flow configuration surfaces for retries, cache settings, and execution control instead of hiding them behind adapter-specific magic.

## Who This Is For

- platform teams defining retry and cache policy
- flow authors tuning expensive quality checks
- operators standardizing concurrency and replay behavior

## When To Use It

Use this page when:

- validation tasks should retry transient failures
- repeated profile or learn operations should reuse cached results
- the team needs clear concurrency boundaries for larger validation workloads

## Prerequisites

- familiarity with Prefect tasks and flows
- awareness of the shared Truthound preflight/runtime split
- a clear policy for transient failure vs hard data-quality failure

## Minimal Quickstart

Use a flow config with retries:

```python
from truthound_prefect import QualityFlowConfig, create_quality_flow

flow = create_quality_flow(
    "validate_users",
    cfg=QualityFlowConfig(
        rules=[{"column": "id", "check": "not_null"}],
        retries=2,
        retry_delay_seconds=30,
        engine_name="truthound",
    ),
)
```

Task builders also expose cache-related settings:

```python
from truthound_prefect import create_check_task

check_task = create_check_task(
    retries=1,
    cache_key="dim_users_quality",
    cache_expiration_seconds=300,
)
```

## Production Pattern

Use this decision table:

| Concern | Recommended Policy |
|--------|--------------------|
| transient network or warehouse failure | retry at the task or flow layer |
| known bad dataset | do not retry blindly; fix data or downgrade severity |
| expensive repeated quality summaries | use a short cache window where safe |
| multi-table fan-out | control concurrency at the Prefect deployment/work-pool layer |

Recommended checklist:

- retry only infrastructure failures, not deterministic data failures
- keep cache keys dataset-specific and environment-aware
- document whether a cached result is allowed to gate downstream execution

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| failures repeat across retries | the issue is data quality, not transport | stop retrying and surface the result directly |
| stale quality state appears | cache keys are too broad or cache expiration is too long | narrow the key and shorten expiration |
| flow overwhelms workers | parallel fan-out lacks an explicit concurrency policy | move concurrency control to deployment/work-pool settings |

## Related Pages

- [Prefect Overview](index.md)
- [Tasks](tasks.md)
- [Flows](flows.md)
- [Work Pools and Workers](work-pools-workers.md)
