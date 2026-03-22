---
title: Prefect Work Pools and Workers
---

# Prefect Work Pools and Workers

Truthound does not replace Prefect deployment architecture. When flows move from local execution to shared infrastructure, work pools and workers remain the operational boundary for where and how a validation flow runs.

## Who This Is For

- platform engineers running shared Prefect workers
- teams promoting quality flows from local development to managed execution
- operators who need environment-specific routing for validation jobs

## When To Use It

Use this page when:

- a flow works locally but now needs managed worker execution
- quality flows should be routed by environment, cost, or ownership
- retries and concurrency must align with worker capacity

## Prerequisites

- a working Prefect deployment model
- at least one saved flow or generated quality flow
- a defined environment strategy such as dev/staging/prod

## Minimal Quickstart

Create the flow locally first:

```python
from truthound_prefect import create_quality_flow, QualityFlowConfig

quality_flow = create_quality_flow(
    "users_quality",
    cfg=QualityFlowConfig(
        rules=[{"column": "id", "check": "not_null"}],
        engine_name="truthound",
    ),
)
```

Then deploy it into the worker topology your team already uses. Truthound-specific configuration should stay in the block or flow config, not in ad hoc worker-only environment logic.

## Production Pattern

Use work pools and workers as the execution boundary, not the rule-authoring boundary.

| Concern | Recommended Owner |
|--------|-------------------|
| validation rules and engine selection | flow config or block |
| environment routing | Prefect deployment / work pool |
| capacity and parallelism | worker and pool settings |
| secrets | Prefect blocks or your existing secrets integration |

Recommended checklist:

- keep one deployment per environment or ownership boundary
- avoid hidden worker-only overrides for rules or engine names
- align retry and concurrency policy with worker capacity

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| same flow behaves differently across environments | configuration is split between the flow and worker environment | move stable config into blocks or flow config |
| workers saturate during large fan-out validation | concurrency policy is missing | control concurrency at the pool/deployment layer |
| a deployment is hard to audit | quality configuration lives outside source control | keep rules and flow definitions versioned with code |

## Related Pages

- [Prefect Overview](index.md)
- [Deployment Patterns](deployment-patterns.md)
- [Retries, Caching, and Concurrency](retries-caching-concurrency.md)
- [Enterprise Rollout Topologies](../enterprise/rollout-topologies.md)
