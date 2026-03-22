---
title: Prefect SLA And Hooks
---

# Prefect SLA And Hooks

Prefect deployments often need the same operational guardrails as the other hosts: pass-rate thresholds, execution-time expectations, and a clear escalation path when quality drifts outside policy.

## Main Types

| Type | Purpose |
|------|---------|
| `SLAConfig` | threshold configuration |
| `SLABlock` | persisted SLA configuration |
| `SLAMonitor` | evaluation |
| `LoggingSLAHook`, `MetricsSLAHook`, `CompositeSLAHook` | response and telemetry |

## When To Use SLABlock

Use `SLABlock` when SLA policy should be saved and reused independently of any one flow definition.

## Recommended Pattern

- validation stays in tasks or flows
- SLA policy lives in config or saved blocks
- hooks handle logging, metrics, or notification side effects

## Related Reading

- [Blocks](blocks.md)
- [Deployment Patterns](deployment-patterns.md)
