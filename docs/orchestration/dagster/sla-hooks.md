---
title: Dagster SLA And Hooks
---

# Dagster SLA And Hooks

Dagster workflows often need both data-quality correctness and operational guarantees. The SLA layer helps you enforce thresholds such as pass rate or execution time without mixing alerting logic directly into every asset or op.

## Main Types

| Type | Purpose |
|------|---------|
| `SLAConfig` | threshold definition |
| `SLAMonitor` | evaluation logic |
| `SLAResource` | reusable resource-based SLA integration |
| `SLAHook` and hook implementations | alerting and monitoring side effects |

## Recommended Pattern

- use `DataQualityResource` or asset helpers for execution
- use SLA resources or hooks for policy and escalation
- keep quality semantics and escalation semantics separate

## Hook Types

The Dagster package exposes logging-oriented, metrics-oriented, and composite hooks for SLA handling.

## Related Reading

- [Resources](resources.md)
- [Recipes](recipes.md)
