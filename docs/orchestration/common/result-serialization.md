---
title: Result Serialization
---

# Result Serialization

Truthound Orchestration uses a shared result contract so that each host can wrap results without redefining what they mean.

## Why Serialization Is Shared

Every host has a different native output surface:

- Airflow uses XCom and task results
- Dagster uses metadata and typed outputs
- Prefect can attach artifacts or return structured task results
- Kestra scripts produce outputs for downstream tasks
- dbt macros print summaries or return table-shaped results

If each host invented its own status or counts semantics, the same check could appear to pass in one host and fail in another. Shared serialization prevents that drift.

## What Must Stay Stable

These concepts are shared across hosts:

- check status
- pass/fail counts
- row and column summary information
- learned rule shapes
- drift and anomaly result semantics
- metadata needed for observability and downstream consumption

Hosts can add presentation details, but they should not change those core meanings.

## Practical Guidance

- use first-party host helpers to serialize results instead of custom ad hoc conversion
- keep custom host metadata additive
- prefer structured results over string-only summaries for downstream logic
- if a host has payload-size limits, reduce the wrapped metadata rather than changing the shared result meaning

## Common Failure Mode

If a platform integration can execute a check but downstream consumers reject the result shape, the bug usually lives in the wrapper layer, not the engine itself.

Use the shared serializer contract as the source of truth when debugging that class of failure.

## Related Reading

- [Preflight and Compatibility](preflight-compatibility.md)
- [Observability and Resilience](observability-resilience.md)
