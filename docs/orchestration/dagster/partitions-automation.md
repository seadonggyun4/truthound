---
title: Dagster Partitions and Automation
---

# Dagster Partitions and Automation

Partitioned asset graphs are often where Dagster becomes more valuable than a generic scheduler. Truthound does not redefine partition semantics; it relies on Dagster to decide *what* should run and then executes checks with shared runtime semantics.

## Who This Is For

- teams validating daily, hourly, or tenant-scoped partitions
- operators using Dagster automation rules
- asset owners who need partition-aware failure isolation

## When To Use It

Use this page when:

- asset checks should execute per partition
- only new partitions should be validated
- failures need to stay local to the partition that broke

## Prerequisites

- a Dagster asset graph with partition definitions
- Truthound asset helpers or resource-backed assets
- a clear ownership model for partitions

## Minimal Quickstart

Use `quality_checked_asset` on partitioned assets so the quality contract stays next to the asset:

```python
from dagster import DailyPartitionsDefinition
from truthound_dagster import quality_checked_asset

@quality_checked_asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-01-01"),
    rules=[{"column": "id", "check": "not_null"}],
)
def users_partitioned(context):
    return load_partition(context.partition_key)
```

## Production Pattern

Recommended automation model:

| Need | Recommended Pattern |
|-----|----------------------|
| validate every fresh partition | asset checks tied to the asset definition |
| rerun only failed partitions | use Dagster re-execution on the failed partition keys |
| separate expensive quality work | move profile/learn paths into distinct jobs |

Operational advice:

- keep partition-level rules deterministic
- use partition metadata to help operators locate the failing slice
- avoid shared mutable state between partition checks

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| one bad partition blocks everything | validation is modeled at the whole-asset level | move quality logic to partition-aware assets or jobs |
| partition reruns are expensive | profile or learn logic is mixed into every partition | split exploratory work from steady-state checks |
| operators cannot tell which slice failed | run metadata omits partition context | include partition keys in alert and metadata output |

## Related Pages

- [Dagster Overview](index.md)
- [Assets and Asset Checks](assets.md)
- [Dagster Metadata and Result Payloads](metadata-results.md)
