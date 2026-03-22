---
title: Prefect Artifacts and Result Payloads
---

# Prefect Artifacts and Result Payloads

Prefect users often want a result that is both programmatic and visible in the Prefect UI. Truthound supports that by keeping a shared result contract and offering artifact-oriented helpers such as `to_prefect_artifact`.

## Who This Is For

- flow authors publishing quality summaries to Prefect
- operators debugging why a flow succeeded with warnings
- teams standardizing reporting across tasks and flows

## When To Use It

Use this page when:

- a quality task result should appear as a Prefect artifact
- a downstream step needs structured status and failure counts
- flow-level reporting should look consistent across environments

## Prerequisites

- `truthound-orchestration[prefect]` installed
- a Prefect flow using tasks or a `DataQualityBlock`
- access to the serialized Truthound result

## Minimal Quickstart

Convert a result into a Prefect-friendly artifact payload:

```python
from truthound_prefect import data_quality_check_task, to_prefect_artifact

result = await data_quality_check_task(
    data,
    rules=[{"column": "id", "check": "not_null"}],
)
artifact = to_prefect_artifact(result)
```

Block-based execution can follow the same output path:

```python
from truthound_prefect import DataQualityBlock

block = DataQualityBlock(engine_name="truthound")
result = await block.check(data, rules=[{"column": "email", "check": "email_format"}])
```

## Production Pattern

Use a layered output model:

| Layer | Purpose |
|------|---------|
| shared Truthound result | canonical status, counts, and failure detail |
| Prefect artifact payload | UI-friendly presentation and operator context |

Recommended practice:

- keep the shared result as the machine-readable source of truth
- use artifacts for dashboards and operator visibility
- standardize naming for artifacts so repeated flows are comparable

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| artifact exists but is hard to automate | only display text was persisted | preserve the structured result dictionary too |
| a warning run looks successful but operators miss it | only Prefect state is inspected | include warning counts in the artifact summary |
| outputs differ between flows | each flow formats artifacts differently | route through `to_prefect_artifact` or a shared wrapper |

## Related Pages

- [Prefect Overview](index.md)
- [Blocks](blocks.md)
- [Tasks](tasks.md)
- [Shared Result Serialization](../common/result-serialization.md)
