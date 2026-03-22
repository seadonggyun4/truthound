---
title: Prefect Install And Compatibility
---

# Prefect Install And Compatibility

## Supported Prefect Line

The current support anchors are:

- minimum supported: Prefect `2.14.0` on Python `3.11`
- primary supported: Prefect `3.6.22` on Python `3.12`

See [Compatibility](../compatibility.md) for the release-blocking matrix.

## Install

```bash
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
```

## What To Verify After Install

- `truthound_prefect` imports cleanly
- a flow can call `data_quality_check_task`
- a `DataQualityBlock` can be created and, if needed, saved
- your preferred Prefect version matches a tested support tuple

## Operational Advice

- use ephemeral defaults for local or ad hoc runs
- move to saved blocks when deployments need stable shared configuration
- debug Prefect-specific failures at the block, task, or flow boundary first

## Related Reading

- [Blocks](blocks.md)
- [Deployment Patterns](deployment-patterns.md)
