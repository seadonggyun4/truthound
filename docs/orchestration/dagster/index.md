---
title: Dagster
---

# Dagster

Dagster is the best fit when Truthound quality logic should feel like part of your asset graph rather than a separate orchestration layer. The integration is centered on `ConfigurableResource`, ops, and asset helpers that preserve Dagster-native structure while reusing the shared runtime.

## Why Teams Choose Dagster

- quality checks can sit next to assets and asset checks
- results can flow into Dagster-native metadata
- resource configuration gives a clean place to define engine behavior
- the same resource can power check, profile, learn, and streaming patterns

## Quickstart

Install the supported Dagster surface:

```bash
pip install truthound-orchestration[dagster] "truthound>=3.0,<4.0"
```

Then wire the default resource:

```python
from dagster import Definitions, asset
from truthound_dagster.resources import DataQualityResource

@asset
def validated_users(data_quality: DataQualityResource):
    return data_quality.check(
        load_users(),
        rules=[{"column": "user_id", "type": "not_null"}],
    )

defs = Definitions(resources={"data_quality": DataQualityResource()})
```

`DataQualityResource()` with no arguments is the canonical default.

## What The Resource Buys You

- one place to configure engine selection, timeout, failure policy, and observability
- shared runtime preflight before real execution
- helper methods for `check`, `profile`, `learn`, and `stream_check`
- alignment with Dagster resource lifecycle hooks

## Primary Surfaces

| Surface | Use It For |
|---------|------------|
| `DataQualityResource` | resource-first integration and direct execution |
| prebuilt ops | operation-level composition in jobs |
| asset decorators and factories | quality-aware assets and asset checks |
| SLA helpers | enforcing operational thresholds around quality runs |

## Runtime Shape

- resource setup uses the shared resolver
- preflight runs before engine creation proceeds
- Dagster metadata stays Dagster-native
- result semantics stay shared with the rest of the repository

## Read Next

- [Install and Compatibility](install-compatibility.md)
- [Resources](resources.md)
- [Ops](ops.md)
- [Assets and Asset Checks](assets.md)
- [SLA and Hooks](sla-hooks.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
