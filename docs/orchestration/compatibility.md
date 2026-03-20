---
title: Compatibility
---

# Compatibility

## Version Line

`truthound-orchestration 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-orchestration 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-orchestration 1.x` | Pre-Truthound-3 lines | Legacy |

## Platform Expectations

| Platform | Runtime Expectation |
|----------|---------------------|
| Airflow | `truthound-airflow 3.x` with `truthound-orchestration 3.x` and `Truthound 3.x` |
| Dagster | `truthound-dagster 3.x` with the same shared line |
| Prefect | `truthound-prefect 3.x` with the same shared line |
| Mage | `truthound-mage 3.x` with the same shared line |
| Kestra | `truthound-kestra 3.x` with the same shared line |
| dbt | `packages/dbt` version `3.0.0` aligned to the Truthound 3.x line |

## Supported Scope

This release line is intentionally scoped to:

- Truthound 3.x runtime/result contracts
- first-party platform packages in this repository
- advanced-tier Great Expectations and Pandera adapters

It does not promise:

- backward runtime compatibility with Truthound 1.x or 2.x
- equal docs prominence for advanced engines
- hidden auto-migration of old workspaces or result formats

## Compatibility APIs

Use the shared runtime helpers when you need explicit validation:

```python
from common.engines import EngineCreationRequest, normalize_runtime_context, run_preflight

report = run_preflight(
    EngineCreationRequest(
        engine_name="truthound",
        runtime_context=normalize_runtime_context(platform="prefect"),
    )
)
```

`run_preflight(...)` checks engine resolution, serializer readiness, and whether the normalized source needs a connection or profile.
