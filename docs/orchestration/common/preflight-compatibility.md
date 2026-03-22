---
title: Preflight And Compatibility
---

# Preflight And Compatibility

Preflight is the shared runtime checkpoint that runs before real execution. It exists to fail early, explain clearly, and keep host-specific wrappers from improvising different compatibility behavior.

## What Preflight Checks

`run_preflight(...)` and related helpers validate:

- engine resolution
- engine capability support for the requested operation
- source normalization
- whether the source requires a host connection or profile
- serializer readiness
- host metadata needed for a consistent runtime context

## The Main Contracts

These public contracts are the core of the preflight story:

| Type | Purpose |
|------|---------|
| `EngineCreationRequest` | declares engine name and runtime context |
| `PlatformRuntimeContext` | describes the host and zero-config policy |
| `CompatibilityCheck` | one named check with pass, warning, or failure state |
| `CompatibilityReport` | engine and host compatibility summary |
| `PreflightReport` | compatibility plus source and serializer readiness |

## Why This Layer Matters

Without shared preflight, each host would answer different versions of the same question:

- Airflow might reject a source at runtime
- Prefect might create a block and only fail later
- Dagster might surface a resource error with different semantics
- dbt might compile correctly and fail much later in execution

Shared preflight keeps those failure classes aligned.

## Typical Failure Classes

## Unsupported Operation

Example:

- asking a selected engine to perform drift detection, anomaly detection, or streaming when it only supports check/profile/learn

What to do:

- verify the engine capability in [Engines](../engines/index.md)
- switch to Truthound if you expected the first-party default path
- make the requested operation explicit in the platform docs

## Source Needs A Connection

Example:

- Airflow receives SQL without a connection
- dbt execution is missing a matching profile
- a remote URI is reachable only through a host-managed secret or connection contract

What to do:

- keep the source explicit
- let the host provide the connection/profile
- treat zero-config as a local-path onboarding path, not a connection synthesizer

## Serializer Readiness

Example:

- the host wants a result shape that the shared serializer path does not expose
- custom metadata would create an inconsistent output contract

What to do:

- keep the host on the first-party serializer path
- review [Result Serialization](result-serialization.md)

## Recommended Usage Pattern

Use explicit preflight when you are building reusable deployment code, CI checks, or custom wrappers:

```python
from common.engines import EngineCreationRequest, normalize_runtime_context, run_preflight

request = EngineCreationRequest(
    engine_name="truthound",
    runtime_context=normalize_runtime_context(platform="airflow"),
)

report = run_preflight(request)
```

Treat failures as actionable configuration feedback, not as opaque internal errors.

## Related Reading

- [Compatibility](../compatibility.md)
- [Source Resolution](source-resolution.md)
- [Troubleshooting](../troubleshooting.md)
