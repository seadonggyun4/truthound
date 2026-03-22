---
title: Engine Resolution and Selection
---

# Engine Resolution and Selection

The shared runtime deliberately separates host-native orchestration boundaries
from engine creation. Airflow operators, Dagster resources, Prefect blocks,
Mage blocks, Kestra scripts, and dbt macros all converge on the same
engine-resolution model.

## Who This Is For

- teams choosing between Truthound, Pandera, and Great Expectations
- operators debugging why a host created one engine instead of another
- contributors wiring a new adapter into the shared runtime

## When To Use It

Use this page when you need to understand how `create_engine(...)`,
`EngineCreationRequest`, and `PlatformRuntimeContext` interact.

## Prerequisites

- familiarity with [Shared Runtime](index.md)
- a host-level understanding of where engine configuration lives in your chosen
  adapter
- the desired execution engine name

## Minimal Quickstart

The common resolver accepts either a bare engine name or an
`EngineCreationRequest` with runtime metadata:

```python
from common.engines import EngineCreationRequest, create_engine
from common.runtime import normalize_runtime_context

runtime_context = normalize_runtime_context(
    platform="prefect",
    host_metadata={"deployment": "daily-quality"},
)

engine = create_engine(
    EngineCreationRequest(
        engine_name="truthound",
        runtime_context=runtime_context,
    )
)
```

For a pure default path, most adapters simply rely on the default engine:

```python
from common.engines import create_engine

engine = create_engine("truthound")
```

## How Resolution Works

The resolver is centered on these primitives:

| Primitive | Responsibility |
|-----------|----------------|
| `EngineCreationRequest` | captures engine name, runtime context, observability config, and source intent |
| `PlatformRuntimeContext` | records which host is calling into the runtime and which auto-config policy applies |
| `create_engine(...)` | creates the concrete engine instance from the registry and request |
| `build_compatibility_report(...)` | checks host-engine compatibility without executing an operation |
| `run_preflight(...)` | extends compatibility checks with source resolution and serializer readiness |

The default behavior in this repository is:

- Truthound is the default engine
- runtime context is host-specific when the host adapter supplies it
- preflight runs before execution in the host-native adapters
- host packages keep engine creation in shared code rather than reimplementing
  resolver logic locally

## Decision Guide

| If you need... | Prefer... | Why |
|----------------|-----------|-----|
| the default first-party path | `truthound` | best aligned with docs, CI tuples, and release guarantees |
| dataframe-model validation with a schema-first surface | `pandera` | strong fit for Python dataframe contracts |
| an expectations-style validation model or GE reuse | `great_expectations` | best for teams already carrying GE concepts forward |
| more than one engine | chain or custom routing | keep host integration stable and route in the shared runtime |

## Production Pattern

- Set the engine explicitly in production configs even if it matches the default.
- Keep the host-level configuration surface small and pass rich context through
  `EngineCreationRequest`.
- Treat `run_preflight(...)` as a release gate, not just a debugging helper.
- Use a chain or registry extension only when there is a real operational need.

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | What To Check |
|---------|--------------|---------------|
| wrong engine created | host default applied because engine name was omitted | inspect the host config surface and request object |
| preflight fails before execution | capability mismatch or unsupported operation | compare requested operation to the engine's capability matrix |
| behavior differs between hosts | runtime context changes auto-config or source expectations | compare `PlatformRuntimeContext` values |

## Related Pages

- [Preflight and Compatibility](preflight-compatibility.md)
- [Capability Matrix](../engines/capability-matrix.md)
- [Engine Selection Guide](../engines/selection-guide.md)
- [Custom Engines](../engines/custom-engines.md)
