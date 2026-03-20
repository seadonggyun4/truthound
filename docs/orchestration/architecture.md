---
title: Architecture
---

# Architecture

## Design Posture

Truthound Orchestration 3.x is not positioned as a neutral multi-engine shell first. It is the official orchestration line for Truthound 3.x, with advanced extension points preserved behind the shared `common` runtime.

## Shared Runtime Boundary

`common` is the only layer that should decide:

- engine creation and alias normalization
- Truthound 3.x version enforcement
- zero-config policy defaults
- source normalization
- compatibility and preflight checks
- shared wire serialization

Key public runtime types:

- `EngineCreationRequest`
- `PlatformRuntimeContext`
- `AutoConfigPolicy`
- `CompatibilityReport`
- `PreflightReport`

## Platform Boundaries

| Platform | Boundary Object | What Stays Platform-Native |
|----------|-----------------|----------------------------|
| Airflow | Operators and Sensors | DAG authoring, hooks, XCom metadata |
| Dagster | `ConfigurableResource` | asset/op wiring, metadata emission |
| Prefect | Blocks | saved block lifecycle, task/flow ergonomics |
| dbt | Macros and generic tests | adapter dispatch, YAML test authoring |
| Mage | Transformer and sensor blocks | block semantics, project config conventions |
| Kestra | Scripts and flow templates | YAML flow composition and task outputs |

## Zero-Config Contract

The runtime defaults are intentionally strict:

- default engine: `truthound`
- default policy: `safe_auto`
- default Truthound context: `ephemeral`
- default persistence: `persist_runs=False`, `persist_docs=False`
- default workspace creation: `auto_create_workspace=False`

That combination keeps the first run easy while avoiding hidden writes into user projects.

## Plugin And Advanced Engine Story

Built-in engines are now described through shared factory specs instead of hand-written branch chains. Third-party engines still enter through the plugin or registry path, but the first-party platform packages do not depend on Truthound-specific internals directly.

## Result Contract

All supported platforms should consume the shared wire format emitted by `common.serializers.serialize_result_wire(...)`. Platform wrappers may add host metadata around that payload, but they should not redefine result semantics.
