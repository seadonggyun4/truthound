---
title: Architecture
---

# Architecture

Truthound Orchestration 3.x is designed as a first-party compatibility line for Truthound, not as a neutral shell that happens to call several validation engines. The architecture is intentionally split between host-native adapters and a shared runtime so that platform ergonomics stay natural while release guarantees stay centralized.

## Design Posture

Truthound Orchestration 3.x is not positioned as a neutral multi-engine shell first. It is the official orchestration line for Truthound 3.x, with advanced extension points preserved behind the shared `common` runtime.

## Architecture At A Glance

```text
Host package
  -> host-native boundary (operator / resource / block / macro / script)
  -> shared runtime normalization
  -> engine creation and capability checks
  -> execution
  -> shared wire serialization
  -> host-specific metadata wrapping
```

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

## What Each Platform Owns

Each host package is responsible for the parts users expect to remain native:

| Platform | Boundary Object | What Stays Host-Native |
|----------|-----------------|------------------------|
| Airflow | Operators, Sensors, Hooks | DAG authoring, XCom usage, connections, task semantics |
| Dagster | `ConfigurableResource`, ops, assets | definitions, metadata emission, asset/op wiring |
| Prefect | Blocks, Tasks, Flows | block lifecycle, flow ergonomics, deployment reuse |
| dbt | Generic tests, macros, run operations | YAML authoring, adapter dispatch, model/test compilation |
| Mage | Transformer, condition, and sensor blocks | project conventions, `io_config.yaml` discovery |
| Kestra | scripts and flow generators | YAML flows, task outputs, Kestra runtime conventions |

## Platform Boundaries

| Platform | Boundary Object | What Stays Platform-Native |
|----------|-----------------|----------------------------|
| Airflow | Operators and Sensors | DAG authoring, hooks, XCom metadata |
| Dagster | `ConfigurableResource` | asset/op wiring, metadata emission |
| Prefect | Blocks | saved block lifecycle, task/flow ergonomics |
| dbt | Macros and generic tests | adapter dispatch, YAML test authoring |
| Mage | Transformer and sensor blocks | block semantics, project config conventions |
| Kestra | Scripts and flow templates | YAML flow composition and task outputs |

## Runtime Lifecycle

Most host flows follow the same sequence:

1. The host boundary receives user input such as a local path, DataFrame, SQL string, or rule list.
2. The shared runtime normalizes the host context into `PlatformRuntimeContext`.
3. The source is normalized into a `ResolvedDataSource`.
4. The runtime builds an engine from `create_engine(...)` or `EngineCreationRequest`.
5. `run_preflight(...)` validates engine support, source shape, and serialization readiness.
6. The operation runs through the engine.
7. The result is serialized through the shared wire contract.
8. The host package adds host metadata or rendering wrappers.

This is the core reason platform packages can stay thin while still behaving consistently.

## Zero-Config Contract

The runtime defaults are intentionally strict:

- default engine: `truthound`
- default policy: `safe_auto`
- default Truthound context: `ephemeral`
- default persistence: `persist_runs=False`, `persist_docs=False`
- default workspace creation: `auto_create_workspace=False`

That combination keeps the first run easy while avoiding hidden writes into user projects.

## Capability Enforcement

Capability checks live in the shared runtime rather than the host package. That gives every adapter the same answers to questions like:

- does this engine support streaming?
- can this engine perform drift or anomaly detection?
- can this source be resolved without a host connection?
- is the serializer contract safe for this host output surface?

The public entry points for that layer are described in [Shared Runtime](common/index.md), especially [Preflight and Compatibility](common/preflight-compatibility.md).

## Plugin And Advanced Engine Story

Built-in engines are now described through shared factory specs instead of hand-written branch chains. Third-party engines still enter through the plugin or registry path, but the first-party platform packages do not depend on Truthound-specific internals directly.

## Result Contract

All supported platforms should consume the shared wire format emitted by shared serializers. Platform wrappers may add host metadata around that payload, but they should not redefine status, counts, rule semantics, or execution metadata.

## Result Contract

All supported platforms should consume the shared wire format emitted by `common.serializers.serialize_result_wire(...)`. Platform wrappers may add host metadata around that payload, but they should not redefine result semantics.

## Why This Split Matters Operationally

This architecture gives you three important benefits:

- platform-specific docs can stay practical and native without duplicating engine rules or compatibility logic
- CI can validate compatibility once at the shared runtime layer and then prove each platform still respects it
- docs can explain where to look when something fails: host wiring, shared runtime, engine support, or serializer contract

## Related Reading

- [Getting Started](getting-started.md)
- [Choose a Platform](choose-a-platform.md)
- [Zero-Config](zero-config.md)
- [Shared Runtime](common/index.md)
- [Engines](engines/index.md)
