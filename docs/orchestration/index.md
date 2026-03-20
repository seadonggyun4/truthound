---
title: Truthound Orchestration
---

# Truthound Orchestration

Truthound Orchestration is the official first-party orchestration compatibility line for Truthound 3.x. It keeps the public experience Truthound-first and zero-config by default, while preserving protocol and plugin boundaries for advanced teams that need custom engines.

## What This Line Optimizes For

- Truthound 3.x is the primary supported runtime and documentation path.
- Platform adapters stay thin and native to their host ecosystems.
- Shared runtime behavior lives in `common`, including engine creation, preflight, result serialization, version checks, and source normalization.
- Zero-config means "easy to run without hidden persistent side effects", not "silently write files everywhere".

## Supported First-Party Platforms

| Platform | Primary Boundary | Zero-Config Default |
|----------|------------------|---------------------|
| Airflow | Operators, Sensors, Hooks | Local file paths work without a connection; SQL requires a connection |
| Dagster | `DataQualityResource()` | Zero-arg resource resolves to Truthound safe auto |
| Prefect | Blocks, Tasks, Flows | Omitted block creates an in-memory Truthound path |
| dbt | Generic tests, macros, adapter dispatch | Package install plus YAML rules is the standard path |
| Mage | Project blocks and `io_config.yaml` discovery | Walks project root for `io_config.yaml`, never writes back |
| Kestra | Python scripts and YAML flow templates | Generated flows default to Truthound safe auto |

## Shared Runtime Boundary

The common runtime owns the behavior that should stay consistent across platforms:

- `create_engine(...)` with a simple engine name or a structured `EngineCreationRequest`
- `PlatformRuntimeContext` for platform metadata and zero-config policy
- `resolve_data_source(...)` for DataFrame, path, URI, SQL, and callable inputs
- `run_preflight(...)` / `build_compatibility_report(...)` for compatibility and installation checks
- Shared wire serialization so platforms do not invent separate result semantics

## Read This Next

- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [Compatibility](compatibility.md)
- [Zero-Config](zero-config.md)
