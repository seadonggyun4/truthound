---
title: Truthound Orchestration
---

# Truthound Orchestration

Truthound Orchestration is the official first-party orchestration line for Truthound 3.x. It gives Airflow, Dagster, Prefect, dbt, Mage, and Kestra users a native host experience while keeping Truthound-first defaults, shared runtime contracts, and release-grade compatibility testing.

This site is written for engineers who need more than a package install command. It explains how the shared runtime works, what each host adapter owns, how zero-config behaves in production, and where to look when a workflow fails in CI or at runtime.

## Who This Site Is For

- platform engineers choosing a supported host for Truthound 3.x
- data platform teams wiring quality checks into schedulers, assets, flows, and SQL builds
- operators who need compatibility expectations, rollout guidance, and troubleshooting steps
- contributors who need to understand the public runtime contract without reverse-engineering tests

## What This Line Optimizes For

- Truthound 3.x is the primary engine and documentation path.
- First-party platform packages stay native to their host ecosystems instead of inventing a separate orchestration DSL.
- Shared runtime behavior is centralized in `common`, including engine creation, preflight, result serialization, compatibility reporting, and source normalization.
- Zero-config means "fast first run without hidden state", not "silent workspace writes or surprise persistence".
- Release guarantees are tested as concrete host-plus-Python tuples in CI, not as vague version promises.

## Supported First-Party Platforms

| Platform | Primary Boundary | Best Fit | Zero-Config Default |
|----------|------------------|----------|---------------------|
| Airflow | Operators, Sensors, Hooks | DAG-based scheduling and quality gates in task graphs | Local file paths run without a connection; SQL still requires one |
| Dagster | `DataQualityResource`, ops, asset helpers | Asset-centric pipelines and metadata-rich quality checks | `DataQualityResource()` resolves to Truthound safe auto |
| Prefect | Blocks, Tasks, Flows | Python-first orchestration with optional persisted configuration | Omitting a block creates an in-memory Truthound-backed path |
| dbt | Generic tests, macros, operations | Warehouse-native validation with YAML authoring | Install the package, add tests, and rely on adapter dispatch |
| Mage | Blocks and `io_config.yaml` discovery | Mage projects that want lightweight block-level quality checks | Project config discovery is read-only and falls back to safe auto |
| Kestra | Python scripts and YAML flow templates | Kestra task runners and generated validation flows | Scripts and templates default to Truthound safe auto |

## Shared Runtime Boundary

The shared runtime owns the behavior that should remain consistent regardless of host:

- `create_engine(...)` and `EngineCreationRequest` for deterministic engine resolution
- `PlatformRuntimeContext` and `AutoConfigPolicy` for host metadata and zero-config rules
- `resolve_data_source(...)` for DataFrames, local paths, URIs, SQL strings, streams, and callables
- `run_preflight(...)` and `build_compatibility_report(...)` for capability and compatibility checks
- shared wire serialization so Airflow XCom, Dagster metadata, Prefect artifacts, and script outputs do not invent incompatible payloads
- shared observability events and OpenLineage-compatible execution metadata

## Documentation Spine

Use this path if you are new to the repository:

1. [Getting Started](getting-started.md) for the fastest first run on each platform.
2. [Choose a Platform](choose-a-platform.md) if you are still deciding where Truthound should live.
3. [Architecture](architecture.md) to understand the adapter/runtime split.
4. [Zero-Config](zero-config.md) to learn what is automatic and what still requires explicit configuration.
5. [Compatibility](compatibility.md) before planning upgrades, release gates, or host-version support.

Then go deeper into:

- [Shared Runtime](common/index.md) for source resolution, preflight, serialization, logging, retries, and resilience
- platform sections for implementation patterns and troubleshooting
- [Engines](engines/index.md) if you need Pandera, Great Expectations, streaming, drift, anomaly detection, or engine lifecycle controls
- [Enterprise & Operations](enterprise/index.md) for secrets, notifications, tenant isolation, and production rollout guidance

## What "Production Ready" Means Here

Truthound Orchestration is considered production-ready when all of the following are true:

- the host package is installed from a supported compatibility tuple
- the shared runtime resolves the source shape you actually provide
- preflight passes before execution starts
- result payloads are consumed through the documented host contract
- persistence, secrets, SLA thresholds, and notifications are explicit where needed

The docs in this site are organized to help you validate each of those layers in order.
