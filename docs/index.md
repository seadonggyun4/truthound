<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound 3.0

Truthound is a layered data quality system. The center of that system is
Truthound Core: a validation-first, Polars-first kernel built around
`TruthoundContext`, `ValidationRunResult`, deterministic auto-suites, and a
planner/runtime execution boundary. Around that core sit two first-party
layers: Truthound Orchestration for host-native execution in schedulers and
workflow systems, and Truthound Dashboard for operating Truthound through a
control-plane UI.

This portal combines those layers in one documentation site without flattening
them into one undifferentiated platform.

## Truthound By Layer

| Layer | Owns | Start here when you need to... |
|--------------|------------|--------------------------------|
| **Truthound Core** | Validation kernel, zero-config workspace, result model, reporters, Data Docs, checkpoint runtime, profiling, benchmarked execution path | run your first validation, use the Python API or CLI, understand `ValidationRunResult`, or evaluate the core contract |
| **Truthound Orchestration** | First-party execution integration layer for Airflow, Dagster, Prefect, dbt, Mage, and Kestra | run Truthound inside a scheduler, asset graph, flow system, or warehouse-native orchestration surface |
| **Truthound Dashboard** | First-party control-plane for sessions, RBAC, sources, artifacts, incidents, secrets, and observability | operate Truthound deployments, manage teams and ownership, or review runs through a web UI |

## Choose Your Entry Point

| I want to... | Start here |
|--------------|------------|
| Run my first validation with almost no setup | [Core Getting Started](getting-started/quickstart.md) |
| Learn the core workflow end to end | [Core Tutorials](tutorials/index.md) |
| Use Truthound from scripts or services | [Core Python API](python-api/index.md) |
| Use Truthound from a terminal or CI job | [Core CLI Reference](cli/index.md) |
| Design scheduler-native execution | [Truthound Orchestration](orchestration/index.md) |
| Operate a control-plane UI for Truthound | [Truthound Dashboard](dashboard/index.md) |
| Understand how the layers fit together | [Concepts & Architecture](concepts/index.md) |

## Why The Core Comes First

Truthound Core is the most rigorously validated layer in the product line. It
is where the primary runtime contracts, benchmark evidence, and release-grade
behavior are fixed.

- `ValidationRunResult` is the canonical runtime output
- deterministic auto-suite selection replaces "run everything" defaults
- planner/runtime boundaries keep execution exact-by-default and maintainable
- `TruthoundContext` owns the zero-config `.truthound/` workspace
- benchmark claims are intentionally bounded to comparable core workloads

## Verified Core Benchmark Snapshot

The latest fixed-runner benchmark verification shows:

- Truthound Core finished ahead of Great Expectations on all eight comparable release-grade workloads
- local speedups ranged from `1.51x` to `11.70x`
- SQLite pushdown speedups ranged from `3.69x` to `7.58x`
- local peak RSS stayed between `35.88%` and `48.16%` of Great Expectations
- correctness parity was preserved across the full comparable workload set

Read the evidence in [Latest Verified Benchmark Summary](releases/latest-benchmark-summary.md).

## How This Portal Is Organized

### Core

Use `Core` when you need the kernel itself:

- [Getting Started](getting-started/index.md)
- [Tutorials](tutorials/index.md)
- [Guides](guides/index.md)
- [Reference](reference/index.md)
- [Concepts & Architecture](concepts/index.md)

### Orchestration

Use `Orchestration` when Truthound should feel native inside Airflow, Dagster,
Prefect, dbt, Mage, or Kestra:

- [Orchestration Overview](orchestration/index.md)
- [Choose a Platform](orchestration/choose-a-platform.md)
- [Shared Runtime](orchestration/common/index.md)

### Dashboard

Use `Dashboard` when you need the operational control-plane:

- [Dashboard Overview](dashboard/index.md)
- [Quickstart](dashboard/quickstart/install-and-run.md)
- [Operations](dashboard/operations/ci-and-quality-gates.md)

## Keep Reading

- [Core Getting Started](getting-started/index.md)
- [Truthound Orchestration](orchestration/index.md)
- [Truthound Dashboard](dashboard/index.md)
- [Release Notes](releases/truthound-3.0.md)
- [Migration to 3.0](guides/migration-3.0.md)
