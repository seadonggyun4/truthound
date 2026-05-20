<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound 3.x

Truthound is a layered data quality system. The center of that system is
Truthound Core: a validation-first, Polars-first kernel built around
`TruthoundContext`, `ValidationRunResult`, deterministic auto-suites, and a
planner/runtime execution boundary. Around that core sit Truthound
Orchestration for host-native execution in schedulers and workflow systems, an
additive `truthound.ai` namespace for reviewable AI proposal and analysis
workflows, and Truthound Depot for operating dataset repositories through an
installation-managed console.

This portal keeps the core and orchestration surfaces fully documented, adds
the public `truthound.ai` contract, and keeps the Depot surface visible at the
boundary level without reproducing the full console implementation here.

## Truthound By Layer

| Layer | Owns | Start here when you need to... |
|--------------|------------|--------------------------------|
| **Truthound Core** | Validation kernel, zero-config workspace, result model, reporters, Data Docs, checkpoint runtime, profiling, benchmarked execution path | run your first validation, use the Python API or CLI, understand `ValidationRunResult`, or evaluate the core contract |
| **Truthound AI** | Optional review-layer APIs for proposal generation, run analysis, approval logs, and controlled apply flows | understand the AI boundary, compile prompt-driven suite proposals, or analyze run evidence without mutating core state by default |
| **Truthound Orchestration** | First-party execution integration layer for Airflow, Dagster, Prefect, dbt, Mage, and Kestra | run Truthound inside a scheduler, asset graph, flow system, or warehouse-native orchestration surface |
| **Truthound Depot** | Installation-managed dataset repository console for branch, push, compare, merge request, quality gate, release, rollback, evidence, RBAC, and observability workflows | operate dataset repositories, manage team access, review merge/release decisions, or inspect evidence through a web UI |

## Choose Your Entry Point

| I want to... | Start here |
|--------------|------------|
| Run my first validation with almost no setup | [Core Getting Started](getting-started/quickstart.md) |
| Learn the core workflow end to end | [Core Tutorials](tutorials/index.md) |
| Use Truthound from scripts or services | [Core Python API](python-api/index.md) |
| Use Truthound from a terminal or CI job | [Core CLI Reference](cli/index.md) |
| Learn the optional AI proposal and analysis contract | [Truthound AI](ai/index.md) |
| Design scheduler-native execution | [Truthound Orchestration](orchestration/index.md) |
| Operate a dataset repository console for Truthound | [Truthound Depot](dashboard/index.md) |
| Understand how the layers fit together | [Concepts & Architecture](concepts/index.md) |
| Understand private Depot engine contracts before public API promotion | [Depot Engine Primitives](concepts/depot-engine-primitives.md) |

## Why The Core Comes First

Truthound Core is the most rigorously validated layer in the product line. It
is where the primary runtime contracts, benchmark evidence, and release-grade
behavior are fixed.

- `ValidationRunResult` is the canonical runtime output
- deterministic auto-suite selection replaces "run everything" defaults
- planner/runtime boundaries keep execution exact-by-default and maintainable
- `TruthoundContext` owns the zero-config `.truthound/` workspace
- benchmark claims are intentionally bounded to comparable core workloads
- private Depot engine primitives provide redacted artifact envelopes,
  deterministic fingerprint/diff contracts, and quality gate projections for
  Truthound Depot and Truthound Orchestration without creating a public
  `truthound.datasets` or `truthound.depot` API

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
- [Depot Engine Primitives](concepts/depot-engine-primitives.md)

### Orchestration

Use `Orchestration` when Truthound should feel native inside Airflow, Dagster,
Prefect, dbt, Mage, or Kestra:

- [Orchestration Overview](orchestration/index.md)
- [Choose a Platform](orchestration/choose-a-platform.md)
- [Shared Runtime](orchestration/common/index.md)

### AI

Use `AI` when you need a reviewable, artifact-driven AI layer on top of the
core validation contract:

- [AI Overview](ai/index.md)
- [System Boundary](ai/system-boundary.md)
- [Proposal Compiler](ai/proposal-compiler.md)
- [Run Analysis Evidence Model](ai/run-analysis-evidence.md)
- [Approval and Apply Semantics](ai/approval-apply.md)

### Depot

Use `Depot` when you need the dataset repository console:

- [Depot Overview](dashboard/index.md)
- branch, push, compare, pull request, release, and rollback workflows
- deployment-aligned review, approval, evidence, and access surfaces
- AI Evidence workflows built on `truthound.ai` review artifacts

## Keep Reading

- [Core Getting Started](getting-started/index.md)
- [Truthound AI](ai/index.md)
- [Truthound Orchestration](orchestration/index.md)
- [Truthound Depot](dashboard/index.md)
- [Release Notes](releases/truthound-3.1.3.md)
- [Migration to 3.0](guides/migration-3.0.md)
