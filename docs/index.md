<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound 3.0

Truthound is a zero-configuration data quality framework powered by Polars.
The current release is designed around one practical promise: `th.check(data)` should be enough to get exact validation, reusable project state, and trustworthy artifacts without configuration ceremony.

## Why Truthound

- Zero-config by default through `TruthoundContext` and the local `.truthound/` workspace
- Polars-first metric planning and backend-aware execution instead of repeated validator-loop scans
- One canonical runtime result, `ValidationRunResult`, shared across checkpoints, reporters, and validation docs
- Deterministic auto-suite selection for schema, nullability, type, range, and key-like signals
- Release-grade benchmark verification against Great Expectations on comparable deterministic workloads

## Verified Benchmark Snapshot

The latest fixed-runner benchmark verification shows:

- Truthound finished ahead of Great Expectations on all eight comparable release-grade workloads
- local speedups ranged from `1.51x` to `11.70x`
- SQLite pushdown speedups ranged from `3.69x` to `7.58x`
- local peak RSS stayed between `35.88%` and `48.16%` of Great Expectations
- correctness parity was preserved across the full comparable workload set

Read the full evidence in [Latest Verified Benchmark Summary](releases/latest-benchmark-summary.md).

## Start Here

- [Quick Start](getting-started/quickstart.md)
- [Architecture](concepts/architecture.md)
- [Zero-Config Context](concepts/zero-config.md)
- [Orchestration Overview](orchestration/index.md)
- [Performance and Benchmarks](guides/performance.md)
- [Great Expectations Comparison](guides/gx-parity.md)
- [Latest Verified Benchmark Summary](releases/latest-benchmark-summary.md)

## Core Platform Shape

Truthound 3.0 centers the public platform around:

- `TruthoundContext` for project discovery, baselines, artifacts, and defaults
- `ValidationSuite`, `CheckSpec`, and `SchemaSpec` for immutable validation intent
- planner/runtime boundaries that compile checks into metric-oriented execution
- `ValidationRunResult` and `CheckResult` as the canonical runtime output

The result is a smaller and more maintainable kernel while keeping the public facade approachable for day-one use.

## Documentation Paths

- [Getting Started](getting-started/index.md)
- [Plugin Platform](concepts/plugins.md)
- [Orchestration](orchestration/index.md)
- [Benchmark Methodology](guides/benchmark-methodology.md)
- [Docs Deployment Verification](guides/docs-deployment-verification.md)
- [Migration to 3.0](guides/migration-3.0.md)
