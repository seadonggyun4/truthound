<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound 3.0

Truthound is a zero-configuration, Polars-first data quality platform for teams that want fast validation, strong defaults, and a clear path from local checks to production workflows.

The 3.0 documentation site is organized for adoption first:

- start with a first validation in minutes
- choose Python or CLI based on how you work
- grow into checkpoints, profiling, reporting, orchestration, and plugins without switching mental models
- keep one canonical runtime result model across validation, docs, reporters, and automation

## Choose Your Path

| I want to... | Start here |
|--------------|------------|
| Run my first validation with almost no setup | [Getting Started](getting-started/quickstart.md) |
| Understand how Truthound differs from GX-style workflows | [Great Expectations Comparison](guides/gx-parity.md) |
| Learn by building something end to end | [Tutorials](tutorials/index.md) |
| Use Truthound from scripts or services | [Python API Reference](python-api/index.md) |
| Use Truthound from a terminal or CI job | [CLI Reference](cli/index.md) |
| Design production automation and orchestration | [Checkpoints Guide](guides/checkpoints.md) and [Orchestration](orchestration/index.md) |

## What Truthound Optimizes For

- Zero-config startup through `TruthoundContext` and a local `.truthound/` workspace
- Polars-first execution planning instead of slow validator-by-validator scan loops
- One canonical runtime result, `ValidationRunResult`, across checkpoints, reporters, and docs
- Deterministic auto-suite selection for schema, nullability, type, range, uniqueness, and key-like checks
- Release-grade benchmark verification against Great Expectations on comparable deterministic workloads

## Start With The Core Workflow

### 1. First validation

Run one file or dataframe and let Truthound infer the practical first checks:

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [First Validation](getting-started/first-validation.md)

### 2. Learn the mental model

Understand the boundaries that make 3.0 consistent:

- [Concepts Overview](concepts/index.md)
- [Architecture](concepts/architecture.md)
- [Zero-Config Context](concepts/zero-config.md)
- [Plugin Platform](concepts/plugins.md)

### 3. Move into task-oriented workflows

Use guides when you already know the job to be done:

- [Guides Overview](guides/index.md)
- [Validators](guides/validators/index.md)
- [Datasources](guides/datasources/index.md)
- [Checkpoints](guides/checkpoint/index.md)
- [Reporters](guides/reporters/index.md)
- [Profiler](guides/profiler/index.md)
- [Stores](guides/stores/index.md)
- [Data Docs](guides/datadocs/index.md)

## Verified Benchmark Snapshot

The latest fixed-runner benchmark verification shows:

- Truthound finished ahead of Great Expectations on all eight comparable release-grade workloads
- local speedups ranged from `1.51x` to `11.70x`
- SQLite pushdown speedups ranged from `3.69x` to `7.58x`
- local peak RSS stayed between `35.88%` and `48.16%` of Great Expectations
- correctness parity was preserved across the full comparable workload set

Read the full evidence in [Latest Verified Benchmark Summary](releases/latest-benchmark-summary.md).

## Core Platform Shape

Truthound 3.0 centers the public platform around:

- `TruthoundContext` for project discovery, baselines, artifacts, and defaults
- `ValidationSuite`, `CheckSpec`, and `SchemaSpec` for immutable validation intent
- planner/runtime boundaries that compile checks into metric-oriented execution
- `ValidationRunResult` and `CheckResult` as the canonical runtime output

The result is a smaller and more maintainable kernel while keeping the public facade approachable for day-one use.

## Documentation Map

- [Getting Started](getting-started/index.md): installation, first validation, zero-config basics
- [Tutorials](tutorials/index.md): sequential, runnable learning paths
- [Guides](guides/index.md): task-oriented documentation by feature family
- [Reference](reference/index.md): Python API and CLI lookup hubs
- [Orchestration](orchestration/index.md): external schedulers, engines, and workflow integration
- [Release Notes](releases/truthound-3.0.md): release narrative and verified benchmark surface
- [Migration to 3.0](guides/migration-3.0.md): workflow and compatibility guidance for upgrades
