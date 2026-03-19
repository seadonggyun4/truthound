# Truthound 3.0

Truthound 3.0 is a Polars-first validation framework centered on three promises:

- zero-configuration is the default user experience
- correctness and operational stability are first-class kernel responsibilities
- extension points stay small, explicit, and durable under change

## Start Here

- [Getting Started](getting-started/index.md)
- [Quick Start](getting-started/quickstart.md)
- [Architecture](concepts/architecture.md)
- [Zero-Config Context](concepts/zero-config.md)
- [Plugin Platform](concepts/plugins.md)
- [Migration to 3.0](guides/migration-3.0.md)

## Core Shift

Truthound 3.0 is a hard contract reset, not a 2.x compatibility extension:

- `th.check()` returns `ValidationRunResult` directly
- `TruthoundContext` auto-discovers and manages a local `.truthound/` workspace
- `validators=None` triggers deterministic auto-suite synthesis
- planning is backend-aware and metric-oriented instead of validator-loop oriented
- checkpoints, reporters, and validation docs all consume the same canonical result model

## Architectural Lineage

Truthound 3.0 synthesizes several proven ideas:

- Great Expectations: suite, checkpoint, and artifact separation
- Soda: scan planning and backend-aware metric batching
- Deequ: analyzer, constraint, verification, and repository decomposition
- Pandera: schema-centric ergonomics and lazy data interaction

The 3.0 design keeps the friendly facade while making the internals more maintainable, benchmarkable, and predictable at scale.
