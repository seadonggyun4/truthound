# Truthound 2.0

Truthound 2.0 is a Polars-first data validation framework organized around a small validation kernel, an explicit planning/runtime split, and a unified plugin platform.

## What Changed

- Validation now flows through `truthound.core`
- `th.check()` remains the primary public facade
- Structured execution output is exposed through `report.validation_run`
- Plugin lifecycle management is unified behind `PluginManager`
- `EnterprisePluginManager` now layers async and optional capabilities on top of the same runtime

## Read This First

- [Getting Started](getting-started/index.md)
- [Architecture](concepts/architecture.md)
- [Plugin Platform](concepts/plugins.md)
- [Migration to 2.0](guides/migration-2.0.md)

## Architectural Lineage

Truthound 2.0 synthesizes several proven design ideas:

- Great Expectations: suite and artifact separation
- Soda: scan planning and backend-aware routing
- Deequ: analyzer, constraint, verification, and repository decomposition
- Pandera: schema-centric validation ergonomics and lazy evaluation

The resulting design is intentionally conservative: the external API remains approachable, while the internal kernel becomes much easier to extend and test.
