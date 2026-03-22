# Reference

Reference is the lookup layer for **Truthound Core**. Use it when you already
know which command, function, type, or option you need and want the canonical
details without walking through a tutorial.

This section does **not** try to document the whole product line in one place.
For host-native adapters, go to [Truthound Orchestration](../orchestration/index.md).
For control-plane APIs and operations, go to [Truthound Dashboard](../dashboard/index.md).

## What Lives Here

### Python API

Use the [Python API reference](../python-api/index.md) when you want to:

- call the core `truthound` package from notebooks, scripts, services, or jobs
- inspect `ValidationRunResult` and related result types directly
- work with `truthound.drift`, `truthound.checkpoint`, `truthound.reporters`, or `truthound.profiler`
- jump from guides into function-level or class-level documentation

Recommended entry points:

- [Python API Overview](../python-api/index.md)
- [Core Functions](../python-api/core-functions.md)
- [Schema](../python-api/schema.md)
- [Validators](../python-api/validators.md)
- [Datasources](../python-api/datasources.md)
- [Reporters](../python-api/reporters.md)

### CLI

Use the [CLI reference](../cli/index.md) when you want to:

- run core validations without writing Python
- generate suites, docs, and benchmark artifacts from the terminal
- operate checkpoints from shell scripts or CI
- scaffold validators, reporters, and plugins

Recommended entry points:

- [CLI Overview](../cli/index.md)
- [Core Commands](../cli/core/index.md)
- [Checkpoint Commands](../cli/checkpoint/index.md)
- [Profiler Commands](../cli/profiler/index.md)
- [Plugin Commands](../cli/plugin/index.md)
- [Scaffolding Commands](../cli/scaffolding/index.md)

## Layer Boundaries

| Need | Best section |
|------|--------------|
| Install Truthound and run your first validation | [Getting Started](../getting-started/index.md) |
| Learn with end-to-end runnable examples | [Tutorials](../tutorials/index.md) |
| Solve a feature-specific core task | [Guides](../guides/index.md) |
| Understand the layered system and kernel boundaries | [Concepts & Architecture](../concepts/index.md) |
| Run Truthound inside Airflow, Dagster, Prefect, dbt, Mage, or Kestra | [Truthound Orchestration](../orchestration/index.md) |
| Operate a UI, RBAC, artifacts, incidents, and observability | [Truthound Dashboard](../dashboard/index.md) |

## Common Lookup Paths

- I want to validate a file from the terminal:
  Start with [CLI Overview](../cli/index.md), then go to [truthound check](../cli/core/check.md).
- I want to validate a dataframe in Python:
  Start with [Python API Overview](../python-api/index.md), then go to [Core Functions](../python-api/core-functions.md).
- I want host-native execution in a scheduler:
  Go directly to [Truthound Orchestration](../orchestration/index.md).
- I want control-plane REST contracts:
  Go directly to [Truthound Dashboard API Reference](../dashboard/api-reference/overview-and-conventions.md).
- I want to compare Truthound with GX-style workflows:
  Read [Great Expectations Comparison](../guides/gx-parity.md) and [Migration to 3.0](../guides/migration-3.0.md).

## Related Reading

- [Getting Started](../getting-started/index.md)
- [Tutorials](../tutorials/index.md)
- [Guides](../guides/index.md)
- [Concepts & Architecture](../concepts/index.md)
- [Truthound Orchestration](../orchestration/index.md)
- [Truthound Dashboard](../dashboard/index.md)
