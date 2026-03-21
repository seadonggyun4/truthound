# Reference

Truthound's reference section is the fastest way to look up commands, APIs, and operational entry points once you already know what you want to do.

Use this section when you need exact interfaces, option tables, or canonical links to the deeper command and API pages. If you are still learning the workflow, start with [Getting Started](../getting-started/index.md), [Tutorials](../tutorials/index.md), or [Guides](../guides/index.md) first.

## What Lives Here

### Python API

Use the [Python API reference](../python-api/index.md) when you want to:

- call `truthound` from scripts, notebooks, or services
- understand the root facade such as `th.check()`, `th.profile()`, or `th.read()`
- work with `ValidationRunResult`, schemas, reporters, and datasource integrations
- jump from workflow docs into function-level or class-level reference material
- find namespace APIs such as `truthound.drift.compare()` and checkpoint orchestration

Recommended entry points:

- [Python API Overview](../python-api/index.md)
- [Core Functions](../python-api/core-functions.md)
- [Schema](../python-api/schema.md)
- [Validators](../python-api/validators.md)
- [Datasources](../python-api/datasources.md)
- [Reporters](../python-api/reporters.md)

### CLI

Use the [CLI reference](../cli/index.md) when you want to:

- run validations without writing Python
- generate suites, docs, and benchmark reports from the terminal
- operate checkpoints and orchestration-related workflows from shell scripts or CI
- scaffold validators, reporters, and plugins

Recommended entry points:

- [CLI Overview](../cli/index.md)
- [Core Commands](../cli/core/index.md)
- [Checkpoint Commands](../cli/checkpoint/index.md)
- [Profiler Commands](../cli/profiler/index.md)
- [Plugin Commands](../cli/plugin/index.md)
- [Scaffolding Commands](../cli/scaffolding/index.md)

## Choose The Right Doc Type

| Need | Best section |
|------|--------------|
| Install Truthound and run your first validation | [Getting Started](../getting-started/index.md) |
| Learn with end-to-end runnable examples | [Tutorials](../tutorials/index.md) |
| Solve a feature-specific task | [Guides](../guides/index.md) |
| Understand architecture and system boundaries | [Concepts & Architecture](../concepts/index.md) |
| Look up command or API details | `Reference` |

## Common Lookup Paths

- I want to validate a file from the terminal:
  Start with [CLI Overview](../cli/index.md), then go to [truthound check](../cli/core/check.md).
- I want to validate a dataframe in Python:
  Start with [Python API Overview](../python-api/index.md), then go to [Core Functions](../python-api/core-functions.md).
- I want to build a plugin:
  Use [Plugin Commands](../cli/plugin/index.md) and [truthound new plugin](../cli/scaffolding/new-plugin.md), then continue to [Plugin Platform](../concepts/plugins.md).
- I want to compare Truthound with GX-style workflows:
  Read [Great Expectations Comparison](../guides/gx-parity.md) and [Migration to 3.0](../guides/migration-3.0.md).

## Related Reading

- [Getting Started](../getting-started/index.md)
- [Tutorials](../tutorials/index.md)
- [Guides](../guides/index.md)
- [Concepts & Architecture](../concepts/index.md)
