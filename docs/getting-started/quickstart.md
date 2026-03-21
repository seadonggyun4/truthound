# Quick Start

This page gets you from install to a real validation result with both the Python API and the CLI. The examples intentionally start small so you can understand the default behavior before moving into suites, checkpoints, or orchestration.

## What You Will Build

- a zero-config validation run
- a `.truthound/` workspace with reusable local state
- a mental model for when to stay simple and when to reach for deeper guides

## Python Workflow

### Step 1: Validate data with zero configuration

```python
import truthound as th

run = th.check(
    {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
)

print(run.execution_mode)
print([issue.issue_type for issue in run.issues])
print(run.metadata["context_root"])
```

`th.check()` now returns `ValidationRunResult` directly. On first use, Truthound automatically creates `.truthound/`, learns or loads a baseline, synthesizes an auto-suite, executes the run, and persists run/docs artifacts when the active context allows it.

### Step 2: Inspect the canonical result

```python
print(run.source)
print(run.execution_mode)
print([check.name for check in run.checks])
print([issue.issue_type for issue in run.issues])
print(run.metadata.get("context_run_artifact"))
```

`ValidationRunResult` is the canonical runtime output for checkpoints, reporters, validation docs, and plugins.

### Step 3: Inspect the project context

```python
import truthound as th

context = th.get_context()

print(context.workspace_dir)
print(context.baselines_dir)
print(context.docs_dir)
```

Use `TruthoundContext` when you need:

- explicit project-root control
- direct access to baseline and run artifacts
- deterministic local persistence behavior

## CLI Workflow

If you prefer a terminal-first path, Truthound keeps the CLI close to the Python mental model:

```bash
truthound check customers.csv
truthound profile customers.csv
truthound scan customers.csv
```

Use the CLI when you want:

- repeatable shell commands
- CI-friendly validation entry points
- quick profiling, reporting, or benchmark runs without writing Python

Continue into the [CLI Reference](../cli/index.md) once you want full option tables and command groups.

## Move Beyond Zero-Config

### Use the kernel directly when you need explicit control

```python
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset
from truthound.context import TruthoundContext

data = {"id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]}
context = TruthoundContext.discover()
suite = ValidationSuite.from_legacy(context=context, validators=["null", "unique"], data=data)
asset = build_validation_asset(data)
plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)
run = ValidationRuntime().execute(asset=asset, plan=plan)
```

Use the kernel directly when you need explicit suite control, backend-aware planning, or advanced extension work.

### Build docs and reports

```python
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

html = generate_validation_report(run, title="Customer Quality Overview")
json_payload = get_reporter("json").render(run)
run.write("quality-report.json")
run.write("quality-report.html")
```

### Manage plugins

```bash
truthound plugins list
truthound plugins list --json
```

The plugin lifecycle is unified behind one `PluginManager`, with `EnterprisePluginManager` acting as an async capability facade rather than a separate runtime.

## Common Troubleshooting

### I do not see `.truthound/`

The workspace is created when the active context allows local persistence. Check the returned `run.metadata` and the resolved context path before assuming it failed.

### I want stricter or more explicit validation

Start with the defaults first, then move into:

- [First Validation](first-validation.md)
- [Validators Guide](../guides/validators/index.md)
- [Datasources Guide](../guides/datasources/index.md)
- [Checkpoints Guide](../guides/checkpoints.md)

### I am upgrading older 2.x code

Use the migration guide rather than trying to map old result types by hand:

```bash
truthound doctor . --migrate-2to3
truthound doctor . --workspace
```

The doctor command finds common 2.x assumptions such as `truthound.compare`, `Report`, `report.validation_run`, and `CheckpointResult.validation_result`, and it can also verify that your local `.truthound/` workspace is structurally healthy.

## Where To Go Next

- [First Validation](first-validation.md)
- [Tutorials](../tutorials/index.md)
- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Reference](../reference/index.md)
- [Migration to 3.0](../guides/migration-3.0.md)
