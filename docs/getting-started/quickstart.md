# Quick Start

## Validate Data With Zero Configuration

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

## Inspect the Canonical Result

```python
print(run.source)
print(run.execution_mode)
print([check.name for check in run.checks])
print([issue.issue_type for issue in run.issues])
print(run.metadata.get("context_run_artifact"))
```

`ValidationRunResult` is the canonical runtime output for checkpoints, reporters, validation docs, and plugins.

## Use the Project Context

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

## Use the Kernel Directly

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

## Build Docs and Reports

```python
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

html = generate_validation_report(run, title="Customer Quality Overview")
json_payload = get_reporter("json").render(run)
run.write("quality-report.json")
run.write("quality-report.html")
```

## Manage Plugins

```bash
truthound plugins list
truthound plugins list --json
```

The plugin lifecycle is unified behind one `PluginManager`, with `EnterprisePluginManager` acting as an async capability facade rather than a separate runtime.

## Migrate Existing 2.x Code

```bash
truthound doctor . --migrate-2to3
```

The doctor command finds common 2.x assumptions such as `truthound.compare`, `Report`, `report.validation_run`, and `CheckpointResult.validation_result`.

## Where To Go Next

- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Plugin Platform](../concepts/plugins.md)
- [Migration to 3.0](../guides/migration-3.0.md)
