# Quick Start

## Validate With the Compatibility Facade

```python
import truthound as th

report = th.check(
    {"id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
    validators=["null", "unique"],
)

print(report)
print(report.validation_run.execution_mode)
```

`th.check()` is still the easiest way to validate data. In 2.0, it internally builds a `ValidationSuite`, compiles a scan plan, executes it through the runtime, and adapts the structured result back into the legacy `Report` shape.

## Inspect the Structured Result

```python
run = report.validation_run

print(run.source)
print(run.execution_mode)
print([check.name for check in run.checks])
print([issue.issue_type for issue in run.issues])
```

`ValidationRunResult` is now the canonical runtime output and is designed to be stable for reporters, documentation generators, and future backends.

## Use the Kernel Directly

```python
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset

suite = ValidationSuite.from_legacy(validators=["null", "unique"])
asset = build_validation_asset({"id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]})
plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)
run = ValidationRuntime().execute(asset=asset, plan=plan)
```

Use the kernel directly when you need:

- backend-aware planning
- direct access to execution metadata
- controlled extension points for plugins or adapters

## Manage Plugins

```bash
truthound plugins list
truthound plugins list --json
```

The plugin lifecycle is now unified behind one `PluginManager`, with `EnterprisePluginManager` acting as an async capability facade rather than a separate runtime.

## Where To Go Next

- [Architecture](../concepts/architecture.md)
- [Plugin Platform](../concepts/plugins.md)
- [Migration to 2.0](../guides/migration-2.0.md)
