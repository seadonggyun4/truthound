# Migration to Truthound 2.0

## What Stays the Same

- `th.check()`, `th.scan()`, `th.mask()`, `th.profile()`, and `th.learn()` remain available
- CLI data validation still begins with `truthound check`
- legacy `Report` output remains usable

## What Changed

- validation execution now routes through `truthound.core`
- `report.validation_run` exposes the structured `ValidationRunResult`
- built-in reporters and validation datadocs now consume `ValidationRunResult` directly
- top-level exports are intentionally smaller
- `PluginManager` is the single lifecycle runtime
- `EnterprisePluginManager` is an async facade, not a second manager implementation
- `use_engine` and `--use-engine` are removed
- the CLI plugin group is `plugins`

## Recommended Upgrade Path

### 1. Keep the Existing Facade

```python
import truthound as th

report = th.check(data, validators=["null", "unique"])
```

This is still the recommended default.

### 2. Start Reading Structured Results

```python
run = report.validation_run
print(run.execution_mode)
print(run.to_dict())
```

### 3. Move Reporting and Validation Docs to the Result Model

```python
from truthound.datadocs import generate_validation_report
from truthound.reporters import get_reporter

run = report.validation_run
json_report = get_reporter("json").render(run)
validation_docs = generate_validation_report(run)
```

Legacy helpers such as `generate_html_report(report)` still work, but passing `Report` or persisted validation DTOs directly to reporters is now a compatibility path rather than the preferred contract.

### 4. Move Extension Code to the Kernel

```python
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset

suite = ValidationSuite.from_legacy(validators=["null", "unique"])
asset = build_validation_asset(data)
plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)
run = ValidationRuntime().execute(asset=asset, plan=plan)
```

### 5. Update Plugin Integrations

Prefer manager ports such as:

- `register_check_factory()`
- `register_data_asset_provider()`
- `register_reporter()`
- `register_hook()`

Reporter plugins should target `ValidationRunResult` as the canonical input contract.

## Breaking Changes

- `use_engine` argument removed from `th.check()`
- `--use-engine` removed from `truthound check`
- root exports narrowed around the main facade and kernel result types
- `truthound.stores.results.ValidationResult` is no longer the canonical reporter input

## Recommended Reading

- [Architecture](../concepts/architecture.md)
- [Plugin Platform](../concepts/plugins.md)
- [Truthound 2.0 Release Notes](../releases/truthound-2.0.md)
