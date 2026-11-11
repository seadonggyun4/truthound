# Migration to Truthound 3.0

## Summary

Truthound 3.0 is a breaking release that removes 2.x compatibility shims from the core validation path. The upgrade is conceptually simple:

- `th.check()` now returns `ValidationRunResult`
- `TruthoundContext` and `.truthound/` are now default behavior
- `compare` moves to `truthound.drift.compare`
- `CheckpointResult.validation_result` is removed
- validator subclass authoring is no longer the supported core extension model

## Fast Diagnostic

Run the migration doctor first:

```bash
truthound doctor . --migrate-2to3
```

The command finds common 2.x assumptions and prints exact replacements.

## API Mapping

| 2.x Pattern | 3.0 Replacement |
| --- | --- |
| `report = th.check(data)` where `report` is `Report` | `run = th.check(data)` where `run` is `ValidationRunResult` |
| `report.validation_run` | use the `ValidationRunResult` returned by `th.check()` directly |
| `from truthound import compare` | `from truthound.drift import compare` |
| `CheckpointResult.validation_result` | `CheckpointResult.validation_run` or `CheckpointResult.validation_view` |
| validator subclass authoring | declarative `CheckSpecFactory` registration |

## Before and After

### Validation

```python
# 2.x
report = th.check(data, validators=["null", "unique"])
print(report.validation_run.execution_mode)
```

```python
# 3.0
run = th.check(data, validators=["null", "unique"])
print(run.planned_execution_mode)
print(run.execution_mode)
```

### Drift

```python
# 2.x
from truthound import compare
```

```python
# 3.0
from truthound.drift import compare
```

### Checkpoints

```python
# 2.x
stats = checkpoint_result.validation_result.statistics
```

```python
# 3.0
stats = checkpoint_result.validation_run
view = checkpoint_result.validation_view
print(view.statistics.total_issues)
```

## Zero-Config Changes

Truthound 3.0 creates `.truthound/` automatically. If you previously managed baselines or run artifacts manually, expect these directories:

- `.truthound/config.yaml`
- `.truthound/catalog/`
- `.truthound/baselines/`
- `.truthound/runs/`
- `.truthound/docs/`
- `.truthound/plugins/`

`truthound.yaml` remains valid, but it is now an override layer instead of a bootstrap requirement.

## Reporter and Docs Migration

Reporter plugins and direct rendering code should use:

- canonical input: `ValidationRunResult`
- render contract: `render(run_result, *, context)`
- shared projection: `RunPresentation`
- `execution_mode` for actual runtime behavior and `planned_execution_mode` for planner intent

Do not rely on `Report` or `truthound.stores.results.ValidationResult` as canonical runtime inputs.

## Extension Migration

If you author custom validators through inheritance, plan a rewrite toward declarative factories:

- register `CheckSpecFactory`
- emit `CheckSpec`
- let the planner/runtime handle execution and aggregation

This is the supported path for long-term compatibility in 3.0.

## Acceptance Checklist

- [ ] `truthound doctor . --migrate-2to3` reports no blocking patterns
- [ ] direct `Report` assumptions are removed
- [ ] root-level `compare` imports are replaced
- [ ] checkpoint integrations use `validation_run` or `validation_view`
- [ ] custom reporters accept `ValidationRunResult`
- [ ] docs/examples no longer reference `report.validation_run`

## Related Reading

- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Release Notes: Truthound 3.0](../releases/truthound-3.0.md)

## Result persistence contracts

New persistence DTOs use version 2 of the embedded `_validation_run_contract`.
They retain each check's own issue evidence, including detail selected by the
result format, instead of assigning evidence by display name. Repeated validator
names and display aliases therefore retain their distinct outcomes after a store
round trip. Treat stored results as sensitive artifacts, just like canonical run
files; a summary view is not a redaction policy.

Unversioned DTOs remain readable when issue ownership is unambiguous. An old
contract with repeated check names and matching issue evidence is rejected with
`ValueError`; rerun validation to recover exact ownership. DTOs without an embedded
contract retain the legacy row-level projection. Unknown embedded contract
versions are rejected rather than guessed.

`ValidationRunResult.from_artifact_dict(payload, expected_run_id=...)` is the
strict persisted-run reader. It requires explicit identity, dimensions, checks,
issues and execution issues, including explicit empty lists for an empty run.
AI persisted-run loading validates the requested run ID. Checkpoint restoration
uses the same reader for a non-null canonical `validation_run`; null still means
no run. Malformed or incomplete artifacts raise `ValueError` before analysis.
Existing unversioned canonical files with these fields remain supported.

`ValidationRunResult.from_dict` remains a permissive construction helper for
compatibility, including intentional empty results. Do not use it as a corruption
check. Structural validation is not an authenticity guarantee and cannot detect
all coherent alterations of evidence. Severity filtering narrows the available
view; it does not rerun checks or recover discarded lower-severity details.
