# Truthound 2.0 Release Notes

## Highlights

- Introduced `truthound.core` with explicit `contracts`, `suite`, `planning`, `runtime`, and `results` layers
- Refactored `th.check()` into a compatibility facade over the new kernel
- Added `ValidationRunResult`, `CheckResult`, and `ExecutionIssue`
- Unified plugin lifecycle management behind `PluginManager`
- Recast `EnterprisePluginManager` as an async capability facade
- Removed the dead `use_engine` public surface
- Added architecture, parity, backend conformance, and plugin-platform tests
- Added docs link checking, `mkdocs` configuration cleanup, ADRs, and docs CI hooks

## Breaking Changes

- `use_engine` is no longer accepted by `th.check()`
- `truthound check --use-engine` is no longer supported
- top-level exports are intentionally narrower

## Compatibility Notes

- the legacy `Report` type remains available
- `report.validation_run` exposes the structured kernel result
- advanced systems remain importable by namespace

## Verification Snapshot

The redesign is covered by targeted validation of:

- public API behavior
- planner/runtime parity
- SQL pushdown routing
- plugin runtime unification
- CLI `plugins` snapshots

See the [Migration Guide](../guides/migration-2.0.md) for upgrade details.
