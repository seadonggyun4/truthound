# Zero-Config Context

## Design Goal

Truthound 3.0 treats zero-configuration as a core framework guarantee rather than a convenience wrapper. A user should be able to call `th.check(data)` and get:

- a resolved project context
- a persistent local workspace
- baseline-aware validation
- structured run artifacts
- validation docs artifacts

without creating a config file first.

## TruthoundContext

`TruthoundContext` is the project boundary for this behavior.

It is responsible for:

- discovering the project root
- creating and maintaining `.truthound/`
- loading optional overrides from `truthound.yaml`
- tracking asset fingerprints
- storing baseline schemas and metric history
- persisting `ValidationRunResult` metadata
- persisting validation docs
- exposing the shared plugin manager

The root API exposes it directly:

```python
import truthound as th

context = th.get_context()
print(context.workspace_dir)
```

## Workspace Layout

Truthound 3.0 fixes the local workspace layout:

- `.truthound/config.yaml`
- `.truthound/catalog/`
- `.truthound/baselines/`
- `.truthound/runs/`
- `.truthound/docs/`
- `.truthound/plugins/`

These paths are deliberately stable so tooling, docs, and migration scripts can reason about them without introspecting private implementation details.

## Execution Flow

When a caller runs `th.check(data)` without explicitly providing validators:

1. Truthound detects the asset type and backend.
2. `TruthoundContext` is discovered or created.
3. The asset is fingerprinted and recorded in the catalog.
4. A baseline schema is loaded or safely created if allowed.
5. `AutoSuiteBuilder` synthesizes a deterministic `ValidationSuite`.
6. `ScanPlanner` compiles the suite into an executable plan.
7. `ValidationRuntime` executes with exact-by-default semantics.
8. The run result and validation docs are persisted if context persistence is enabled.

## Auto-Suite Selection

`validators=None` does not mean "run every validator." The auto-suite path is rule-based and deterministic.

Current selection rules are:

- always include schema validation when a baseline or explicit schema exists
- always include nullability coverage
- add type checks where the profile is confident
- add range checks for numeric columns
- add uniqueness only for columns that look key-like from profile and naming heuristics

The builder intentionally avoids expensive or probabilistic checks unless they are explicitly eligible.

## Exact-By-Default Behavior

Truthound 3.0 keeps deterministic validation exact by default.

- core validation verdicts should come from exact checks
- sampling is allowed for profiling, drift, anomaly detection, or explicitly approximate workloads
- large-scale approximations should be treated as prefilters and exact-confirmed before a final failure verdict

This makes zero-configuration usable in production instead of making it a demo-only path.

## Optional Configuration

`truthound.yaml` remains supported, but it is an override layer rather than a required bootstrap file.

Typical overrides include:

- enabling or disabling run persistence
- enabling or disabling validation-doc persistence
- tuning default result format
- selecting a docs theme

If the file does not exist, Truthound uses safe defaults and writes the resolved context into `.truthound/config.yaml`.

## Baselines and History

The baseline/history layer exists to support more than one feature:

- schema validation
- auto-suite reuse
- metric history
- docs summaries
- benchmark baselines

This prevents schema/profile persistence from fragmenting into multiple incompatible storage paths.

## Operational Notes

- corrupt baseline or cache payloads should be quarantined rather than silently reused
- the workspace should be safe for repeated runs in the same project
- context persistence is a default, but callers can still pass an explicit `TruthoundContext` for tests or isolated jobs

## Related Reading

- [Architecture](architecture.md)
- [Checkpoints](../guides/checkpoints.md)
- [Migration to 3.0](../guides/migration-3.0.md)
