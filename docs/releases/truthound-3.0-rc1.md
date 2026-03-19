# Truthound 3.0 RC1 Release Notes

## Highlights

Truthound 3.0 RC1 is the first release candidate for the hard cutoff from the 2.x compatibility era.

Key changes:

- `th.check()` now returns `ValidationRunResult`
- `TruthoundContext` and `.truthound/` are live by default
- deterministic auto-suite selection replaces the old "run everything" fallback
- checkpoints standardize on `validation_run` and `validation_view`
- reporters and validation docs consume the same canonical result model
- `truthound doctor --migrate-2to3` provides migration diagnostics

## Public Contract

The stable root surface is now:

- functions: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- types: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`

Advanced systems such as drift, ML, lineage, and realtime are namespace imports rather than root exports.

## Zero-Config Workspace

Truthound now auto-creates:

- `.truthound/config.yaml`
- `.truthound/catalog/`
- `.truthound/baselines/`
- `.truthound/runs/`
- `.truthound/docs/`
- `.truthound/plugins/`

## Upgrade Guidance

Before upgrading an existing project, run:

```bash
truthound doctor . --migrate-2to3
```

Then review the detailed [Migration Guide](../guides/migration-3.0.md).

## Status

This is a release candidate. The remaining gate before GA is production benchmark and parity validation against the 3.0 cutoff targets.
