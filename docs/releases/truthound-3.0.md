# Truthound 3.0 Release Notes

## Highlights

Truthound 3.0 turns the familiar facade into a native zero-configuration validation platform with a smaller kernel, a clearer result model, and verified benchmark evidence.

Key changes:

- `th.check()` returns `ValidationRunResult`
- `TruthoundContext` and `.truthound/` are active by default
- deterministic auto-suite selection replaces the old "run everything" fallback
- checkpoints standardize on `validation_run` and `validation_view`
- reporters and validation docs consume the same canonical result model
- `truthound doctor --migrate-2to3` provides upgrade diagnostics
- the fixed-runner benchmark artifact set verifies Truthound ahead of Great Expectations on all comparable release-grade workloads in the published comparison set

## Public Surface

The stable root surface is:

- functions: `check`, `scan`, `mask`, `profile`, `learn`, `read`, `get_context`
- types: `TruthoundContext`, `ValidationSuite`, `CheckSpec`, `SchemaSpec`, `ValidationRunResult`, `CheckResult`

Advanced systems such as drift, ML, lineage, and realtime are namespace imports rather than root exports.

## Zero-Config Workspace

Truthound now auto-creates and reuses:

- `.truthound/config.yaml`
- `.truthound/catalog/`
- `.truthound/baselines/`
- `.truthound/runs/`
- `.truthound/docs/`
- `.truthound/plugins/`

## Benchmark Verification

The current release-grade artifact set shows:

- speedups over Great Expectations across all eight comparable workloads
- lower local memory use across the full comparable workload set
- correctness parity preserved for every comparable workload

See [Latest Verified Benchmark Summary](latest-benchmark-summary.md) for the exact measured numbers and artifact links.

## Upgrade Guidance

Before upgrading an existing project, run:

```bash
truthound doctor . --migrate-2to3
```

Then review the detailed [Migration Guide](../guides/migration-3.0.md).
