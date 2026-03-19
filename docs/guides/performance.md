# Performance and Benchmarks

## Goal

Truthound 3.0 treats performance as a release gate, not a marketing claim. Official claims are valid only when the repo-tracked parity suites pass on a fixed self-hosted runner and the published artifacts are attached to the release process.

## What Gets Measured

Truthound benchmarks the public 3.0 path, not an internal shortcut:

- local exact workloads run through `th.check(...)`
- SQL exact workloads run through `th.check(source=..., pushdown=True)`
- zero-config context creation and baseline reuse are part of the measured path
- correctness parity is evaluated before any speedup claim is accepted

## Benchmark Classes

Truthound 3.0 maintains three workload classes:

- `local_exact`: deterministic local workloads on repo-tracked fixtures
- `sql_exact`: deterministic SQL pushdown workloads with SQLite as the canonical backend
- `sql_shadow`: advisory-only SQL workloads, currently DuckDB, used for trend visibility but not GA blocking

## Release Blockers

The 3.0 benchmark gate is considered passing only when all of the following are true:

- local exact workloads show `Truthound >= 1.5x` GX median throughput
- SQL exact workloads show `Truthound >= 1.0x` GX median throughput
- local exact workloads stay at or below `60%` of GX peak RSS
- Truthound and GX both match the workload manifest's expected verdict and issue count
- no workload produces false-positive or false-negative regressions on tier-1 exact checks

## Running the Parity Gate

Install the optional benchmark dependencies:

```bash
uv sync --extra dev --extra benchmarks
```

Run the fast local smoke:

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
```

Run the nightly local parity against GX:

```bash
truthound benchmark parity --suite nightly-core --frameworks both --backend local --strict
```

Run the SQL parity gate:

```bash
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
```

## Published Artifacts

Benchmark artifacts are written under `.truthound/benchmarks/`:

- `results/`: ad hoc and nightly suite outputs
- `baselines/`: saved parity baselines used for regression checks
- `artifacts/`: per-workload child-process workspaces and temporary databases
- `release/`: release-grade JSON, Markdown, and HTML summaries

The latest published release artifact set is tracked in [Latest Benchmark Summary](../releases/latest-benchmark-summary.md).

Official release claims must come from the fixed-runner artifact set, not from ad hoc local runs. The canonical release artifact set includes:

- `release-ga.json`
- `release-ga.md`
- `release-ga.html`
- `env-manifest.json`
- `latest-benchmark-summary.md`

## Detailed References

- [Benchmark Methodology](benchmark-methodology.md)
- [Workload Catalog](benchmark-workloads.md)
- [GX Parity Gate](gx-parity.md)
- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Migration to 3.0](migration-3.0.md)
