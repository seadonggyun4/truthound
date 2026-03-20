# Performance and Benchmarks

## Benchmark Position

Truthound treats benchmark claims as release-grade evidence, not ambient marketing copy. Public comparisons are grounded in the fixed-runner artifact set published for the current release line.

The latest verified result shows:

- Truthound finished ahead of Great Expectations on all comparable deterministic core workloads
- correctness parity held across all comparable workloads
- local memory stayed well below the configured ceiling, at `35.88%` to `48.16%` of Great Expectations peak RSS

See the current numbers in [Latest Verified Benchmark Summary](../releases/latest-benchmark-summary.md).

## What Gets Measured

Truthound benchmarks the real public path:

- local exact workloads run through `th.check(...)`
- SQL exact workloads run through `th.check(source=..., pushdown=True)`
- zero-config context discovery and `.truthound/` reuse are part of the measured path
- correctness is validated before any speedup claim is accepted

## Benchmark Classes

- `local_exact`: deterministic local workloads on repo-tracked fixtures
- `sql_exact`: deterministic SQL pushdown workloads with SQLite as the canonical comparison backend
- `sql_shadow`: advisory-only SQL workloads, currently DuckDB, used for optimization visibility rather than release decisions

## Verified Thresholds

The current release-grade benchmark verification requires all of the following:

- local exact workloads show `Truthound >= 1.5x` Great Expectations median throughput
- SQL exact workloads show `Truthound >= 1.0x` Great Expectations median throughput
- local exact workloads stay at or below `60%` of Great Expectations peak RSS
- Truthound and Great Expectations both match the workload manifest's expected verdict and issue count
- no comparable workload produces false-positive or false-negative regressions

## Running the Verification Suites

Install the optional benchmark dependencies:

```bash
uv sync --extra dev --extra benchmarks
```

Run the fast local smoke:

```bash
truthound benchmark parity --suite pr-fast --frameworks truthound --backend local --strict
```

Run the local comparison suite:

```bash
truthound benchmark parity --suite nightly-core --frameworks both --backend local --strict
```

Run the SQLite comparison suite:

```bash
truthound benchmark parity --suite nightly-sql --frameworks both --backend sqlite --strict
```

Run the fixed-runner release-grade suite:

```bash
truthound benchmark parity --suite release-ga --frameworks both --strict
```

## Artifact Layout

Benchmark artifacts are written under `.truthound/benchmarks/`:

- `results/`: ad hoc and nightly suite outputs
- `baselines/`: saved parity baselines used for regression checks
- `artifacts/`: per-workload child-process workspaces and temporary databases
- `release/`: release-grade JSON, Markdown, and HTML summaries

The canonical published artifact set includes:

- `release-ga.json`
- `release-ga.md`
- `release-ga.html`
- `env-manifest.json`
- `latest-benchmark-summary.md`

## Reading the Claim Carefully

Truthound's published comparison is intentionally narrow and precise:

- it covers comparable deterministic core checks plus SQLite pushdown workloads
- it does not claim blanket superiority over every Great Expectations feature area
- it is meant to describe the current release-grade artifact set, not all possible future workloads

## Related Reading

- [Benchmark Methodology](benchmark-methodology.md)
- [Great Expectations Comparison](gx-parity.md)
- [Workload Catalog](benchmark-workloads.md)
- [Docs Deployment Verification](docs-deployment-verification.md)
- [Architecture](../concepts/architecture.md)
