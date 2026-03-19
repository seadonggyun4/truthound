# GX Parity Gate

## Scope

Truthound 3.0 does not claim parity across every possible validator. The initial GA gate compares only deterministic core semantics with a clear one-to-one mapping.

Supported mappings:

- `not_null` -> GX non-null expectation
- `unique` -> GX uniqueness expectation
- `between` -> GX range expectation
- `schema` -> GX column existence and type expectations

If a workload cannot be mapped honestly, it should be marked non-comparable and kept out of the GA gate.

## What Counts as Passing

A comparable workload passes parity only when:

- Truthound matches the manifest's expected verdict
- GX matches the manifest's expected verdict
- Truthound and GX observe the same issue count

Performance assertions are evaluated only after correctness parity is satisfied.

For `release-ga`, parity is not enough on its own. The run must also come from the fixed self-hosted runner policy with documented hardware and storage metadata.

## Why Issue Count Matters

Warm timing alone is not enough. The gate compares the number of detected failing elements so that a faster run cannot hide:

- dropped violations
- mismatched null semantics
- duplicate-count drift
- range-bound interpretation differences

## SQL Policy

SQLite is the canonical SQL parity backend because it is deterministic, lightweight, and easy to reproduce in CI.

DuckDB is tracked separately as a shadow benchmark. It is useful for optimization work, but it does not currently decide the 3.0 GA verdict.

## Interpreting Results

- `correctness` assertions tell you whether a framework matched the workload manifest
- `issue-parity` assertions tell you whether Truthound and GX agreed on the observed issue count
- `local-speedup` and `sql-speedup` assertions tell you whether Truthound cleared the release threshold
- `local-memory-ratio` tells you whether Truthound stayed below the memory ceiling
- `release-ga:*` assertions tell you whether the result came from the full fixed-runner release procedure

## Related Reading

- [Performance and Benchmarks](performance.md)
- [Benchmark Methodology](benchmark-methodology.md)
- [Workload Catalog](benchmark-workloads.md)
