# Great Expectations Comparison

## Scope

Truthound does not claim equivalence across every possible validator family. The current release-grade comparison covers deterministic core semantics with a clear one-to-one mapping.

Supported mappings:

- `not_null` -> Great Expectations non-null expectation
- `unique` -> Great Expectations uniqueness expectation
- `between` -> Great Expectations range expectation
- `schema` -> Great Expectations column existence and type expectations

If a workload cannot be mapped honestly, it should be marked non-comparable and kept out of the release-grade comparison set.

## What Counts as Passing

A comparable workload passes only when:

- Truthound matches the manifest's expected verdict
- Great Expectations matches the manifest's expected verdict
- Truthound and Great Expectations observe the same issue count

Performance assertions are evaluated only after correctness parity is satisfied.

For the fixed-runner release verification, parity is necessary but not sufficient. The run must also come from the documented self-hosted runner policy with hardware and storage metadata.

## Why Issue Count Matters

Warm timing alone is not enough. The comparison tracks the number of detected failing elements so that a faster run cannot hide:

- dropped violations
- mismatched null semantics
- duplicate-count drift
- range-bound interpretation differences

## SQL Policy

SQLite is the canonical SQL comparison backend because it is deterministic, lightweight, and easy to reproduce in CI.

DuckDB is tracked separately as a shadow benchmark. It is useful for optimization work, but it does not currently decide the release-grade comparison verdict.

## Interpreting Results

- `correctness` assertions tell you whether a framework matched the workload manifest
- `issue-parity` assertions tell you whether Truthound and Great Expectations agreed on the observed issue count
- `local-speedup` and `sql-speedup` assertions tell you whether Truthound cleared the release thresholds
- `local-memory-ratio` tells you whether Truthound stayed below the memory ceiling
- `release-ga:*` assertions tell you whether the result came from the full fixed-runner release procedure

## Current Verified Outcome

The latest verified artifact set shows Truthound ahead of Great Expectations on every comparable workload in the current release-grade catalog while preserving correctness parity and a lower local memory footprint.

Read the exact measured numbers in [Latest Verified Benchmark Summary](../releases/latest-benchmark-summary.md).

## Related Reading

- [Performance and Benchmarks](performance.md)
- [Benchmark Methodology](benchmark-methodology.md)
- [Workload Catalog](benchmark-workloads.md)
