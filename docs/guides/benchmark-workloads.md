# Workload Catalog

## Source of Truth

Truthound 3.0 parity workloads are repo-tracked under `benchmarks/workloads/`. Each manifest pins:

- the fixture dataset
- the expected verdict
- the expected issue count
- the Truthound mapping
- the GX mapping
- the suite membership

## Tier-1 Local Exact Workloads

| Workload | Purpose | Expected Issues |
| --- | --- | ---: |
| `local-null` | Non-null parity on `email` | 1 |
| `local-unique` | Uniqueness parity on `customer_id` | 1 |
| `local-range` | Numeric bounds parity on `age` | 1 |
| `local-schema` | Schema presence parity on `status` | 1 |
| `local-mixed-core-suite` | Combined null, unique, and range checks with a passing schema contract | 3 |

## Tier-1 SQL Exact Workloads

SQLite is the canonical SQL backend for the 3.0 release gate.

| Workload | Purpose | Expected Issues |
| --- | --- | ---: |
| `sqlite-null` | Pushdown non-null parity on `email` | 1 |
| `sqlite-unique` | Pushdown uniqueness parity on `customer_id` | 1 |
| `sqlite-range` | Pushdown numeric bounds parity on `age` | 1 |

## SQL Shadow Workloads

DuckDB shadow workloads are tracked for advisory trend visibility only:

- `duckdb-null`
- `duckdb-unique`
- `duckdb-range`

These are Truthound-only in the initial 3.0 gate and do not block GA.

## Suite Membership

| Suite | Included Workloads |
| --- | --- |
| `pr-fast` | local null, local unique, local mixed core suite |
| `nightly-core` | all local exact workloads |
| `nightly-sql` | all SQLite exact workloads plus DuckDB shadow workloads |
| `release-ga` | all local exact workloads plus all SQLite exact workloads |

## Why Repo-Tracked Fixtures

Truthound keeps the release gate on repo-tracked fixtures because it makes the benchmark:

- reviewable in pull requests
- reproducible in CI
- versionable alongside code changes
- diffable when a workload meaning changes

## Related Reading

- [Benchmark Methodology](benchmark-methodology.md)
- [GX Parity Gate](gx-parity.md)
