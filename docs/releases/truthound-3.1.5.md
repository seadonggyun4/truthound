# Truthound 3.1.5 Release Notes

## Highlights

Truthound 3.1.5 repairs the shared SQL DataSource contract used by Truthound
itself and by consumer products such as Truthound Depot.

The release separates Core defects from consumer adapter defects. A successful
Core release does not by itself certify a Depot connector; Depot must install
this published artifact and rerun its own source, worker, persistence, and UI
matrix.

## SQL DataSource fixes

- `row_count`, scalar queries, and result conversion now accept tuple rows,
  mapping rows such as PyMySQL `DictCursor`, and SQLAlchemy-style `_mapping`
  rows.
- DB-API provider query execution shares one column-name normalization and
  cursor cleanup contract.
- Polars fallback materialization uses provider-compatible bounded queries and
  `fetchmany()` batches instead of an unrestricted `fetchall()`.
- the fallback reads `materialization_row_limit + 1` rows so concurrent source
  growth cannot be silently truncated and reported as complete data.
- SQLite query mode strips a trailing semicolon before embedding the query and
  sampled query-mode sources preserve the parent provider contract.
- BigQuery, Snowflake, Redshift, Databricks, Oracle, and SQL Server now route
  fallback conversion through the shared bounded implementation. Oracle uses
  `ROWNUM`; SQL Server uses `TOP`; other providers use `LIMIT`.

## Installation extras

Truthound 3.1.5 publishes explicit SQL provider extras:

```bash
pip install truthound[postgresql]
pip install truthound[mysql]
pip install truthound[duckdb]
pip install truthound[snowflake]
pip install truthound[bigquery]
pip install truthound[redshift]
pip install truthound[databricks]
pip install truthound[oracle]
pip install truthound[sqlserver]
pip install truthound[sql-connectors]
```

## Public API contract

SQL DataSources are passed with the `source` keyword:

```python
import truthound as th

validation = th.check(source=source)
profile = th.profile(source=source)
```

For a source larger than `materialization_row_limit`, use pushdown-capable
validation or an explicit bounded source such as `source.sample(10_000)`.
Truthound raises `DataSourceSizeError` rather than returning an incomplete
Polars fallback.

## Verification status

The release gate distinguishes implementation coverage from provider
certification.

| Evidence | Status in the source change |
|----------|-----------------------------|
| tuple, mapping, and `_mapping` row contract | Contract tested |
| bounded batching and no-silent-truncation behavior | Contract tested |
| SQLite public `check` and `profile` path | 3.1.5 wheel integration tested |
| DuckDB public source path | 3.1.5 wheel integration tested |
| PostgreSQL and MySQL local service path | 3.1.5 wheel integration tested |
| Snowflake, BigQuery, Redshift, Databricks, Oracle, SQL Server real accounts | Unverified until credential-backed certification runs |

An unavailable credential, skipped test, or import-only check is not a pass.
See the release evidence and consumer repository QA for the exact artifact and
provider matrix used after publication.

## Known limitation

The 3.1.5 artifact still leaves Snowflake, BigQuery, Redshift, Databricks,
Oracle, and SQL Server as abstract classes because their provider-native schema
implementations do not satisfy the base class's query-only abstract hook.
Upgrade to 3.1.6 before using those providers. No credential-backed support is
implied by this constructor fix.

## Consumer upgrade gate

Consumer repositories must not keep a permanent local workaround for a Core
defect. They must:

1. install the published 3.1.5 wheel from PyPI,
2. verify the installed version and artifact hash,
3. remove the superseded workaround,
4. rerun public API, worker, persistence, re-entry, and provider QA,
5. record consumer-specific failures separately from Core failures.
