# Truthound 3.1.6 Release Notes

## Highlights

Truthound 3.1.6 repairs the SQL provider construction contract exposed by the
3.1.5 consumer integration gate. Snowflake, BigQuery, Redshift, Databricks,
Oracle, and SQL Server implemented provider-native schema discovery but could
not be instantiated because `BaseSQLDataSource` required an unrelated abstract
schema-query hook.

## Schema discovery strategies

SQL providers now satisfy one of two explicit strategies:

- query strategy: implement `_get_table_schema_query()` and use the common
  DB-API schema runner;
- native strategy: implement `_fetch_schema()` for a provider metadata API or
  dialect-specific operation.

Construction validates that at least one strategy exists. The common query
runner normalizes tuple, mapping, and `_mapping` schema rows and closes the
cursor on success and failure.

## Provider class contract

The following advertised classes are verified as concrete in source contract
tests and in the built-artifact release smoke:

| Provider | Schema strategy |
|----------|-----------------|
| PostgreSQL | Shared schema query |
| MySQL | Shared schema query |
| SQLite | Provider query/pragma path |
| DuckDB | Shared schema query |
| Snowflake | Native information schema |
| BigQuery | Native client metadata |
| Redshift | Native information schema |
| Databricks | Native `DESCRIBE` |
| Oracle | Native catalog query |
| SQL Server | Native information schema |

The release workflow installs the wheel with `truthound[sql-connectors]`,
imports every driver and public provider class, and fails if any class is
abstract.

## Evidence boundary

Concrete class and constructor/config tests are non-network package evidence.
They do not certify a provider account. Snowflake, BigQuery, Redshift,
Databricks, Oracle, and SQL Server remain unverified at the real-provider level
until credential-backed schema, row count, bounded read, validation, profile,
serialization or re-entry, and cleanup complete.

## Consumer upgrade gate

Consumers such as Truthound Depot must install the published 3.1.6 artifact,
verify the runtime version and hash, and rerun their provider lifecycle. A local
checkout or unpublished wheel is not consumer certification evidence.
