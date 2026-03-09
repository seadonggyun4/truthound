# CLI Data Source Guide

All Truthound CLI commands support reading data from databases and external sources in addition to local files. This guide covers the shared data source options available across all core commands.

## Overview

Truthound CLI commands accept data from three input modes:

1. **File mode** (default): Pass a file path as a positional argument
2. **Connection string mode**: Use `--connection` and `--table` (or `--query`) to connect to a database
3. **Source config mode**: Use `--source-config` to load connection details from a JSON or YAML file

These modes are mutually exclusive. If a file argument is provided alongside connection options, the file takes precedence.

## Data Source Options

The following options are available on all core commands (`check`, `scan`, `mask`, `profile`, `learn`, `compare`, `read`):

| Option | Short | Description |
|--------|-------|-------------|
| `--connection` | `--conn` | Database connection string (see formats below) |
| `--table` | | Database table name to read |
| `--query` | | SQL query to execute (alternative to `--table`) |
| `--source-config` | `--sc` | Path to a data source config file (JSON or YAML) |
| `--source-name` | | Custom label for the data source (used in reports) |

## Connection String Formats

### PostgreSQL

```
postgresql://user:password@host:5432/dbname
```

Install the PostgreSQL backend:

```bash
pip install truthound[postgresql]
```

### MySQL

```
mysql://user:password@host:3306/dbname
```

Install the MySQL backend:

```bash
pip install truthound[mysql]
```

### SQLite

```
sqlite:///path/to/database.db
sqlite:///./relative/path.db
```

SQLite is included by default; no extra install is needed.

### DuckDB

```
duckdb:///path/to/database.duckdb
duckdb:///:memory:
```

Install the DuckDB backend:

```bash
pip install truthound[duckdb]
```

### Microsoft SQL Server

```
mssql://user:password@host:1433/dbname
```

Install the SQL Server backend:

```bash
pip install truthound[mssql]
```

## Source Config File Format

For repeatable or complex connection setups, use a source config file with `--source-config`.

### JSON Example

```json
{
  "type": "postgresql",
  "connection": "postgresql://user:password@host:5432/dbname",
  "table": "users",
  "source_name": "production-users"
}
```

### YAML Example

```yaml
type: postgresql
connection: postgresql://user:password@host:5432/dbname
table: users
source_name: production-users
```

### Using a SQL Query

```yaml
type: postgresql
connection: postgresql://user:password@host:5432/dbname
query: "SELECT id, name, email FROM users WHERE active = true"
source_name: active-users
```

## Dual-Source Config (for `compare`)

The `compare` command accepts two data sources. You can provide a source config file that defines both `baseline` and `current`:

```yaml
baseline:
  type: postgresql
  connection: postgresql://user:pass@host/db
  table: users_baseline

current:
  type: postgresql
  connection: postgresql://user:pass@host/db
  table: users_current
```

Usage:

```bash
truthound compare --source-config compare_sources.yaml --method psi
```

Alternatively, you can specify individual files or connections for each source on the command line.

## Per-Backend Install Hints

Truthound uses optional dependency groups for database backends. Install only what you need:

| Backend | Install Command |
|---------|----------------|
| PostgreSQL | `pip install truthound[postgresql]` |
| MySQL | `pip install truthound[mysql]` |
| DuckDB | `pip install truthound[duckdb]` |
| SQL Server | `pip install truthound[mssql]` |
| BigQuery | `pip install truthound[bigquery]` |
| Snowflake | `pip install truthound[snowflake]` |
| All databases | `pip install truthound[databases]` |

SQLite support is included in the base install.

## Security Considerations

**Do not put passwords directly in CLI history.** Connection strings with embedded credentials are visible in shell history and process listings.

Recommended practices:

1. **Use environment variables:**

    ```bash
    export DB_CONN="postgresql://user:password@host/db"
    truthound check --connection "$DB_CONN" --table users
    ```

2. **Use source config files** with restricted file permissions:

    ```bash
    chmod 600 db_config.yaml
    truthound check --source-config db_config.yaml
    ```

3. **Use `.pgpass` or equivalent** credential files supported by your database client.

4. **Avoid inline passwords** in CI/CD pipelines. Use secrets management (GitHub Secrets, Vault, etc.) and inject via environment variables.

## Examples for Each Command

### check

```bash
# Validate a PostgreSQL table
truthound check --connection "postgresql://user:pass@host/db" --table orders

# Validate with source config
truthound check --source-config prod_db.yaml --strict
```

### scan

```bash
# Scan a database table for PII
truthound scan --connection "postgresql://user:pass@host/db" --table customers
```

### mask

```bash
# Mask PII in a database table and write to a file
truthound mask --connection "sqlite:///data.db" --table users -o masked_users.csv
```

### profile

```bash
# Profile a database table
truthound profile --connection "postgresql://user:pass@host/db" --table transactions
```

### learn

```bash
# Learn schema from a database table
truthound learn --connection "postgresql://user:pass@host/db" --table products -o schema.yaml
```

### compare

```bash
# Compare two database tables
truthound compare --source-config compare_sources.yaml --method psi --strict
```

### read

```bash
# Preview a database table
truthound read --connection "postgresql://user:pass@host/db" --table users --head 20

# Run a SQL query and export as CSV
truthound read --connection "sqlite:///data.db" --query "SELECT * FROM orders WHERE total > 100" --format csv -o high_orders.csv
```

## See Also

- [Data Sources Overview](index.md)
- [Database Connections](databases.md)
- [CLI Core Commands](../../cli/core/index.md)
