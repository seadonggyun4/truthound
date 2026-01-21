# Traditional Database Data Sources

This document covers traditional relational database data sources in Truthound: SQLite, PostgreSQL, and MySQL.

## Overview

SQL database sources provide connection pooling, query pushdown optimization, and both table and query modes for validation.

| Database | Driver | Installation | Built-in |
|----------|--------|--------------|----------|
| SQLite | `sqlite3` | Built-in | Yes |
| PostgreSQL | `psycopg2` | `pip install psycopg2-binary` | No |
| MySQL | `pymysql` | `pip install pymysql` | No |
| Oracle | `oracledb` | `pip install oracledb` | No |
| SQL Server | `pyodbc` | `pip install pyodbc` | No |

## Common Features

All SQL data sources share these capabilities:

- **Connection Pooling**: Thread-safe connection pool
- **Table Mode**: Validate existing tables
- **Query Mode**: Validate custom SQL query results
- **SQL Pushdown**: Run operations directly on the database
- **Schema Inference**: Automatic column type detection

### Capabilities

```python
from truthound.datasources import DataSourceCapability

# All SQL sources have these capabilities
source.capabilities
# {
#     DataSourceCapability.SQL_PUSHDOWN,
#     DataSourceCapability.SAMPLING,
#     DataSourceCapability.SCHEMA_INFERENCE,
#     DataSourceCapability.ROW_COUNT,
# }
```

## SQLite

SQLite is a file-based database included in Python's standard library.

### Basic Usage

```python
from truthound.datasources.sql import SQLiteDataSource

# From database file
source = SQLiteDataSource(table="users", database="data.db")

# In-memory database
source = SQLiteDataSource(table="users", database=":memory:")
```

### Query Mode

Validate custom SQL query results instead of a table:

```python
# Table mode (validate existing table)
source = SQLiteDataSource(table="users", database="data.db")

# Query mode (validate query results)
source = SQLiteDataSource(
    database="data.db",
    query="SELECT id, name, email FROM users WHERE active = 1",
)

# Access query properties
print(source.is_query_mode)    # True
print(source.query_sql)        # "SELECT id, name FROM users WHERE active = 1"
print(source.table_name)       # None
```

### From DataFrame

Create a SQLite source from a Pandas or Polars DataFrame:

```python
import pandas as pd
from truthound.datasources.sql import SQLiteDataSource

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
})

# Creates a temporary database file
source = SQLiteDataSource.from_dataframe(df, "test_table")

# Or specify database path
source = SQLiteDataSource.from_dataframe(df, "users", database="test.db")
```

### Configuration

```python
from truthound.datasources.sql import SQLiteDataSource, SQLiteDataSourceConfig
import sqlite3

config = SQLiteDataSourceConfig(
    database="data.db",
    timeout=5.0,                                                # Connection timeout
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    isolation_level=None,                                        # Autocommit mode
)

source = SQLiteDataSource(table="users", config=config)
```

### SQLite-Specific Methods

```python
source = SQLiteDataSource(table="users", database="data.db")

# Get table info (PRAGMA table_info)
info = source.get_table_info()
# [{'cid': 0, 'name': 'id', 'type': 'INTEGER', 'notnull': 1, ...}, ...]

# Get index info (PRAGMA index_list)
indexes = source.get_index_info()

# Get foreign keys (PRAGMA foreign_key_list)
fks = source.get_foreign_keys()

# Optimize database
source.vacuum()     # Reclaim storage
source.analyze()    # Update statistics
```

## PostgreSQL

PostgreSQL is a powerful open-source relational database.

### Installation

```bash
pip install psycopg2-binary
# or for production
pip install psycopg2
```

### Basic Usage

```python
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    schema_name="public",  # Default schema
)
```

### Connection String

```python
source = PostgreSQLDataSource.from_connection_string(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table="users",
    schema_name="public",
)
```

### Query Mode

```python
# Validate results of a complex query
source = PostgreSQLDataSource(
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
    query="""
        SELECT u.id, u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at > '2024-01-01'
    """,
)
```

### Configuration

```python
from truthound.datasources.sql import PostgreSQLDataSource, PostgreSQLDataSourceConfig

config = PostgreSQLDataSourceConfig(
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    schema_name="public",
    sslmode="prefer",            # SSL mode: disable, require, verify-ca, verify-full
    application_name="truthound",

    # Connection pool settings
    pool_size=5,                  # Connections in pool
    pool_timeout=30.0,            # Timeout for acquiring connection
    query_timeout=300.0,          # Query execution timeout
    fetch_size=10000,             # Rows to fetch at a time
)

source = PostgreSQLDataSource(table="users", config=config)
```

### PostgreSQL-Specific Methods

```python
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
)

# Get table size information
size = source.get_table_size()
# {'total_size': '1024 MB', 'table_size': '512 MB', 'indexes_size': '512 MB'}

# Get table statistics from pg_stat_user_tables
stats = source.get_table_statistics()
# {'live_rows': 1000000, 'dead_rows': 500, 'last_vacuum': ..., ...}

# Get index information
indexes = source.get_index_info()
# [{'index_name': 'users_pkey', 'column_name': 'id', 'is_unique': True, ...}, ...]

# Get constraints
constraints = source.get_constraints()
# [{'constraint_name': 'users_pkey', 'constraint_type': 'p', ...}, ...]

# Maintenance operations
source.analyze()              # Update statistics
source.vacuum()               # Reclaim storage
source.vacuum(full=True)      # Full vacuum (locks table)
```

## MySQL

MySQL is a popular open-source relational database.

### Installation

```bash
pip install pymysql
```

### Basic Usage

```python
from truthound.datasources.sql import MySQLDataSource

source = MySQLDataSource(
    table="users",
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
)
```

### Connection String

```python
source = MySQLDataSource.from_connection_string(
    connection_string="mysql://root:pass@localhost:3306/mydb",
    table="users",
)
```

### Configuration

```python
from truthound.datasources.sql import MySQLDataSource, MySQLDataSourceConfig

config = MySQLDataSourceConfig(
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
    charset="utf8mb4",           # Character set
    autocommit=True,             # Auto-commit mode
    ssl={                        # SSL configuration
        "ca": "/path/to/ca.pem",
        "cert": "/path/to/client-cert.pem",
        "key": "/path/to/client-key.pem",
    },

    # Connection pool settings
    pool_size=5,
    pool_timeout=30.0,
    query_timeout=300.0,
    fetch_size=10000,
)

source = MySQLDataSource(table="users", config=config)
```

### MySQL-Specific Methods

```python
source = MySQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="root",
    password="secret",
)

# Get table status (SHOW TABLE STATUS)
status = source.get_table_status()
# {'Rows': 1000000, 'Avg_row_length': 128, 'Data_length': ..., ...}

# Get table size
size = source.get_table_size()
# {'data_size_mb': 128.5, 'index_size_mb': 32.1, 'total_size_mb': 160.6, 'approx_rows': 1000000}

# Get index information (SHOW INDEX)
indexes = source.get_index_info()

# Get CREATE TABLE statement
create_sql = source.get_create_table()
print(create_sql)
# CREATE TABLE `users` (
#   `id` int NOT NULL AUTO_INCREMENT,
#   ...
# )

# Maintenance operations
source.analyze()    # Update statistics
source.optimize()   # Defragment table
```

## Connection Pooling

All SQL sources use a thread-safe connection pool:

```python
from truthound.datasources.sql import SQLConnectionPool

# Pool is managed automatically, but you can access it
source = PostgreSQLDataSource(table="users", ...)

# Pool properties
print(source._pool.size)       # Pool size
print(source._pool.available)  # Available connections

# Context manager for connections
with source._get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    cursor.close()
```

### Configuration

```python
from truthound.datasources.sql import SQLDataSourceConfig

config = SQLDataSourceConfig(
    pool_size=5,           # Maximum connections in pool
    pool_timeout=30.0,     # Timeout waiting for connection
    query_timeout=300.0,   # Query execution timeout
    fetch_size=10000,      # Batch size for fetching
    use_server_side_cursor=False,  # Server-side cursors for large results
    schema_name=None,      # Database schema
)
```

## Query Execution

Execute custom SQL queries:

```python
source = PostgreSQLDataSource(table="users", ...)

# Execute query returning rows
results = source.execute_query("SELECT * FROM users WHERE age > %s", (30,))
# [{'id': 1, 'name': 'Alice', 'age': 35}, ...]

# Execute query returning single value
count = source.execute_scalar("SELECT COUNT(*) FROM users WHERE active = %s", (True,))
# 1000

# Built-in query builders
count_query = source.build_count_query("age > 30")
# "SELECT COUNT(*) FROM public.users WHERE age > 30"

distinct_query = source.build_distinct_count_query("email")
# "SELECT COUNT(DISTINCT email) FROM public.users"

null_query = source.build_null_count_query("phone")
# "SELECT COUNT(*) FROM public.users WHERE phone IS NULL"

stats_query = source.build_stats_query("age")
# "SELECT COUNT(age) as count, AVG(age) as mean, ..."
```

## Validation Example

Using SQL sources with the validation API:

```python
import truthound as th
from truthound.datasources.sql import PostgreSQLDataSource

# Create source
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
)

# Run validation - SQL pushdown executes queries on the database
report = th.check(
    source=source,
    validators=["null", "unique", "duplicate"],
    columns=["id", "email", "phone"],
)

# With rules
report = th.check(
    source=source,
    rules={
        "id": ["not_null", "unique"],
        "email": ["not_null", {"type": "regex", "pattern": r".*@.*"}],
        "status": [{"type": "allowed_values", "values": ["active", "inactive"]}],
    },
)

print(f"Found {len(report.issues)} issues")
```

## Sampling

SQL sources support server-side sampling:

```python
source = PostgreSQLDataSource(table="users", ...)

# Create sampled source (uses LIMIT)
sampled = source.sample(n=10000)

# Sampled source wraps queries with LIMIT
print(sampled.full_table_name)
# "(SELECT * FROM public.users LIMIT 10000) AS sampled"
```

## Converting to Polars

Fetch all data as a Polars LazyFrame:

```python
source = PostgreSQLDataSource(table="users", ...)

# Convert to LazyFrame (fetches all data)
lf = source.to_polars_lazyframe()

# Apply Polars operations
result = (
    lf
    .filter(pl.col("age") > 25)
    .group_by("department")
    .agg(pl.col("salary").mean())
    .collect()
)
```

> **Warning**: `to_polars_lazyframe()` fetches all data into memory. Use sampling for large tables.

## Factory Functions

Convenience functions for creating SQL sources:

```python
from truthound.datasources import get_sql_datasource

# SQLite
source = get_sql_datasource("data.db", table="users")

# PostgreSQL
source = get_sql_datasource(
    "postgresql://user:pass@localhost/mydb",
    table="users",
)

# MySQL
source = get_sql_datasource(
    "mysql://root:pass@localhost/mydb",
    table="users",
)
```

## Error Handling

```python
from truthound.datasources.sql import PostgreSQLDataSource
from truthound.datasources.base import (
    DataSourceError,
    DataSourceConnectionError,
)

try:
    source = PostgreSQLDataSource(
        table="users",
        host="nonexistent.host",
        database="mydb",
        user="postgres",
        password="wrong",
    )
    source.validate_connection()
except DataSourceConnectionError as e:
    print(f"Connection failed: {e}")

try:
    source = PostgreSQLDataSource(
        table="nonexistent_table",
        host="localhost",
        database="mydb",
        user="postgres",
        password="secret",
    )
    schema = source.schema  # Triggers schema fetch
except DataSourceError as e:
    print(f"Error: {e}")
```

## Checking Availability

Check if database drivers are installed:

```python
from truthound.datasources.sql import get_available_sources, check_source_available

# Get all SQL sources and their availability
sources = get_available_sources()
for name, cls in sources.items():
    status = "available" if cls is not None else "not installed"
    print(f"{name}: {status}")
# sqlite: available
# postgresql: available
# mysql: not installed
# ...

# Check specific source
if check_source_available("postgresql"):
    from truthound.datasources.sql import PostgreSQLDataSource
    # Use PostgreSQL...
else:
    print("Install psycopg2-binary: pip install psycopg2-binary")
```

## Best Practices

1. **Use connection pooling** - Configure appropriate pool size for your workload
2. **Use query mode for complex joins** - Validate query results directly
3. **Sample large tables** - Use `sample()` before validation
4. **Set query timeouts** - Prevent long-running queries from blocking
5. **Use SSL in production** - Configure `sslmode` for PostgreSQL, `ssl` for MySQL
6. **Check driver availability** - Use `check_source_available()` for graceful fallback
