# Data Sources and Execution Engines

This document describes the multi-data source architecture introduced in Phase 5 of Truthound, enabling validation across different data backends.

## Overview

Truthound supports multiple data backends through a unified abstraction layer:

- **DataSource**: Abstraction for data storage and access
- **ExecutionEngine**: Abstraction for running validation operations

This architecture enables validators to work seamlessly across Polars, Pandas, SQL databases, and Spark.

## Quick Start with DataSource

All core API functions (`check`, `scan`, `mask`, `profile`) support the `source` parameter for DataSource integration:

```python
import truthound as th
from truthound.datasources import get_sql_datasource

# Create a DataSource
source = get_sql_datasource("mydb.db", table="users")

# Use with all core API functions
report = th.check(source=source)           # Data validation
pii_report = th.scan(source=source)        # PII detection
profile = th.profile(source=source)        # Statistical profiling
masked_df = th.mask(source=source)         # Data masking

# You can also combine with other options
report = th.check(
    source=source,
    validators=["null", "duplicate"],
    min_severity="medium",
    parallel=True,
)
```

## Supported Data Sources

Truthound supports a comprehensive range of data backends:

| Category | Sources |
|----------|---------|
| DataFrame | Polars, Pandas, PySpark |
| Core SQL | SQLite, PostgreSQL, MySQL |
| Cloud DW | BigQuery, Snowflake, Redshift, Databricks |
| Enterprise | Oracle, SQL Server |
| File | CSV, Parquet, JSON, NDJSON |

### Polars (Primary)

Polars is the primary and most feature-complete backend.

```python
import polars as pl
from truthound import check
from truthound.datasources import PolarsDataSource

df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
})

# Direct usage with check()
report = check(df, rules={...})

# Explicit DataSource usage
source = PolarsDataSource(df)
engine = source.get_execution_engine()
print(f"Row count: {engine.count_rows()}")
```

### Pandas

Native Pandas support with automatic conversion when needed.

```python
import pandas as pd
from truthound.datasources import PandasDataSource

df = pd.DataFrame({
    "id": [1, 2, 3],
    "value": [100, 200, 300],
})

source = PandasDataSource(df)
engine = source.get_execution_engine()
print(f"Null counts: {engine.count_nulls_all()}")
```

### SQL Databases

SQL sources with connection pooling and query pushdown optimization.

#### SQLite

```python
from truthound.datasources.sql import SQLiteDataSource

# From file
source = SQLiteDataSource(table="users", database="data.db")

# From DataFrame (creates temporary DB)
import polars as pl
df = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
source = SQLiteDataSource.from_dataframe(df, "test_table")

engine = source.get_execution_engine()
print(f"Distinct values: {engine.count_distinct('name')}")
```

#### PostgreSQL

```python
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
    schema="public",
)

# Or using connection string
source = PostgreSQLDataSource(
    table="users",
    connection_string="postgresql://user:pass@localhost/db",
)
```

#### MySQL

```python
from truthound.datasources.sql import MySQLDataSource

source = MySQLDataSource(
    table="customers",
    host="localhost",
    database="mydb",
    user="root",
    password="secret",
)
```

### Cloud Data Warehouses

#### Google BigQuery

```python
from truthound.datasources.sql import BigQueryDataSource

# Using service account credentials
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
    credentials_path="/path/to/service-account.json",
)

# Using default credentials (ADC)
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
)

# Cost-aware query execution
engine = source.get_execution_engine()
results = source.execute_with_cost_check(
    "SELECT * FROM large_table",
    max_bytes=1_000_000_000,  # 1GB limit
    max_cost_usd=1.0,         # $1 limit
)
```

#### Snowflake

```python
from truthound.datasources.sql import SnowflakeDataSource

# Password authentication
source = SnowflakeDataSource(
    table="users",
    account="myaccount",
    database="MY_DB",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    user="myuser",
    password="mypassword",
)

# SSO authentication
source = SnowflakeDataSource(
    table="users",
    account="myaccount",
    database="MY_DB",
    authenticator="externalbrowser",
)

# Key-pair authentication
source = SnowflakeDataSource(
    table="users",
    account="myaccount",
    database="MY_DB",
    user="myuser",
    private_key_path="/path/to/rsa_key.pem",
)

# From environment variables
source = SnowflakeDataSource.from_env(
    table="users",
    database="MY_DB",
    env_prefix="SNOWFLAKE",  # Reads SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, etc.
)
```

#### Amazon Redshift

```python
from truthound.datasources.sql import RedshiftDataSource

# Password authentication
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    user="admin",
    password="password",
    schema="public",
)

# IAM authentication
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    cluster_identifier="my-cluster",
    db_user="admin",
    iam_auth=True,
)

# From environment variables
source = RedshiftDataSource.from_env(
    table="users",
    env_prefix="REDSHIFT",
)
```

#### Databricks SQL

```python
from truthound.datasources.sql import DatabricksDataSource

# Personal Access Token (PAT)
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",  # Unity Catalog
    schema="default",
)

# OAuth M2M
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    client_id="...",
    client_secret="...",
    use_oauth=True,
    catalog="main",
    schema="default",
)

# Delta Lake features
history = source.get_table_history(limit=10)
source.optimize(zorder_by=["user_id"])
old_data = source.time_travel_query(version=5)
```

### Enterprise Databases

#### Oracle Database

```python
from truthound.datasources.sql import OracleDataSource

# Using service name
source = OracleDataSource(
    table="USERS",
    host="oracle.example.com",
    service_name="ORCL",
    user="myuser",
    password="mypassword",
)

# Using SID
source = OracleDataSource(
    table="USERS",
    host="oracle.example.com",
    sid="ORCL",
    user="myuser",
    password="mypassword",
)

# Using TNS name
source = OracleDataSource.from_tns(
    table="USERS",
    tns_name="MYDB",
    user="myuser",
    password="mypassword",
)

# Oracle Wallet (for cloud databases)
source = OracleDataSource(
    table="USERS",
    dsn="mydb_high",
    user="ADMIN",
    password="password",
    wallet_location="/path/to/wallet",
)
```

#### Microsoft SQL Server

```python
from truthound.datasources.sql import SQLServerDataSource

# SQL Server Authentication
source = SQLServerDataSource(
    table="Users",
    host="sqlserver.example.com",
    database="MyDB",
    user="sa",
    password="password",
    schema="dbo",
)

# Windows Authentication
source = SQLServerDataSource(
    table="Users",
    host="sqlserver.example.com",
    database="MyDB",
    trusted_connection=True,
)

# Azure SQL Database
source = SQLServerDataSource(
    table="Users",
    host="myserver.database.windows.net",
    database="MyDB",
    user="admin@myserver",
    password="password",
    encrypt=True,
)

# From connection string
source = SQLServerDataSource.from_connection_string(
    table="Users",
    connection_string="SERVER=myserver,1433;DATABASE=mydb;UID=user;PWD=pass",
)
```

### Spark

PySpark support with automatic sampling for large datasets.

```python
from pyspark.sql import SparkSession
from truthound.datasources import SparkDataSource

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("large_data.parquet")

# Create source with automatic sampling
source = SparkDataSource(df)

# Check if sampling is needed
if source.needs_sampling():
    source = source.sample(n=100_000)

engine = source.get_execution_engine()
```

### File-Based Sources

Direct loading from CSV, JSON, Parquet, and NDJSON files.

```python
from truthound.datasources import FileDataSource

# CSV
source = FileDataSource("data.csv")

# Parquet
source = FileDataSource("data.parquet")

# JSON
source = FileDataSource("data.json")

# NDJSON (newline-delimited JSON)
source = FileDataSource("data.ndjson")
```

## Factory Functions

### Auto-Detection

The `get_datasource()` function auto-detects the input type:

```python
from truthound import get_datasource

# Polars DataFrame
source = get_datasource(pl.DataFrame(...))

# Pandas DataFrame
source = get_datasource(pd.DataFrame(...))

# File path
source = get_datasource("data.csv")

# Dictionary
source = get_datasource({"a": [1, 2, 3], "b": ["x", "y", "z"]})
```

### SQL Factory

```python
from truthound import get_sql_datasource

# SQLite
source = get_sql_datasource("data.db", table="users")

# PostgreSQL
source = get_sql_datasource(
    "postgresql://user:pass@localhost/db",
    table="users",
)
```

### Convenience Functions

```python
from truthound.datasources import from_polars, from_pandas, from_file, from_dict

source = from_polars(polars_df)
source = from_pandas(pandas_df)
source = from_file("data.csv")
source = from_dict({"col1": [1, 2], "col2": ["a", "b"]})
```

## Execution Engines

Execution engines handle the actual validation operations.

### PolarsExecutionEngine

Primary engine using Polars' lazy evaluation.

```python
from truthound.execution import PolarsExecutionEngine

engine = PolarsExecutionEngine(lazyframe)

# Core operations
engine.count_rows()
engine.get_columns()
engine.count_nulls("column")
engine.count_distinct("column")

# Statistics
engine.get_stats("numeric_column")
# Returns: {count, null_count, mean, min, max, sum, std, median, q25, q75}

# Aggregations
from truthound.execution import AggregationType
engine.aggregate({
    "col1": AggregationType.SUM,
    "col2": AggregationType.COUNT_DISTINCT,
})
```

### SQLExecutionEngine

SQL engine with query pushdown.

```python
from truthound.execution import SQLExecutionEngine

engine = SQLExecutionEngine(sql_datasource)

# Operations are pushed to the database
engine.count_matching("age > 30")  # Runs: SELECT COUNT(*) WHERE age > 30
engine.count_in_range("salary", min_value=50000, max_value=100000)
engine.count_duplicates(["email"])
```

### PandasExecutionEngine

Native Pandas operations for compatibility.

```python
from truthound.execution import PandasExecutionEngine

engine = PandasExecutionEngine(pandas_df)
engine.get_value_counts("category")
```

## Size Limits and Sampling

### Configuration

```python
from truthound.datasources.base import DataSourceConfig

config = DataSourceConfig(
    max_rows=10_000_000,      # Maximum rows allowed
    max_memory_mb=4096,        # Maximum memory in MB
    sample_size=100_000,       # Default sample size
    sample_seed=42,            # Reproducible sampling
)
```

### Checking Limits

```python
source = PolarsDataSource(large_df)

# Check if sampling is needed
if source.needs_sampling():
    print("Data exceeds limits, sampling recommended")
    source = source.sample(n=100_000)

# Get safe sample (auto-samples if needed)
safe_source = source.get_safe_sample()
```

### Size Warnings

When using the `check()` API, you'll get warnings for large datasets:

```python
from truthound import check

# Warning will be emitted if data exceeds limits
report = check(large_source, rules={...})
# WARNING: Data source exceeds recommended size limits...
```

## Column Types

Unified column type representation across backends:

```python
from truthound.datasources import ColumnType

# Available types
ColumnType.INTEGER
ColumnType.FLOAT
ColumnType.DECIMAL
ColumnType.STRING
ColumnType.DATE
ColumnType.DATETIME
ColumnType.TIME
ColumnType.DURATION
ColumnType.BOOLEAN
ColumnType.BINARY
ColumnType.LIST
ColumnType.STRUCT
ColumnType.NULL
ColumnType.UNKNOWN
```

### Type Inspection

```python
source = PolarsDataSource(df)

# Get schema
schema = source.schema
# {'id': ColumnType.INTEGER, 'name': ColumnType.STRING, ...}

# Get columns by type
numeric_cols = source.get_numeric_columns()
string_cols = source.get_string_columns()
datetime_cols = source.get_datetime_columns()

# Check specific column type
col_type = source.get_column_type("my_column")
```

## Connection Pooling (SQL)

SQL data sources use connection pooling for efficiency:

```python
from truthound.datasources.sql import SQLDataSourceConfig

config = SQLDataSourceConfig(
    pool_size=5,              # Connection pool size
    pool_timeout=30.0,        # Timeout for acquiring connection
    query_timeout=300.0,      # Query execution timeout
    fetch_size=10000,         # Rows to fetch at a time
)

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    config=config,
)
```

## Spark-Specific Features

### Partitioning

```python
source = SparkDataSource(spark_df)

# Repartition for better distribution
source = source.repartition(num_partitions=10)

# Coalesce to reduce partitions
source = source.coalesce(num_partitions=2)
```

### Persistence

```python
# Persist in memory for repeated access
source = source.persist()

# Get execution plan
print(source.explain(extended=True))

# Unpersist when done
source = source.unpersist()
```

### Safety Limits

Spark sources have conservative limits to prevent driver OOM:

```python
from truthound.datasources.spark_source import SparkDataSourceConfig

config = SparkDataSourceConfig(
    max_rows_for_local=100_000,    # Max rows to collect to driver
    persist_sampled=True,           # Persist sampled data
    force_sampling=False,           # Always sample
)
```

## Using with Validators

DataSources integrate seamlessly with the validation API:

```python
from truthound import check
from truthound.datasources import get_datasource

# Create source
source = get_datasource("data.csv")

# Use with check() API
report = check(
    source,  # Pass source directly
    rules={
        "id": ["not_null", "unique"],
        "email": ["not_null", {"type": "regex", "pattern": r".*@.*"}],
    },
)

# Or pass to check() with source parameter
report = check(
    df,  # Original data
    source=source,  # Explicit source
    rules={...},
)
```

## Best Practices

1. **Use Polars** when possible - it's the most optimized backend
2. **Sample large datasets** before validation to avoid memory issues
3. **Use SQL pushdown** for database sources - operations run on the server
4. **Configure connection pools** appropriately for your workload
5. **Check `needs_sampling()`** before processing unknown-size data
6. **Use factory functions** for convenience and type auto-detection

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         Truthound Validators                               │
├───────────────────────────────────────────────────────────────────────────┤
│                         Execution Engines                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│   │   Polars    │   │   Pandas    │   │    SQL      │   │   Spark     │  │
│   │   Engine    │   │   Engine    │   │   Engine    │   │   Engine    │  │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘  │
├───────────────────────────────────────────────────────────────────────────┤
│                          Data Sources                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ DataFrame: Polars, Pandas, PySpark                                   │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ Core SQL: SQLite, PostgreSQL, MySQL                                  │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ Cloud DW: BigQuery, Snowflake, Redshift, Databricks                  │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ Enterprise: Oracle, SQL Server                                       │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ File: CSV, Parquet, JSON, NDJSON                                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

## Installation

Different data sources require different dependencies:

```bash
# Core (always available)
pip install truthound

# Cloud Data Warehouses
pip install truthound[bigquery]     # Google BigQuery
pip install truthound[snowflake]    # Snowflake
pip install truthound[redshift]     # Amazon Redshift
pip install truthound[databricks]   # Databricks SQL

# Enterprise Databases
pip install truthound[oracle]       # Oracle Database
pip install truthound[sqlserver]    # SQL Server

# DataFrame Libraries
pip install truthound[spark]        # PySpark

# All enterprise sources
pip install truthound[enterprise]
```

## Checking Available Sources

```python
from truthound.datasources.sql import get_available_sources, check_source_available

# Get all sources and their availability
sources = get_available_sources()
for name, cls in sources.items():
    status = "available" if cls is not None else "not installed"
    print(f"{name}: {status}")

# Check specific source
if check_source_available("bigquery"):
    from truthound.datasources.sql import BigQueryDataSource
    # Use BigQuery...
```

---

## Further Reading

- **[Data Sources Architecture](DATASOURCES_ARCHITECTURE.md)** - Design patterns, extensibility guide, and quality assessment
- **[Architecture Overview](ARCHITECTURE.md)** - Overall system architecture
- **[README](../README.md)** - Main documentation
