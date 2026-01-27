# Data Sources Guide

This guide covers connecting Truthound to various data backends through the Python API. It includes connection patterns, authentication methods, and practical workflows for database validation.

---

## Quick Start

```python
import truthound as th
from truthound.datasources import get_sql_datasource

# Quick data loading with th.read()
df = th.read("data.csv")                                     # File path
df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})         # Dict data
df = th.read("large_data.parquet", sample_size=10000)        # With sampling
df = th.read({"path": "data.csv", "delimiter": ";"})         # Config dict

# File-based validation
report = th.check("data.csv")

# Database validation
source = get_sql_datasource("mydb.db", table="users")
report = th.check(source=source)

# With query pushdown (runs on database server)
from truthound.datasources import PostgreSQLDataSource
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
report = th.check(source=source, pushdown=True)
```

---

## Common Workflows

### Workflow 1: Multi-Source Validation Pipeline

```python
import truthound as th
from truthound.datasources import PostgreSQLDataSource, BigQueryDataSource

# Define sources
sources = {
    "postgres_users": PostgreSQLDataSource(
        table="users", host="localhost", database="app"
    ),
    "bigquery_events": BigQueryDataSource(
        project="my-project", dataset="analytics", table="events"
    ),
}

# Validate all sources
results = {}
for name, source in sources.items():
    report = th.check(source=source)
    results[name] = report
    print(f"{name}: {len(report.issues)} issues found")
```

### Workflow 2: Cross-Database Comparison

```python
import truthound as th
from truthound.validators import CrossTableRowCountValidator
from truthound.datasources import PostgreSQLDataSource, MySQLDataSource

# Source and target databases
source_db = PostgreSQLDataSource(table="users", host="source-db", database="app")
target_db = MySQLDataSource(table="users", host="target-db", database="app")

# Load data
source_data = source_db.to_polars_lazyframe().collect()
target_data = target_db.to_polars_lazyframe().collect()

# Compare row counts
validator = CrossTableRowCountValidator(
    reference_data=source_data,
    operator="==",
    tolerance=0.0
)
issues = validator.validate(target_data.lazy())
```

### Workflow 3: Query Mode Validation

```python
from truthound.datasources.sql import PostgreSQLDataSource

# Validate custom query results instead of entire table
source = PostgreSQLDataSource(
    database="analytics",
    host="localhost",
    query="""
        SELECT user_id, email, created_at
        FROM users
        WHERE status = 'active'
        AND created_at > '2024-01-01'
    """
)

# Validate query results
report = th.check(source=source, validators=["null", "email"])
```

### Workflow 4: Environment-Based Connection

```python
import os
from truthound.datasources.sql import PostgreSQLDataSource

# Set environment variables
# export DB_HOST=localhost
# export DB_NAME=myapp
# export DB_USER=readonly
# export DB_PASSWORD=secret

source = PostgreSQLDataSource(
    table="users",
    host=os.environ["DB_HOST"],
    database=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
)

# Or use from_env() where available
from truthound.datasources.sql import SnowflakeDataSource
source = SnowflakeDataSource.from_env(
    table="users",
    database="MY_DB",
    env_prefix="SNOWFLAKE",  # Reads SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, etc.
)
```

---

## Overview

Truthound supports multiple data backends through a unified abstraction layer:

- **DataSource**: Abstraction for data storage and access
- **ExecutionEngine**: Abstraction for running validation operations

This architecture enables validators to work seamlessly across Polars, Pandas, SQL databases, and Spark.

> **Important: Python API Only**
>
> DataSource integration is available **only through the Python API**. The CLI (`truthound check`, `truthound scan`, etc.) only supports file-based inputs (CSV, JSON, Parquet, NDJSON, JSONL).
>
> For SQL databases, Spark, or Cloud Data Warehouses, you must use the Python API with the `source=` parameter as shown below.

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
| File | CSV, JSON, Parquet, NDJSON, JSONL |

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
    schema_name="public",
)

# Or using connection string
source = PostgreSQLDataSource.from_connection_string(
    connection_string="postgresql://user:pass@localhost/db",
    table="users",
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

Direct loading from CSV, JSON, Parquet, NDJSON, and JSONL files.

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

# JSONL (JSON Lines - same as NDJSON)
source = FileDataSource("data.jsonl")
```

## Factory Functions

### th.read() - Convenience Data Loading

The `th.read()` function provides a simple way to load data from various sources:

```python
import truthound as th

# File paths
df = th.read("data.csv")
df = th.read("data.parquet")
df = th.read("data.json")

# Raw data dict (column-oriented)
df = th.read({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Config dict with options
df = th.read({"path": "data.csv", "delimiter": ";"})

# With sampling for large datasets
df = th.read("large_data.csv", sample_size=10000)

# Polars DataFrame/LazyFrame passthrough
df = th.read(existing_df)
df = th.read(existing_lf)
```

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

# Numeric types
ColumnType.INTEGER
ColumnType.FLOAT
ColumnType.DECIMAL

# String types
ColumnType.STRING
ColumnType.TEXT

# Date/Time types
ColumnType.DATE
ColumnType.DATETIME
ColumnType.TIME
ColumnType.DURATION

# Boolean
ColumnType.BOOLEAN

# Binary
ColumnType.BINARY

# Complex types
ColumnType.LIST
ColumnType.STRUCT
ColumnType.JSON

# Other
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

## NoSQL Data Sources

Truthound supports NoSQL databases for document-based and search-based data validation.

### MongoDB

**Installation**:
```bash
pip install truthound[mongodb]
# or
pip install motor
```

**Usage**:
```python
from truthound.datasources import from_mongodb

# Async MongoDB data source
source = await from_mongodb(
    connection_string="mongodb://localhost:27017",
    database="mydb",
    collection="users",
)

async with source:
    schema = await source.get_schema_async()
    lf = await source.to_polars_lazyframe_async()
```

**MongoDB Atlas**:
```python
import os
from truthound.datasources import from_atlas

# Set MONGODB_ATLAS_URI environment variable with your connection string
# export MONGODB_ATLAS_URI="mongodb+srv://<username>:<password>@<cluster>.mongodb.net"

source = await from_atlas(
    connection_string=os.environ["MONGODB_ATLAS_URI"],
    database="mydb",
    collection="users",
)
```

### Elasticsearch

**Installation**:
```bash
pip install truthound[elasticsearch]
# or
pip install elasticsearch[async]
```

**Usage**:
```python
from truthound.datasources import from_elasticsearch

source = await from_elasticsearch(
    hosts=["http://localhost:9200"],
    index="users",
    query={"match_all": {}},  # Optional query filter
)

async with source:
    lf = await source.to_polars_lazyframe_async()
```

---

## Streaming Data Sources

Truthound supports streaming platforms for real-time data validation.

### Apache Kafka

**Installation**:
```bash
pip install truthound[kafka]
# or
pip install aiokafka
```

**Usage**:
```python
from truthound.datasources import from_kafka

# Kafka consumer data source
source = await from_kafka(
    bootstrap_servers="localhost:9092",
    topic="user-events",
    group_id="truthound-validator",
    auto_offset_reset="earliest",
)

async with source:
    # Consume messages as DataFrame
    async for batch in source.consume_batches(batch_size=1000):
        report = th.check(batch, validators=["null", "format"])
```

### Confluent Kafka

**Usage**:
```python
from truthound.datasources import from_confluent

source = await from_confluent(
    bootstrap_servers="pkc-xxx.confluent.cloud:9092",
    topic="user-events",
    api_key="your-api-key",
    api_secret="your-api-secret",
)
```

---

## Async Data Sources

For I/O-bound workloads, Truthound provides async data source support with full async/await compatibility.

### Async Protocols

```python
from truthound.datasources import (
    AsyncDataSourceProtocol,
    AsyncConnectableProtocol,
    AsyncStreamableProtocol,
    AsyncQueryableProtocol,
)
```

### Async Factory Functions

```python
from truthound.datasources import (
    get_async_datasource,
    from_mongodb,
    from_elasticsearch,
    from_kafka,
    from_confluent,
    from_atlas,
    detect_async_datasource_type,
    is_native_async_source,
)

# Auto-detect async data source
source = await get_async_datasource(
    "mongodb://localhost:27017",
    database="mydb",
    collection="users",
)
```

### Sync-Async Adapters

Convert between sync and async data sources:

```python
from truthound.datasources import (
    SyncToAsyncAdapter,
    AsyncToSyncAdapter,
    adapt_to_async,
    adapt_to_sync,
    is_async_source,
    is_sync_source,
)

# Wrap sync source for async usage
sync_source = PolarsDataSource(df)
async_source = adapt_to_async(sync_source)

# Wrap async source for sync usage
async_source = await from_mongodb(...)
sync_source = adapt_to_sync(async_source)
```

### Async Base Classes

```python
from truthound.datasources import (
    AsyncBaseDataSource,
    AsyncDataSourceConfig,
    AsyncConnectionPool,
)

class MyAsyncDataSource(AsyncBaseDataSource):
    async def connect(self) -> None:
        ...

    async def to_polars_lazyframe_async(self) -> pl.LazyFrame:
        ...
```

---

## Optimized Pandas Conversion

For large Pandas DataFrames, use the optimized converter for better memory efficiency:

```python
from truthound.datasources import (
    OptimizedPandasDataSource,
    OptimizedPandasConfig,
    DataFrameOptimizer,
    optimize_pandas_to_polars,
    estimate_polars_memory,
    get_optimal_chunk_size,
)

# Estimate memory before conversion
memory_estimate = estimate_polars_memory(pandas_df)
print(f"Estimated Polars memory: {memory_estimate / 1024**2:.1f} MB")

# Get optimal chunk size for conversion
chunk_size = get_optimal_chunk_size(pandas_df)

# Optimized conversion
config = OptimizedPandasConfig(
    chunk_size=chunk_size,
    use_pyarrow=True,
    preserve_index=False,
)
source = OptimizedPandasDataSource(pandas_df, config=config)
```

---

## Further Reading

- **[Data Sources Architecture](DATASOURCES_ARCHITECTURE.md)** - Design patterns, extensibility guide, and quality assessment
- **[Architecture Overview](ARCHITECTURE.md)** - Overall system architecture
- **[README](../README.md)** - Main documentation
