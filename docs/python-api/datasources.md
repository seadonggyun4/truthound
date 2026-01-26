# Data Sources

Multi-backend data source support for connecting to various data systems.

## Overview

Truthound supports multiple data backends through the DataSource abstraction:

| Category | Sources |
|----------|---------|
| **Files** | CSV, Parquet, JSON, NDJSON |
| **DataFrames** | Polars, Pandas |
| **SQL Databases** | PostgreSQL, MySQL, SQLite, DuckDB |
| **Cloud Warehouses** | BigQuery, Snowflake, Redshift, Databricks |
| **Enterprise** | Oracle, SQL Server |
| **Big Data** | Apache Spark |
| **NoSQL** | MongoDB, Elasticsearch (async) |
| **Streaming** | Apache Kafka, Kinesis (async) |

## Installation & Dependencies

Each data source requires specific Python packages. Core file formats (CSV, JSON, Parquet, NDJSON, JSONL) and SQLite are built-in with no additional dependencies.

| Category | Data Source | Required Package | Install Command |
|----------|-------------|------------------|-----------------|
| **DataFrame** | Polars | (built-in) | - |
| | Pandas | `pandas` | `pip install truthound[pandas]` |
| | Spark | `pyspark` | `pip install truthound[spark]` |
| **Core SQL** | SQLite | (built-in) | - |
| | DuckDB | `duckdb` | `pip install truthound[duckdb]` |
| | PostgreSQL | `psycopg2` | `pip install truthound[postgresql]` |
| | MySQL | `mysql-connector-python` | `pip install truthound[mysql]` |
| **Cloud DW** | BigQuery | `google-cloud-bigquery` | `pip install truthound[bigquery]` |
| | Snowflake | `snowflake-connector-python` | `pip install truthound[snowflake]` |
| | Redshift | `redshift-connector` | `pip install truthound[redshift]` |
| | Databricks | `databricks-sql-connector` | `pip install truthound[databricks]` |
| **Enterprise** | Oracle | `oracledb` | `pip install truthound[oracle]` |
| | SQL Server | `pyodbc` | `pip install truthound[sqlserver]` |
| **NoSQL** | MongoDB | `pymongo` | `pip install truthound[mongodb]` |
| | Elasticsearch | `elasticsearch` | `pip install truthound[elasticsearch]` |
| **Streaming** | Kafka | `confluent-kafka` | `pip install truthound[kafka]` |
| | Kinesis | `aiobotocore` | `pip install truthound[kinesis]` |
| **File** | CSV, JSON, Parquet, NDJSON, JSONL | (built-in) | - |

### Install Multiple Dependencies

```bash
# Install all enterprise data sources
pip install truthound[enterprise]

# Install specific combinations
pip install truthound[postgresql,bigquery,spark]

# Install all optional dependencies
pip install truthound[all]
```

### Source Code Locations

| Category | Data Source | Source File |
|----------|-------------|-------------|
| DataFrame | Polars | `datasources/polars_source.py` |
| | Pandas | `datasources/pandas_source.py` |
| | Spark | `datasources/spark_source.py` |
| Core SQL | SQLite | `datasources/sql/sqlite.py` |
| | DuckDB | `datasources/sql/duckdb.py` |
| | PostgreSQL | `datasources/sql/postgresql.py` |
| | MySQL | `datasources/sql/mysql.py` |
| Cloud DW | BigQuery | `datasources/sql/bigquery.py` |
| | Snowflake | `datasources/sql/snowflake.py` |
| | Redshift | `datasources/sql/redshift.py` |
| | Databricks | `datasources/sql/databricks.py` |
| Enterprise | Oracle | `datasources/sql/oracle.py` |
| | SQL Server | `datasources/sql/sqlserver.py` |
| NoSQL | MongoDB | `datasources/nosql/mongodb.py` |
| | Elasticsearch | `datasources/nosql/elasticsearch.py` |
| Streaming | Kafka | `datasources/streaming/kafka.py` |
| | Kinesis | `datasources/streaming/kinesis.py` |
| File | adapters | `datasources/adapters.py` |

## BaseDataSource

All data sources inherit from `BaseDataSource`.

### Definition

```python
from truthound.datasources.base import BaseDataSource

class BaseDataSource(ABC, Generic[ConfigT]):
    """Abstract base class for all data sources."""

    source_type: str = "base"

    @property
    def name(self) -> str:
        """Get the data source name."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, ColumnType]:
        """Column name to type mapping."""

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""

    @property
    def row_count(self) -> int | None:
        """Get row count if efficiently available."""

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get supported capabilities."""

    @abstractmethod
    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Convert to Polars LazyFrame."""

    @abstractmethod
    def get_execution_engine(self) -> BaseExecutionEngine:
        """Return execution engine."""

    def needs_sampling(self) -> bool:
        """Check if sampling is needed for large datasets."""

    @abstractmethod
    def sample(self, n: int, seed: int | None = None) -> BaseDataSource:
        """Return sampled data source."""
```

### Capabilities

```python
from truthound.datasources import DataSourceCapability

class DataSourceCapability(Enum):
    LAZY_EVALUATION = "lazy_evaluation"  # Supports lazy/deferred execution
    SQL_PUSHDOWN = "sql_pushdown"        # Can push operations to database
    SAMPLING = "sampling"                # Supports efficient sampling
    STREAMING = "streaming"              # Supports streaming/chunked reads
    SCHEMA_INFERENCE = "schema_inference"  # Can infer schema without full scan
    ROW_COUNT = "row_count"              # Can get row count efficiently
```

### ColumnType

```python
from truthound.datasources import ColumnType

class ColumnType(Enum):
    # Numeric types
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"

    # String types
    STRING = "string"
    TEXT = "text"

    # Date/Time types
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"

    # Boolean
    BOOLEAN = "boolean"

    # Binary
    BINARY = "binary"

    # Complex types
    LIST = "list"
    STRUCT = "struct"
    JSON = "json"

    # Other
    NULL = "null"
    UNKNOWN = "unknown"
```

---

## File Sources

### FileDataSource

Automatically detects file format based on extension.

```python
from truthound.datasources.polars_source import FileDataSource

# CSV
source = FileDataSource("data.csv")

# Parquet
source = FileDataSource("data.parquet")

# JSON
source = FileDataSource("data.json")

# NDJSON (newline-delimited JSON)
source = FileDataSource("data.ndjson")
```

### FileDataSourceConfig

```python
from truthound.datasources.polars_source import FileDataSourceConfig

config = FileDataSourceConfig(
    path="data.csv",
    name="my_data",
    max_rows=1_000_000,
    infer_schema_length=1000,
    n_rows=None,  # Limit rows to read
    encoding="utf-8",
)
source = FileDataSource(config=config)
```

---

## DataFrame Sources

### PolarsDataSource

```python
from truthound.datasources.polars_source import PolarsDataSource
import polars as pl

# From DataFrame
df = pl.read_csv("data.csv")
source = PolarsDataSource(df)

# From LazyFrame
lf = pl.scan_csv("data.csv")
source = PolarsDataSource(lf)
```

### PandasDataSource

```python
from truthound.datasources.pandas_source import PandasDataSource
import pandas as pd

pdf = pd.read_csv("data.csv")
source = PandasDataSource(pdf)
```

### DictDataSource

```python
from truthound.datasources.polars_source import DictDataSource

data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
source = DictDataSource(data)
```

---

## SQL Databases

### SQLiteDataSource

```python
from truthound.datasources.sql import SQLiteDataSource

# From table
source = SQLiteDataSource(
    database="mydb.db",
    table="users",
)

# With custom query
source = SQLiteDataSource(
    database="mydb.db",
    query="SELECT * FROM users WHERE active = 1",
)
```

### DuckDBDataSource

DuckDB is an in-process analytical database with excellent Polars integration.

```python
from truthound.datasources.sql import DuckDBDataSource

# From table
source = DuckDBDataSource(
    database="analytics.duckdb",
    table="events",
)

# With custom query
source = DuckDBDataSource(
    database="analytics.duckdb",
    query="SELECT * FROM events WHERE date > '2024-01-01'",
)

# In-memory database
source = DuckDBDataSource(
    database=":memory:",
    query="SELECT 1 as id, 'test' as value",
)

# Read directly from Parquet file (DuckDB feature)
source = DuckDBDataSource.from_parquet("data/*.parquet")

# Read from CSV file
source = DuckDBDataSource.from_csv("data.csv")

# Create from Polars DataFrame
import polars as pl
df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
source = DuckDBDataSource.from_dataframe(df, "my_table")
```

### PostgreSQLDataSource

```python
from truthound.datasources.sql import PostgreSQLDataSource

# Basic connection
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
)

# With schema
source = PostgreSQLDataSource(
    table="users",
    schema="public",
    host="localhost",
    database="mydb",
    user="postgres",
)

# With connection URL
source = PostgreSQLDataSource.from_connection_string(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table="users",
)

# With custom query
source = PostgreSQLDataSource(
    host="localhost",
    database="mydb",
    user="postgres",
    query="SELECT id, email FROM users WHERE created_at > '2024-01-01'",
)
```

### MySQLDataSource

```python
from truthound.datasources.sql import MySQLDataSource

source = MySQLDataSource(
    table="orders",
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
)
```

---

## Cloud Data Warehouses

### BigQueryDataSource

```python
from truthound.datasources.sql import BigQueryDataSource

# Basic usage
source = BigQueryDataSource(
    project="my-gcp-project",
    dataset="analytics",
    table="events",
)

# With credentials file
source = BigQueryDataSource(
    project="my-gcp-project",
    dataset="analytics",
    table="events",
    credentials_path="/path/to/credentials.json",
)

# With custom query
source = BigQueryDataSource(
    project="my-gcp-project",
    query="SELECT * FROM `analytics.events` WHERE date > '2024-01-01'",
)
```

### SnowflakeDataSource

```python
from truthound.datasources.sql import SnowflakeDataSource

# Password authentication
source = SnowflakeDataSource(
    account="myaccount",
    user="myuser",
    password="mypassword",
    database="MYDB",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    table="USERS",
)

# Key-pair authentication
source = SnowflakeDataSource(
    account="myaccount",
    user="myuser",
    private_key_path="/path/to/key.p8",
    database="MYDB",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    table="USERS",
)
```

### RedshiftDataSource

```python
from truthound.datasources.sql import RedshiftDataSource

# Password authentication
source = RedshiftDataSource(
    host="cluster.region.redshift.amazonaws.com",
    port=5439,
    database="mydb",
    user="myuser",
    password="mypassword",
    table="users",
)

# IAM authentication
source = RedshiftDataSource(
    host="cluster.region.redshift.amazonaws.com",
    database="mydb",
    iam_role="arn:aws:iam::123456789:role/RedshiftRole",
    table="users",
)
```

### DatabricksDataSource

```python
from truthound.datasources.sql import DatabricksDataSource

source = DatabricksDataSource(
    server_hostname="adb-123456789.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",
    schema="default",
    table="users",
)
```

---

## Enterprise Databases

### OracleDataSource

```python
from truthound.datasources.sql import OracleDataSource

# With service name
source = OracleDataSource(
    host="localhost",
    port=1521,
    service_name="ORCL",
    user="myuser",
    password="mypassword",
    table="USERS",
)
```

### SQLServerDataSource

```python
from truthound.datasources.sql import SQLServerDataSource

# SQL authentication
source = SQLServerDataSource(
    host="localhost",
    port=1433,
    database="mydb",
    user="sa",
    password="mypassword",
    table="users",
)

# Windows authentication
source = SQLServerDataSource(
    host="localhost",
    database="mydb",
    trusted_connection=True,
    table="users",
)
```

---

## Apache Spark

### SparkDataSource

```python
from truthound.datasources.spark_source import SparkDataSource
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("truthound").getOrCreate()
spark_df = spark.read.parquet("data.parquet")

source = SparkDataSource(spark_df)
```

---

## Factory Functions

### get_datasource()

Auto-detect source type from input.

```python
from truthound.datasources import get_datasource

# File path
source = get_datasource("data.csv")           # FileDataSource
source = get_datasource("data.parquet")       # FileDataSource
source = get_datasource("data.json")          # FileDataSource
source = get_datasource("data.ndjson")        # FileDataSource

# Polars DataFrame
source = get_datasource(polars_df)            # PolarsDataSource

# Pandas DataFrame
source = get_datasource(pandas_df)            # PandasDataSource

# Dictionary
source = get_datasource({"col": [1, 2, 3]})   # DictDataSource
```

> **Note**: `get_datasource()` does NOT support `.db` (SQLite) files directly. Use `get_sql_datasource()` for SQLite databases.

### get_sql_datasource()

Create SQL source from connection URL or file path.

```python
from truthound.datasources import get_sql_datasource

# SQLite database file (requires table parameter)
source = get_sql_datasource("mydb.db", table="users")

# PostgreSQL connection URL
source = get_sql_datasource(
    "postgresql://user:pass@localhost/db",
    table="users"
)
```

---

## Usage with Core Functions

```python
import truthound as th
from truthound.datasources.sql import PostgreSQLDataSource, BigQueryDataSource, SQLiteDataSource

# Check
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
report = th.check(source=source)

# Learn schema from database
source = SQLiteDataSource(database="mydb.db", table="users")
schema = th.learn(source=source)
schema.save("schema.yaml")

# Scan for PII
source = BigQueryDataSource(project="...", dataset="...", table="customers")
pii_report = th.scan(source=source)

# Profile
source = PostgreSQLDataSource(table="orders", host="localhost", database="mydb")
profile = th.profile(source=source)

# Mask
source = PostgreSQLDataSource(table="users", host="localhost", database="mydb")
masked_df = th.mask(source=source, strategy="hash")
```

---

## Query Pushdown

For SQL sources, Truthound can push validation predicates to the database:

```python
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="large_table",
    host="localhost",
    database="mydb",
)

# Enable pushdown - validations run on the database server
report = th.check(source=source, pushdown=True)

# Example: null_check becomes:
# SELECT COUNT(*) FROM table WHERE column IS NULL
```

---

## Sampling

For large datasets, use sampling to improve performance:

```python
source = PostgreSQLDataSource(table="huge_table", host="localhost", database="mydb")

# Check if sampling is needed
if source.needs_sampling():
    print(f"Dataset has {source.row_count:,} rows, sampling recommended")
    source = source.sample(n=100_000, seed=42)

report = th.check(source=source)
```

---

## Connection Pooling

SQL sources support connection pooling:

```python
from truthound.datasources.sql.base import SQLDataSourceConfig

config = SQLDataSourceConfig(
    table="users",
    host="localhost",
    database="mydb",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
)
source = PostgreSQLDataSource(config=config)
```

---

## NoSQL Databases

### MongoDBDataSource (Async)

```python
from truthound.datasources import from_mongodb

# Connect to MongoDB
source = await from_mongodb(
    connection_string="mongodb://localhost:27017",
    database="mydb",
    collection="users",
)

async with source:
    lf = await source.to_polars_lazyframe_async()
    report = th.check(lf)
```

### ElasticsearchDataSource (Async)

```python
from truthound.datasources import from_elasticsearch

source = await from_elasticsearch(
    hosts=["http://localhost:9200"],
    index="logs",
    query={"match_all": {}},
)

async with source:
    lf = await source.to_polars_lazyframe_async()
```

---

## Streaming Platforms

### KafkaDataSource (Async)

```python
from truthound.datasources import from_kafka

source = await from_kafka(
    bootstrap_servers="localhost:9092",
    topic="events",
    group_id="truthound-validators",
)

async with source:
    async for batch in source.consume_batches(batch_size=1000):
        report = th.check(batch)
```

---

## Async Factory Functions

For async data sources (MongoDB, Elasticsearch, Kafka):

```python
from truthound.datasources import get_async_datasource

# Auto-detect async source type
source = await get_async_datasource(
    "mongodb://localhost:27017",
    database="mydb",
    collection="users",
)

# Check if source is async
from truthound.datasources import is_async_source, is_sync_source

if is_async_source(source):
    async with source:
        lf = await source.to_polars_lazyframe_async()
```

## See Also

- [Data Sources Guide](../guides/datasources.md) - Detailed configuration
- [DataSources Architecture](../concepts/datasources-architecture.md) - Technical deep dive
- [Performance Guide](../guides/performance.md) - Optimization tips
