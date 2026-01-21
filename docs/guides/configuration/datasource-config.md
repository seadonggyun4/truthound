# DataSource Configuration

Truthound supports multiple data source backends with unified configuration patterns.

## Quick Start

```python
from truthound.datasources import get_datasource

# Auto-detect source type
ds = get_datasource(df)  # Polars/Pandas DataFrame
ds = get_datasource("data.parquet")  # File path
ds = get_datasource("postgresql://user:pass@localhost/db", table="users")
```

## DataSourceConfig

Base configuration for all data sources.

```python
from truthound.datasources.base import DataSourceConfig

config = DataSourceConfig(
    name="my_source",
    max_rows=10_000_000,        # Size limit before sampling
    max_memory_mb=4096,         # Memory limit (4GB)
    sample_size=100_000,        # Default sample size
    sample_seed=42,             # Reproducible sampling
    cache_schema=True,          # Cache schema info
    strict_types=False,         # Strict type checking
    metadata={},                # Custom metadata
)
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_rows` | 10,000,000 | Max rows before requiring sampling |
| `max_memory_mb` | 4,096 | Memory limit in MB |
| `sample_size` | 100,000 | Default sample size |
| `sample_seed` | 42 | Seed for reproducible sampling |
| `cache_schema` | True | Cache schema information |
| `strict_types` | False | Enable strict type checking |

## SQL DataSource Configuration

### SQLDataSourceConfig

Extended configuration for SQL data sources.

```python
from truthound.datasources.sql.base import SQLDataSourceConfig

config = SQLDataSourceConfig(
    # Inherited from DataSourceConfig
    max_rows=10_000_000,

    # SQL-specific
    pool_size=5,                    # Connection pool size
    pool_timeout=30.0,              # Pool acquire timeout
    query_timeout=300.0,            # Query timeout (5 min)
    fetch_size=10000,               # Rows per fetch batch
    use_server_side_cursor=False,   # Server-side cursors
    schema_name=None,               # Database schema
)
```

### Connection Pooling

```python
from truthound.datasources.sql.base import SQLConnectionPool

pool = SQLConnectionPool(
    connection_factory=create_connection,
    size=5,                # Max connections
    timeout=30.0,          # Acquire timeout
)

# Context manager usage
with pool.acquire() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
```

## Database-Specific Configuration

### PostgreSQL

```python
from truthound.datasources.sql.postgresql import (
    PostgreSQLDataSource,
    PostgreSQLDataSourceConfig,
)

# Option 1: Direct parameters
ds = PostgreSQLDataSource(
    table="users",
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    schema_name="public",
)

# Option 2: Connection string
ds = PostgreSQLDataSource.from_connection_string(
    "postgresql://postgres:secret@localhost:5432/mydb",
    table="users",
    schema_name="public",
)

# Option 3: Full config
config = PostgreSQLDataSourceConfig(
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    sslmode="require",              # SSL mode
    application_name="truthound",   # Application name
    schema_name="public",
    pool_size=10,
    query_timeout=600.0,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | localhost | Database host |
| `port` | 5432 | PostgreSQL port |
| `database` | postgres | Database name |
| `user` | postgres | Username |
| `sslmode` | prefer | SSL mode (disable, require, verify-ca, verify-full) |
| `application_name` | truthound | Application identifier |
| `schema_name` | public | Default schema |

### MySQL

```python
from truthound.datasources.sql.mysql import (
    MySQLDataSource,
    MySQLDataSourceConfig,
)

ds = MySQLDataSource(
    table="users",
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
)

# Full config
config = MySQLDataSourceConfig(
    host="localhost",
    port=3306,
    database="mysql",
    user="root",
    password="",
    charset="utf8mb4",
    ssl=None,                # SSL config dict
    autocommit=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | localhost | Database host |
| `port` | 3306 | MySQL port |
| `database` | mysql | Database name |
| `user` | root | Username |
| `charset` | utf8mb4 | Character set |
| `autocommit` | True | Auto-commit mode |

### SQLite

```python
from truthound.datasources.sql.sqlite import (
    SQLiteDataSource,
    SQLiteDataSourceConfig,
)

# In-memory database
ds = SQLiteDataSource(
    table="users",
    database=":memory:",
)

# File database
ds = SQLiteDataSource(
    table="users",
    database="./data.db",
)

# Query mode
ds = SQLiteDataSource(
    query="SELECT * FROM users WHERE active = 1",
    database="./data.db",
)

# Full config
config = SQLiteDataSourceConfig(
    database=":memory:",
    timeout=5.0,
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    isolation_level=None,
)
```

## Cloud Data Warehouse Configuration

### Base Cloud Configuration

```python
from truthound.datasources.sql.cloud_base import CloudDWConfig

config = CloudDWConfig(
    project=None,              # Cloud project
    warehouse=None,            # Warehouse name
    region=None,               # Cloud region
    role=None,                 # Access role
    timeout=300,               # Query timeout
    use_cache=True,            # Result caching
    credentials_path=None,     # Path to credentials file
    credentials_dict={},       # Credentials as dict
)
```

### Google BigQuery

```python
from truthound.datasources.sql.bigquery import (
    BigQueryDataSource,
    BigQueryConfig,
)

ds = BigQueryDataSource(
    table="users",
    project="my-project",
    dataset="my_dataset",
    credentials_path="./service-account.json",
    location="US",
)

# Full config
config = BigQueryConfig(
    dataset="my_dataset",
    location="US",
    use_legacy_sql=False,
    maximum_bytes_billed=10_000_000_000,  # Cost control
    job_timeout=300,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | None | BigQuery dataset |
| `location` | None | Data location |
| `use_legacy_sql` | False | Use legacy SQL syntax |
| `maximum_bytes_billed` | None | Cost control limit |
| `job_timeout` | 300 | Job timeout in seconds |

### Snowflake

```python
from truthound.datasources.sql.snowflake import (
    SnowflakeDataSource,
    SnowflakeConfig,
)

ds = SnowflakeDataSource(
    table="users",
    account="xy12345.us-east-1",
    user="myuser",
    database="MYDB",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    password="secret",
    role="ANALYST",
)

# Full config
config = SnowflakeConfig(
    account=None,
    user=None,
    password=None,
    database=None,
    schema_name="PUBLIC",
    warehouse=None,
    role=None,
    authenticator="snowflake",       # snowflake, externalbrowser, oauth
    private_key_path=None,           # Key-pair auth
    private_key_passphrase=None,
    token=None,                       # OAuth token
    client_session_keep_alive=True,
)
```

### Amazon Redshift

```python
from truthound.datasources.sql.redshift import (
    RedshiftDataSource,
    RedshiftConfig,
)

ds = RedshiftDataSource(
    table="users",
    host="cluster.region.redshift.amazonaws.com",
    port=5439,
    database="mydb",
    user="admin",
    password="secret",
)

# IAM authentication
config = RedshiftConfig(
    host=None,
    port=5439,
    database=None,
    user=None,
    password=None,
    iam_auth=False,                  # Use IAM auth
    cluster_identifier=None,
    db_user=None,
    access_key_id=None,
    secret_access_key=None,
    session_token=None,
    ssl=True,
    ssl_mode="verify-ca",
)
```

### Databricks

```python
from truthound.datasources.sql.databricks import (
    DatabricksDataSource,
    DatabricksConfig,
)

ds = DatabricksDataSource(
    table="users",
    host="workspace.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",
)

# Full config
config = DatabricksConfig(
    host=None,
    http_path=None,                  # SQL warehouse path
    access_token=None,               # Personal access token
    catalog=None,                    # Unity Catalog
    use_cloud_fetch=True,            # Optimize large results
    max_download_threads=10,
    client_id=None,                  # OAuth
    client_secret=None,
    use_oauth=False,
)
```

## In-Memory & File DataSources

### Polars

```python
from truthound.datasources.polars_source import (
    PolarsDataSource,
    PolarsDataSourceConfig,
)

ds = PolarsDataSource(
    data=pl.DataFrame({"a": [1, 2, 3]}),
)

config = PolarsDataSourceConfig(
    rechunk=False,           # Rechunk for performance
    streaming=False,         # Streaming mode for large files
)
```

### File-Based Sources

```python
from truthound.datasources.polars_source import (
    FileDataSource,
    FileDataSourceConfig,
)

# Auto-detect format
ds = FileDataSource("data.csv")
ds = FileDataSource("data.parquet")
ds = FileDataSource("data.json")

config = FileDataSourceConfig(
    infer_schema_length=10000,    # Rows for schema inference
    ignore_errors=False,
    encoding="utf8",
    separator=",",                # CSV separator
)
```

### Dictionary Source

```python
from truthound.datasources.polars_source import DictDataSource

ds = DictDataSource({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
})
```

### Pandas Source

```python
from truthound.datasources.pandas_source import PandasDataSource

ds = PandasDataSource(pandas_df)
```

### PySpark Source

```python
from truthound.datasources.spark_source import (
    SparkDataSource,
    SparkDataSourceConfig,
)

ds = SparkDataSource(spark_df)

config = SparkDataSourceConfig(
    max_rows_for_local=100_000,     # Conservative limit
    sampling_fraction=None,          # Auto-calculate
    persist_sampled=True,
    force_sampling=False,
    repartition_for_sampling=None,
)
```

## Factory Functions

### Auto-Detection

```python
from truthound.datasources import get_datasource

# Polars DataFrame
ds = get_datasource(pl_df)

# Pandas DataFrame
ds = get_datasource(pd_df)

# PySpark DataFrame
ds = get_datasource(spark_df)

# Dictionary
ds = get_datasource({"a": [1, 2, 3]})

# File path
ds = get_datasource("data.parquet")

# SQL connection string
ds = get_datasource(
    "postgresql://user:pass@localhost/db",
    table="users",
)
```

### Convenience Functions

```python
from truthound.datasources import (
    from_polars,
    from_pandas,
    from_spark,
    from_file,
    from_dict,
)

ds = from_polars(pl_df)
ds = from_pandas(pd_df)
ds = from_spark(spark_df, force_sampling=True)
ds = from_file("data.csv")
ds = from_dict({"col": [1, 2, 3]})
```

## DataSource Capabilities

Each data source declares its capabilities:

```python
from truthound.datasources._protocols import DataSourceCapability

# Check capabilities
if DataSourceCapability.SQL_PUSHDOWN in ds.capabilities:
    # Use query pushdown optimization
    pass

if DataSourceCapability.LAZY_EVALUATION in ds.capabilities:
    # Use lazy evaluation
    pass
```

| Capability | Description |
|------------|-------------|
| `LAZY_EVALUATION` | Supports lazy/deferred execution |
| `SQL_PUSHDOWN` | Can push operations to database |
| `SAMPLING` | Supports data sampling |
| `STREAMING` | Supports streaming processing |
| `SCHEMA_INFERENCE` | Can infer schema automatically |
| `ROW_COUNT` | Can efficiently count rows |

## Table vs Query Mode

SQL data sources support both table and query modes:

```python
# Table mode (default)
ds = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
)

# Query mode
ds = PostgreSQLDataSource(
    query="SELECT * FROM users WHERE active = 1",
    host="localhost",
    database="mydb",
)

# Check mode
if ds.is_query_mode:
    print(f"Query: {ds.query_sql}")
else:
    print(f"Table: {ds.full_table_name}")
```

## Cost-Aware Execution

Cloud data warehouses support cost-aware execution:

```python
# Check cost before execution
result = ds.execute_with_cost_check(
    query="SELECT * FROM large_table",
    max_bytes=10_000_000_000,      # 10GB limit
    max_cost_usd=1.0,              # $1 limit
)
```

## Type Mapping

Unified type mapping across backends:

| Source Type | ColumnType |
|------------|------------|
| INT, INTEGER | INTEGER |
| FLOAT, DOUBLE | FLOAT |
| VARCHAR, TEXT | STRING |
| DATE | DATE |
| TIMESTAMP, DATETIME | DATETIME |
| BOOLEAN, BOOL | BOOLEAN |
| JSON, JSONB | JSON |
| ARRAY | LIST |
| STRUCT | STRUCT |

## Environment Variables

```bash
# PostgreSQL
export TRUTHOUND_DATABASE_HOST=localhost
export TRUTHOUND_DATABASE_PORT=5432
export TRUTHOUND_DATABASE_POOL_SIZE=10

# Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
export AWS_REGION=us-east-1
export SNOWFLAKE_ACCOUNT=xy12345
```
