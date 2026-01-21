# Cloud Data Warehouse Data Sources

This document covers cloud data warehouse data sources in Truthound: BigQuery, Snowflake, Redshift, and Databricks.

## Overview

Cloud data warehouse sources inherit from `CloudDWDataSource` and share common features:

- Credential management (service accounts, tokens, environment variables)
- Cost-aware query execution
- Cloud-native authentication methods

| Platform | Driver | Installation |
|----------|--------|--------------|
| BigQuery | `google-cloud-bigquery` | `pip install google-cloud-bigquery db-dtypes` |
| Snowflake | `snowflake-connector-python` | `pip install snowflake-connector-python` |
| Redshift | `redshift-connector` | `pip install redshift-connector` |
| Databricks | `databricks-sql-connector` | `pip install databricks-sql-connector` |

## Google BigQuery

BigQuery is Google Cloud's serverless data warehouse.

### Installation

```bash
pip install google-cloud-bigquery db-dtypes
```

### Basic Usage

```python
from truthound.datasources.sql import BigQueryDataSource

source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
)
```

### Authentication Methods

#### Service Account JSON

```python
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
    credentials_path="/path/to/service-account.json",
)
```

#### Application Default Credentials (ADC)

```python
# Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
# or gcloud auth application-default login
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
)
```

### Configuration

```python
from truthound.datasources.sql import BigQueryDataSource, BigQueryConfig

config = BigQueryConfig(
    # Location settings
    location="US",                    # Dataset location (US, EU, etc.)

    # Query settings
    use_legacy_sql=False,             # Use standard SQL (default)
    job_timeout=300,                  # Query timeout in seconds

    # Cost control
    maximum_bytes_billed=1_000_000_000,  # 1GB limit per query

    # General settings
    max_rows=10_000_000,
    use_cache=True,
)

source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
    config=config,
)
```

### Cost Estimation

BigQuery supports dry-run cost estimation:

```python
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
)

# Get cost estimate before running query
estimate = source._get_cost_estimate("SELECT * FROM my_table")
# {'bytes_processed': 1073741824, 'estimated_cost_usd': 0.005}
```

### BigQuery-Specific Methods

```python
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
    credentials_path="service-account.json",
)

# Get detailed table information
info = source.get_table_info()
# {
#     'num_rows': 1000000,
#     'num_bytes': 128000000,
#     'created': datetime(...),
#     'modified': datetime(...),
#     'description': 'User table',
#     'labels': {'env': 'prod'},
#     'partitioning': 'DAY',
#     'clustering': ['user_id'],
# }

# Get partition information
partitions = source.get_partition_info()
# [{'partition_id': '20240101', 'total_rows': 10000, ...}, ...]

# Export to GCS
job_id = source.export_to_gcs(
    destination_uri="gs://my-bucket/exports/users-*.parquet",
    format="PARQUET",  # PARQUET, CSV, JSON, AVRO
)
```

### From Query

Create a BigQuery view from a query:

```python
source = BigQueryDataSource.from_query(
    query="""
        SELECT user_id, COUNT(*) as order_count
        FROM orders
        GROUP BY user_id
    """,
    project="my-gcp-project",
    dataset="my_dataset",
    table_name="user_orders_view",
    credentials_path="service-account.json",
)
```

## Snowflake

Snowflake is a cloud-native data warehouse.

### Installation

```bash
pip install snowflake-connector-python
```

### Basic Usage

```python
from truthound.datasources.sql import SnowflakeDataSource

source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",  # Account identifier
    user="myuser",
    password="mypassword",
    database="MY_DB",
    schema="PUBLIC",              # Default schema
    warehouse="COMPUTE_WH",
)
```

### Authentication Methods

#### Password Authentication

```python
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser",
    password="mypassword",
    database="MY_DB",
)
```

#### SSO (External Browser)

```python
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser@company.com",
    database="MY_DB",
    authenticator="externalbrowser",
)
```

#### Key-Pair Authentication

```python
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser",
    database="MY_DB",
    private_key_path="/path/to/rsa_key.pem",
)

# With encrypted private key
from truthound.datasources.sql import SnowflakeConfig

config = SnowflakeConfig(
    private_key_path="/path/to/rsa_key.pem",
    private_key_passphrase="my_passphrase",
)
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser",
    database="MY_DB",
    config=config,
)
```

#### OAuth Authentication

```python
from truthound.datasources.sql import SnowflakeConfig

config = SnowflakeConfig(
    token="<oauth_token>",
)
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser",
    database="MY_DB",
    authenticator="oauth",
    config=config,
)
```

#### Environment Variables

```python
# Reads: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
#        SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE
source = SnowflakeDataSource.from_env(
    table="USERS",
    schema="PUBLIC",
    env_prefix="SNOWFLAKE",
)
```

### Configuration

```python
from truthound.datasources.sql import SnowflakeConfig

config = SnowflakeConfig(
    # Connection settings
    client_session_keep_alive=True,  # Keep connection alive

    # Connection pool
    pool_size=5,
    pool_timeout=30.0,
    query_timeout=300.0,
)
```

### Snowflake-Specific Methods

```python
source = SnowflakeDataSource(
    table="USERS",
    account="xy12345.us-east-1",
    user="myuser",
    password="secret",
    database="MY_DB",
)

# Get detailed table information
info = source.get_table_info()
# {
#     'TABLE_CATALOG': 'MY_DB',
#     'TABLE_SCHEMA': 'PUBLIC',
#     'TABLE_NAME': 'USERS',
#     'TABLE_TYPE': 'BASE TABLE',
#     'ROW_COUNT': 1000000,
#     'BYTES': 128000000,
#     'CREATED': datetime(...),
#     'LAST_ALTERED': datetime(...),
#     'COMMENT': 'User table',
# }

# Get query history
history = source.get_query_history(limit=10)
# [{'QUERY_ID': '...', 'QUERY_TEXT': '...', 'EXECUTION_STATUS': 'SUCCESS', ...}]

# Clone table (zero-copy)
source.clone_table(
    target_table="USERS_BACKUP",
    schema="BACKUP",  # Optional, defaults to same schema
)

# Get clustering information
clustering = source.get_clustering_info()
# {'clustering_key': 'LINEAR(user_id)'}
```

## Amazon Redshift

Redshift is AWS's data warehouse service.

### Installation

```bash
pip install redshift-connector
# or use psycopg2
pip install psycopg2-binary
```

### Basic Usage

```python
from truthound.datasources.sql import RedshiftDataSource

source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    user="admin",
    password="password",
    port=5439,       # Default Redshift port
    schema="public",
)
```

### Authentication Methods

#### Password Authentication

```python
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    user="admin",
    password="password",
)
```

#### IAM Authentication

```python
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    cluster_identifier="my-cluster",
    db_user="admin",
    iam_auth=True,
)

# With explicit AWS credentials
from truthound.datasources.sql import RedshiftConfig

config = RedshiftConfig(
    access_key_id="AKIAXXXXXXXX",
    secret_access_key="secret",
    session_token="token",  # For temporary credentials
)
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    cluster_identifier="my-cluster",
    db_user="admin",
    iam_auth=True,
    config=config,
)
```

#### Environment Variables

```python
# Reads: REDSHIFT_HOST, REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD
source = RedshiftDataSource.from_env(
    table="users",
    schema="public",
    env_prefix="REDSHIFT",
)
```

### Configuration

```python
from truthound.datasources.sql import RedshiftConfig

config = RedshiftConfig(
    # SSL settings
    ssl=True,                # Use SSL connection
    ssl_mode="verify-ca",    # SSL verification mode

    # Connection pool
    pool_size=5,
    pool_timeout=30.0,
    query_timeout=300.0,
)
```

### Redshift-Specific Methods

```python
source = RedshiftDataSource(
    table="users",
    host="cluster.abc123.us-east-1.redshift.amazonaws.com",
    database="mydb",
    user="admin",
    password="password",
)

# Get detailed table information from svv_table_info
info = source.get_table_info()
# {
#     'table_schema': 'public',
#     'table_name': 'users',
#     'diststyle': 'KEY(user_id)',
#     'sortkey1': 'created_at',
#     'size_mb': 128,
#     'pct_used': 0.5,
#     'unsorted': 0.1,
#     'stats_off': 0.0,
#     'tbl_rows': 1000000,
# }

# Get distribution style
dist_style = source.get_dist_style()
# 'KEY(user_id)'

# Get sort keys
sort_keys = source.get_sort_keys()
# ['created_at', 'user_id']

# Maintenance operations
source.analyze()                  # Update statistics
source.vacuum()                   # Reclaim space
source.vacuum(full=True)          # Full vacuum
source.vacuum(sort_only=True)     # Sort only

# Unload to S3
source.unload_to_s3(
    s3_path="s3://my-bucket/exports/users/",
    iam_role="arn:aws:iam::123456789:role/RedshiftS3Access",
    format="PARQUET",             # PARQUET, CSV, JSON
    partition_by=["year", "month"],
)
```

## Databricks

Databricks is a unified analytics platform with Delta Lake support.

### Installation

```bash
pip install databricks-sql-connector
```

### Basic Usage

```python
from truthound.datasources.sql import DatabricksDataSource

source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",       # Unity Catalog
    schema="default",
)
```

### Authentication Methods

#### Personal Access Token (PAT)

```python
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",
    schema="default",
)
```

#### OAuth M2M (Machine-to-Machine)

```python
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    client_id="client_id",
    client_secret="client_secret",
    use_oauth=True,
    catalog="main",
    schema="default",
)
```

#### Environment Variables

```python
# Reads: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
source = DatabricksDataSource.from_env(
    table="users",
    schema="default",
    catalog="main",
    env_prefix="DATABRICKS",
)
```

### Configuration

```python
from truthound.datasources.sql import DatabricksConfig

config = DatabricksConfig(
    # Cloud Fetch for large results
    use_cloud_fetch=True,        # Enable cloud fetch
    max_download_threads=10,     # Parallel download threads

    # Connection settings
    pool_size=5,
    pool_timeout=30.0,
    query_timeout=300.0,
)
```

### Databricks-Specific Methods (Delta Lake)

```python
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
    catalog="main",
    schema="default",
)

# Get detailed table information
info = source.get_table_info()
# {'format': 'delta', 'location': 's3://...', 'numFiles': 100, ...}

# Get table history (Delta Lake versioning)
history = source.get_table_history(limit=10)
# [{'version': 5, 'timestamp': ..., 'operation': 'WRITE', ...}]

# Get table properties
props = source.get_table_properties()
# {'delta.minReaderVersion': '1', 'delta.minWriterVersion': '2', ...}

# Optimize table (Delta Lake)
source.optimize(zorder_by=["user_id", "created_at"])

# Vacuum old files
source.vacuum(retention_hours=168)  # 7 days default

# Time travel queries
old_data = source.time_travel_query(version=5)
# or
old_data = source.time_travel_query(timestamp="2024-01-01")

# Restore table to previous version
source.restore_table(version=5)
# or
source.restore_table(timestamp="2024-01-01")

# List catalogs, schemas, tables
catalogs = source.get_catalogs()
schemas = source.get_schemas(catalog="main")
tables = source.get_tables(schema="default", catalog="main")
```

### SQL Warehouse vs Cluster

```python
# SQL Warehouse (recommended for BI/analytics)
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",  # SQL warehouse
    access_token="dapi...",
)

# All-Purpose Cluster
source = DatabricksDataSource(
    table="users",
    host="adb-12345.azuredatabricks.net",
    http_path="/sql/protocolv1/o/0/0123-456789-abc123",  # Cluster
    access_token="dapi...",
)
```

## Validation Example

Using cloud warehouse sources with the validation API:

```python
import truthound as th
from truthound.datasources.sql import BigQueryDataSource

# Create source
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
    credentials_path="service-account.json",
)

# Run validation
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

## Converting to Polars

Fetch data as a Polars LazyFrame:

```python
source = BigQueryDataSource(
    table="users",
    project="my-gcp-project",
    dataset="my_dataset",
)

# Convert to LazyFrame
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

> **Warning**: `to_polars_lazyframe()` fetches all data into memory. Use sampling or `max_rows` for large tables.

## Best Practices

1. **Use service accounts** - Don't use personal credentials in production
2. **Set cost limits** - Configure `maximum_bytes_billed` for BigQuery
3. **Use environment variables** - Store credentials securely with `from_env()`
4. **Configure connection pools** - Adjust pool size for your workload
5. **Sample large tables** - Use `max_rows` or `sample()` before validation
6. **Use appropriate warehouses** - SQL Warehouses for analytics, clusters for ETL
