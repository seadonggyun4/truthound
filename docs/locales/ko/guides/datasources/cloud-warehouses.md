# Cloud Data 웨어하우스 Data Sources

## 설치와 실행 증거

| 플랫폼 | 드라이버 | 설치 |
|--------|----------|------|
| BigQuery | `google-cloud-bigquery` | `pip install truthound[bigquery]` |
| Snowflake | `snowflake-connector-python` | `pip install truthound[snowflake]` |
| Redshift | `redshift-connector` | `pip install truthound[redshift]` |
| Databricks | `databricks-sql-connector` | `pip install truthound[databricks]` |

Cloud warehouse도 관계형 SQL source와 같은 row 정규화 및 bounded Polars
fallback 계약을 사용합니다. 일부 결과를 전체 결과로 오인하지 않도록 제한을
넘으면 `DataSourceSizeError`를 발생시킵니다.

provider class import나 mock contract test만으로 실제 계정 지원을 선언할 수
없습니다. 실제 credential 기반 read, `th.check(source=...)`,
`th.profile(source=...)`, cleanup, 재진입 결과와 설치 artifact를 함께 기록해야
합니다. credential이 없으면 통과가 아니라 **미검증**입니다.

3.1.6부터 BigQuery, Snowflake, Redshift, Databricks 공개 class는 모두 concrete이며,
사용하지 않는 schema-query abstract method 대신 기존 provider-native schema 조회를
정식 전략으로 사용합니다. 이는 package 생성 계약이며 실제 credential 기반
provider 인증을 대신하지 않습니다.

실무 운영 가이드에서 Databricks, Truthound, Snowflake, Redshift, BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `CloudDWDataSource`, Cloud, CloudDWDataSource을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Credential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Cost-aware을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Cloud-native을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 플랫폼 | 실무 운영 가이드에서 Driver을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 설치 |
|----------|--------|--------------|
| 실무 운영 가이드에서 BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `google-cloud-bigquery`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install google-cloud-bigquery db-dtypes`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Snowflake을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `snowflake-connector-python`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install snowflake-connector-python`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Redshift을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `redshift-connector`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install redshift-connector`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Databricks을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `databricks-sql-connector`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install databricks-sql-connector`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Google BigQuery

실무 운영 가이드에서 BigQuery, Google, Cloud을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설치

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

### 설정

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

실무 운영 가이드에서 BigQuery을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 BigQuery, Create을(를) 다루는 항목입니다:

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

Snowflake is a cloud-native data 웨어하우스.

### 설치

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

#### 환경 변수

```python
# Reads: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
#        SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE
source = SnowflakeDataSource.from_env(
    table="USERS",
    schema="PUBLIC",
    env_prefix="SNOWFLAKE",
)
```

### 설정

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

실무 운영 가이드에서 Redshift, AWS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설치

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

#### 환경 변수

```python
# Reads: REDSHIFT_HOST, REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD
source = RedshiftDataSource.from_env(
    table="users",
    schema="public",
    env_prefix="REDSHIFT",
)
```

### 설정

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

실무 운영 가이드에서 Databricks, Delta, Lake을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설치

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

#### 환경 변수

```python
# Reads: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
source = DatabricksDataSource.from_env(
    table="users",
    schema="default",
    catalog="main",
    env_prefix="DATABRICKS",
)
```

### 설정

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

### SQL 웨어하우스 vs Cluster

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

## 검증 Example

실무 운영 가이드에서 API을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 Polars, Fetch, LazyFrame을(를) 다루는 항목입니다:

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

> 실무 운영 가이드에서 `to_polars_lazyframe()`, `max_rows`, Warning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 권장 방식

1. 실무 운영 가이드에서 Don을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 BigQuery, `maximum_bytes_billed`, Set, Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `from_env()`, Store을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Configure, Adjust을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 `max_rows`, `sample()`, Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 SQL, Warehouses, ETL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
