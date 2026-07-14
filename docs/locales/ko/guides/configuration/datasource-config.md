# DataSource 설정

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
from truthound.datasources import get_datasource

# Auto-detect source type
ds = get_datasource(df)  # Polars/Pandas DataFrame
ds = get_datasource("data.parquet")  # File path
ds = get_datasource("postgresql://user:pass@localhost/db", table="users")
```

## DataSourceConfig

실무 운영 가이드에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `max_rows`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_memory_mb`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sample_seed`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Seed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `cache_schema`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 스키마 information |
| 실무 운영 가이드에서 `strict_types`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## SQL DataSource 설정

### SQLDataSourceConfig

실무 운영 가이드에서 SQL, Extended을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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
    materialization_row_limit=100_000,  # Polars fallback 최대 행 수
    use_server_side_cursor=False,   # Server-side cursors
    schema_name=None,               # Database schema
)
```

`max_rows`는 DataSource 전체 안전 한도이고, `materialization_row_limit`는
Python 메모리에서 Polars fallback을 수행할 때의 더 엄격한 한도입니다. fallback은
`fetch_size` batch로 `limit + 1`행까지 확인하므로 동시 데이터 증가를 감지하며,
일부 데이터가 전체 결과인 것처럼 성공 처리되지 않습니다.

| 설정 | 기본값 | 의미 |
|------|--------|------|
| `fetch_size` | 10,000 | fallback batch당 요청할 최대 행 수 |
| `materialization_row_limit` | 100,000 | 완전한 Polars fallback으로 허용할 최대 행 수 |
| `use_server_side_cursor` | False | provider별 server cursor 선호 설정 |

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

## 데이터베이스-Specific 설정

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `host`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 host |
| 실무 운영 가이드에서 `port`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PostgreSQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `database`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 name |
| 실무 운영 가이드에서 `user`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Username을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sslmode`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SSL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `application_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Application을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `schema_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Default 스키마 |

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `host`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 host |
| 실무 운영 가이드에서 `port`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 MySQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `database`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 name |
| 실무 운영 가이드에서 `user`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Username을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `charset`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Character을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `autocommit`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Auto-commit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

## Cloud Data 웨어하우스 설정

### Base Cloud 설정

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|---------|-------------|
| 실무 운영 가이드에서 `dataset`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `location`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `use_legacy_sql`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `maximum_bytes_billed`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cost을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `job_timeout`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 작업 timeout in seconds |

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

## In-Memory & 파일 DataSources

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

### 파일-Based Sources

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

### Dictionary 소스

```python
from truthound.datasources.polars_source import DictDataSource

ds = DictDataSource({
    "name": ["Alice", "Bob"],
    "age": [30, 25],
})
```

### Pandas 소스

```python
from truthound.datasources.pandas_source import PandasDataSource

ds = PandasDataSource(pandas_df)
```

### PySpark 소스

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

| 실무 운영 가이드에서 Capability을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------|
| 실무 운영 가이드에서 `LAZY_EVALUATION`, LAZY_EVALUATION을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 SQL, `SQL_PUSHDOWN`, SQL_PUSHDOWN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Can push operations to 데이터베이스 |
| 실무 운영 가이드에서 `SAMPLING`, SAMPLING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `STREAMING`, STREAMING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SCHEMA_INFERENCE`, SCHEMA_INFERENCE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Can infer 스키마 automatically |
| 실무 운영 가이드에서 `ROW_COUNT`, ROW_COUNT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Can을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 테이블 vs Query Mode

실무 운영 가이드에서 SQL을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 Cloud을(를) 다루는 항목입니다:

```python
# Check cost before execution
result = ds.execute_with_cost_check(
    query="SELECT * FROM large_table",
    max_bytes=10_000_000_000,      # 10GB limit
    max_cost_usd=1.0,              # $1 limit
)
```

## Type Mapping

실무 운영 가이드에서 Unified을(를) 다루는 항목입니다:

| 소스 Type | 실무 운영 가이드에서 ColumnType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|------------|
| 실무 운영 가이드에서 INT, INTEGER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 INTEGER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 FLOAT, DOUBLE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 FLOAT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VARCHAR, TEXT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 STRING을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 DATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 DATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 TIMESTAMP, DATETIME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 DATETIME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 BOOLEAN, BOOL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 BOOLEAN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 JSON, JSONB을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 ARRAY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 LIST을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STRUCT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 STRUCT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 환경 변수

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
