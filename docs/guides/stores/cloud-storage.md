# Cloud Storage Backends

Enterprise storage backends for distributed systems: AWS S3, Google Cloud Storage, Azure Blob, and SQL databases.

## Installation

```bash
# AWS S3
pip install truthound[s3]

# Google Cloud Storage
pip install truthound[gcs]

# Azure Blob Storage
pip install truthound[azure]

# SQL Database (SQLAlchemy)
pip install truthound[db]

# All cloud backends
pip install truthound[all-stores]
```

## S3 Store

AWS S3 backend using `boto3`.

### Basic Usage

```python
from truthound.stores.backends.s3 import S3Store

store = S3Store(
    bucket="my-validation-bucket",
    prefix="validations/",
    region="us-east-1",
)

# Save and retrieve
run_id = store.save(result)
result = store.get(run_id)
```

### Using the Factory

```python
from truthound.stores import get_store

store = get_store(
    "s3",
    bucket="my-bucket",
    prefix="validations/",
    region="us-east-1",
)
```

### Configuration

```python
from truthound.stores.backends.s3 import S3Config

config = S3Config(
    bucket="my-bucket",                    # S3 bucket name
    prefix="truthound/",                   # Key prefix
    namespace="default",                   # Namespace for organization
    region="us-east-1",                    # AWS region
    endpoint_url=None,                     # Custom endpoint (MinIO, LocalStack)
    use_compression=True,                  # Gzip compression
    storage_class="STANDARD",              # S3 storage class
    server_side_encryption="AES256",       # SSE setting
    kms_key_id=None,                       # KMS key for aws:kms
    tags={"env": "prod"},                  # Object tags
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bucket` | `str` | `""` | S3 bucket name (required) |
| `prefix` | `str` | `truthound/` | Key prefix for objects |
| `region` | `str \| None` | `None` | AWS region name |
| `endpoint_url` | `str \| None` | `None` | Custom endpoint URL |
| `use_compression` | `bool` | `True` | Compress with gzip |
| `storage_class` | `str` | `STANDARD` | S3 storage class |
| `server_side_encryption` | `str \| None` | `None` | SSE type (AES256, aws:kms) |
| `kms_key_id` | `str \| None` | `None` | KMS key ID |
| `tags` | `dict[str, str]` | `{}` | Object tags |

### S3-Compatible Services

```python
# MinIO
store = S3Store(
    bucket="my-bucket",
    endpoint_url="http://localhost:9000",
    region="us-east-1",
)

# LocalStack
store = S3Store(
    bucket="my-bucket",
    endpoint_url="http://localhost:4566",
    region="us-east-1",
)
```

### Server-Side Encryption

```python
# SSE-S3
store = S3Store(
    bucket="my-bucket",
    server_side_encryption="AES256",
)

# SSE-KMS
store = S3Store(
    bucket="my-bucket",
    server_side_encryption="aws:kms",
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/abc123",
)
```

## GCS Store

Google Cloud Storage backend using `google-cloud-storage`.

### Basic Usage

```python
from truthound.stores.backends.gcs import GCSStore

store = GCSStore(
    bucket="my-validation-bucket",
    prefix="validations/",
    project="my-gcp-project",
)

run_id = store.save(result)
result = store.get(run_id)
```

### Configuration

```python
from truthound.stores.backends.gcs import GCSConfig

config = GCSConfig(
    bucket="my-bucket",                    # GCS bucket name
    prefix="truthound/",                   # Object prefix
    namespace="default",                   # Namespace
    project="my-project",                  # GCP project ID
    credentials_path=None,                 # Service account JSON path
    use_compression=True,                  # Gzip compression
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bucket` | `str` | `""` | GCS bucket name (required) |
| `prefix` | `str` | `truthound/` | Object name prefix |
| `project` | `str \| None` | `None` | GCP project ID |
| `credentials_path` | `str \| None` | `None` | Service account JSON path |
| `use_compression` | `bool` | `True` | Compress with gzip |

### Authentication

```python
# Default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
store = GCSStore(bucket="my-bucket")

# Explicit service account
store = GCSStore(
    bucket="my-bucket",
    credentials_path="/path/to/service-account.json",
)

# With project ID
store = GCSStore(
    bucket="my-bucket",
    project="my-gcp-project",
)
```

## Azure Blob Store

Azure Blob Storage backend using `azure-storage-blob`.

### Basic Usage

```python
from truthound.stores.backends.azure_blob import AzureBlobStore

store = AzureBlobStore(
    container="my-container",
    connection_string="DefaultEndpointsProtocol=https;...",
)

run_id = store.save(result)
result = store.get(run_id)
```

### Configuration

```python
from truthound.stores.backends.azure_blob import AzureBlobConfig

config = AzureBlobConfig(
    container="my-container",              # Container name
    prefix="truthound/",                   # Blob name prefix
    namespace="default",                   # Namespace
    connection_string=None,                # Connection string
    account_url=None,                      # Account URL
    account_name=None,                     # Account name
    account_key=None,                      # Account key
    sas_token=None,                        # SAS token
    use_compression=True,                  # Gzip compression
    content_type="application/json",       # Content type
    access_tier="Hot",                     # Access tier
    metadata={"env": "prod"},              # Blob metadata
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `container` | `str` | `""` | Container name (required) |
| `prefix` | `str` | `truthound/` | Blob name prefix |
| `connection_string` | `str \| None` | `None` | Connection string |
| `account_url` | `str \| None` | `None` | Account URL |
| `account_name` | `str \| None` | `None` | Account name |
| `account_key` | `str \| None` | `None` | Account key |
| `sas_token` | `str \| None` | `None` | SAS token |
| `use_compression` | `bool` | `True` | Compress with gzip |
| `access_tier` | `str \| None` | `None` | Hot, Cool, Archive |
| `metadata` | `dict[str, str]` | `{}` | Blob metadata |

### Authentication Methods

```python
# Connection string (recommended for development)
store = AzureBlobStore(
    container="my-container",
    connection_string="DefaultEndpointsProtocol=https;...",
)

# Account URL with SAS token
store = AzureBlobStore(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
    sas_token="sv=2021-06-08&ss=b&srt=sco&sp=rwdlacx&...",
)

# Account name and key
store = AzureBlobStore(
    container="my-container",
    account_name="myaccount",
    account_key="...",
)

# Managed identity (DefaultAzureCredential)
store = AzureBlobStore(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
)
```

### Access Tiers

```python
store = AzureBlobStore(
    container="my-container",
    connection_string="...",
    access_tier="Cool",  # Hot, Cool, Archive
)

# Change tier for existing result
store.set_access_tier("run-123", "Archive")
```

## Database Store

SQL database backend using SQLAlchemy. Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.

### Basic Usage

```python
from truthound.stores.backends.database import DatabaseStore

# SQLite
store = DatabaseStore(
    connection_url="sqlite:///validations.db",
)

# PostgreSQL
store = DatabaseStore(
    connection_url="postgresql://user:pass@localhost/validations",
)

run_id = store.save(result)
result = store.get(run_id)
```

### Configuration

```python
from truthound.stores.backends.database import DatabaseConfig, PoolingConfig

config = DatabaseConfig(
    connection_url="postgresql://user:pass@localhost/db",
    namespace="default",                   # Namespace
    table_prefix="",                       # Table name prefix
    pool_size=5,                           # Connection pool size
    max_overflow=10,                       # Max overflow connections
    echo=False,                            # Echo SQL statements
    create_tables=True,                    # Auto-create tables
    use_pool_manager=True,                 # Use enterprise pool manager
    pooling=PoolingConfig(                 # Advanced pooling
        pool_size=5,
        max_overflow=10,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
        enable_circuit_breaker=True,
        enable_health_checks=True,
        enable_retry=True,
        max_retries=3,
        retry_base_delay=0.1,
    ),
)
```

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `connection_url` | `str` | `sqlite:///.truthound/store.db` | SQLAlchemy connection URL |
| `table_prefix` | `str` | `""` | Table name prefix |
| `pool_size` | `int` | `5` | Connection pool size |
| `max_overflow` | `int` | `10` | Max overflow connections |
| `echo` | `bool` | `False` | Echo SQL statements |
| `create_tables` | `bool` | `True` | Auto-create tables |
| `use_pool_manager` | `bool` | `True` | Use enterprise pool manager |

### Connection URLs

```python
# SQLite
DatabaseStore(connection_url="sqlite:///data.db")
DatabaseStore(connection_url="sqlite:///:memory:")

# PostgreSQL
DatabaseStore(connection_url="postgresql://user:pass@host:5432/db")
DatabaseStore(connection_url="postgresql+psycopg2://user:pass@host/db")

# MySQL
DatabaseStore(connection_url="mysql+pymysql://user:pass@host:3306/db")

# SQL Server
DatabaseStore(connection_url="mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17")
```

### Enterprise Pooling

The database store includes enterprise-grade connection pooling:

```python
from truthound.stores.backends.database import DatabaseStore, PoolingConfig

store = DatabaseStore(
    connection_url="postgresql://user:pass@localhost/db",
    pooling=PoolingConfig(
        pool_size=10,
        max_overflow=20,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
        enable_circuit_breaker=True,
        enable_health_checks=True,
        enable_retry=True,
        max_retries=3,
    ),
)

# Check health
if store.is_healthy:
    print("Database connection is healthy")

# Get pool metrics
metrics = store.pool_metrics
if metrics:
    print(f"Active connections: {metrics.active_connections}")

# Get comprehensive status
status = store.get_pool_status()

# Manual connection recycling
recycled = store.recycle_connections()
```

### Retry with Exponential Backoff

```python
from sqlalchemy import text

def insert_and_query(session):
    session.execute(text("INSERT INTO my_table ..."))
    return session.execute(text("SELECT * FROM my_table")).fetchall()

# Automatic retry on transient errors
result = store.execute_with_retry(insert_and_query)
```

## Common Features

### Compression

All cloud stores support gzip compression (enabled by default):

```python
# Enabled (default for cloud stores)
store = S3Store(bucket="my-bucket", compression=True)

# Disabled
store = S3Store(bucket="my-bucket", compression=False)
```

### Namespace Organization

```python
# Development namespace
dev_store = S3Store(bucket="my-bucket", namespace="development")

# Production namespace
prod_store = S3Store(bucket="my-bucket", namespace="production")
```

### Error Handling

```python
from truthound.stores.base import (
    StoreConnectionError,
    StoreNotFoundError,
    StoreReadError,
    StoreWriteError,
)

try:
    store.initialize()
except StoreConnectionError as e:
    print(f"Connection failed: {e}")

try:
    result = store.get("nonexistent")
except StoreNotFoundError as e:
    print(f"Not found: {e.item_id}")
```

## Backend Comparison

| Feature | S3 | GCS | Azure Blob | Database |
|---------|-----|-----|------------|----------|
| Compression | Yes | Yes | Yes | No |
| Server-side encryption | Yes | Yes | Yes | N/A |
| Access tiers | Yes | No | Yes | N/A |
| Custom metadata | Tags | No | Yes | Yes |
| Connection pooling | N/A | N/A | N/A | Yes |
| Circuit breaker | No | No | No | Yes |
| Retry with backoff | No | No | No | Yes |

## Next Steps

- [FileSystem Store](filesystem.md) - Local development
- [Versioning](versioning.md) - Version history for results
- [Tiering](tiering.md) - Hot/Warm/Cold/Archive storage
- [Caching](caching.md) - In-memory caching layer
