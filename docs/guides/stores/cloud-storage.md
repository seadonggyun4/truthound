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

### Conditional publication and recovery

`S3Store(config=S3Config(...))` captures the supplied settings, including nested
metadata and object tags. Individual constructor settings also accept JSON or
safe YAML (`serialization_format`) and a native compression name such as
`compression="zstd"`; `True` retains gzip and `False` disables it. Pickle and
unknown settings fail before provider work. Configured metadata supplies record
defaults, and each record's own metadata takes precedence. Object tags use URL
encoding for both names and values.

Initialization loads the index and validates the actual record inventory,
identities and metadata; new indices also retain SHA-256 and opaque ETag receipts.
Legacy indices without hashes remain readable when their records agree. Reads
close native response bodies. `max_object_bytes` (default 128 MiB) bounds encoded
objects/indexes and checks decoded record size; the codec may allocate before
the decoded-size check. Query errors propagate, `limit=0` is empty, and sorting
honors `run_time`, `data_asset` or `status`. Only a provider not-found response
means absence; access/connection failures remain errors.

Record and index writes use native conditional publication with confirmed ETags;
new keys use `IfNoneMatch="*"` and replacements use the prior `IfMatch` receipt.
Deletion uses the record's ETag and requires HTTP 204. A changed index blocks an
already stale owner before its next record write. A failed or unconfirmed write
fences further mutation. These are separate object operations: concurrent races
can leave unconfirmed objects and are not an atomic multi-object transaction or
an exclusive remote lease. Use one active writer per namespace and reconcile
before reusing uncertain state.

`rebuild_index()` is explicit and can run even when initialization rejects a
corrupt or inconsistent index. It validates every current selected-format record
before replacing the index conditionally. Malformed records remain errors and
are never silently omitted. Rebuilding adopts the current valid record inventory,
including unindexed records, and drops entries for missing records; inspect the
namespace before choosing this recovery. Closing releases the owned native
client, while a close failure retains its handle for an explicit retry.
Approved MCP S3 store factories use the same native paths, including
`rebuild_index`, and withhold provider keys, tags and private metadata from the
client projection. S3-compatible services must implement the required
[conditional write](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html)
and [delete](https://docs.aws.amazon.com/en_en/AmazonS3/latest/API/API_DeleteObject.html)
contracts. Current development evidence uses the actual botocore request model
and persistent synthetic objects through worker restart; live cloud services
remain separately unverified.

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

### Generation receipts and recovery

`GCSStore(config=GCSConfig(...))` captures settings and nested record metadata.
Individual settings accept `serialization_format="json"` or `"yaml"` and native
codec names such as `compression="zstd"`; `True` means gzip and `False` disables
compression. Unknown settings and pickle are rejected. Record metadata overrides
configured defaults. Connections remain lazy until an operation needs the SDK.

GCS uses the same Core record/index, query, checksum and explicit rebuild logic
as S3. New publications use `if_generation_match=0`; updates, index replacement
and deletion use the exact prior generation. Metadata is reloaded before media
reads, and `raw_download=True` plus a pinned generation preserves stored bytes:
Google's HTTP decoder and Core must not both decompress the same gzip object.
Missing generation receipts cannot commit an index and fence further mutations.
Read failures and corrupt indices remain errors; access denial is not absence.
`rebuild_index()` explicitly validates/adopts the current valid record inventory,
including orphaned records, before a conditional index replacement. Historical
`blob_name` index entries without hashes/generations remain readable when their
identities and metadata agree; corrupt records are never silently skipped.

`max_object_bytes` defaults to 128 MiB and checks encoded objects/indexes and
decoded records; it is not a codec heap sandbox. Query sorting and `limit=0` follow
the shared object-store contract. Close releases the actual native client and
retains a failed cleanup handle for explicit retry. Use one writer per namespace;
separate record/index operations can leave uncertain state and do not form a
multi-object transaction or exclusive remote lease. Reconcile before reuse.
Approved MCP factories expose the same native lifecycle/rebuild behavior, while
provider keys, project/credential details and private metadata stay out of the
client projection. Development evidence executes actual Google SDK JSON/media
serialization and gzip decoding against an owned persistent HTTP transport,
including installed-worker restart; it does not establish live GCS verification.
See Google's [generation preconditions](https://docs.cloud.google.com/python/docs/reference/storage/latest/generation_metageneration)
and [Blob API](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob).

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
| `access_tier` | `str \| None` | `None` | Hot, Cool, Cold, Archive |
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
    access_tier="Cool",  # Hot, Cool, Cold, Archive
)

# Change tier for existing result
store.set_access_tier("run-123", "Archive")
```

### ETag receipts, recovery and client ownership

Install `truthound[azure]` for both Blob Storage and `azure-identity`. Select one
connection source: connection string, account URL, or account name. A URL/name
without a key or SAS uses an owned `DefaultAzureCredential`; credential discovery
failures never fall back to anonymous access. Set `anonymous=True` explicitly for
public containers. Conflicting authentication settings are rejected.

`AzureBlobStore(config=AzureBlobConfig(...))` captures nested configuration.
`metadata` contains ASCII Azure blob headers; `record_metadata` contains defaults
for the persisted validation record. JSON and safe YAML support configured codecs
such as gzip and zstd. Reads disable the SDK's automatic gzip decoding so Core
restores stored bytes exactly once. `max_object_bytes` bounds stored and restored
record/index bytes; it is not a process heap limit.

Azure shares Core's S3/GCS inventory, record validation, SHA-256, query and index
rebuild logic. New blobs use `If-None-Match: *`; updates, reads and deletes use
confirmed ETags. Missing write receipts, concurrent replacement, unreadable indexed
records and corrupt indexes fence further writes. Result and index publication
are separate conditional operations, not a multi-blob transaction or remote lease.
Prefer one writer per namespace. After investigating an interrupted publication,
use `rebuild_index()` explicitly on a fresh owner; it validates every candidate
before publishing the new index. Legacy `blob_name` indexes remain readable when
the record inventory and metadata agree. Corruption never becomes an empty store.

`access_tier` accepts `Hot`, `Cool`, `Cold` and `Archive` and is supplied with the
native upload. Tier rejection fails the save. `set_access_tier(run_id, tier)`
requires a conditional native acknowledgement; `get_access_tier(run_id)` returns
observed `tier` and `archive_status`. An acknowledgement does not mean archive
rehydration has completed. See the [Azure BlobClient API](https://learn.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python).

Call `close()` to release both the owned SDK client and default credential. Cleanup
failure retains the failed handle and blocks new work until an explicit successful
close retry. Offline tests exercise the real SDK pipeline with synthetic service
responses; they do not establish live-account authorization or rehydration timing.

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

S3 and GCS also retain failed close handles and reject new IO until an explicit
close retry succeeds. An unconfirmed shutdown cannot be reused as a healthy client.
