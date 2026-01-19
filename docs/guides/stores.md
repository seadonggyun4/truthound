# Truthound Stores

This document provides comprehensive documentation for Truthound's storage backends, which persist validation results and expectations.

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Built-in Backends](#3-built-in-backends)
4. [Cloud Backends](#4-cloud-backends)
5. [Database Backend](#5-database-backend)
6. [Streaming Storage](#6-streaming-storage)
7. [Compression](#7-compression)
8. [Encryption](#8-encryption)
9. [Configuration Reference](#9-configuration-reference)
10. [Custom Backends](#10-custom-backends)
11. [Best Practices](#11-best-practices)

---

## 1. Overview

Truthound stores provide a unified interface for persisting validation results across different backends. This enables:

- **Result History**: Track validation results over time
- **Trend Analysis**: Compare results across runs
- **Audit Trail**: Maintain compliance records
- **Integration**: Connect with existing infrastructure (S3, GCS, databases)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Store Factory                           │
│              get_store(backend, **config)                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ FileSystem  │    │   Memory    │    │     S3      │
    │   Store     │    │   Store     │    │   Store     │
    └─────────────┘    └─────────────┘    └─────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    GCS      │    │  Database   │    │   Custom    │
    │   Store     │    │   Store     │    │   Store     │
    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 2. Quick Start

### Basic Usage

```python
from truthound.stores import get_store, ValidationResult
import truthound as th

# Create a store
store = get_store("filesystem", base_path=".truthound/results")
store.initialize()

# Run validation and save results
report = th.check("data.csv")
result = ValidationResult.from_report(report, "data.csv")
run_id = store.save(result)

# Retrieve results
retrieved = store.get(run_id)
print(f"Status: {retrieved.status}")

# List all runs
run_ids = store.list_ids()
```

### Available Backends

| Backend | Package Required | Use Case |
|---------|-----------------|----------|
| `filesystem` | (built-in) | Local development, single machine |
| `memory` | (built-in) | Testing, temporary storage |
| `s3` | `boto3` | AWS infrastructure, scalable storage |
| `gcs` | `google-cloud-storage` | GCP infrastructure |
| `database` | `sqlalchemy` | Queryable storage, existing DB infrastructure |

### Check Available Backends

```python
from truthound.stores.factory import list_available_backends, is_backend_available

# List all backends that can be used
print(list_available_backends())
# ['filesystem', 'memory', 's3']  # depends on installed packages

# Check specific backend
if is_backend_available("s3"):
    store = get_store("s3", bucket="my-bucket")
```

---

## 3. Built-in Backends

### 3.1 FileSystem Store

Local filesystem storage with JSON serialization.

```python
from truthound.stores import get_store

store = get_store(
    "filesystem",
    base_path=".truthound/results",  # Storage directory
    namespace="production",           # Organize by namespace
    use_compression=True,             # Gzip compression
)
store.initialize()
```

**File Structure**:
```
.truthound/results/
└── production/
    ├── index.json              # Metadata index
    ├── run_001.json.gz         # Compressed result
    ├── run_002.json.gz
    └── ...
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_path` | str | `.truthound/results` | Base directory for storage |
| `namespace` | str | `default` | Namespace for organizing results |
| `use_compression` | bool | `False` | Enable gzip compression |

### 3.2 Memory Store

In-memory storage for testing and temporary use.

```python
from truthound.stores import get_store

store = get_store("memory")
store.initialize()

# Data persists only during session
run_id = store.save(result)
retrieved = store.get(run_id)

# Clears on Python exit or explicit clear
store.clear()
```

**Use Cases**:
- Unit testing
- Temporary validation pipelines
- Development without file I/O

---

## 4. Cloud Backends

### 4.1 S3 Store

AWS S3 storage for scalable, durable result storage.

**Installation**:
```bash
pip install truthound[s3]
# or
pip install boto3
```

**Usage**:
```python
from truthound.stores import get_store

store = get_store(
    "s3",
    bucket="my-validation-results",
    prefix="truthound/",              # Key prefix
    namespace="production",
    compression=True,                 # Gzip compression
    region_name="us-east-1",          # AWS region
)
store.initialize()

# Save result
run_id = store.save(result)

# Results stored at: s3://my-validation-results/truthound/production/run_id.json.gz
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bucket` | str | (required) | S3 bucket name |
| `prefix` | str | `""` | Key prefix for all objects |
| `namespace` | str | `default` | Namespace subdirectory |
| `compression` | bool | `False` | Enable gzip compression |
| `region_name` | str | None | AWS region |

**AWS Credentials**:
The store uses standard boto3 credential resolution:
1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (for EC2/ECS/Lambda)

### 4.2 GCS Store

Google Cloud Storage backend.

**Installation**:
```bash
pip install truthound[gcs]
# or
pip install google-cloud-storage
```

**Usage**:
```python
from truthound.stores import get_store

store = get_store(
    "gcs",
    bucket="my-validation-results",
    prefix="truthound/",
    project="my-gcp-project",
    credentials_path="/path/to/service-account.json",  # Optional
    compression=True,
)
store.initialize()
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bucket` | str | (required) | GCS bucket name |
| `prefix` | str | `""` | Object prefix |
| `project` | str | None | GCP project ID |
| `credentials_path` | str | None | Path to service account JSON |
| `compression` | bool | `False` | Enable gzip compression |

**GCP Authentication**:
1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
2. Explicit `credentials_path` parameter
3. Default application credentials (gcloud auth)

---

## 5. Database Backend

SQL database storage using SQLAlchemy.

**Installation**:
```bash
pip install truthound[database]
# or
pip install sqlalchemy

# Plus database driver:
pip install psycopg2-binary  # PostgreSQL
pip install pymysql          # MySQL
pip install sqlite3          # SQLite (built-in)
```

**Usage**:
```python
from truthound.stores import get_store

# SQLite (simple, file-based)
store = get_store(
    "database",
    connection_url="sqlite:///validations.db",
)

# PostgreSQL
store = get_store(
    "database",
    connection_url="postgresql://user:pass@localhost:5432/validations",
    pool_size=10,
    echo=False,  # Set True for SQL debugging
)

# MySQL
store = get_store(
    "database",
    connection_url="mysql+pymysql://user:pass@localhost/validations",
)

store.initialize()  # Creates tables if needed
```

**Configuration**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `connection_url` | str | `sqlite:///.truthound/store.db` | SQLAlchemy connection URL |
| `namespace` | str | `default` | Namespace for filtering |
| `pool_size` | int | `5` | Connection pool size |
| `max_overflow` | int | `10` | Max overflow connections |
| `echo` | bool | `False` | Echo SQL statements |
| `create_tables` | bool | `True` | Auto-create tables |

**Database Schema**:
```sql
CREATE TABLE validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    data_asset VARCHAR(500) NOT NULL,
    run_time DATETIME NOT NULL,
    status VARCHAR(50) NOT NULL,
    namespace VARCHAR(255) DEFAULT 'default',
    tags_json TEXT,
    data_json TEXT NOT NULL,
    created_at DATETIME,
    updated_at DATETIME
);

-- Indexes for common queries
CREATE INDEX ix_run_id ON validation_results(run_id);
CREATE INDEX ix_data_asset ON validation_results(data_asset);
CREATE INDEX ix_run_time ON validation_results(run_time);
CREATE INDEX ix_status ON validation_results(status);
CREATE INDEX ix_namespace ON validation_results(namespace);
```

---

## 6. Streaming Storage

For large-scale validation results that cannot fit in memory, use streaming stores.

### 6.1 StreamingFileSystemStore

JSONL-based streaming with memory-efficient reads and writes.

```python
from truthound.stores.streaming import StreamingFileSystemStore

# Create streaming store
store = StreamingFileSystemStore(base_path=".truthound/streaming")

# Create a session for incremental writes
session = store.create_session("run_001", "large_dataset.csv")

# Write results incrementally (memory-efficient)
with store.create_writer(session) as writer:
    for batch in validation_batches:  # Millions of results
        for result in batch:
            writer.write_result(result)
        writer.checkpoint()  # Save progress for recovery

# Finalize the session
final_result = store.finalize_result(session)

# Read results back efficiently (streaming)
for result in store.iter_results("run_001"):
    process(result)
```

### 6.2 Key Features

| Feature | Description |
|---------|-------------|
| **Incremental Writes** | Write results without memory overflow |
| **Checkpointing** | Save progress for failure recovery |
| **Streaming Reads** | Chunk-based memory-efficient reads |
| **Compression** | gzip, zstd, lz4 compression support |
| **Async Support** | Async read/write APIs |

### 6.3 StreamingS3Store

Multipart upload-based S3 streaming for cloud environments.

```python
from truthound.stores.streaming import StreamingS3Store

store = StreamingS3Store(
    bucket="my-validation-results",
    prefix="streaming/",
    compression="gzip",
)

# Same API as StreamingFileSystemStore
session = store.create_session("run_001", "dataset.csv")
# ...
```

---

## 7. Compression

Enterprise-grade compression for validation results.

### 7.1 Supported Algorithms

| Algorithm | Package | Ratio | Speed | Use Case |
|-----------|---------|-------|-------|----------|
| gzip | (built-in) | Good | Moderate | General purpose |
| zstd | `zstandard` | Excellent | Fast | Large datasets |
| lz4 | `lz4` | Moderate | Very Fast | Real-time |
| snappy | `python-snappy` | Moderate | Fast | Streaming |
| brotli | `brotli` | Excellent | Slow | Archival |
| lzma | (built-in) | Best | Slowest | Maximum compression |

### 7.2 Basic Usage

```python
from truthound.stores.compression import (
    get_compressor,
    CompressionAlgorithm,
    CompressionLevel,
)

# Simple compression
compressor = get_compressor(CompressionAlgorithm.GZIP, level=CompressionLevel.HIGH)
compressed = compressor.compress(data)
original = compressor.decompress(compressed)
```

### 7.3 Adaptive Compression

Automatically selects the best algorithm based on data characteristics:

```python
from truthound.stores.compression import AdaptiveCompressor, CompressionGoal

adaptive = AdaptiveCompressor(goal=CompressionGoal.BALANCED)
result = adaptive.compress_with_metrics(data)

print(f"Selected: {result.metrics.algorithm}")
print(f"Ratio: {result.metrics.compression_ratio:.2f}x")
print(f"Time: {result.metrics.compression_time_ms:.1f}ms")
```

### 7.4 Compression Pipeline

Chain transforms with compression:

```python
from truthound.stores.compression import (
    CompressionPipeline,
    DeduplicationTransform,
    get_compressor,
    CompressionAlgorithm,
)

pipeline = (
    CompressionPipeline("json_optimized")
    .add_transform(DeduplicationTransform(block_size=1024))
    .add_compression(get_compressor(CompressionAlgorithm.ZSTD))
    .add_checksum()
)

result = pipeline.process(json_data)
original = pipeline.reverse(result.data)
```

---

## 8. Encryption

Enterprise-grade encryption for protecting sensitive validation results.

### 8.1 Overview

Truthound provides comprehensive encryption capabilities:

- **AEAD Algorithms**: AES-256-GCM, ChaCha20-Poly1305, XChaCha20-Poly1305
- **Key Derivation**: Argon2id, PBKDF2, scrypt, HKDF
- **Key Management**: Storage backends, rotation, expiration
- **Streaming Encryption**: Memory-efficient large file handling
- **Envelope Encryption**: DEK/KEK pattern for enterprise use

### 8.2 Quick Start

```python
from truthound.stores.encryption import (
    get_encryptor,
    generate_key,
    EncryptionAlgorithm,
)

# Generate a key
key = generate_key(EncryptionAlgorithm.AES_256_GCM)

# Encrypt data
encryptor = get_encryptor("aes-256-gcm")
encrypted = encryptor.encrypt(b"sensitive validation results", key)

# Decrypt data
decrypted = encryptor.decrypt(encrypted, key)
```

### 8.3 Supported Algorithms

| Algorithm | Key Size | Nonce Size | Use Case |
|-----------|----------|------------|----------|
| AES-128-GCM | 128-bit | 96-bit | Lightweight encryption |
| AES-256-GCM | 256-bit | 96-bit | Default, widely supported |
| ChaCha20-Poly1305 | 256-bit | 96-bit | Fast on CPUs without AES-NI |
| XChaCha20-Poly1305 | 256-bit | 192-bit | Extended nonce for high-volume |
| Fernet | 128-bit | - | Simple symmetric encryption |

### 8.4 Password-Based Encryption

Derive encryption keys from passwords using secure KDFs:

```python
from truthound.stores.encryption import derive_key, KeyDerivation

# Derive key from password (Argon2id recommended)
key, salt = derive_key(
    "my_secure_password",
    kdf=KeyDerivation.ARGON2ID,
)

# Store salt alongside encrypted data for decryption
# Later, derive the same key:
key2, _ = derive_key("my_secure_password", salt=salt, kdf=KeyDerivation.ARGON2ID)
```

**Supported KDFs:**

| KDF | Security | Speed | Use Case |
|-----|----------|-------|----------|
| Argon2id | Highest | Slow | Password hashing (recommended) |
| scrypt | High | Slow | Memory-hard derivation |
| PBKDF2-SHA256 | Good | Fast | Compatibility |
| HKDF-SHA256 | N/A | Fast | Key expansion (not for passwords) |

### 8.5 Encryption Pipeline

Combine compression and encryption (recommended pattern):

```python
from truthound.stores.encryption import (
    create_secure_pipeline,
    create_fast_pipeline,
    create_password_pipeline,
    generate_key,
    EncryptionAlgorithm,
)

# Secure pipeline: compress → encrypt → checksum
key = generate_key(EncryptionAlgorithm.AES_256_GCM)
pipeline = create_secure_pipeline(key, compression="gzip")

# Process data
result = pipeline.process(sensitive_data)
print(f"Original: {len(sensitive_data)} bytes")
print(f"Encrypted: {len(result.data)} bytes")
print(f"Compression ratio: {result.metrics.compression_ratio:.2f}x")

# Reverse the pipeline
original = pipeline.reverse(result.data, result.header)
assert original == sensitive_data

# Fast pipeline: uses ChaCha20-Poly1305 for better performance
fast_pipeline = create_fast_pipeline(key)

# Password-based pipeline
password_pipeline, salt = create_password_pipeline("my_password")
# Store `salt` for later decryption
```

### 8.6 Custom Pipeline Configuration

Build pipelines with custom stages:

```python
from truthound.stores.encryption import (
    EncryptionPipeline,
    EncryptionAlgorithm,
)

pipeline = (
    EncryptionPipeline()
    .add_compression("zstd")  # Compress first
    .add_encryption(EncryptionAlgorithm.AES_256_GCM, key=key)
    .add_checksum("sha256")  # Integrity verification
)

# Fluent API for configuration
result = pipeline.process(data)
serialized = pipeline.process_to_bytes(data)  # With header
original = pipeline.reverse_from_bytes(serialized)
```

### 8.7 Key Management

Manage encryption keys with built-in key stores:

```python
from truthound.stores.encryption import (
    KeyManager,
    KeyManagerConfig,
    FileKeyStore,
    InMemoryKeyStore,
    EnvironmentKeyStore,
)

# File-based key store (encrypted with master password)
store = FileKeyStore(
    path=".keys",
    master_password="master_secret_password",
)

# Create key manager
manager = KeyManager(
    store=store,
    config=KeyManagerConfig(
        default_algorithm=EncryptionAlgorithm.AES_256_GCM,
        auto_rotate_days=90,  # Auto-rotate after 90 days
    ),
)

# Create a new key
key = manager.create_key(key_id="validation_key_2024")

# Retrieve key
key = manager.get_key("validation_key_2024")

# Rotate key (creates new version)
new_key = manager.rotate_key("validation_key_2024")

# List all keys
keys = manager.list_keys()

# Delete key (secure wipe)
manager.delete_key("old_key")
```

**Available Key Stores:**

| Store | Use Case |
|-------|----------|
| `InMemoryKeyStore` | Testing, temporary keys |
| `FileKeyStore` | Local development, small deployments |
| `EnvironmentKeyStore` | Container environments, CI/CD |

### 8.8 Envelope Encryption

For large datasets, use envelope encryption (DEK/KEK pattern):

```python
from truthound.stores.encryption import EnvelopeEncryption, generate_key

# Master key (KEK - Key Encryption Key)
master_key = generate_key(EncryptionAlgorithm.AES_256_GCM)

# Create envelope encryption
envelope = EnvelopeEncryption(master_key)

# Encrypt data (generates random DEK, encrypts data, wraps DEK)
encrypted_data, wrapped_dek = envelope.encrypt(large_dataset)

# Store encrypted_data and wrapped_dek
# ...

# Decrypt data
original = envelope.decrypt(encrypted_data, wrapped_dek)
```

**Benefits:**
- DEK (Data Encryption Key) is random per encryption
- Only master key needs secure storage
- Key rotation only re-wraps DEKs
- Supports key hierarchy (multiple master keys)

### 8.9 Streaming Encryption

For large files that don't fit in memory:

```python
from truthound.stores.encryption import (
    StreamingEncryptor,
    StreamingDecryptor,
    generate_key,
    EncryptionAlgorithm,
)

key = generate_key(EncryptionAlgorithm.AES_256_GCM)

# Encrypt large file
with StreamingEncryptor(key, "output.enc") as enc:
    for chunk in read_large_file("input.bin", chunk_size=1024*1024):
        enc.write(chunk)

# Decrypt
with StreamingDecryptor(key, "output.enc") as dec:
    for chunk in dec:
        process_chunk(chunk)
```

### 8.10 Chunked Encryption (Random Access)

For files requiring random access to encrypted content:

```python
from truthound.stores.encryption import (
    ChunkedEncryptor,
    ChunkedDecryptor,
    generate_key,
    EncryptionAlgorithm,
)

key = generate_key(EncryptionAlgorithm.AES_256_GCM)

# Encrypt with chunk index
encryptor = ChunkedEncryptor(key, chunk_size=64*1024)
encrypted_data, chunk_index = encryptor.encrypt(large_data)

# Save chunk_index for random access
index_bytes = chunk_index.to_bytes()

# Decrypt specific chunk (random access)
decryptor = ChunkedDecryptor(key)
chunk_5 = decryptor.decrypt_chunk(encrypted_data, chunk_index, chunk_index=5)

# Decrypt range of chunks
chunks_10_to_20 = decryptor.decrypt_range(encrypted_data, chunk_index, start=10, end=20)

# Decrypt all
all_data = decryptor.decrypt_all(encrypted_data, chunk_index)
```

### 8.11 Metrics and Monitoring

Track encryption performance:

```python
from truthound.stores.encryption import EncryptionStats

stats = EncryptionStats()

# Record metrics from operations
encryptor = get_encryptor("aes-256-gcm")
result = encryptor.encrypt(data, key)
stats.record(result.metrics)

# View statistics
print(f"Total operations: {stats.total_operations}")
print(f"Total plaintext bytes: {stats.total_plaintext_bytes}")
print(f"Average throughput: {stats.average_throughput_mbps:.2f} MB/s")
print(f"Algorithm usage: {stats.algorithm_usage}")
```

### 8.12 Security Best Practices

1. **Always use AEAD**: All supported algorithms provide authenticated encryption
2. **Never reuse nonces**: The library handles this automatically
3. **Compress before encrypting**: Use `create_secure_pipeline()` for optimal security
4. **Use strong KDFs for passwords**: Prefer Argon2id over PBKDF2
5. **Rotate keys regularly**: Use `KeyManager.rotate_key()`
6. **Secure key storage**: Use `FileKeyStore` with strong master password
7. **Clear sensitive data**: Call `key.clear()` when done with keys

---

## 9. Configuration Reference

### Common Store Methods

All stores implement the same interface:

```python
class BaseStore(Generic[T, C], ABC):
    # Lifecycle
    def initialize(self) -> None: ...
    def close(self) -> None: ...

    # CRUD operations
    def save(self, item: ValidationResult) -> str: ...
    def get(self, item_id: str) -> ValidationResult: ...
    def exists(self, item_id: str) -> bool: ...
    def delete(self, item_id: str) -> bool: ...

    # Query operations
    def list_ids(self, query: StoreQuery | None = None) -> list[str]: ...
    def query(self, query: StoreQuery) -> list[ValidationResult]: ...
    def count(self, query: StoreQuery | None = None) -> int: ...
```

### StoreQuery

Query validation results with filters:

```python
from truthound.stores.base import StoreQuery
from datetime import datetime, timedelta

# Query by data asset
query = StoreQuery(data_asset="customers.csv")
results = store.query(query)

# Query by status
query = StoreQuery(status="failure")
failed_results = store.query(query)

# Query by time range
query = StoreQuery(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    order_by="run_time",
    ascending=False,
    limit=10,
)
recent_results = store.query(query)

# Pagination
query = StoreQuery(limit=20, offset=40)  # Page 3 with 20 per page
page_3 = store.query(query)
```

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `data_asset` | str | Filter by data asset name |
| `status` | str | Filter by status ("success", "failure") |
| `start_time` | datetime | Results after this time |
| `end_time` | datetime | Results before this time |
| `order_by` | str | Field to sort by |
| `ascending` | bool | Sort direction |
| `limit` | int | Maximum results to return |
| `offset` | int | Skip first N results |

### ValidationResult

Structure of stored validation results:

```python
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)

result = ValidationResult(
    run_id="run_2024_001",
    run_time=datetime.now(),
    data_asset="customers.csv",
    status=ResultStatus.SUCCESS,
    results=[
        ValidatorResult(
            validator_name="null_check",
            success=True,
            column="email",
        ),
        ValidatorResult(
            validator_name="duplicate_check",
            success=False,
            column="id",
            issue_type="duplicates",
            count=15,
            severity="high",
            message="Found 15 duplicate values",
        ),
    ],
    statistics=ResultStatistics(
        total_validators=2,
        passed_validators=1,
        failed_validators=1,
        total_rows=10000,
        total_columns=20,
        total_issues=1,
        high_issues=1,
    ),
    tags={"environment": "production", "team": "data"},
)
```

---

## 10. Custom Backends

Create custom storage backends by implementing the `BaseStore` interface:

```python
from dataclasses import dataclass
from truthound.stores import register_store, BaseStore, StoreConfig
from truthound.stores.results import ValidationResult

@dataclass
class RedisConfig(StoreConfig):
    """Configuration for Redis store."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    key_prefix: str = "truthound:"
    ttl_seconds: int | None = None  # Optional expiration

@register_store("redis")
class RedisStore(BaseStore[ValidationResult, RedisConfig]):
    """Redis-based validation result store."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        namespace: str = "default",
        **kwargs,
    ):
        config = RedisConfig(
            host=host,
            port=port,
            namespace=namespace,
            **kwargs,
        )
        super().__init__(config)
        self._client = None

    @classmethod
    def _default_config(cls) -> RedisConfig:
        return RedisConfig()

    def _do_initialize(self) -> None:
        import redis
        self._client = redis.Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=self._config.password,
        )
        self._client.ping()  # Test connection

    def save(self, item: ValidationResult) -> str:
        self.initialize()
        key = f"{self._config.key_prefix}{item.run_id}"
        data = json.dumps(item.to_dict(), default=str)
        self._client.set(key, data, ex=self._config.ttl_seconds)
        return item.run_id

    def get(self, item_id: str) -> ValidationResult:
        self.initialize()
        key = f"{self._config.key_prefix}{item_id}"
        data = self._client.get(key)
        if not data:
            raise StoreNotFoundError("ValidationResult", item_id)
        return ValidationResult.from_dict(json.loads(data))

    def exists(self, item_id: str) -> bool:
        self.initialize()
        key = f"{self._config.key_prefix}{item_id}"
        return bool(self._client.exists(key))

    def delete(self, item_id: str) -> bool:
        self.initialize()
        key = f"{self._config.key_prefix}{item_id}"
        return self._client.delete(key) > 0

    def list_ids(self, query=None) -> list[str]:
        self.initialize()
        pattern = f"{self._config.key_prefix}*"
        keys = self._client.keys(pattern)
        prefix_len = len(self._config.key_prefix)
        return [k.decode()[prefix_len:] for k in keys]

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

# Usage
store = get_store("redis", host="localhost", port=6379)
```

---

## 11. Best Practices

### 11.1 Choosing a Backend

| Scenario | Recommended Backend |
|----------|---------------------|
| Local development | `filesystem` |
| Unit tests | `memory` |
| AWS infrastructure | `s3` |
| GCP infrastructure | `gcs` |
| Need SQL queries | `database` |
| Multi-region, high availability | `s3` or `gcs` |
| Existing PostgreSQL | `database` |

### 11.2 Namespace Organization

Use namespaces to organize results:

```python
# By environment
prod_store = get_store("s3", bucket="results", namespace="production")
staging_store = get_store("s3", bucket="results", namespace="staging")

# By team
team_a_store = get_store("filesystem", base_path="./results", namespace="team-a")
team_b_store = get_store("filesystem", base_path="./results", namespace="team-b")

# By data domain
finance_store = get_store("database", connection_url="...", namespace="finance")
marketing_store = get_store("database", connection_url="...", namespace="marketing")
```

### 11.3 Compression

Enable compression for cloud storage:

```python
# S3 with compression
store = get_store("s3", bucket="results", compression=True)

# Filesystem with compression
store = get_store("filesystem", base_path="./results", use_compression=True)
```

Benefits:
- 60-80% storage reduction for JSON data
- Lower storage costs
- Faster network transfer

### 11.4 Error Handling

```python
from truthound.stores.base import (
    StoreError,
    StoreNotFoundError,
    StoreConnectionError,
    StoreWriteError,
    StoreReadError,
)

try:
    store = get_store("s3", bucket="my-bucket")
    store.initialize()
except StoreConnectionError as e:
    print(f"Could not connect to storage: {e}")

try:
    result = store.get("non_existent_run")
except StoreNotFoundError:
    print("Result not found")

try:
    store.save(result)
except StoreWriteError as e:
    print(f"Failed to save: {e}")
```

### 11.5 Resource Cleanup

Always close stores when done:

```python
# Manual cleanup
store = get_store("database", connection_url="...")
try:
    store.initialize()
    # ... use store ...
finally:
    store.close()

# Context manager (recommended)
# Note: BaseStore supports context manager protocol
with get_store("database", connection_url="...") as store:
    store.initialize()
    # ... use store ...
# Automatically closed
```

---

## Summary

Truthound stores provide flexible, extensible storage for validation results:

- **5 built-in backends**: filesystem, memory, S3, GCS, database
- **Unified interface**: Same API across all backends
- **Rich queries**: Filter by asset, status, time range
- **Streaming storage**: Memory-efficient handling of large results
- **Enterprise compression**: 8 algorithms with adaptive selection
- **Enterprise encryption**: AES-256-GCM, ChaCha20-Poly1305, key management
- **Easy extension**: Register custom backends with `@register_store`

### Feature Matrix

| Feature | Module | Description |
|---------|--------|-------------|
| Streaming Storage | `truthound.stores.streaming` | Incremental writes, checkpointing |
| Compression | `truthound.stores.compression` | gzip, zstd, lz4, adaptive selection |
| Encryption | `truthound.stores.encryption` | AEAD, key derivation, pipelines |

For more information:
- [Architecture Overview](ARCHITECTURE.md)
- [Reporters Documentation](REPORTERS.md)
- [Main README](../README.md)
