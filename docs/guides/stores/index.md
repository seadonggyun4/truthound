# Storage Backends Guide

This guide covers persisting validation results with Truthound's Python API. It includes practical workflows for configuring storage backends, combining enterprise features, and managing result lifecycle.

---

## Quick Start

```python
from truthound.stores import get_store
from truthound.stores.results import ValidationResult

# Create store
store = get_store("filesystem", base_path=".truthound/store")

# Save validation result
result = ValidationResult.from_report(report, "customers.csv")
run_id = store.save(result)

# Retrieve result
loaded = store.get(run_id)
```

---

## Common Workflows

### Workflow 1: Production Storage with S3

```python
from truthound.stores import get_store
from truthound.stores.results import ValidationResult
import truthound as th

# Configure S3 store
store = get_store(
    "s3",
    bucket="company-validation-results",
    prefix="prod/daily/",
    region="us-east-1",
)

# Run validation
report = th.check("data.csv")

# Save with metadata
result = ValidationResult.from_report(
    report,
    data_asset="data.csv",
    tags={"environment": "production", "team": "data-eng"},
)
run_id = store.save(result)
print(f"Saved as: {run_id}")
```

### Workflow 2: Combining Versioning + Caching + Observability

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore, CacheMode
from truthound.stores.caching.backends import LRUCache
from truthound.stores.observability import ObservableStore
from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    MetricsConfig,
)

# Layer 1: Base storage
base = get_store("s3", bucket="validation-results", prefix="prod/")

# Layer 2: Add version history (keep last 10 versions)
versioned = VersionedStore(base, VersioningConfig(max_versions=10))

# Layer 3: Add caching (LRU cache with 1-hour TTL)
cache = LRUCache(max_size=1000, ttl_seconds=3600)
cached = CachedStore(versioned, cache, mode=CacheMode.READ_WRITE)

# Layer 4: Add observability (audit logs + metrics)
store = ObservableStore(
    cached,
    ObservabilityConfig(
        audit=AuditConfig(enabled=True),
        metrics=MetricsConfig(enabled=True),
    ),
)

# Use the fully-featured store
result = store.save(validation_result)

# Version operations
history = versioned.get_history(run_id, limit=5)
versioned.rollback(run_id, version=3)
```

### Workflow 3: Query Historical Results

```python
from truthound.stores import get_store
from truthound.stores.base import StoreQuery
from datetime import datetime, timedelta

store = get_store("database", connection_url="postgresql://localhost/truthound")

# Query last 7 days of failures
query = StoreQuery(
    data_asset="customers.csv",
    start_time=datetime.now() - timedelta(days=7),
    status="failure",
    limit=100,
    order_by="run_time",
    ascending=False,
)

results = store.query(query)
for result in results:
    print(f"{result.run_time}: {result.issue_count} issues")
```

### Workflow 4: Retention Policy with Auto-Cleanup

```python
from truthound.stores import get_store
from truthound.stores.retention import (
    RetentionStore,
    TimeBasedPolicy,
    CountBasedPolicy,
    CompositePolicy,
)
from datetime import timedelta

# Base store
base = get_store("filesystem", base_path=".truthound/store")

# Define retention policies
policy = CompositePolicy([
    TimeBasedPolicy(max_age=timedelta(days=90)),  # Delete after 90 days
    CountBasedPolicy(max_count=1000),              # Keep max 1000 results
])

# Wrap with retention
store = RetentionStore(base, policy, cleanup_interval_hours=24)

# Results older than 90 days are automatically cleaned up
```

---

## Full Documentation

Truthound provides flexible storage backends for persisting validation results, schemas, and profiles. All stores implement a common interface with support for CRUD operations, querying, and optional features like versioning, caching, and replication.

## Backend Comparison

| Backend | Use Case | Persistence | Scalability | Dependencies |
|---------|----------|-------------|-------------|--------------|
| [FileSystem](filesystem.md) | Local development, single-node | File-based | Single machine | None |
| [Memory](filesystem.md#memory-store) | Testing, ephemeral | None | Single process | None |
| [S3](cloud-storage.md#s3-store) | AWS production | Cloud | Unlimited | `boto3` |
| [GCS](cloud-storage.md#gcs-store) | GCP production | Cloud | Unlimited | `google-cloud-storage` |
| [Azure Blob](cloud-storage.md#azure-blob-store) | Azure production | Cloud | Unlimited | `azure-storage-blob` |
| [Database](cloud-storage.md#database-store) | SQL-based storage | Database | Depends on DB | `sqlalchemy` |

## Installation

```bash
# Core (includes FileSystem and Memory stores)
pip install truthound

# Cloud storage backends
pip install truthound[s3]      # AWS S3
pip install truthound[gcs]     # Google Cloud Storage
pip install truthound[azure]   # Azure Blob Storage
pip install truthound[db]      # Database (SQLAlchemy)

# All storage backends
pip install truthound[all-stores]
```

## Quick Start

### Using the Factory

```python
from truthound.stores import get_store

# FileSystem (default)
store = get_store("filesystem", base_path=".truthound/store")

# S3
store = get_store(
    "s3",
    bucket="my-bucket",
    prefix="validations/",
    region="us-east-1",
)

# Database
store = get_store(
    "database",
    connection_url="postgresql://user:pass@localhost/db",
)
```

### Direct Instantiation

```python
from truthound.stores.backends.filesystem import FileSystemStore, FileSystemConfig
from truthound.stores.backends.s3 import S3Store

# FileSystem with config
config = FileSystemConfig(
    base_path=".truthound/store",
    use_compression=True,
    pretty_print=False,
)
store = FileSystemStore(config=config)

# S3 with direct parameters
store = S3Store(
    bucket="my-validation-results",
    prefix="prod/",
    region="us-west-2",
    compression=True,
)
```

## Common Operations

All stores implement the `ValidationStore` protocol:

```python
from truthound.stores.results import ValidationResult
from truthound.stores.base import StoreQuery

# Initialize store (lazy initialization)
store.initialize()

# Save a result
result = ValidationResult.from_report(report, "customers.csv")
run_id = store.save(result)

# Retrieve a result
result = store.get(run_id)

# Check existence
exists = store.exists(run_id)

# Delete a result
deleted = store.delete(run_id)

# List all IDs
ids = store.list_ids()

# Query with filters
query = StoreQuery(
    data_asset="customers.csv",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    status="failure",
    limit=100,
)
results = store.query(query)

# Close connection
store.close()
```

## StoreQuery Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_asset` | `str \| None` | Filter by data asset name |
| `start_time` | `datetime \| None` | Filter results after this time |
| `end_time` | `datetime \| None` | Filter results before this time |
| `status` | `str \| None` | Filter by status (`success`, `failure`, `error`) |
| `tags` | `dict[str, str] \| None` | Filter by tags |
| `limit` | `int \| None` | Maximum results to return |
| `offset` | `int` | Skip first N results (default: 0) |
| `order_by` | `str` | Sort field (default: `run_time`) |
| `ascending` | `bool` | Sort order (default: `False`) |

## Enterprise Features

Truthound stores support advanced features through wrapper stores:

| Feature | Module | Description |
|---------|--------|-------------|
| [Versioning](versioning.md) | `stores/versioning/` | Version history, diff, rollback |
| [Retention](retention.md) | `stores/retention/` | TTL policies, automatic cleanup |
| [Tiering](tiering.md) | `stores/tiering/` | Hot/Warm/Cold/Archive storage |
| [Caching](caching.md) | `stores/caching/` | LRU, LFU, TTL caches |
| [Replication](replication.md) | `stores/replication/` | Cross-region sync |
| [Observability](observability.md) | `stores/observability/` | Audit, metrics, tracing |

### Composing Features

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore, CacheConfig, CacheMode
from truthound.stores.observability import ObservableStore
from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    MetricsConfig,
    TracingConfig,
)

# Base store
base = get_store("s3", bucket="my-bucket", prefix="results/")

# Add versioning
versioned = VersionedStore(base, VersioningConfig(max_versions=10))

# Add caching
from truthound.stores.caching.backends import LRUCache
cache = LRUCache(max_size=1000, ttl_seconds=3600)
cached = CachedStore(versioned, cache, mode=CacheMode.READ_WRITE)

# Add observability
store = ObservableStore(
    cached,
    ObservabilityConfig(
        audit=AuditConfig(enabled=True),
        metrics=MetricsConfig(enabled=True),
        tracing=TracingConfig(enabled=True),
    ),
)
```

## Error Handling

All stores raise consistent exceptions:

```python
from truthound.stores.base import (
    StoreError,           # Base exception
    StoreNotFoundError,   # Item not found
    StoreConnectionError, # Connection failed
    StoreWriteError,      # Write operation failed
    StoreReadError,       # Read operation failed
)

try:
    result = store.get("nonexistent-id")
except StoreNotFoundError as e:
    print(f"Result not found: {e.identifier}")
except StoreConnectionError as e:
    print(f"Connection failed to {e.backend}: {e}")
```

## Configuration Reference

### Base Configuration

All stores extend `StoreConfig`:

```python
from truthound.stores.base import StoreConfig

config = StoreConfig(
    namespace="production",    # Logical grouping
    prefix="validations/",     # Path/key prefix
)
```

### Compression

Most stores support gzip compression:

```python
# FileSystem
FileSystemConfig(use_compression=True)

# S3
S3Config(use_compression=True)

# Azure
AzureBlobConfig(use_compression=True)
```

## Architecture

```
stores/
├── backends/               # Storage backend implementations
│   ├── filesystem.py       # Local filesystem
│   ├── memory.py           # In-memory (testing)
│   ├── s3.py               # AWS S3
│   ├── gcs.py              # Google Cloud Storage
│   ├── azure_blob.py       # Azure Blob Storage
│   └── database.py         # SQL databases
│
├── versioning/             # Result versioning
├── retention/              # TTL and retention policies
├── tiering/                # Hot/Warm/Cold/Archive
├── caching/                # Result caching layer
├── replication/            # Cross-region replication
├── observability/          # Audit, metrics, tracing
├── batching/               # Batch write optimization
└── backpressure/           # Flow control
```

## Next Steps

- [FileSystem Store](filesystem.md) - Local file storage
- [Cloud Storage](cloud-storage.md) - S3, GCS, Azure Blob
- [Versioning](versioning.md) - Version control for results
- [Caching](caching.md) - In-memory caching layer
