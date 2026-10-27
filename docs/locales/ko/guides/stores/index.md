# Storage Backends Guide

실무 운영 가이드에서 Truthound, API, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

```python
import truthound as th
from truthound.stores import get_store
from truthound.stores.results import ValidationResult

# Create store
store = get_store("filesystem", base_path=".truthound/store")

# Run validation and persist the storage DTO
run = th.check("customers.csv")
stored_result = ValidationResult.from_report(run, "customers.csv")
run_id = store.save(stored_result)

# Retrieve result
loaded = store.get(run_id)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common 워크플로우s

### 워크플로우 1: Production Storage with S3

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
run = th.check("data.csv")

# Save with metadata
stored_result = ValidationResult.from_report(
    run,
    data_asset="data.csv",
    tags={"environment": "production", "team": "data-eng"},
)
run_id = store.save(stored_result)
print(f"Saved as: {run_id}")
```

### 워크플로우 2: Combining 버전 관리 + 캐싱 + 관측성

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
from truthound.stores.results import ValidationResult
import truthound as th

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

# Run validation and persist the storage DTO
run = th.check("data.csv")
stored_result = ValidationResult.from_report(run, data_asset="data.csv")
run_id = store.save(stored_result)

# Version operations
history = versioned.get_history(run_id, limit=5)
versioned.rollback(run_id, version=3)
```

### 워크플로우 3: Query Historical 결과

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

### 워크플로우 4: 보존 Policy with Auto-Cleanup

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Full Documentation

실무 운영 가이드에서 Truthound, CRUD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Backend Comparison

| 실무 운영 가이드에서 Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Persistence을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Scalability을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Dependencies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|----------|-------------|-------------|--------------|
| 실무 운영 가이드에서 FileSystem을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 파일-based | 실무 운영 가이드에서 Single을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Testing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Single을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 AWS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cloud을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unlimited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `boto3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 GCS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 GCP을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cloud을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unlimited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `google-cloud-storage`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Azure, Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Azure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cloud을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unlimited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `azure-storage-blob`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| [데이터베이스](cloud-storage.md#database-store) | 실무 운영 가이드에서 SQL, SQL-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 데이터베이스 | 실무 운영 가이드에서 Depends을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `sqlalchemy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 설치

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

## 빠른 시작

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

실무 운영 가이드에서 `ValidationStore`, ValidationStore을(를) 다루는 항목입니다:

```python
from truthound.stores.results import ValidationResult
from truthound.stores.base import StoreQuery
import truthound as th

# Initialize store (lazy initialization)
store.initialize()

# Save a result
run = th.check("customers.csv")
stored_result = ValidationResult.from_report(run, "customers.csv")
run_id = store.save(stored_result)

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 실무 운영 가이드에서 `data_asset`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Filter by data 자산 name |
| 실무 운영 가이드에서 `start_time`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `end_time`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `status`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `success`, `failure`, `error`, Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `tags`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `limit`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Maximum 결과 to return |
| 실무 운영 가이드에서 `offset`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Skip first N 결과 (default: 0) |
| 실무 운영 가이드에서 `order_by`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `run_time`, Sort을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ascending`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, Sort, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Enterprise Features

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Module을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|--------|-------------|
| [버전 관리](versioning.md) | 실무 운영 가이드에서 `stores/versioning/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| [보존](retention.md) | 실무 운영 가이드에서 `stores/retention/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 TTL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Tiering을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stores/tiering/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hot/Warm/Cold/Archive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| [캐싱](caching.md) | 실무 운영 가이드에서 `stores/caching/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 LRU, LFU, TTL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Replication을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stores/replication/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cross-region을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| [관측성](observability.md) | 실무 운영 가이드에서 `stores/observability/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 감사, metrics, tracing |

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

## 설정 레퍼런스

### Base 설정

실무 운영 가이드에서 `StoreConfig`, StoreConfig을(를) 다루는 항목입니다:

```python
from truthound.stores.base import StoreConfig

config = StoreConfig(
    namespace="production",    # Logical grouping
    prefix="validations/",     # Path/key prefix
)
```

### Compression

실무 운영 가이드에서 Most을(를) 다루는 항목입니다:

```python
# FileSystem
FileSystemConfig(use_compression=True)

# S3
S3Config(use_compression=True)

# Azure
AzureBlobConfig(use_compression=True)
```

## 아키텍처

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

## 다음 단계

- 실무 운영 가이드에서 FileSystem, Store, Local을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Cloud, Storage, GCS, Azure, Blob을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Versioning, Version을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Caching, In-memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.


## 제한 반복 조회와 MCP

`BaseStore.iter_query(query, batch_size)`는 양수 batch 크기를 요구하며 페이지 전체에 `query.limit`을 정확히 적용합니다. limit=0이면 반환하지 않습니다. `VersionedStore.config`와 `RetentionStore.config`는 내부 backend 설정을 제공합니다. opt-in [Core MCP 프로필](../mcp.md)은 운영자 factory의 backend 설정으로 승인된 네이티브 저장소 동작을 제공합니다.

### 네이티브 스키마 마이그레이션 복구

`MigratableStore`는 lazy·batch 변환과 재열기 후에도 버전을 `ValidationResult.metadata`에 보존합니다. 필수 backup·검증 실패는 원본 갱신을 차단합니다. lazy 저장 실패를 전파하고 query가 변환 실패 항목을 조용히 누락하지 않습니다. 생성한 원본 backup은 버전을 바꾸지 않고 읽으며 후속 migration에서 제외하고 backup ID 충돌 시 실패합니다. 상태 조회는 표본 실패를 명시합니다. `batch_size`·`max_retries`는 양수이고 현재 버전은 최소 지원 버전 이상이어야 합니다. 선택적인 신뢰된 `set_io_guard`는 callback·backend 효과 직전에 실행합니다. Core MCP는 승인된 저장소 binding으로 상태·record별 변환 경로·preview와 실제 migration을 제공하며 callback·경로는 운영자가 소유합니다. 여러 record 변환은 부분 저장될 수 있으며 저장소 전체 rollback을 보장하지 않습니다.

### 관측 저장소 수명주기

`ObservableStore.initialize()`는 내부 backend 설정으로 `BaseStore`를 통해 한 번 초기화합니다. save·query observer는 작업당 한 번 집계합니다. 독립적인 `StoreMetrics.record_save`·`record_query`는 기존 동작을 유지하고 외부 observer가 이미 집계하면 `record_operation=False`를 사용할 수 있습니다. backend close 실패 후에도 관측 shutdown을 실행합니다. audit·tracer·소유 metrics 서버 종료를 모두 시도하고 정지한 metrics HTTP 서버는 socket을 해제합니다. Core MCP는 승인된 저장소 binding으로 비공개 metric 별칭·네이티브 감사 결과 투영·flush를 지원하며 물리 ID·label·actor·오류 원문은 제외합니다. flush 완료만으로 원격 전달을 증명하지 않습니다.

### 동시 commit 실패

동시 파일 저장소 작업은 index 갱신까지 데이터 잠금을 유지합니다. index commit은 프로세스 잠금 안에서 디스크 상태를 갱신하고 충돌한 추가를 거부합니다. 실패한 index 쓰기를 성공으로 응답하지 않고 불확실한 소유자는 계속 사용할 수 없습니다. 손상된 index·WAL 증거를 유지하며 WAL 복구 시 기존 commit 항목을 보존합니다. 잠금 탐지는 실제 호환되지 않는 대기 순환과 일반 경합을 구분합니다.

`with store.batch() as batch`는 명시적인 `commit`이 필요하고 나머지 종료는 buffer를 rollback합니다. index 이전 실패는 변경한 파일을 복원하고 후속 action을 중단합니다. index commit이 불명확하거나 복원이 불완전하면 비공개 backup hash·manifest를 보관하고 인스턴스와 재열기를 격리합니다. 명시적 복구에는 `list_batch_recovery(limit=100,offset=0)`와 `reconcile_batch(recovery_id,resolution="rollback"|"keep_current",expected_fingerprint=...)`를 사용합니다. batch-owner·변경 파일 잠금 안에서 원본 hash·현재 파일 지문을 확인합니다. rollback은 원본 바이트를 복원하고 keep-current는 실제 부분 파일의 index만 맞춥니다. 복구·commit 확정 상태는 삭제 전에 `.batch-resolved-*`로 이동하여 정리 중 중단되어도 정상 데이터를 유지합니다. `cleanup_batch_recovery(recovery_id,expected_fingerprint=...)`가 남은 확정 증거를 정리합니다. 복구에는 실제 네이티브 프로세스 잠금이 필요하며 변경·손상·경로 이탈 증거는 거부합니다. preparing receipt는 모든 파일 효과에 앞서 저장하고 명시적으로 해결할 때 현재 파일을 유지합니다. 자동 multi-file crash 원자성이나 전원 장애 내구성 증명은 아닙니다. Core MCP는 승인된 저장소 별칭으로 제한된 preview·commit, 전체 lock 통계와 네이티브 index rebuild를 제공합니다.

`pip install 'truthound[stores-concurrency]'`로 플랫폼 공통 `filelock`·`portalocker` 잠금을 설치합니다. MCP extra에도 기본 복구 provider용 filelock을 포함합니다. 네이티브 비차단·0 제한 시간을 무기한 대기로 바꾸지 않습니다. portalocker는 지원되는 lock signature와 동시 공유 reader별 독립 descriptor를 사용합니다. open 실패 시 로컬 소유권을 해제하고 fcntl 잠금 파일의 inode를 인계 후에도 유지합니다. 다른 프로세스가 대기할 수 있는 동안 잠금 파일을 삭제하지 않아야 합니다. 실제 로컬 프로세스로 provider·중단 계약을 실행하며 전체 Windows 검증은 최종 CI gate에 남아 있습니다.


### 네이티브 스트리밍 저장소

`StreamingFileSystemStore`, `StreamingDatabaseStore`, `StreamingS3Store`는 `BaseStore`와 별개의 스트리밍 검증 저장소입니다. 저장소가 직접 생성한 writer·reader도 소유하며 하나의 종료가 실패해도 나머지 정리를 시도합니다. 실패한 세션을 완료로 덮어쓰지 않습니다. 쓰기 실패 시 버퍼를 보존하고 존재하지 않는 청크를 성공 처리하지 않습니다. backend의 명시적 멱등 쓰기 구현이 있어야 재시도하며, 불명확한 append를 자동 재실행하지 않습니다. 체크포인트는 기본 스트림 설정과 datetime 지표를 보존하며 backend 주소·자격증명은 생성자에서만 공급합니다.

파일 manifest는 임시 파일 flush 후 교체하며 청크는 기존 증거를 덮어쓰지 않고 게시합니다. run 식별자와 읽기 경로는 선택한 디렉터리 내부로 제한합니다. 재개 전 미등록·누락·손상·불일치 청크를 거부합니다. manifest 누락·손상을 추정 레코드 수로 대신하거나 조용히 삭제하지 않고 조사할 수 있게 보존합니다. 전체 스트림 원자성이나 전원 손실 내구성을 보장하지 않습니다. SQLite는 청크 commit 후 별도 트랜잭션으로 세션을 저장하므로 사이에 중단되면 재개 시 불일치를 감지하여 명시적 조정을 요구합니다. 새 DB와 재열기 DB 모두 run당 하나의 세션을 강제합니다. CSV 청크에는 JSON 타입·중첩 값 복원을 위한 `__truthound_record_v1__` payload 열이 추가되며 이름 충돌 시 접미사를 붙입니다. 예약된 payload marker가 없는 기존 일반 CSV는 문자열 값으로 읽습니다.

`finalize_result`는 저장소 소유 writer의 남은 데이터를 반영하며 실패·중단 세션을 거부합니다. issue count의 합계를 포함한 영속화용 `ValidationResult`를 생성합니다. 최종 집계는 결과 목록을 메모리에 구성하므로 청크 단위 소비에는 reader를 사용합니다. 파일·실제 로컬 SQLite 경로는 해당 범위의 테스트로 검증했고, S3는 로컬 대역 클라이언트의 프로토콜 검사이며 실제 클라우드 검증이 아닙니다.

스트리밍 선택 코덱과 이식 가능한 파일 잠금은 `pip install 'truthound[stores-streaming]'`으로 설치합니다. Parquet는 Core의 Polars writer로 독립 파일을 생성하며 `__truthound_record_v1__` JSON payload 열로 빈·서로 다른 중첩 구조도 보존합니다. Snappy는 기존 Core compression provider를 재사용합니다. `BufferedStreamWriter`는 seek 가능한 binary 출력에서 하나의 문서를 유지하며 CSV·Parquet·압축 프레임을 단순 연결하지 않습니다. 전체 출력은 메모리 한도를 적용하며 외부 출력의 소유권은 호출자에게 남습니다. `get_output`은 메모리 출력에만 사용할 수 있습니다. `writer.abort()`는 버퍼를 flush하지 않으며 이미 시작된 I/O를 되돌리지 않습니다. 파일과 실제 파일 SQLite writer의 소유권은 네이티브 프로세스 잠금으로 관리하며 전원 손실·원격 서비스 fencing 보장을 뜻하지 않습니다.

파일 저장소의 `recovery_status(run_id)`는 실제 writer가 없는 상태에서 manifest·청크의 현재 bytes를 해시합니다. `reconcile_stream(run_id, resolution="adopt"|"discard", expected_fingerprint=...)`는 같은 네이티브 잠금 안에서 fingerprint를 다시 확인하고, 이미 확인된 prefix를 보존한 채 미확정 청크만 명시적으로 채택하거나 폐기합니다. 채택에는 연속 인덱스·설정된 코덱·실제 레코드 검증이 필요합니다. 결과는 paused이며 이후 별도 `resume_session`이 필요합니다. fingerprint가 바뀌거나 원래 manifest·확정 청크가 손상·누락되면 변경을 거부합니다. 여러 파일 폐기는 중간 실패할 수 있으므로 전체 복구 원자성으로 해석하지 않습니다. 미확정 청크를 자동 조정하지 않습니다. S3의 명시적 조정은 아직 제공하지 않습니다.

파일 SQLite도 동일한 `recovery_status`·`reconcile_stream` API를 제공합니다. 삭제와 세션 갱신을 `BEGIN IMMEDIATE` 트랜잭션으로 묶고 run 잠금을 유지합니다. 검사 중 session `max_memory_mb`를 넘으면 변경하지 않고 거부합니다. 새 DB 청크는 저장된 레코드의 canonical JSON에 `db-json-sha256:` 체크섬을 기록하므로 레코드 수가 같은 변조도 읽기·재개 전에 거부합니다. 이전 형식의 DB 청크는 기존 개수 검사만 가능하며 새 체크섬을 자동 생성해 과거 무결성을 주장하지 않습니다. 원격 DB·S3의 fencing·복구 동등성은 미검증입니다.

S3 스트리밍은 조건부 요청을 지원하는 SDK·provider가 필요합니다. 새 청크는 `IfNoneMatch="*"`, manifest는 보존한 opaque ETag의 `IfMatch`로 게시하며 쓰기 전 `HeadObject`로 현재 receipt를 확인합니다. ETag receipt 누락·변경·청크 inventory 불일치는 성공으로 처리하지 않고 무조건 덮어쓰기로 우회하지 않습니다. multipart는 알고 있는 소유 upload만 abort하며 part·complete receipt 누락 시 레코드를 승인하지 않습니다. 실행과 abort가 모두 실패하면 두 오류를 보존하고 나머지 소유 upload 정리를 시도합니다. 이 동작은 로컬 프로토콜 대역 검사이며 실제 S3 검증·여러 객체의 원자성·원격 단독 lease 보장이 아닙니다. 요청 계약은 AWS [PutObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html), [HeadObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html), [CompleteMultipartUpload](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html)를 따릅니다.

메모리 SQLite는 소유 엔진의 단일 연결을 async executor thread에서도 공유하며, 저장소가 만든 writer·reader의 연결 작업을 직렬화합니다. 서로 다른 writer의 commit/rollback이 섞이지 않도록 하고 메모리 DB가 thread마다 분리되는 문제를 막습니다. 파일·SQLite 청크 삭제도 실제 writer 소유권을 확인합니다. 단일 청크의 직렬화·압축 bytes가 `max_memory_mb`를 넘으면 저장하기 전에 거부합니다. 디코더의 엄격한 최대 heap 사용을 보장하는 OS sandbox는 아닙니다.

타입이 있는 `iter_results`·`read_result`·`get_chunk`·최종 집계·청크 채택은 실제 bool verdict와 음수가 아닌 정수 count를 요구합니다. 기존 문자열 `"False"`를 truthy 값으로 집계하거나 문자열 count를 조용히 변환하지 않습니다. 잘못된 타입의 원본은 변경하지 않으며 일반 `reader.read`·`read_batch`로 원시 레코드를 조사할 수 있습니다.

잠금 해제에는 실제 acquire가 반환한 opaque `LockHandle` 인스턴스를 사용합니다. 이미 해제한 handle이나 다시 만든 복사본은 같은 경로·재사용된 descriptor의 새 소유자를 해제하지 못합니다. filelock은 thread 공유 context로 async executor thread의 종료를 지원하며, SDK unlock 실패 시 소유권을 보존하여 같은 자원의 정리를 재시도할 수 있게 합니다. 실제 로컬 filelock·portalocker 검사는 최종 Windows CI를 대신하지 않습니다.

스트리밍 설정은 실제 bool, 양의 정수 크기, 유한한 음수 아닌 재시도·timer 값을 검증합니다. 문자열 `"False"`가 checkpoint를 활성화하지 않습니다. MCP의 session·writer·reader·stats 결과는 과거 네이티브 오류 원문 대신 개수를 표시하며 원래 진단 상태는 운영자 소유 저장소에 보존합니다.

S3도 `recovery_status`·`reconcile_stream(adopt|discard)`를 제공합니다. 선택한 run의 manifest·객체 SHA/ETag 지문, 확인된 prefix·offset·codec·count를 재검사하고 채택한 세션은 paused로 돌려 별도 resume을 요구합니다. 폐기는 manifest에 조건부 진행 기록을 남긴 뒤 조사한 ETag의 객체만 조건부 삭제합니다. 부분 실패·응답 receipt 누락 시 일반 읽기·쓰기와 재개는 차단되며 새 지문 조사와 명시적 폐기가 필요합니다. 전체 `delete_chunks`도 마지막에 manifest를 제거하며 부분 실패는 `pending_resolution=delete`로 남아 새 승인/호출로 완료해야 합니다. S3 삭제 개수에는 해당 호출이 확인한 현재 객체와 manifest가 포함되며 과거 버전·delete marker를 영구 제거하지 않습니다. 조건부 헤더의 의미는 AWS [DeleteObject](https://docs.aws.amazon.com/AmazonS3/latest/API/API_DeleteObject.html)를 따릅니다. 여러 객체의 원자성이나 원격 단독 lease는 제공하지 않습니다. 읽는 동안 manifest receipt가 바뀌면 reader는 실패합니다.

`truthound[s3,stores-streaming]`는 조건부 API 모델을 지원하는 boto3 1.42.26+를 요구합니다. 실제 botocore 오프라인 모델 검사는 일반/multipart 쓰기 헤더를 확인하고, 로컬 합성 객체 프로토콜은 동시 업로드·부분 폐기·변경된 객체·작업 프로세스 재생성 복구를 확인합니다. 둘 다 실제 S3 서비스 검증이 아닙니다. Multipart는 설정된 `max_concurrency`만큼만 요청을 진행하고 시작된 요청을 정리한 뒤 소유 upload를 abort합니다. SDK 네트워크 호출은 provider의 timeout 설정을 따릅니다.

이 미출시 checkout의 일반 `DatabaseStore` 조회·갱신은 namespace에 한정합니다. 새 테이블은 `(namespace, run_id)` 복합 고유 키를 사용합니다. 전역 run ID 고유 키를 가진 기존 테이블은 계속 읽을 수 있지만 namespace 간 충돌 시 기존 레코드를 옮기거나 테이블 스키마를 조용히 변경하지 않고 실패합니다. 이런 기존 스키마에서 같은 run ID를 함께 쓰려면 운영자가 별도 마이그레이션해야 합니다. exists/list/query/count의 SQL 오류와 손상된 결과 JSON은 false·빈 목록·0의 성공으로 바꾸지 않습니다. 태그는 offset/limit 전에 걸러 내고 정렬을 적용하며 limit 0은 빈 결과입니다. 데이터베이스 저장소 종료 시 초기화 상태를 정리해 파일 저장소가 자기 연결 풀을 다시 열 수 있습니다.

풀 지표는 초기 연결도 집계합니다. dispose는 모든 소유 Session과 engine의 정리를 시도하며 실패하면 소유권을 유지하고 새 Session을 차단하여 명시적으로 정리를 재시도할 수 있게 합니다. 초기 연결 실패 시 소유 engine을 정리합니다. 예상하지 못한 health-check 오류도 실패 지표에 반영하고 예외 원문은 기록하지 않습니다. 실제 로컬 SQLite 검사 범위이며 원격 SQL runtime은 별도 미검증입니다.

`DatabaseStore(..., table_prefix="tenant_")`는 `tenant_validation_results`와
각각 접두사를 붙인 독립 인덱스·제약을 선택합니다. 빈 접두사는 기본 테이블을 유지합니다.
접두사는 빈 문자열 또는 영문자·밑줄로 시작하는 24자 이하의 단순 ASCII 식별자입니다.
초기화는 선택한 테이블만 만들며 그 안에서도 namespace를 분리합니다. 기존 테이블의
이름이나 레코드를 자동 마이그레이션하지 않습니다. 풀·레거시 SQLite 모두 접두사
레코드를 종료·재열기 후 유지합니다. MCP에서는 팩토리가 이 설정을 소유하고 호출자가
테이블명을 입력하지 않습니다. SQL 초기화 실패는 고정 오류 메시지를 사용하고 부분
소유 연결을 정리하며 SQL·provider 상세나 초기화 성공 상태를 노출하지 않습니다.


## 네이티브 코덱·흐름 제어 유틸리티

`CompressionPipeline`은 네이티브 변환·압축·체크섬을 구성합니다.
`EncryptionPipeline`은 압축·암호화·체크섬을 구성하고 `THEP` 헤더를 직렬화합니다.
알고리즘·키·사용자 구성요소는 호출자 또는 승인 팩토리가 소유합니다.

```python
from truthound.stores.encryption.base import EncryptionAlgorithm, generate_key
from truthound.stores.encryption.pipeline import EncryptionPipeline

key = generate_key(EncryptionAlgorithm.AES_256_GCM)
pipeline = EncryptionPipeline().add_compression("gzip").add_encryption(key=key)
encoded = pipeline.process_to_bytes(b"synthetic-data", max_output_bytes=1024 * 1024)
assert pipeline.reverse_from_bytes(encoded, max_output_bytes=1024 * 1024) == b"synthetic-data"
```

암호화 복원은 단계 실행 전에 선택한 전체 단계 수·순서·종류·알고리즘·헤더 버전을
검증합니다. 잘리거나 잘못된 헤더는 `DecryptionError`이며 체크섬과 선언한 압축 해제
크기도 검사합니다. 헤더 CRC는 손상 탐지이며 임의 헤더 메타데이터의 인증 보장이
아닙니다. 입력 헤더에서 코드를 만들지 않고 선택한 네이티브 파이프라인·키를 사용합니다.
압축·암호화 파이프라인의 양의 `max_output_bytes`는 입력과 각 단계 반환 후에 검사합니다.
허용 결과 크기를 제한하며 디코더가 반환하기 전의 메모리 할당까지 제한하지 않습니다.
각 단계 앞뒤의 실행 guard는 철회 후 다음 단계를 차단합니다. 압축 파이프라인 오류
메시지에 provider 오류 원문을 넣지 않습니다.

네이티브 백프레셔는 지표와 실행 허가를 제어합니다. 복합 전략은 하위 실패를 제외하고
성공으로 바꾸지 않고 모은 오류를 발생시킵니다. boolean이 아닌 허가 결과는 잘못된
계약입니다. 취소 시 일시정지 표시를 해제합니다. 토큰 버킷 reset은 구성한 허가 수를
복원하며 명시적 용량은 유한하고 1 이상이어야 합니다. 이 기능은 로컬 흐름 제어이며
분산 잠금이나 외부 서비스 트랜잭션이 아닙니다.

미배포 MCP Core 프로파일은 비공개 `store_utilities` 바인딩으로
`store_utilities_open/execute`를 제공합니다. 승인된 설치 팩토리는
`StoreUtilityRuntime(compression={}, encryption={}, backpressure={}, payloads={},
slots={})`를 반환합니다. 공개 별칭으로 네이티브 파이프라인·백프레셔 전략·초기 bytes·
고정 절대 슬롯 경로를 선택합니다. 호출자는 팩토리 코드·provider·키·물리 경로를
wire 값으로 지정하지 않습니다. 구성요소 정리는 팩토리가 소유합니다.

명령은 `inspect`, `payload_capture`, `payload_import`, `payload_export`,
`payload_release`, `payload_status`, `codec_process`, `codec_reverse`, `slot_save`,
`slot_load`, `slot_delete`, `pressure_status`, `pressure_update`,
`pressure_acquire`, `pressure_release`, `pressure_reset`, `shutdown`입니다.
Capture는 고정 자산 또는 산출물 하나의 정확한 bytes를 읽습니다. 자산은 JSON으로
재구성하지 않고 캡처된 IPC bytes입니다. 코덱은 `codec_kind="compression"|"encryption"`,
`codec_id`와 소유자 로컬 `payload_id`를 선택합니다. 결과는 불투명 참조·크기·SHA256이며
export는 비공개 산출물에 태그된 binary를 반환합니다. Import는 이 export 형식만
받아 원본 산출물을 고정합니다. 기존 소유자의 파생 산출물을 import할 때는 다른
소유자를 사용해야 하며 resident 의존성 순환은 거부합니다.

소유자당 최대 32개·합계 16 MiB payload와 활성 자원 예산을 적용합니다. 파일은
materialize 전에 크기를 검사합니다. 단계 반환·보존 bytes 제한이며 OS 메모리
sandbox는 아닙니다. 암호화 출력은 네이티브 헤더·암호문을 포함하며 키는 제외합니다.
지표 응답은 provider 설정·사용자 단계 이름·오류 원문을 제외합니다. 평문 export를
포함한 각 동작에 정확한 승인과 민감정보 공개 허용을 요구합니다. 캡처 입력을 해제하면
소유자·파생 산출물도 철회합니다. 고정 슬롯은 별도로 남으며 새 승인으로 읽습니다.

슬롯 저장은 비공개 sibling 파일에 flush한 뒤 고정 대상만 교체합니다. 전체 작업의
전원 손실 내구성이나 안전한 소거를 보장하지 않습니다. 슬롯 삭제는 선택한 현재 파일을
제거합니다. 네이티브 흐름 제어 오류는 실패 `native_pressure_incomplete` 산출물과
상태로 남고 후속 acquire가 다시 실행하지 않습니다. 명시적 `pressure_reset`은 native
reset 성공 뒤에만 상태를 해제하며 이전 허가 소비를 되돌리지는 않습니다.
`acquired=false`는 유효한 거부 결정입니다. 승인 shutdown은 소유자·팩토리 정리 후에
응답합니다. 리소스 해제만으로 느린 finalizer 완료가 증명되지는 않습니다. 기본 payload·
흐름 상태는 메모리에 있으며 실제 외부 provider·원격 키 저장소는 별도 미검증입니다.
[MCP](../mcp.md)를 참조하십시오.


### 키 영속화와 승인된 키 사용

네이티브 `KeyManager.create_key`는 새 ID를 생성하며 기존 ID는 거부합니다. 현재 키
교체에는 명시적 rotation을 사용합니다. Rotation은 보관본을 먼저 보호한 뒤 새 버전을
게시합니다. `FileKeyStore`는 비공개 sibling 파일을 flush한 뒤 완성된 파일을 게시하며
새 ID 생성으로 기존 값을 덮어쓰지 않습니다. 교체 실패 시 이전 기록을 유지합니다.
손상·읽기 실패·ID 불일치는 고정 `EncryptionError`이며 `get_or_create_key`에서
키 없음으로 처리하지 않습니다. 명시적 TTL 0은 즉시 만료합니다. 만료 정리는 확인한
삭제만 집계하며 불명확 응답은 실패입니다. unlink 실패도 이전 파일을 보존합니다.
Unlink·Python 객체 초기화는 안전한 소거 보장이 아니며 보관본·현재 키 회전은 두 파일
전체의 트랜잭션이 아닙니다. 부분 실패는 무작정 rotation을 재시도하지 않고 운영자가
조사해야 합니다. 일반 확장 키 저장소의 동시성 범위는 해당 backend의 계약입니다.
`EncryptionKey` repr은 키 자료를 제외합니다. 명시적 네이티브 export는 운영자 코드에
남아 있지만 MCP에서는 제공하지 않습니다.

`FileKeyStore(master_key=...)`는 32-byte master key를 요구하며 master password는
저장한 salt·네이티브 KDF를 사용합니다. Master 없이 생성하는 기존 네이티브 방식은
비공개 파일에 평문 키를 저장하므로 암호화 backend를 명시적으로 선택해야 합니다.
키·암호·salt 파일·메타데이터·경로는 팩토리 소유이며 모델 입력이나 반환 키 메타데이터가
아닙니다.

`StoreUtilityRuntime.keys`는 별칭을 `StoreKeyBinding(manager, native_id)`에 연결합니다.
Manager는 네이티브 `KeyManager`입니다. `key_status`, `key_history`, `key_create`,
`key_get_or_create`, `key_rotate`, `key_delete`, `key_apply`, `key_cleanup_expired`는
각각 정확한 승인을 요구합니다. 출력은 공개 별칭·알고리즘·종류·버전·시간·만료·불명확
상태뿐이며 키 bytes·실제 ID·임의 메타데이터를 제외합니다. `key_history(limit=...)`는
해당 고정 ID와 네이티브 버전 보관본만 포함합니다. `key_version`은 조회·적용·삭제할
보관본/현재 버전을 선택합니다. 회전 기본값은 `archive_old=true`이며 명시적 false는
이전 키 보존을 바꿉니다. 만료 정리는 별칭 없는 소유 키를 포함한 팩토리 선택 manager
전체를 대상으로 하며 승인 효과에 이 범위가 표시됩니다.

`key_apply(key_id=..., codec_id=...)`는 바인딩된 암호화 파이프라인만 선택하고 암호
알고리즘 일치를 요구하며 키 자료를 반환하지 않습니다. 이후 코덱 실행도 적용한 키의
존재·유효기간·고정 버전/자료 일치를 검사합니다. 회전 후에는 명시적으로 다시 적용하고,
이전 암호문은 별도로 선택한 보관본을 사용할 수 있습니다. 키 감사 callback·네이티브
저장 작업은 팩토리 권한 guard 아래에서 실행합니다.

확인되지 않은 키 동작은 실패 `native_key_incomplete`이며 같은 manager의 소유자
로컬 별칭을 모두 차단합니다. 현재 상태 읽기는 허용하지만 변경·연결 코덱이 자동으로
반복하지 않습니다. 이 상태는 소유자 로컬이며 영구 알림 journal이 아닙니다. 새로
승인한 소유자를 열고 추가 작업을 선택하기 전 네이티브 backend를 조사해야 합니다.
설치형 로컬 검사는 합성 키와 실제 암호화 파일 저장소로 회전·보관본 복원·새 소유자
재열기·삭제를 검사합니다. 원격 키 서비스는 별도 미검증입니다.

### 스트리밍·청크 암호화의 인증

`StreamingEncryptor`는 THSE v2를 작성합니다. 각 네이티브 AEAD 레코드는 헤더,
레코드 종류·순서와 호출자의 associated data를 인증합니다. 마지막 빈 인증
레코드는 관측한 청크 수·평문 크기를 묶습니다.
`StreamingDecryptor.authenticated_complete`는 이 레코드와 EOF까지 검증한 뒤에만
true가 됩니다. 개별 청크 소비는 그 청크를 인증하며, 전체 스트림 결과를 확정하려면
iterator를 끝까지 소비해야 합니다. 완료 누락·헤더 변경·잘리거나 후행하는 레코드,
빈 스트림의 잘못된 키는 형식화된 오류가 됩니다. 짧은 read/write와 seek 불가능한
스트리밍 입력을 지원합니다. 평문 청크 크기는 1–16,777,216바이트이며 잘못된 설정은
출력 파일을 열기 전에 거부합니다.

쓰기 실패는 writer를 중단 상태로 고정합니다. context 내부 예외는 완료 레코드를
추가하지 않고 abort합니다. 이미 쓴 바이트를 지우거나 기존 파일의 원자적 교체를
보장하지는 않습니다. 호출자 소유 stream은 열어 두고, 네이티브 파일 핸들은 정상
종료·abort·초기화 실패 때 닫습니다. 소유자 guard는 네이티브 암호화·복호화 전과
후속 효과를 수락하기 전에 실행합니다.

`ChunkedEncryptor`의 v2 `ChunkIndex`는 canonical metadata 전체에 대한 AEAD
인증 태그를 포함합니다. `ChunkedDecryptor`는 빈 payload를 포함해 평문을 노출하기
전에 index를 복사·검증합니다. 연속 offset, nonce 도출, 크기·경계와 청크 인증을
확인합니다. `decrypt_all`은 payload 후행 바이트를 거부합니다. 선택 범위 조회는
index·선택한 청크를 인증하며 읽지 않은 청크를 검증하지 않습니다. 유효한 빈 범위는
빈 바이트를 반환하고 음수 청크 번호·잘못된 범위는 실패합니다.

새 reader는 v1 스트림·서명 없는 구형 청크 index를 기본 거부합니다. 원래 key/AAD를
사용하는 통제된 이전에만 `allow_legacy=True`를 명시해야 하며,
`authenticated_complete`·`authenticated_index`는 false로 유지됩니다. 구형 데이터의
완전성·metadata 인증을 증명할 수 없으므로 자동으로 인증됐다고 표시하지 않습니다.
암호문·index를 별도 파일에 쓰는 동작은 다중 파일 transaction이 아닙니다. 이 형식은
진행 중인 MCP 개발 변경으로 아직 게시되지 않았습니다.

`StoreUtilityRuntime`은 추가로 네이티브 `adaptive`·`chunked_compression`·
`chunked_encryption` 코덱 별칭을 바인딩합니다. `codec_process`·
`codec_reverse`에서 이 `codec_kind`를 선택합니다. 청크 암호화 bundle은 v2
인증 index와 암호문을 포함합니다. 청크 압축 bundle의 헤더·index·체크섬 검사는
손상을 찾으며 암호학적 인증은 아닙니다. `codec_range`는 두 청크 종류에서
`start`·`end` 바이트 위치를 받습니다. 복원 결과는 기존 보관 바이트 예산을
공유합니다. 네이티브 선언 크기를 전체 복원 전에 확인하고 반환 결과도 다시
검사하되 provider의 decoder heap 최대 할당을 제한하지는 않습니다.

`adaptive`의 `codec_analyze`는 네이티브 수치 특성·알고리즘별 표본 압축률·실패를,
`codec_recommend`는 알고리즘·점수·압축률·속도 필드를 반환합니다. 원시 표본,
provider 진단·자유 형식 추천 사유를 제외합니다. 팩토리의 `enable_sampling`·
`sample_size`·허용 알고리즘·fallback·level·분석 예산을 네이티브 선택에 적용합니다.
`max_analysis_time_ms`는 검사 지점에서 추가 probe를 멈추고
`analysis_budget_exhausted`를 기록하며 진행 중인 네이티브 probe를 중단하지는
못합니다. 크기·flags·알고리즘 번호·체크섬 손상과 iterator 청크 누락/추가는
오류입니다. 선택한 코덱과 bundle metadata가 일치해야 합니다. 증분 스트리밍
코덱 세션은 아래에 설명합니다.

### 증분 코덱 세션

팩토리는 `StoreCompressionStreamBinding(compressor_factory,
decompressor_factory)`으로 `compression_streams`,
`StoreEncryptionStreamBinding(encryptor_factory, decryptor_factory)`으로
`encryption_streams`를 바인딩합니다. 압축 팩토리는 인자 없이 새 네이티브
`StreamingCompressor`·`StreamingDecompressor`를 반환합니다. 암호화 팩토리는
소유된 binary buffer를 받아 운영자의 key/AAD/알고리즘으로 설정한 네이티브
`StreamingEncryptor`·`StreamingDecryptor`를 반환합니다. 코드·비밀 값은
바인딩이 선택하며 호출자는 별칭만 제공합니다.

`stream_open`은 `codec_id`·`codec_kind`(`compression` 또는 `encryption`)·
`direction`(`encode` 또는 `decode`)을 선택합니다. 암호화 decode는 암호문
`payload_id`도 선택합니다. `stream_write`는 보관 payload를 읽으며,
`stream_flush`·`stream_read`·`stream_finalize`·`stream_abort`·
`stream_status`·`stream_release`는 반환된 불투명 `stream_id`를 선택합니다.
기존 승인·민감 정보 공개 동의·부모 grant 수명주기를 유지합니다. 네이티브 stream은
최대 8개이며 미처리 입력·출력 buffer도 소유자의 바이트 예산에 포함됩니다.
암호화 출력 buffer는 예산을 넘는 쓰기를 거부하되 provider의 최대 할당은 이
한도 밖입니다.

압축 `write`·`flush` 결과는 독립적으로 완성된 네이티브 프레임입니다. payload를
반환하거나 아직 buffer 중이면 null입니다. 완성된 각 프레임을 decode stream의
`write`에 넣어야 하며 임의 transport 조각을 받는 네이티브 API는 아닙니다.
`stream_finalize`는 남은 프레임을 flush합니다. 암호화 encode는 최종 확정까지
출력을 보관한 뒤 완성된 THSE v2 payload를 반환합니다. Decode `stream_read`는
인증한 청크 하나를 반환하며 전체 완료를 주장하지 않습니다. `stream_finalize`는
남은 평문을 반환하고 최종 인증 레코드·EOF를 확인합니다. 남은 payload가 0바이트여도
유효합니다.

결과는 상태·유한 수치 metric·buffer 크기·현재 출력 receipt를 포함합니다.
`authenticated_complete`는 EOF를 확인한 암호화 decode에 적용됩니다. 네이티브
실패는 `native_codec_incomplete`로 반환하고 후속 처리를 차단하며 명시적
abort/release가 필요합니다. 확정된 handle에는 다시 쓸 수 없습니다. 선택된
네이티브 context 종료를 한 번 실행하며 불명확한 정리는 `cleanup_uncertain`으로
표시하고 자동 재시도하지 않습니다. Shutdown은 모든 stream 정리를 시도하고 확인된
종료만 승인합니다. 철회는 소유자·파생 산출물을 무효화합니다. 미완료 세션 종료는
미처리 바이트를 자동 확정하지 않고 abort합니다. 상태는 소유자 로컬이며 프로세스
손실 후 이어쓰기를 보장하지 않습니다.

네이티브 백프레셔 전략 생성은 유한하지 않은 시간·rate 값, 숫자 위치의 boolean,
잘못된 window와 0 이하 sampling interval을 거부합니다. `update_metrics`는 전체
관측 묶음을 변경 전에 검증합니다. 메모리는 유한한 0–100 비율, 지연은 유한한
0 이상 값, queue depth는 0 이상 정수여야 합니다. 거부된 관측 묶음은 기존
metric을 변경하지 않습니다.


### S3 multipart 소유 정보의 영구 복구

스트리밍 S3는 upload 생성 전에 조건부 manifest intent를 게시하고 part 제출
전에 정확한 upload receipt를 저장합니다. 생성 도중 worker가 종료되면 미해결
intent를, part 전송 도중 종료되면 소유 receipt를 유지합니다. intent가 남아 있는
동안 일반 open·resume·write·run 삭제와 chunk 복구를 막고 실패한 multipart 시도는 새
upload를 자동 시작하지 않습니다. intent가 없는 구형 manifest만으로 과거의
미완료 upload가 없다고 증명할 수는 없습니다.

`recovery_status(run_id)`는 chunk 증거와 함께 `multipart_uploads`·
`unresolved_upload_intents`를 반환합니다. 명시적
`reconcile_multipart(run_id, expected_fingerprint=..., receipts=...)`는 정확한
소유 upload를 abort하고 확인 후 각 journal 항목을 조건부 제거합니다. abort
receipt가 없으면 journal을 보존합니다. 새로 확인한 뒤 명시적으로 복구하면
`NoSuchUpload`를 이미 없어진 상태로 처리하며, 새 abort가 있었다고 보고하지
않습니다. 결과는 조정한 `resolved_uploads`와 HTTP 204로 확인한
`aborted_uploads`를 구분합니다. 새 운영자 receipt도 정리 전에 기록합니다.
생성 receipt가 없으면 운영자가 독립적으로 확인한 정확한 object key→upload ID
비공개 매핑을 제공해야 합니다. 이는 운영자의 소유 확인이며 Core는 bucket 목록에서
ID를 추측하거나 다른 upload를 abort하지 않습니다. 잘못된 확인으로 원래 upload의
상태를 증명할 수는 없습니다.

multipart 정리 후 불명확한 완료 chunk가 있으면 이를 확인·조정한 뒤 재개합니다.
이 경로는 네이티브 abort 확인을 사용하는 낙관적 object 프로토콜이며 독점 원격
lease나 진행 중 원격 요청·과금 종료를 보장하지 않습니다. 복구 전에 이전 소유자의
작업을 중지해야 합니다. 생성·part 단계 실제 프로세스 종료 검증은 재시작 후에도
남는 합성 object·upload로 수행했으며 botocore 모델·실서비스 검증은 별개입니다.
