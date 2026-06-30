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
