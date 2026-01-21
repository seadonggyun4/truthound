# Store Configuration

Truthound provides multiple storage backends for validation results with enterprise features.

## Quick Start

```python
from truthound.stores import get_store

# Filesystem (default)
store = get_store("filesystem", base_path=".truthound/results")

# S3
store = get_store("s3", bucket="my-bucket", region="us-east-1")

# Database
store = get_store("database", connection_url="postgresql://localhost/db")
```

## StoreConfig

Base configuration for all stores.

```python
from truthound.stores.base import StoreConfig

config = StoreConfig(
    namespace="default",           # Logical grouping
    prefix="",                     # Path prefix
    serialization_format="json",   # json, yaml, pickle
    compression=None,              # gzip, lz4, zstd
    metadata={},                   # Custom metadata
)
```

## Backend Configurations

### FileSystem Store

```python
from truthound.stores.backends.filesystem import (
    FileSystemStore,
    FileSystemConfig,
)

config = FileSystemConfig(
    base_path=".truthound/store",  # Base directory
    file_extension=".json",        # File extension
    create_dirs=True,              # Auto-create directories
    pretty_print=True,             # Pretty JSON output
    use_compression=False,         # Enable gzip compression
)

store = FileSystemStore(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_path` | .truthound/store | Base storage directory |
| `file_extension` | .json | File extension |
| `create_dirs` | True | Auto-create directories |
| `pretty_print` | True | Human-readable JSON |
| `use_compression` | False | Enable gzip |

### S3 Store

```python
from truthound.stores.backends.s3 import S3Store, S3Config

config = S3Config(
    bucket="my-bucket",
    prefix="truthound/",
    region="us-east-1",
    endpoint_url=None,                  # Custom endpoint (MinIO, etc.)
    use_compression=True,
    storage_class="STANDARD",           # STANDARD, INTELLIGENT_TIERING, etc.
    server_side_encryption=None,        # AES256, aws:kms
    kms_key_id=None,                    # KMS key ARN
    tags={},                            # S3 object tags
)

store = S3Store(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bucket` | "" | S3 bucket name |
| `prefix` | truthound/ | Object key prefix |
| `region` | None | AWS region |
| `use_compression` | True | Compress objects |
| `storage_class` | STANDARD | S3 storage class |
| `server_side_encryption` | None | SSE type |

### GCS Store

```python
from truthound.stores.backends.gcs import GCSStore, GCSConfig

config = GCSConfig(
    bucket="my-bucket",
    prefix="truthound/",
    project=None,                       # GCP project ID
    credentials_path=None,              # Service account JSON path
    use_compression=True,
)

store = GCSStore(config)
```

### Azure Blob Store

```python
from truthound.stores.backends.azure_blob import (
    AzureBlobStore,
    AzureBlobConfig,
)

config = AzureBlobConfig(
    container="my-container",
    prefix="truthound/",
    # Authentication (choose one)
    connection_string=None,             # Full connection string
    account_url=None,                   # Account URL
    account_name=None,                  # Account name
    account_key=None,                   # Account key
    sas_token=None,                     # SAS token
    # Options
    use_compression=True,
    content_type="application/json",
    access_tier=None,                   # Hot, Cool, Archive
    metadata={},                        # Blob metadata
)

store = AzureBlobStore(config)
```

**Authentication Methods:**

```python
# Connection string (development)
config = AzureBlobConfig(
    container="my-container",
    connection_string="DefaultEndpointsProtocol=https;..."
)

# Account URL + SAS token
config = AzureBlobConfig(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
    sas_token="sv=2021-06-08&..."
)

# Managed identity (production)
config = AzureBlobConfig(
    container="my-container",
    account_url="https://myaccount.blob.core.windows.net",
    # Uses DefaultAzureCredential automatically
)
```

### Database Store

```python
from truthound.stores.backends.database import (
    DatabaseStore,
    DatabaseConfig,
    PoolingConfig,
)

config = DatabaseConfig(
    connection_url="postgresql://user:pass@localhost/db",
    table_prefix="",                    # Table name prefix
    pool_size=5,                        # Connection pool size
    max_overflow=10,                    # Additional connections
    echo=False,                         # SQLAlchemy echo
    create_tables=True,                 # Auto-create tables
    pooling=PoolingConfig(),
    use_pool_manager=True,              # Enterprise pool manager
)

store = DatabaseStore(config)
```

## Connection Pool Configuration

Advanced connection pooling for database stores.

```python
from truthound.stores.backends.connection_pool import (
    ConnectionPoolConfig,
    PoolConfig,
    PoolStrategy,
    RetryConfig,
    CircuitBreakerConfig,
    HealthCheckConfig,
)

# Pool settings
pool = PoolConfig(
    strategy=PoolStrategy.QUEUE_POOL,   # Pool strategy
    pool_size=5,                        # Max connections
    max_overflow=10,                    # Extra connections
    pool_timeout=30.0,                  # Acquire timeout
    pool_recycle=3600,                  # Recycle connections (1hr)
    pool_pre_ping=True,                 # Ping before use
    echo_pool=False,                    # Log pool events
    reset_on_return="rollback",         # Reset strategy
)

# Retry settings
retry = RetryConfig(
    max_retries=3,
    base_delay=0.1,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)

# Circuit breaker
circuit_breaker = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=3,
    timeout=60.0,
    half_open_max_calls=3,
)

# Health checks
health_check = HealthCheckConfig(
    enabled=True,
    interval=30.0,
    timeout=5.0,
    query="SELECT 1",
)

# Full configuration
config = ConnectionPoolConfig(
    connection_url="postgresql://localhost/db",
    pool=pool,
    retry=retry,
    circuit_breaker=circuit_breaker,
    health_check=health_check,
)
```

**Pool Strategies:**

| Strategy | Description |
|----------|-------------|
| `QUEUE_POOL` | Standard pool with overflow (default) |
| `NULL_POOL` | No pooling, new connection each time |
| `STATIC_POOL` | Single connection for all requests |
| `SINGLETON_THREAD` | One connection per thread |
| `ASYNC_QUEUE` | Async-compatible queue pool |

## Versioning Configuration

Enable version control for stored results.

```python
from truthound.stores.versioning.base import (
    VersioningConfig,
    VersioningMode,
)

config = VersioningConfig(
    mode=VersioningMode.INCREMENTAL,    # Versioning strategy
    max_versions=0,                     # 0 = unlimited
    auto_cleanup=True,                  # Auto-remove old versions
    track_changes=True,                 # Track change history
    require_message=False,              # Require commit message
    enable_branching=False,             # Enable branches
    checksum_algorithm="sha256",        # Integrity checksum
)
```

**Versioning Modes:**

| Mode | Format | Example |
|------|--------|---------|
| `INCREMENTAL` | v{n} | v1, v2, v3 |
| `SEMANTIC` | X.Y.Z | 1.0.0, 1.1.0 |
| `TIMESTAMP` | ISO timestamp | 2024-01-15T10:30:45 |
| `GIT_LIKE` | Short hash | abc1234 |

```python
from truthound.stores.versioning import VersionedStore

# Wrap any store with versioning
versioned_store = VersionedStore(
    store=base_store,
    config=VersioningConfig(mode=VersioningMode.SEMANTIC),
)

# Save with version
versioned_store.save(result, message="Initial validation")

# List versions
versions = versioned_store.list_versions(item_id)

# Get specific version
result = versioned_store.get_version(item_id, version=2)

# Rollback
versioned_store.rollback(item_id, version=1)
```

## Retention Configuration

Configure retention policies for automatic cleanup.

```python
from truthound.stores.retention.policies import (
    TimeBasedPolicy,
    CountBasedPolicy,
    SizeBasedPolicy,
    StatusBasedPolicy,
    TagBasedPolicy,
    CompositePolicy,
    RetentionAction,
)

# Time-based: Delete after 90 days
time_policy = TimeBasedPolicy(
    max_age_days=90,
    action=RetentionAction.DELETE,
)

# Count-based: Keep last 100 per asset
count_policy = CountBasedPolicy(
    max_count=100,
    per_asset=True,
    action=RetentionAction.ARCHIVE,
)

# Size-based: Max 1GB
size_policy = SizeBasedPolicy(
    max_size_bytes=1_000_000_000,
    action=RetentionAction.DELETE,
)

# Status-based: Delete failed after 7 days
status_policy = StatusBasedPolicy(
    status="failed",
    max_age_days=7,
    action=RetentionAction.DELETE,
)

# Tag-based: Keep production results longer
tag_policy = TagBasedPolicy(
    tag_key="environment",
    tag_value="production",
    max_age_days=365,
    action=RetentionAction.ARCHIVE,
)

# Composite: Combine multiple policies
composite = CompositePolicy(
    policies=[time_policy, count_policy],
    mode="any",  # "any" or "all"
)
```

**Retention Actions:**

| Action | Description |
|--------|-------------|
| `DELETE` | Permanently remove |
| `ARCHIVE` | Move to archive tier |
| `QUARANTINE` | Move to quarantine |

## Caching Configuration

Add caching layer for improved performance.

```python
from truthound.stores.caching.base import (
    CacheConfig,
    EvictionPolicy,
)

config = CacheConfig(
    max_size=10000,                     # Max cache entries
    max_memory_mb=100.0,                # Max memory usage
    ttl_seconds=3600.0,                 # TTL (1 hour)
    eviction_policy=EvictionPolicy.LRU, # Eviction strategy
    eviction_batch_size=100,            # Batch eviction size
    enable_statistics=True,             # Track cache stats
    warm_on_startup=False,              # Warm cache on startup
    background_refresh=False,           # Background refresh
    refresh_threshold_percent=20.0,     # Refresh threshold
)
```

**Eviction Policies:**

| Policy | Description |
|--------|-------------|
| `LRU` | Least Recently Used |
| `LFU` | Least Frequently Used |
| `TTL` | Time To Live |
| `FIFO` | First In First Out |
| `RANDOM` | Random eviction |

```python
from truthound.stores.caching import CachedStore

cached_store = CachedStore(
    store=base_store,
    config=config,
)

# Cache statistics
stats = cached_store.get_stats()
print(f"Hit rate: {stats.hits / (stats.hits + stats.misses):.2%}")
```

## Replication Configuration

Configure cross-region replication.

```python
from truthound.stores.replication.base import (
    ReplicationConfig,
    ReplicationMode,
    ReadPreference,
    ConflictResolution,
    ReplicaTarget,
)

config = ReplicationConfig(
    mode=ReplicationMode.ASYNC,
    targets=[],
    read_preference=ReadPreference.PRIMARY,
    conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
    min_sync_replicas=1,
    enable_health_checks=True,
    health_check_interval_seconds=30.0,
    max_replication_lag_ms=5000.0,
    enable_failover=True,
    enable_metrics=True,
)

# Add replica target
target = ReplicaTarget(
    name="us-west-replica",
    store=west_store,
    region="us-west-2",
    priority=1,
    is_read_replica=True,
    sync_timeout_seconds=30.0,
    max_retry_attempts=3,
)
```

**Replication Modes:**

| Mode | Description |
|------|-------------|
| `SYNC` | Wait for all replicas |
| `ASYNC` | Fire and forget |
| `SEMI_SYNC` | Wait for at least one |

**Read Preferences:**

| Preference | Description |
|------------|-------------|
| `PRIMARY` | Always read from primary |
| `SECONDARY` | Prefer secondary replicas |
| `NEAREST` | Read from nearest |
| `ANY` | Read from any available |

## Backpressure Configuration

Manage system load with backpressure.

```python
from truthound.stores.backpressure.base import BackpressureConfig

config = BackpressureConfig(
    enabled=True,
    memory_threshold_percent=80.0,      # Memory pressure threshold
    queue_depth_threshold=10000,        # Queue depth threshold
    latency_threshold_ms=100.0,         # Latency threshold
    min_pause_ms=10.0,                  # Minimum pause
    max_pause_ms=5000.0,                # Maximum pause
    base_rate=10000.0,                  # Base ops/sec
    min_rate=100.0,                     # Minimum ops/sec
    adaptive_window_size=100,           # Adaptive window
    recovery_rate=0.1,                  # Recovery rate
    sampling_interval_ms=100.0,         # Sampling interval
    pressure_decay_factor=0.95,         # Decay factor
)
```

## Observability Configuration

Configure audit logging, metrics, and tracing.

```python
from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    AuditLogLevel,
    MetricsConfig,
    TracingConfig,
)

# Audit logging
audit = AuditConfig(
    enabled=True,
    level=AuditLogLevel.STANDARD,       # MINIMAL, STANDARD, VERBOSE, DEBUG
    backend="json",                     # memory, file, json, elasticsearch, kafka
    file_path="./audit.log",
    include_data_preview=False,
    max_data_preview_size=1024,
    redact_sensitive=True,
    sensitive_fields=["password", "secret", "token", "api_key"],
    retention_days=90,
    batch_size=100,
    flush_interval_seconds=5.0,
)

# Prometheus metrics
metrics = MetricsConfig(
    enabled=True,
    prefix="truthound_store",
    labels={"service": "data-quality"},
    histogram_buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    enable_http_server=True,
    http_port=9090,
    http_path="/metrics",
    push_gateway_url=None,
    push_interval_seconds=10.0,
)

# Distributed tracing
tracing = TracingConfig(
    enabled=True,
    service_name="truthound-store",
    sampler="parent_based",             # parent_based, always_on, always_off
    sample_ratio=1.0,
    exporter="otlp",                    # otlp, jaeger, zipkin, console
    endpoint=None,
    propagators=["tracecontext", "baggage"],
    record_exceptions=True,
)

# Combined configuration
observability = ObservabilityConfig(
    audit=audit,
    metrics=metrics,
    tracing=tracing,
    correlation_id_header="X-Correlation-ID",
    environment="production",
)

# Presets
config = ObservabilityConfig.production()  # Production defaults
config = ObservabilityConfig.minimal()     # Minimal overhead
config = ObservabilityConfig.disabled()    # All disabled
```

## Factory Functions

```python
from truthound.stores import get_store, list_available_backends

# List available backends
backends = list_available_backends()
# ['filesystem', 's3', 'gcs', 'azure', 'database', 'memory']

# Create store by backend name
store = get_store("filesystem", base_path=".truthound/results")
store = get_store("s3", bucket="my-bucket", prefix="results/")
store = get_store("database", connection_url="postgresql://localhost/db")
store = get_store("azure", container="my-container", connection_string="...")
store = get_store("memory")  # For testing

# Check backend availability
from truthound.stores import is_backend_available

if is_backend_available("s3"):
    store = get_store("s3", bucket="my-bucket")
```

## Store Operations

```python
from truthound.stores.base import StoreQuery

# Save result
result_id = store.save(validation_result)

# Get result
result = store.get(result_id)

# Check existence
if store.exists(result_id):
    store.delete(result_id)

# Query results
query = StoreQuery(
    data_asset="users_table",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
    status="failure",
    tags={"environment": "production"},
    limit=100,
    offset=0,
    order_by="run_time",
    ascending=False,
)

results = store.query(query)

# Iterate with batching
for result in store.iter_query(query, batch_size=100):
    process(result)
```
