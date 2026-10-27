# Result Caching

In-memory caching layer for faster validation result retrieval.

## Overview

The caching module provides high-performance in-memory caches with various eviction policies:

- **LRU** (Least Recently Used) - Evicts entries not accessed recently
- **LFU** (Least Frequently Used) - Evicts entries accessed least often
- **TTL** (Time To Live) - Evicts entries based on expiration time
- **FIFO** (First In First Out) - Evicts oldest entries
- **RANDOM** - Random eviction

## Quick Start

```python
from truthound.stores import get_store
from truthound.stores.caching.backends import LRUCache
from truthound.stores.caching import CachedStore

# Create base store
base = get_store("s3", bucket="my-bucket")

# Create cache
cache = LRUCache(max_size=1000, ttl_seconds=3600)

# Wrap store with cache
store = CachedStore(base, cache)

# Operations now use cache
result = store.get("run-123")  # Cached on first access
result = store.get("run-123")  # Served from cache
```

## Cache Implementations

### LRUCache

Least Recently Used - evicts entries not accessed recently.

```python
from truthound.stores.caching.backends import LRUCache

# Basic usage
cache = LRUCache(max_size=1000)

# With TTL
cache = LRUCache(max_size=1000, ttl_seconds=3600)

# With full config
from truthound.stores.caching.base import CacheConfig, EvictionPolicy

config = CacheConfig(
    max_size=5000,
    ttl_seconds=7200,
    eviction_policy=EvictionPolicy.LRU,
)
cache = LRUCache(config=config)
```

Uses `OrderedDict` for O(1) access and eviction.

### LFUCache

Least Frequently Used - evicts entries with lowest access count.

```python
from truthound.stores.caching.backends import LFUCache

cache = LFUCache(max_size=1000, ttl_seconds=3600)

# Frequently accessed entries stay cached
cache.set("popular", data)
for _ in range(100):
    cache.get("popular")  # High access count

cache.set("rarely_used", data)
cache.get("rarely_used")  # Low access count

# "rarely_used" evicted first when cache is full
```

Uses a min-heap for efficient frequency-based eviction.

### TTLCache

Time-based eviction with automatic expiration cleanup.

```python
from truthound.stores.caching.backends import TTLCache

# 5-minute TTL
cache = TTLCache(ttl_seconds=300)

# Set with default TTL
cache.set("key1", data)

# Set with custom TTL
cache.set("key2", data, ttl_seconds=60)  # 1 minute

# Manual cleanup of expired entries
expired_count = cache.cleanup_expired()
```

### InMemoryCache

Basic cache with configurable eviction policy.

```python
from truthound.stores.caching.backends import InMemoryCache
from truthound.stores.caching.base import CacheConfig, EvictionPolicy

# FIFO eviction
config = CacheConfig(
    max_size=1000,
    eviction_policy=EvictionPolicy.FIFO,
)
cache = InMemoryCache(config)

# Random eviction
config = CacheConfig(
    max_size=1000,
    eviction_policy=EvictionPolicy.RANDOM,
)
cache = InMemoryCache(config)
```

## Configuration

### CacheConfig

```python
from truthound.stores.caching.base import CacheConfig, EvictionPolicy

config = CacheConfig(
    max_size=10000,                        # Max entries
    max_memory_mb=100.0,                   # Max memory (MB)
    ttl_seconds=3600.0,                    # Default TTL (1 hour)
    eviction_policy=EvictionPolicy.LRU,    # Eviction strategy
    eviction_batch_size=100,               # Entries to evict at once
    enable_statistics=True,                # Track metrics
    warm_on_startup=False,                 # Pre-warm cache
    background_refresh=False,              # Background refresh
    refresh_threshold_percent=20.0,        # Refresh at 20% TTL remaining
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_size` | `int` | `10000` | Maximum entries |
| `max_memory_mb` | `float` | `100.0` | Maximum memory in MB |
| `ttl_seconds` | `float` | `3600.0` | Default TTL in seconds |
| `eviction_policy` | `EvictionPolicy` | `LRU` | Eviction strategy |
| `eviction_batch_size` | `int` | `100` | Entries to evict per batch |
| `enable_statistics` | `bool` | `True` | Enable metrics tracking |
| `warm_on_startup` | `bool` | `False` | Pre-warm cache on startup |
| `background_refresh` | `bool` | `False` | Refresh entries in background |
| `refresh_threshold_percent` | `float` | `20.0` | Refresh threshold |

## Eviction Policies

```python
from truthound.stores.caching.base import EvictionPolicy

EvictionPolicy.LRU      # Least Recently Used
EvictionPolicy.LFU      # Least Frequently Used
EvictionPolicy.TTL      # Time To Live (expired first)
EvictionPolicy.FIFO     # First In First Out
EvictionPolicy.RANDOM   # Random eviction
```

## Cache Operations

### Basic Operations

```python
# Set
cache.set("key", value)
cache.set("key", value, ttl_seconds=300)

# Get
value = cache.get("key")  # Returns None if not found

# Delete
deleted = cache.delete("key")  # Returns True/False

# Check existence
exists = cache.exists("key")

# Clear all
count = cache.clear()
```

### Batch Operations

```python
# Set multiple
cache.set_many({
    "key1": value1,
    "key2": value2,
}, ttl_seconds=3600)

# Get multiple
values = cache.get_many(["key1", "key2", "key3"])
# Returns dict of found entries: {"key1": value1, "key2": value2}

# Delete multiple
deleted_count = cache.delete_many(["key1", "key2"])
```

### Metrics

```python
# Get metrics
metrics = cache.metrics

print(f"Hits: {metrics.hits}")
print(f"Misses: {metrics.misses}")
print(f"Hit rate: {metrics.hit_rate:.1f}%")
print(f"Evictions: {metrics.evictions}")
print(f"Expirations: {metrics.expirations}")
print(f"Size: {metrics.size}")
print(f"Memory: {metrics.memory_bytes} bytes")
print(f"Avg get time: {metrics.average_get_time_ms:.2f}ms")
print(f"Avg set time: {metrics.average_set_time_ms:.2f}ms")

# Full stats
stats = cache.get_stats()
```

### CacheMetrics Fields

| Field | Description |
|-------|-------------|
| `hits` | Cache hit count |
| `misses` | Cache miss count |
| `sets` | Set operation count |
| `evictions` | Eviction count |
| `expirations` | Expiration count |
| `size` | Current entry count |
| `memory_bytes` | Current memory usage |
| `hit_rate` | Hit rate percentage |
| `average_get_time_ms` | Average get latency |
| `average_set_time_ms` | Average set latency |

## CacheEntry

Internal cache entry structure:

```python
from truthound.stores.caching.base import CacheEntry
from datetime import datetime, timedelta

entry = CacheEntry(
    key="my-key",
    value=data,
    created_at=datetime.now(),
    expires_at=datetime.now() + timedelta(hours=1),
    access_count=0,
    last_accessed=datetime.now(),
    size_bytes=1024,
)

# Check expiration
if entry.is_expired:
    print("Entry expired")

# Get remaining TTL
remaining = entry.ttl_remaining_seconds

# Update access metadata
entry.touch()  # Increments access_count, updates last_accessed
```

## CachedStore Wrapper

Wrap any store with caching:

```python
from truthound.stores.caching import CachedStore, CacheMode
from truthound.stores.caching.backends import LRUCache

# Create components
base_store = get_store("s3", bucket="my-bucket")
cache = LRUCache(max_size=1000)

# Create cached store
store = CachedStore(base_store, cache, mode=CacheMode.WRITE_THROUGH)

# Cache modes
CacheMode.READ_THROUGH   # Read from cache, fallback to store
CacheMode.WRITE_THROUGH  # Write to both cache and store synchronously
CacheMode.WRITE_BEHIND   # Write to cache, async write to store
CacheMode.CACHE_ASIDE    # Application manages cache explicitly
```

## Real-World Examples

### High-Throughput API

```python
# Large cache with long TTL for API responses
config = CacheConfig(
    max_size=50000,
    max_memory_mb=500.0,
    ttl_seconds=3600,  # 1 hour
    eviction_policy=EvictionPolicy.LRU,
)
cache = LRUCache(config=config)
```

### Development Environment

```python
# Small cache with short TTL
cache = LRUCache(max_size=100, ttl_seconds=300)  # 5 min
```

### Analytics Workload

```python
# LFU for frequently accessed reports
cache = LFUCache(max_size=1000, ttl_seconds=7200)  # 2 hours
```

### Short-Lived Sessions

```python
# TTL cache for session data
cache = TTLCache(ttl_seconds=900, max_size=10000)  # 15 min
```

## Thread Safety

All cache implementations are thread-safe using `threading.RLock`:

```python
import threading

cache = LRUCache(max_size=1000)

def worker():
    for i in range(1000):
        cache.set(f"key-{i}", f"value-{i}")
        cache.get(f"key-{i}")

# Safe for concurrent access
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Composing with Other Features

```python
from truthound.stores import get_store
from truthound.stores.versioning import VersionedStore, VersioningConfig
from truthound.stores.caching import CachedStore
from truthound.stores.caching.backends import LRUCache

# Base -> Versioning -> Cache
base = get_store("s3", bucket="my-bucket")
versioned = VersionedStore(base, VersioningConfig(max_versions=10))
cache = LRUCache(max_size=1000)
store = CachedStore(versioned, cache)
```

## Next Steps

- [Replication](replication.md) - Cross-region data replication
- [Observability](observability.md) - Audit, metrics, tracing
- [Tiering](tiering.md) - Hot/Warm/Cold/Archive storage


CachedStore get, get_many, refresh and warm treat only StoreNotFoundError as absence. Other underlying read failures propagate instead of returning None or a successful partial batch. Earlier entries may already have warmed the cache before a later read fails. Refreshing a genuinely deleted record evicts its stale cache entry. Cache failures themselves remain visible.


## Cache modes and owned background writes

READ_THROUGH fills on reads and invalidates on saves. WRITE_THROUGH writes the
store before caching a snapshot. CACHE_ASIDE bypasses implicit caching; use the
cache or explicit warm/refresh/invalidate methods. WRITE_BEHIND requires a stable
nonempty item.id, snapshots each submitted item and uses one instance-owned
worker thread. save returns an accepted ID, not a durable-write receipt. The
queue is bounded by max_pending_writes (default 1000 items); it is not a byte
reservation or durable journal. Cache eviction can remove a queued value before
the write completes. list_ids lists durable records. Process loss can lose queued
writes; flush before handing data off, and close the wrapper before its provider.
Providers used in this mode must permit calls on the worker thread.

flush waits for owned work and returns the cumulative completed count. close and
context exit flush and join the worker; they do not close the borrowed underlying
store. Delete and refresh flush first to preserve ordering. A failed background
write may already have changed storage: it becomes outcome_unknown, blocks new
submissions and prevents automatic replay. get_stats().write_behind exposes only
counts, closed/outcome_unknown and an opaque uncertain_token. After operator
reconciliation, reconcile_write(token, "confirm" | "discard" | "retry") acknowledges
or drops that queue entry, or retries it with accept_duplicate_risk=True. Confirm
is an operator assertion, discard is not a rollback, and retry may duplicate a
write. The exact token prevents resolving a different queued write. Failed close
retains the queue for explicit reconciliation. set_io_guard checks immediately
before worker I/O. No raw exception message is included in queue statistics.

Approved MCP cache_flush and cache_reconcile expose these native controls.
Reconciliation requires write_token and cache_resolution; retry additionally
requires accept_duplicate_risk: true. Provider configuration, queue limits and
physical paths remain factory-owned. MCP shutdown drains the wrapper before
factory teardown. Actual installed-worker tests cover file-store flush, cache
invalidation, new-owner reading and owned-process termination.

The default wrapper cache honors `CacheConfig.eviction_policy` through native
InMemoryCache. All four native backends enforce `max_memory_mb` using bounded
estimates of retained Python values (including built-in containers and dataclass
fields, avoiding cycles). Oversized replacement rejects before removing the old
entry; ordinary additions evict by the selected backend policy. This is an
estimate, not a whole-process heap reservation: opaque objects, external buffers,
mutable values and allocator overhead can differ. Configuration is copied and
nonfinite limits/TTLs are rejected. `enable_statistics=False` disables hit/miss/
set/eviction timing counters while retaining size/memory accounting.

`warm_on_startup=True` loads at most `max_size` durable IDs when constructing the
wrapper. `background_refresh=True` uses an owned periodic thread and native
`expiring_keys(threshold_percent)` metadata; it refreshes `store:` entries whose
remaining TTL falls below the configured percentage. TTL zero does not start an
expiration refresh thread. Custom caches must implement that metadata port if
refresh is enabled. Providers must permit calls on this thread. Refresh serializes
with wrapper operations, honors the I/O guard and flushes pending writes first.
Statistics expose maintenance running/failures/refreshes; failures are counted
without raw exception text and subsequent sweeps retain their delay. close stops
and joins maintenance before draining writes. Process exit does not retain cache,
refresh history or pending write-behind entries.
