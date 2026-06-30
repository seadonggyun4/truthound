# 결과 캐싱

실무 운영 가이드에서 In-memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 LRU, Least, Recently, Used, Evicts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 LFU, Least, Frequently, Used, Evicts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 TTL, Time, Live, Evicts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 FIFO, First, Out, Evicts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 RANDOM, Random을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

## 캐시 Implementations

### LRUCache

실무 운영 가이드에서 Least, Recently, Used을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 `OrderedDict`, OrderedDict을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### LFUCache

실무 운영 가이드에서 Least, Frequently, Used을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### TTLCache

실무 운영 가이드에서 Time-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Basic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## 설정

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

### 설정 Options

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|------|---------|-------------|
| 실무 운영 가이드에서 `max_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `10000`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_memory_mb`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `100.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ttl_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `3600.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default, TTL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `eviction_policy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `EvictionPolicy`, EvictionPolicy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `LRU`, LRU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Eviction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `eviction_batch_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `100`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Entries을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `enable_statistics`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `warm_on_startup`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Pre-warm 캐시 on startup |
| 실무 운영 가이드에서 `background_refresh`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `False`, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Refresh을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `refresh_threshold_percent`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `20.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Refresh을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Eviction Policies

```python
from truthound.stores.caching.base import EvictionPolicy

EvictionPolicy.LRU      # Least Recently Used
EvictionPolicy.LFU      # Least Frequently Used
EvictionPolicy.TTL      # Time To Live (expired first)
EvictionPolicy.FIFO     # First In First Out
EvictionPolicy.RANDOM   # Random eviction
```

## 캐시 Operations

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

### 메트릭

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

### Cache메트릭 Fields

| 실무 운영 가이드에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 실무 운영 가이드에서 `hits`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 hit count |
| 실무 운영 가이드에서 `misses`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 캐시 miss count |
| 실무 운영 가이드에서 `sets`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `evictions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Eviction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `expirations`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Expiration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Current을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `memory_bytes`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Current을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `hit_rate`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hit을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `average_get_time_ms`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Average을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `average_set_time_ms`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Average을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## CacheEntry

Internal 캐시 entry structure:

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

실무 운영 가이드에서 Wrap을(를) 다루는 항목입니다:

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

## Real-World 예시

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

실무 운영 가이드에서 `threading.RLock`, RLock을(를) 다루는 항목입니다:

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

## 다음 단계

- 실무 운영 가이드에서 Replication, Cross-region을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [관측성](observability.md) - 감사, metrics, tracing
- 실무 운영 가이드에서 Tiering, Hot/Warm/Cold/Archive을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
