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


CachedStore의 get/get_many/refresh/warm은 StoreNotFoundError만 부재로 처리합니다. 다른 하위 저장소 읽기 실패는 None이나 부분 성공으로 바꾸지 않고 전달합니다. 뒤 항목 읽기 실패 전에 앞 항목이 캐시에 반영됐을 수 있습니다. 실제 삭제된 기록을 refresh하면 오래된 캐시 항목을 제거합니다. 캐시 자체 오류도 숨기지 않습니다.


## 캐시 모드와 소유한 백그라운드 쓰기

READ_THROUGH는 읽을 때 채우고 저장 시 무효화합니다. WRITE_THROUGH는 저장소에
쓴 뒤 사본을 캐시합니다. CACHE_ASIDE는 암묵적 캐싱을 건너뛰며 cache 또는 명시적
warm/refresh/invalidate를 사용합니다. WRITE_BEHIND는 안정적인 비어 있지 않은
item.id를 요구하고 제출 항목의 사본을 인스턴스 소유 작업 스레드 하나로 씁니다.
save 결과는 접수 ID이지 영속 쓰기 완료 증명이 아닙니다. max_pending_writes
(기본 1000개)로 개수를 제한하며 바이트 예약이나 영속 journal은 아닙니다. 쓰기 완료
전 캐시에서 항목이 퇴출될 수 있고 list_ids는 영속 기록만 나열합니다. 프로세스 종료로
대기 쓰기가 유실될 수 있으므로 전달 전에 flush하고 provider보다 wrapper를 먼저
닫아야 합니다. provider는 해당 작업 스레드 호출을 지원해야 합니다.

flush는 소유한 쓰기를 기다리고 누적 완료 개수를 반환합니다. close/context 종료는
flush 후 스레드를 종료하며 빌린 하위 저장소는 닫지 않습니다. delete/refresh는 먼저
flush하여 순서를 지킵니다. 백그라운드 쓰기 실패는 이미 저장됐을 수 있으므로
outcome_unknown으로 표시하고 새 제출·자동 재시도를 막습니다. get_stats의
write_behind는 개수·closed/outcome_unknown·불투명 uncertain_token만 노출합니다.
운영자 확인 후 reconcile_write(token, confirm/discard/retry)로 항목을 확인·제거·재시도하며
retry는 accept_duplicate_risk=True를 요구합니다. confirm은 운영자 주장이고 discard는
rollback이 아니며 retry는 중복 쓰기를 만들 수 있습니다. 정확한 token으로 다른 작업을
잘못 처리하지 않습니다. close 실패는 명시적 복구용 대기열을 보존합니다. set_io_guard는
작업 스레드 I/O 직전에 검사하며 통계에는 원시 예외 본문을 넣지 않습니다.

승인된 MCP cache_flush/cache_reconcile이 이 제어를 제공합니다. 복구에는 write_token과
cache_resolution, retry에는 accept_duplicate_risk: true도 필요합니다. provider·큐 한도·
물리 경로는 팩토리가 소유합니다. MCP 종료는 팩토리 teardown 전에 wrapper를 비웁니다.
설치 worker 검증은 파일 저장소 flush·캐시 무효화·새 소유자의 읽기·소유 프로세스 종료를
확인합니다.

wrapper의 기본 캐시는 네이티브 InMemoryCache로 CacheConfig.eviction_policy를
적용합니다. 네이티브 backend 4종 모두 내장 컨테이너·dataclass 필드·순환 참조 처리를
포함한 제한된 Python 값 크기 추정으로 max_memory_mb를 적용합니다. 너무 큰 교체는
기존 항목 삭제 전에 거부하고 일반 추가는 선택한 backend 정책으로 퇴출합니다. 전체
프로세스 메모리 예약이 아닌 추정치이며 불투명 객체·외부 버퍼·변경 가능한 값·할당기
부담은 다를 수 있습니다. 설정은 복사하고 유한하지 않은 한도/TTL은 거부합니다.
enable_statistics=False는 hit/miss/set/퇴출·시간 집계를 끄되 크기·메모리 계산은 유지합니다.

warm_on_startup=True는 wrapper 생성 시 영속 ID를 max_size개까지 읽습니다.
background_refresh=True는 소유한 주기 스레드와 네이티브 expiring_keys 메타데이터로
TTL이 설정 비율 이하인 store: 항목을 갱신합니다. TTL 0은 만료 갱신 스레드를 시작하지
않습니다. 사용자 캐시는 갱신 활성화 시 이 메타데이터 port를 구현해야 하고 provider는
작업 스레드 호출을 허용해야 합니다. 갱신은 wrapper 작업과 직렬화하고 I/O guard를
검사하며 대기 쓰기를 먼저 flush합니다. 통계는 maintenance의 running/failures/refreshes를
제공하고 원시 예외 본문 없이 실패를 집계하며 실패 후에도 순회 간격을 유지합니다.
close는 갱신 스레드를 종료한 뒤 쓰기를 비웁니다. 프로세스 종료 후 캐시·갱신 이력·대기
지연 쓰기는 남지 않습니다.
