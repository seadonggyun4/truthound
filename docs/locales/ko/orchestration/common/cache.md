---
title: 캐싱
---

# 캐싱

오케스트레이션 실행에서 Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from common.cache import cached, CacheConfig, EvictionPolicy

# Simple decorator usage
@cached(ttl_seconds=300.0)
def fetch_user(user_id: str) -> dict:
    return db.query(user_id)

# Detailed control with configuration object
config = CacheConfig(
    max_size=1000,
    ttl_seconds=3600.0,
    eviction_policy=EvictionPolicy.LRU,
)

@cached(config=config)
async def async_fetch(user_id: str) -> dict:
    return await db.async_query(user_id)
```

## Preset 설정s

```python
from common.cache import (
    DEFAULT_CACHE_CONFIG,    # Default: 1000 items, LRU
    SMALL_CACHE_CONFIG,      # Small: 100 items, 60s TTL
    LARGE_CACHE_CONFIG,      # Large: 10000 items, 1 hour TTL
    SHORT_TTL_CACHE_CONFIG,  # Short: 30s TTL
    LONG_TTL_CACHE_CONFIG,   # Long: 24 hour TTL
    NO_EVICTION_CACHE_CONFIG,  # Unlimited (use with caution)
)

@cached(config=SMALL_CACHE_CONFIG)
def frequently_accessed_data():
    return compute()
```

## 캐시 Backends

| 오케스트레이션 실행에서 Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|-------------|----------|
| 오케스트레이션 실행에서 `LRUCache`, LRUCache을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Least, Recently, Used을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LFUCache`, LFUCache을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Least, Frequently, Used을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Prioritize을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `TTLCache`, TTLCache을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Time, Live을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Time-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `InMemoryCache`, InMemoryCache을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Simple memory 캐시 | 오케스트레이션 실행에서 Unlimited을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Eviction Policies

| 오케스트레이션 실행에서 Policy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 오케스트레이션 실행에서 `LRU`, LRU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Remove을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LFU`, LFU을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Remove을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `TTL`, TTL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Remove을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `FIFO`, FIFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Remove을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `NONE`, NONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Builder Pattern

```python
from common.cache import CacheConfig, EvictionPolicy

config = CacheConfig()
config = config.with_max_size(500)
config = config.with_ttl(300.0)
config = config.with_eviction_policy(EvictionPolicy.LFU)
config = config.with_namespace("users")
config = config.with_name("user_cache")
```

## Key Generators

```python
from common.cache import (
    cached,
    DefaultKeyGenerator,
    ArgumentKeyGenerator,
    CallableKeyGenerator,
)

# Use specific arguments for key generation
@cached(
    key_generator=ArgumentKeyGenerator(arg_names=("user_id",)),
)
def get_user(user_id: str, include_details: bool = False):
    return db.query(user_id)

# Custom key generation
def custom_extractor(func, args, kwargs):
    return f"user:{kwargs.get('user_id', 'default')}"

@cached(
    key_generator=CallableKeyGenerator(custom_extractor),
)
def get_user_data(user_id: str):
    return db.query(user_id)
```

## Hooks

```python
from common.cache import cached, LoggingCacheHook, MetricsCacheHook

# Logging hook
logging_hook = LoggingCacheHook()

# Metrics hook
metrics_hook = MetricsCacheHook()

@cached(ttl_seconds=60.0, hooks=[logging_hook, metrics_hook])
def monitored_fetch(key: str):
    return compute(key)

# Query metrics
print(metrics_hook.hits)       # Cache hit count
print(metrics_hook.misses)     # Cache miss count
print(metrics_hook.hit_rate)   # Hit rate (0.0 ~ 1.0)
print(metrics_hook.evictions)  # Evicted item count
```

## Registry Usage

```python
from common.cache import get_cache, CacheRegistry

# Get cache from global registry
cache = get_cache("users", config=CacheConfig(max_size=1000))
cache.set("user:1", user_data)
user = cache.get("user:1")

# Registry management
registry = CacheRegistry()
registry.clear_all()
stats = registry.get_all_stats()
```

## Utility Functions

```python
from common.cache import cache_get, cache_set, cache_delete, cache_clear, cache_stats

# Use convenience functions
cache_set("users", "user:1", user_data, ttl_seconds=300.0)
user = cache_get("users", "user:1", default=None)
cache_delete("users", "user:1")
cache_clear("users")

# Query statistics
stats = cache_stats("users")
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Size: {stats.size}/{stats.max_size}")
```

## Exception Handling

```python
from common.cache import cached, CacheKeyError, CacheAction

# RAISE action: raise exception on cache miss
config = CacheConfig(on_miss=CacheAction.RAISE)

@cached(config=config)
def strict_fetch(key: str):
    return compute(key)

try:
    result = strict_fetch("missing")
except CacheKeyError as e:
    print(f"Cache key '{e.key}' not found")
    result = default_value
```

## CacheExecutor

```python
from common.cache import CacheExecutor, CacheConfig

config = CacheConfig(max_size=100, ttl_seconds=60.0)
executor = CacheExecutor(config)

# Synchronous function
result = executor.execute(compute_function, arg1, arg2)

# Asynchronous function
result = await executor.execute_async(async_compute, arg1, arg2)

# Cache control
executor.invalidate("key")
executor.clear()
```

## CacheStats

```python
from common.cache import CacheStats

stats: CacheStats = cache.get_stats()
print(f"Size: {stats.size}")
print(f"Max size: {stats.max_size}")
print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Evictions: {stats.evictions}")
```
