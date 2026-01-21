# Caching & Incremental Profiling

This document describes the profile result caching and incremental profiling system.

## Overview

The caching system implemented in `src/truthound/profiler/caching.py` caches profile results based on file fingerprints to save re-profiling time.

## CacheKey Protocol

```python
from typing import Protocol

class CacheKey(Protocol):
    """Cache key protocol"""

    def to_string(self) -> str:
        """Convert cache key to string"""
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...
```

## FileHashCacheKey

SHA-256 based file hash cache key.

```python
from truthound.profiler.caching import FileHashCacheKey

# Create cache key from file
cache_key = FileHashCacheKey.from_file("data.csv")

print(cache_key.file_path)      # data.csv
print(cache_key.file_hash)      # SHA-256 hash
print(cache_key.file_size)      # File size
print(cache_key.modified_time)  # Modification time
print(cache_key.to_string())    # Cache key string
```

### Hash Computation

```python
# Internal implementation - using SHA-256
def _compute_hash(self, path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
```

## Cache Backends

### MemoryCacheBackend

Memory-based LRU cache.

```python
from truthound.profiler.caching import MemoryCacheBackend

cache = MemoryCacheBackend(max_size=100)

# Store
cache.set(cache_key, profile)

# Retrieve
profile = cache.get(cache_key)

# Check existence
if cache.exists(cache_key):
    print("Cache hit!")

# Delete
cache.delete(cache_key)

# Clear all
cache.clear()
```

### FileCacheBackend

Disk-based JSON cache.

```python
from truthound.profiler.caching import FileCacheBackend

cache = FileCacheBackend(
    cache_dir=".truthound/cache",
    max_age_days=30,  # Expire after 30 days
)

# Store (auto JSON serialization)
cache.set(cache_key, profile)

# Retrieve (auto JSON deserialization)
profile = cache.get(cache_key)
```

### RedisCacheBackend

Redis-based distributed cache.

```python
from truthound.profiler.caching import RedisCacheBackend

cache = RedisCacheBackend(
    host="localhost",
    port=6379,
    db=0,
    prefix="truthound:profile:",
    ttl_seconds=86400,  # 24-hour TTL
)

cache.set(cache_key, profile)
profile = cache.get(cache_key)
```

## ProfileCache

Unified cache interface.

```python
from truthound.profiler.caching import ProfileCache

# Use file system cache
cache = ProfileCache(cache_dir=".truthound/cache")

# Compute fingerprint
fingerprint = cache.compute_fingerprint("data.csv")

# Check and use cache
if cache.exists(fingerprint):
    profile = cache.get(fingerprint)
    print("Cache hit!")
else:
    profile = profiler.profile_file("data.csv")
    cache.set(fingerprint, profile)
    print("Cache miss, computed and cached")
```

### get_or_compute Pattern

```python
from truthound.profiler.caching import ProfileCache

cache = ProfileCache()

# Automatic computation and storage on cache miss
profile = cache.get_or_compute(
    key=cache_key,
    compute_fn=lambda: profiler.profile_file("data.csv"),
)
```

## TTL (Time-To-Live)

```python
from truthound.profiler.caching import FileCacheBackend
from datetime import timedelta

cache = FileCacheBackend(
    cache_dir=".truthound/cache",
    default_ttl=timedelta(days=7),
)

# Specify individual TTL
cache.set(cache_key, profile, ttl=timedelta(hours=1))

# Clean up expired cache
cache.cleanup_expired()
```

## Incremental Profiling

Re-profiles only the changed parts by comparing with the previous profile.

```python
from truthound.profiler import IncrementalProfiler

inc_profiler = IncrementalProfiler(cache=cache)

# Initial profile
profile_v1 = inc_profiler.profile("data_v1.csv")

# Incremental update (only re-profile changed columns)
profile_v2 = inc_profiler.update(
    "data_v2.csv",
    previous=profile_v1,
)

print(f"Columns re-profiled: {profile_v2.columns_updated}")
print(f"Columns reused: {profile_v2.columns_cached}")
print(f"Time saved: {profile_v2.time_saved_ms}ms")
```

## Cache Statistics

```python
from truthound.profiler.caching import CacheStatistics

# Retrieve cache statistics
stats = cache.get_statistics()

print(f"Total entries: {stats.total_entries}")
print(f"Cache hits: {stats.hit_count}")
print(f"Cache misses: {stats.miss_count}")
print(f"Hit ratio: {stats.hit_ratio:.2%}")
print(f"Total size: {stats.total_size_bytes / 1024 / 1024:.2f} MB")
```

## Cache Invalidation

```python
# Invalidate specific key
cache.invalidate(cache_key)

# Pattern-based invalidation
cache.invalidate_pattern("data_*.csv")

# Clear all
cache.clear()

# Clean up only expired entries
cache.cleanup_expired()
```

## Cache Chaining

Chain multiple cache backends together.

```python
from truthound.profiler.caching import CacheChain, MemoryCacheBackend, FileCacheBackend

# Memory -> File system chain
cache = CacheChain([
    MemoryCacheBackend(max_size=50),      # L1: Fast memory
    FileCacheBackend(".cache"),           # L2: Persistent storage
])

# Lookup L1 -> on miss lookup L2 -> copy to L1
profile = cache.get(cache_key)
```

## CLI Usage

```bash
# Profile with caching
th profile data.csv --cache

# Specify cache directory
th profile data.csv --cache --cache-dir .my_cache

# Ignore cache (force re-profiling)
th profile data.csv --no-cache

# View cache statistics
th cache stats

# Clear cache
th cache clear

# Clean up only expired cache
th cache cleanup
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_CACHE_DIR` | Cache directory | `.truthound/cache` |
| `TRUTHOUND_CACHE_TTL_DAYS` | Cache TTL (days) | `30` |
| `TRUTHOUND_CACHE_MAX_SIZE_MB` | Maximum cache size | `1000` |
| `TRUTHOUND_REDIS_URL` | Redis URL | `None` |

## Next Steps

- [Drift Detection](drift-detection.md) - Compare cached profiles
- [Distributed Processing](distributed.md) - Share cache in distributed environments
