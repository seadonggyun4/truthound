# Caching & Incremental Profiling

이 문서는 프로파일 결과 캐싱 및 증분 프로파일링 시스템을 설명합니다.

## 개요

`src/truthound/profiler/caching.py`에 구현된 캐싱 시스템은 파일 핑거프린트 기반으로 프로파일 결과를 캐싱하여 재프로파일링 시간을 절약합니다.

## CacheKey Protocol

```python
from typing import Protocol

class CacheKey(Protocol):
    """캐시 키 프로토콜"""

    def to_string(self) -> str:
        """캐시 키를 문자열로 변환"""
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...
```

## FileHashCacheKey

SHA-256 기반 파일 해시 캐시 키입니다.

```python
from truthound.profiler.caching import FileHashCacheKey

# 파일에서 캐시 키 생성
cache_key = FileHashCacheKey.from_file("data.csv")

print(cache_key.file_path)      # data.csv
print(cache_key.file_hash)      # SHA-256 해시
print(cache_key.file_size)      # 파일 크기
print(cache_key.modified_time)  # 수정 시간
print(cache_key.to_string())    # 캐시 키 문자열
```

### 해시 계산

```python
# 내부 구현 - SHA-256 사용
def _compute_hash(self, path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
```

## 캐시 백엔드

### MemoryCacheBackend

메모리 기반 LRU 캐시입니다.

```python
from truthound.profiler.caching import MemoryCacheBackend

cache = MemoryCacheBackend(max_size=100)

# 저장
cache.set(cache_key, profile)

# 조회
profile = cache.get(cache_key)

# 존재 확인
if cache.exists(cache_key):
    print("Cache hit!")

# 삭제
cache.delete(cache_key)

# 전체 삭제
cache.clear()
```

### FileCacheBackend

디스크 기반 JSON 캐시입니다.

```python
from truthound.profiler.caching import FileCacheBackend

cache = FileCacheBackend(
    cache_dir=".truthound/cache",
    max_age_days=30,  # 30일 후 만료
)

# 저장 (자동 JSON 직렬화)
cache.set(cache_key, profile)

# 조회 (자동 JSON 역직렬화)
profile = cache.get(cache_key)
```

### RedisCacheBackend

Redis 기반 분산 캐시입니다.

```python
from truthound.profiler.caching import RedisCacheBackend

cache = RedisCacheBackend(
    host="localhost",
    port=6379,
    db=0,
    prefix="truthound:profile:",
    ttl_seconds=86400,  # 24시간 TTL
)

cache.set(cache_key, profile)
profile = cache.get(cache_key)
```

## ProfileCache

통합 캐시 인터페이스입니다.

```python
from truthound.profiler.caching import ProfileCache

# 파일 시스템 캐시 사용
cache = ProfileCache(cache_dir=".truthound/cache")

# 핑거프린트 계산
fingerprint = cache.compute_fingerprint("data.csv")

# 캐시 확인 및 사용
if cache.exists(fingerprint):
    profile = cache.get(fingerprint)
    print("Cache hit!")
else:
    profile = profiler.profile_file("data.csv")
    cache.set(fingerprint, profile)
    print("Cache miss, computed and cached")
```

### get_or_compute 패턴

```python
from truthound.profiler.caching import ProfileCache

cache = ProfileCache()

# 캐시 미스 시 자동 계산 및 저장
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

# 개별 TTL 지정
cache.set(cache_key, profile, ttl=timedelta(hours=1))

# 만료된 캐시 정리
cache.cleanup_expired()
```

## 증분 프로파일링

이전 프로파일과 비교하여 변경된 부분만 재프로파일링합니다.

```python
from truthound.profiler import IncrementalProfiler

inc_profiler = IncrementalProfiler(cache=cache)

# 초기 프로파일
profile_v1 = inc_profiler.profile("data_v1.csv")

# 증분 업데이트 (변경된 컬럼만 재프로파일링)
profile_v2 = inc_profiler.update(
    "data_v2.csv",
    previous=profile_v1,
)

print(f"Columns re-profiled: {profile_v2.columns_updated}")
print(f"Columns reused: {profile_v2.columns_cached}")
print(f"Time saved: {profile_v2.time_saved_ms}ms")
```

## 캐시 통계

```python
from truthound.profiler.caching import CacheStatistics

# 캐시 통계 조회
stats = cache.get_statistics()

print(f"Total entries: {stats.total_entries}")
print(f"Cache hits: {stats.hit_count}")
print(f"Cache misses: {stats.miss_count}")
print(f"Hit ratio: {stats.hit_ratio:.2%}")
print(f"Total size: {stats.total_size_bytes / 1024 / 1024:.2f} MB")
```

## 캐시 무효화

```python
# 특정 키 무효화
cache.invalidate(cache_key)

# 패턴 기반 무효화
cache.invalidate_pattern("data_*.csv")

# 전체 무효화
cache.clear()

# 만료된 항목만 정리
cache.cleanup_expired()
```

## 캐시 체이닝

여러 캐시 백엔드를 체인으로 연결합니다.

```python
from truthound.profiler.caching import CacheChain, MemoryCacheBackend, FileCacheBackend

# 메모리 -> 파일 시스템 체인
cache = CacheChain([
    MemoryCacheBackend(max_size=50),      # L1: 빠른 메모리
    FileCacheBackend(".cache"),        # L2: 영구 저장
])

# L1에서 조회 -> 미스 시 L2에서 조회 -> L1에 복사
profile = cache.get(cache_key)
```

## CLI 사용법

```bash
# 캐시 사용 프로파일링
th profile data.csv --cache

# 캐시 디렉토리 지정
th profile data.csv --cache --cache-dir .my_cache

# 캐시 무시 (강제 재프로파일링)
th profile data.csv --no-cache

# 캐시 통계 확인
th cache stats

# 캐시 정리
th cache clear

# 만료된 캐시만 정리
th cache cleanup
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `TRUTHOUND_CACHE_DIR` | 캐시 디렉토리 | `.truthound/cache` |
| `TRUTHOUND_CACHE_TTL_DAYS` | 캐시 TTL (일) | `30` |
| `TRUTHOUND_CACHE_MAX_SIZE_MB` | 최대 캐시 크기 | `1000` |
| `TRUTHOUND_REDIS_URL` | Redis URL | `None` |

## 다음 단계

- [드리프트 감지](drift-detection.md) - 캐시된 프로파일 비교
- [분산 처리](distributed.md) - 분산 환경에서 캐시 공유
