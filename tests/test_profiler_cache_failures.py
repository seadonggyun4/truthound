"""Native cache corruption and failed writes retain evidence; zero TTL expires."""

import json
from datetime import timedelta

import polars as pl
import pytest

from truthound.profiler import DataProfiler
from truthound.profiler.caching import (
    CacheEntry,
    CacheKey,
    FileCacheBackend,
    MemoryCacheBackend,
    ProfileCache,
)

pytestmark = pytest.mark.fault


@pytest.mark.parametrize("compressed", [False, True])
def test_native_file_cache_corruption_is_failure_and_does_not_delete_evidence(tmp_path, compressed):
    backend = FileCacheBackend(tmp_path, compress=compressed)
    path = backend._get_path("local")
    original = b"synthetic-corrupt-cache"
    path.write_bytes(original)
    with pytest.raises(RuntimeError):
        backend.get("local")
    assert path.read_bytes() == original


@pytest.mark.parametrize("reading", [False, True])
def test_native_failed_cache_serialization_never_truncates_or_deletes_committed_entry(tmp_path, monkeypatch, reading):
    backend = FileCacheBackend(tmp_path, compress=False)
    profile = DataProfiler().profile(pl.DataFrame({"x": [1, 2]}).lazy())
    backend.set("local", CacheEntry(profile))
    path = backend._get_path("local")
    original = path.read_bytes()

    def failed_dump(value, stream, *args, **kwargs):
        stream.write("synthetic-partial-cache")
        raise OSError("synthetic-private-cache-write")

    monkeypatch.setattr(json, "dump", failed_dump)
    with pytest.raises((RuntimeError, OSError)):
        backend.get("local") if reading else backend.set("local", CacheEntry(profile))
    assert path.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["memory", "file"])
def test_native_zero_ttl_overrides_default_and_exists_checks_expiration(tmp_path, kind):
    backend = MemoryCacheBackend() if kind == "memory" else FileCacheBackend(tmp_path)
    cache = ProfileCache(backend=backend, default_ttl=timedelta(days=1))
    profile = DataProfiler().profile(pl.DataFrame({"x": [1]}).lazy())
    key = CacheKey("local")
    cache.set(key, profile, ttl=timedelta(0))
    assert not backend.exists(key.to_string())
    assert cache.get(key) is None


@pytest.mark.parametrize("kind", ["memory", "file"])
def test_native_pattern_invalidation_deletes_only_matching_logical_keys_after_reopen(tmp_path, kind):
    backend = MemoryCacheBackend() if kind == "memory" else FileCacheBackend(tmp_path)
    cache = ProfileCache(backend=backend)
    profile = DataProfiler().profile(pl.DataFrame({"x": [1]}).lazy())
    for name in ("selected-a", "selected-b", "other"):
        cache.set(CacheKey(name), profile)
    if kind == "file":
        cache = ProfileCache(backend=FileCacheBackend(tmp_path))
    assert cache.invalidate_by_pattern("default:1:selected-*") == 2
    assert cache.get(CacheKey("other")) is not None
    assert cache.get(CacheKey("selected-a")) is None


def test_native_corrupt_file_prevents_pattern_deletion_before_any_effect(tmp_path):
    backend = FileCacheBackend(tmp_path)
    cache = ProfileCache(backend=backend)
    profile = DataProfiler().profile(pl.DataFrame({"x": [1]}).lazy())
    cache.set(CacheKey("selected"), profile)
    corrupt = backend._get_path("corrupt")
    corrupt.write_bytes(b"synthetic-corrupt-pattern-entry")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(RuntimeError):
        cache.invalidate_by_pattern("*")
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_native_redis_corrupt_payload_preserves_server_bytes_and_sanitizes_errors():
    import threading

    from truthound.profiler.caching import RedisCacheBackend, RedisConnectionError

    class Client:
        def __init__(self):
            self.data = b"synthetic-corrupt-redis"
            self.failed = False

        def get(self, key):
            if self.failed:
                raise OSError("synthetic-private-redis-credential")
            return self.data

        def delete(self, *args):
            self.data = None
            return 1

    backend = object.__new__(RedisCacheBackend)
    backend._client = Client()
    backend._RedisError = OSError
    backend._lock = threading.RLock()
    backend.prefix = "private:"
    backend._hits = backend._misses = backend._errors = 0
    with pytest.raises(RuntimeError):
        backend.get("local")
    assert backend._client.data == b"synthetic-corrupt-redis"
    backend._client.failed = True
    with pytest.raises(RedisConnectionError) as failure:
        backend.get("local")
    assert "synthetic-private-redis-credential" not in str(failure.value) + backend._last_error


def test_native_redis_pattern_scan_is_namespace_bounded_and_zero_ttl_never_sets():
    import threading

    from truthound.profiler.caching import RedisCacheBackend

    class Client:
        deleted = []

        def scan(self, **arguments):
            assert arguments == {"cursor": 0, "match": r"private\[*", "count": 100}
            return 0, [b"private[selected", b"outside-selected", b"private[other"]

        def delete(self, *keys):
            self.deleted.extend(keys)
            return len(keys)

        def set(self, *args):
            raise AssertionError("Zero TTL persisted data")

        setex = set

    backend = object.__new__(RedisCacheBackend)
    backend._client = Client()
    backend._RedisError = OSError
    backend._lock = threading.RLock()
    backend.prefix = "private["
    assert backend.invalidate_by_pattern("selected*") == 1
    assert backend._client.deleted == [b"private[selected"]
    profile = DataProfiler().profile(pl.DataFrame({"x": [1]}).lazy())
    backend.set("expires", CacheEntry(profile), ttl=timedelta(0))
    assert backend._client.deleted[-1] == "private[expires"


@pytest.mark.parametrize("lazy", [False, True])
def test_native_redis_failed_client_construction_closes_pool_and_never_returns_broken_backend(monkeypatch, lazy):
    import sys
    from types import ModuleType

    from truthound.profiler.caching import RedisCacheBackend, RedisConnectionError

    closed = []
    redis = ModuleType("redis")
    errors = ModuleType("redis.exceptions")
    errors.RedisError = OSError

    class Pool:
        def __init__(self, **kwargs):
            pass

        def disconnect(self):
            closed.append("pool")

    def client(**kwargs):
        raise OSError("synthetic-private-redis-construction")

    redis.ConnectionPool, redis.Redis = Pool, client
    monkeypatch.setitem(sys.modules, "redis", redis)
    monkeypatch.setitem(sys.modules, "redis.exceptions", errors)
    with pytest.raises(RedisConnectionError) as failure:
        RedisCacheBackend(lazy_connect=lazy)
    assert closed == ["pool"]
    assert "synthetic-private-redis-construction" not in str(failure.value)


def test_native_redis_close_attempts_client_and_pool_and_surfaces_failure():
    from types import SimpleNamespace

    from truthound.profiler.caching import RedisCacheBackend

    events = []

    def client_close():
        events.append("client")
        raise OSError("synthetic-private-redis-close")

    native = object.__new__(RedisCacheBackend)
    native._client = SimpleNamespace(close=client_close)
    native._pool = SimpleNamespace(disconnect=lambda: events.append("pool"))
    native._connected = True
    with pytest.raises(RuntimeError) as failure:
        native.close()
    assert events == ["client", "pool"] and not native._connected
    assert "synthetic-private-redis-close" not in str(failure.value)
