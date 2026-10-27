"""Cached store failures must not become missing records or partial success."""

from datetime import datetime

import pytest

from truthound.stores.backends.memory import MemoryStore
from truthound.stores.caching.store import CachedStore
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("operation", ["get", "get_many", "refresh", "warm"])
def test_cached_reads_preserve_underlying_failure(operation):
    class BrokenStore(MemoryStore):
        def get(self, item_id):
            raise OSError("private native storage failure")

    store = CachedStore(BrokenStore())
    argument = ["missing"] if operation in {"get_many", "warm"} else "missing"
    with pytest.raises(OSError):
        getattr(store, operation)(argument)


def test_refresh_deleted_record_invalidates_old_cache():
    native = MemoryStore()
    store = CachedStore(native)
    item = ValidationResult("old", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    identity = store.save(item)
    native.delete(identity)
    assert store.refresh(identity) is None
    assert store.get(identity) is None and not store.exists(identity)


def test_write_behind_owns_snapshot_and_flushes_before_close():
    from threading import Event

    from truthound.stores.caching.store import CacheMode

    entered, release = Event(), Event()

    class BlockedStore(MemoryStore):
        def save(self, item):
            entered.set()
            assert release.wait(5)
            return super().save(item)

    native = BlockedStore()
    store = CachedStore(native, mode=CacheMode.WRITE_BEHIND)
    item = ValidationResult("queued", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    try:
        identity = store.save(item)
        assert entered.wait(5) and not native.exists(identity)
        item.data_asset = "changed"
        assert store.get_stats()["write_behind"]["pending"] == 1
    finally:
        release.set()
        store.close()
    assert native.get(identity).data_asset == "asset"
    assert store.get_stats()["write_behind"]["closed"]


@pytest.mark.parametrize("action", ["confirm", "discard", "retry"])
def test_uncertain_write_is_fenced_and_requires_exact_reconciliation(action):
    from truthound.stores.base import StoreWriteError
    from truthound.stores.caching.store import CacheMode

    class FailedReceipt(MemoryStore):
        calls = 0

        def save(self, item):
            self.calls += 1
            result = super().save(item)
            if self.calls == 1:
                raise OSError("receipt lost after write")
            return result

    native = FailedReceipt()
    store = CachedStore(native, mode=CacheMode.WRITE_BEHIND)
    item = ValidationResult("uncertain", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    store.save(item)
    with pytest.raises(StoreWriteError):
        store.flush()
    token = store.get_stats()["write_behind"]["uncertain_token"]
    with pytest.raises(StoreWriteError):
        store.reconcile_write("wrong", "confirm")
    with pytest.raises(StoreWriteError):
        store.save(item)
    assert native.calls == 1
    store.reconcile_write(token, action, accept_duplicate_risk=action == "retry")
    store.close()
    assert native.calls == (2 if action == "retry" else 1)


@pytest.mark.parametrize("mode", ["read_through", "cache_aside"])
def test_cache_modes_do_not_populate_on_save(mode):
    from truthound.stores.caching.store import CacheMode

    native = MemoryStore()
    store = CachedStore(native, mode=CacheMode(mode))
    item = ValidationResult("mode", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    identity = store.save(item)
    assert not store.cache.exists("store:" + identity)
    assert store.get(identity).id == identity
    assert store.cache.exists("store:" + identity) == (mode == "read_through")


@pytest.mark.parametrize("backend", ["InMemoryCache", "LRUCache", "LFUCache", "TTLCache"])
def test_native_cache_memory_limit_rejects_oversize_without_losing_previous(backend):
    from truthound.stores.caching import backends
    from truthound.stores.caching.base import CacheConfig

    cache = getattr(backends, backend)(config=CacheConfig(max_memory_mb=0.001))
    cache.set("kept", "small")
    with pytest.raises(MemoryError):
        cache.set("kept", b"x" * 4096)
    assert cache.get("kept") == "small"


def test_configured_startup_warming_and_background_refresh_are_owned():
    from threading import Event

    from truthound.stores.caching.base import CacheConfig

    refreshed = Event()

    class ObservedStore(MemoryStore):
        reads = 0

        def get(self, identity):
            result = super().get(identity)
            self.reads += 1
            if self.reads > 1:
                refreshed.set()
            return result

    native = ObservedStore()
    item = ValidationResult("refresh", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    native.save(item)
    store = CachedStore(
        native,
        cache_config=CacheConfig(
            warm_on_startup=True,
            background_refresh=True,
            ttl_seconds=0.1,
            refresh_threshold_percent=80,
        ),
    )
    try:
        assert store.cache.exists("store:" + item.id)
        assert refreshed.wait(3)
        assert store.get_stats()["maintenance"]["running"]
    finally:
        store.close()
    assert not store.get_stats()["maintenance"]["running"]


def test_background_refresh_failure_remains_visible_and_close_joins():
    from threading import Event

    from truthound.stores.caching.base import CacheConfig

    attempted = Event()

    class FailedRefresh(MemoryStore):
        reads = 0

        def get(self, identity):
            self.reads += 1
            if self.reads > 1:
                attempted.set()
                raise OSError("private refresh failure")
            return super().get(identity)

    native = FailedRefresh()
    native.save(
        ValidationResult("refresh-failure", datetime(2026, 1, 1), "asset", ResultStatus.SUCCESS)
    )
    store = CachedStore(
        native,
        cache_config=CacheConfig(
            warm_on_startup=True,
            background_refresh=True,
            ttl_seconds=0.1,
            refresh_threshold_percent=80,
        ),
    )
    assert attempted.wait(3)
    store.close()
    stats = store.get_stats()["maintenance"]
    assert stats["failures"] >= 1 and not stats["running"]
    assert "private refresh failure" not in str(stats)


@pytest.mark.parametrize("backend", ["InMemoryCache", "LRUCache", "LFUCache", "TTLCache"])
def test_cache_estimated_memory_eviction_and_disabled_counters(backend):
    from truthound.stores.caching import backends
    from truthound.stores.caching.base import CacheConfig

    cache = getattr(backends, backend)(
        config=CacheConfig(max_memory_mb=0.001, enable_statistics=False)
    )
    cache.set("first", b"a" * 600)
    cache.set("second", b"b" * 600)
    assert cache.get("first") is None and cache.get("second") == b"b" * 600
    assert cache.metrics.memory_bytes <= 1024 and cache.metrics.size == 1
    assert (
        cache.metrics.sets
        == cache.metrics.hits
        == cache.metrics.misses
        == cache.metrics.evictions
        == 0
    )
