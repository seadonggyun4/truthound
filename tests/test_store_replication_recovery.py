"""Native replication failures must stay failures and owned work must drain."""

import asyncio
from datetime import datetime

import pytest

from truthound.stores.backends.memory import MemoryStore
from truthound.stores.base import StoreWriteError
from truthound.stores.replication.base import ReplicaTarget, ReplicationConfig, ReplicationMode
from truthound.stores.replication.store import ReplicatedStore
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.contract


class BrokenStore(MemoryStore):
    def save(self, item):
        raise StoreWriteError("synthetic failure")


def fixture_store(mode=ReplicationMode.SYNC, broken=False):
    primary, replica = MemoryStore(), BrokenStore() if broken else MemoryStore()
    target = ReplicaTarget("local", replica, "local", max_retry_attempts=1)
    target.mark_healthy()
    store = ReplicatedStore(
        primary, ReplicationConfig(mode=mode, targets=[target], enable_health_checks=False)
    )
    item = ValidationResult("local-run", datetime(2026, 1, 1), "local", ResultStatus.SUCCESS)
    return store, primary, replica, item


def test_native_failed_full_sync_and_batch_report_failure():
    async def run():
        store, primary, _, item = fixture_store(broken=True)
        primary.save(item)
        result = await store.full_sync()
        assert not result["local"]["success"]
        assert result["local"]["items_failed"] == 1

    asyncio.run(run())


def test_native_synchronous_replica_failure_does_not_return_save_success():
    store, primary, _, item = fixture_store(broken=True)
    with pytest.raises(StoreWriteError):
        store.save(item)
    assert primary.exists(item.id)  # Partial primary effect remains; no retry claim.


def test_native_async_sync_replica_failure_does_not_return_save_success():
    async def run():
        store, primary, _, item = fixture_store(broken=True)
        with pytest.raises(StoreWriteError):
            await store.save_async(item)
        assert primary.exists(item.id)

    asyncio.run(run())


def test_native_stop_drains_save_async_work_without_starting_monitor():
    async def run():
        store, _, replica, item = fixture_store(ReplicationMode.ASYNC)
        await store.save_async(item)
        await store.stop()
        assert replica.exists(item.id)

    asyncio.run(run())


def test_native_stop_drains_sync_save_queue_before_worker_shutdown():
    async def run():
        store, _, replica, item = fixture_store(ReplicationMode.ASYNC)
        await store.start()
        store.save(item)
        await store.stop()
        assert replica.exists(item.id)

    asyncio.run(run())


def test_native_replica_guard_runs_inside_background_io_before_read_or_write():
    async def run():
        store, primary, replica, item = fixture_store(ReplicationMode.ASYNC)
        active = True

        def guard():
            if not active:
                raise PermissionError("revoked")

        store.set_io_guard(guard)
        primary.save(item)
        active = False
        result = await store.full_sync()
        assert not result["local"]["success"]
        assert not replica.exists(item.id)

    asyncio.run(run())


def test_failed_full_sync_keeps_replica_unhealthy_for_read_routing():
    from truthound.stores.replication.base import ReplicaHealth

    async def run():
        store, primary, _, item = fixture_store(broken=True)
        primary.save(item)
        await store.full_sync()
        assert store.config.targets[0].health == ReplicaHealth.UNHEALTHY

    asyncio.run(run())


def test_synchronous_save_in_event_loop_rejects_before_primary_write():
    async def run():
        store, primary, _, item = fixture_store()
        with pytest.raises(RuntimeError, match="save_async"):
            store.save(item)
        assert not primary.exists(item.id)

    asyncio.run(run())


def test_replication_monitor_failure_is_visible_bounded_and_other_targets_checked():
    from truthound.stores.replication.base import ReplicaHealth
    from truthound.stores.replication.monitor import ReplicationMonitor

    async def run():
        targets = [ReplicaTarget(name, MemoryStore(), "local") for name in ("bad", "good")]
        calls = []
        class Syncer:
            async def check_health(self, target):
                calls.append(target.name)
                await asyncio.sleep(0)
                if target.name == "bad":
                    raise RuntimeError("private health details")
                target.mark_healthy()
                return ReplicaHealth.HEALTHY
        monitor = ReplicationMonitor(ReplicationConfig(targets=targets, health_check_interval_seconds=0.05), Syncer())
        await monitor.start()
        await asyncio.sleep(0.02)
        await monitor.stop()
        summary = monitor.get_health_summary()
        assert calls == ["bad", "good"]
        assert summary["monitor_failures"] == 1
        assert summary["running"] is False and summary["errors"]
        assert "private health details" not in str(summary)
    asyncio.run(run())


def test_replication_monitor_awaits_async_handlers_and_records_partial_failures():
    from truthound.stores.replication.monitor import (
        ReplicationEvent,
        ReplicationEventType,
        ReplicationMonitor,
    )

    async def run():
        monitor = ReplicationMonitor(ReplicationConfig(), None)
        observed = []
        def bad(event):
            raise ValueError("private callback details")
        async def good(event):
            await asyncio.sleep(0)
            observed.append(event.event_type)
        monitor.add_handler(bad)
        monitor.add_handler(good)
        event = ReplicationEvent(ReplicationEventType.REPLICA_HEALTHY, "local")
        with pytest.raises(ExceptionGroup):
            await monitor._emit_event(event)
        assert observed == [event.event_type]
        assert monitor.events == [event]
        summary = monitor.get_health_summary()
        assert summary["handler_failures"] == 1
        assert "private callback details" not in str(summary)
    asyncio.run(run())
