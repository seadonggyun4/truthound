"""Native lock detection distinguishes queued contention from a wait cycle."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest

from truthound.stores.concurrency.locks import LockMode
from truthound.stores.concurrency.manager import DeadlockError, FileLockManager

pytestmark = pytest.mark.fault


def test_native_ordered_nested_locks_wait_without_false_deadlock(tmp_path):
    manager = FileLockManager(default_timeout=2)
    held, waiting = Event(), Event()
    a, b = tmp_path / "a", tmp_path / "b"

    def holder():
        with manager.write_lock(b):
            held.set()
            assert waiting.wait(1)

    def waiter():
        assert held.wait(1)
        with manager.write_lock(a):
            waiting.set()
            with manager.write_lock(b):
                return "acquired"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(holder), pool.submit(waiter)
        first.result(timeout=3)
        assert second.result(timeout=3) == "acquired"
    assert manager.statistics.deadlocks == 0


def test_native_real_thread_wait_cycle_is_rejected_and_locks_release(tmp_path):
    manager = FileLockManager(default_timeout=2)
    barrier = Barrier(2)
    paths = [tmp_path / "a", tmp_path / "b"]

    def owner(number):
        try:
            with manager.write_lock(paths[number]):
                barrier.wait(timeout=1)
                with manager.write_lock(paths[1 - number]):
                    return "completed"
        except DeadlockError:
            return "cycle"

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(owner, number) for number in (0, 1)]
        results = [task.result(timeout=4) for task in tasks]
    assert sorted(results) == ["completed", "cycle"]
    assert manager.get_held_locks() == []
    for path in paths:
        handle = manager.lock(Path(path), LockMode.EXCLUSIVE, blocking=False)
        manager.unlock(handle)
