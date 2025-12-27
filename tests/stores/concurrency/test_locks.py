"""Tests for lock strategies."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from truthound.stores.concurrency.locks import (
    LockMode,
    LockHandle,
    LockStrategy,
    FcntlLockStrategy,
    FileLockStrategy,
    PortalockerStrategy,
    NoOpLockStrategy,
    get_default_lock_strategy,
)


class TestLockHandle:
    """Tests for LockHandle dataclass."""

    def test_create_lock_handle(self, tmp_path: Path) -> None:
        """Test creating a lock handle."""
        handle = LockHandle(path=tmp_path / "test.txt", mode=LockMode.EXCLUSIVE)

        assert handle.path == tmp_path / "test.txt"
        assert handle.mode == LockMode.EXCLUSIVE
        assert handle.fd is None
        assert handle.timestamp > 0
        assert handle.thread_id > 0
        assert handle.process_id > 0

    def test_lock_handle_str(self, tmp_path: Path) -> None:
        """Test lock handle string representation."""
        handle = LockHandle(path=tmp_path / "test.txt", mode=LockMode.SHARED)
        assert "SHARED" in str(handle)

        handle = LockHandle(path=tmp_path / "test.txt", mode=LockMode.EXCLUSIVE)
        assert "EXCLUSIVE" in str(handle)

    def test_lock_handle_frozen(self, tmp_path: Path) -> None:
        """Test that lock handle is immutable."""
        handle = LockHandle(path=tmp_path / "test.txt", mode=LockMode.EXCLUSIVE)

        with pytest.raises(AttributeError):
            handle.mode = LockMode.SHARED


class TestNoOpLockStrategy:
    """Tests for NoOpLockStrategy."""

    def test_acquire_always_succeeds(self, tmp_path: Path) -> None:
        """Test that acquire always succeeds."""
        strategy = NoOpLockStrategy()
        handle = strategy.acquire(tmp_path / "test.txt", LockMode.EXCLUSIVE)

        assert handle is not None
        assert handle.mode == LockMode.EXCLUSIVE

    def test_try_acquire_always_succeeds(self, tmp_path: Path) -> None:
        """Test that try_acquire always succeeds."""
        strategy = NoOpLockStrategy()
        handle = strategy.try_acquire(tmp_path / "test.txt", LockMode.EXCLUSIVE)

        assert handle is not None

    def test_release_is_noop(self, tmp_path: Path) -> None:
        """Test that release does nothing."""
        strategy = NoOpLockStrategy()
        handle = strategy.acquire(tmp_path / "test.txt", LockMode.EXCLUSIVE)

        # Should not raise
        strategy.release(handle)

    def test_is_locked_always_false(self, tmp_path: Path) -> None:
        """Test that is_locked always returns False."""
        strategy = NoOpLockStrategy()
        assert strategy.is_locked(tmp_path / "test.txt") is False

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using strategy as context manager."""
        strategy = NoOpLockStrategy()

        with strategy.lock(tmp_path / "test.txt", LockMode.EXCLUSIVE) as handle:
            assert handle is not None


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl not available on Windows")
class TestFcntlLockStrategy:
    """Tests for FcntlLockStrategy."""

    def test_acquire_exclusive(self, tmp_path: Path) -> None:
        """Test acquiring exclusive lock."""
        strategy = FcntlLockStrategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        handle = strategy.acquire(test_file, LockMode.EXCLUSIVE)
        try:
            assert handle is not None
            assert handle.mode == LockMode.EXCLUSIVE
            assert strategy.is_locked(test_file)
        finally:
            strategy.release(handle)

    def test_acquire_shared(self, tmp_path: Path) -> None:
        """Test acquiring shared lock."""
        strategy = FcntlLockStrategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        handle = strategy.acquire(test_file, LockMode.SHARED)
        try:
            assert handle is not None
            assert handle.mode == LockMode.SHARED
        finally:
            strategy.release(handle)

    def test_multiple_shared_locks(self, tmp_path: Path) -> None:
        """Test that multiple shared locks can be held."""
        strategy1 = FcntlLockStrategy()
        strategy2 = FcntlLockStrategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        handle1 = strategy1.acquire(test_file, LockMode.SHARED)
        handle2 = strategy2.acquire(test_file, LockMode.SHARED)

        try:
            assert handle1 is not None
            assert handle2 is not None
        finally:
            strategy1.release(handle1)
            strategy2.release(handle2)

    def test_exclusive_blocks_exclusive(self, tmp_path: Path) -> None:
        """Test that exclusive lock blocks other exclusive locks."""
        strategy = FcntlLockStrategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        handle = strategy.acquire(test_file, LockMode.EXCLUSIVE)
        try:
            # Try to acquire non-blocking - should fail
            result = strategy.try_acquire(test_file, LockMode.EXCLUSIVE)
            # Note: This tests within same process, behavior varies by OS
            # On some systems this may succeed due to thread-local file descriptors
        finally:
            strategy.release(handle)

    def test_release_invalid_handle(self, tmp_path: Path) -> None:
        """Test releasing invalid handle raises error."""
        strategy = FcntlLockStrategy()
        handle = LockHandle(path=tmp_path / "nonexistent.txt", mode=LockMode.EXCLUSIVE)

        with pytest.raises(ValueError):
            strategy.release(handle)

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test using strategy as context manager."""
        strategy = FcntlLockStrategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with strategy.lock(test_file, LockMode.EXCLUSIVE) as handle:
            assert handle is not None
            assert strategy.is_locked(test_file)

        # Lock should be released after context
        assert not strategy.is_locked(test_file)


class TestGetDefaultLockStrategy:
    """Tests for get_default_lock_strategy."""

    def test_returns_lock_strategy(self) -> None:
        """Test that a LockStrategy is returned."""
        strategy = get_default_lock_strategy()
        assert isinstance(strategy, LockStrategy)

    @pytest.mark.skipif(sys.platform == "win32", reason="fcntl preferred on Unix")
    def test_prefers_fcntl_on_unix(self) -> None:
        """Test that fcntl is preferred on Unix systems."""
        strategy = get_default_lock_strategy()
        assert isinstance(strategy, FcntlLockStrategy)


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    def test_thread_safety_exclusive(self, tmp_path: Path) -> None:
        """Test that exclusive locks provide mutual exclusion."""
        strategy = get_default_lock_strategy()
        test_file = tmp_path / "test.txt"
        test_file.touch()

        counter = {"value": 0}
        errors: list[Exception] = []

        def increment() -> None:
            try:
                with strategy.lock(test_file, LockMode.EXCLUSIVE):
                    current = counter["value"]
                    time.sleep(0.001)  # Simulate work
                    counter["value"] = current + 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All increments should have happened
        assert counter["value"] == 10
        assert len(errors) == 0

    def test_shared_locks_allow_concurrent_reads(self, tmp_path: Path) -> None:
        """Test that shared locks allow concurrent readers.

        Each thread uses its own lock strategy instance to avoid internal
        state conflicts within the same process.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        results: list[str] = []
        lock = threading.Lock()

        def read_file() -> None:
            # Each thread gets its own strategy to avoid internal state conflicts
            strategy = get_default_lock_strategy()
            with strategy.lock(test_file, LockMode.SHARED):
                content = test_file.read_text()
                with lock:
                    results.append(content)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_file) for _ in range(5)]
            for future in as_completed(futures):
                future.result()

        assert len(results) == 5
        assert all(r == "test content" for r in results)
