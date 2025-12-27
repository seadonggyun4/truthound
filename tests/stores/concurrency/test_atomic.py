"""Tests for atomic file operations."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from truthound.stores.concurrency.atomic import (
    AtomicFileWriter,
    AtomicFileReader,
    AtomicOperation,
    atomic_write,
    atomic_read,
    safe_rename,
    atomic_update,
)
from truthound.stores.concurrency.manager import FileLockManager
from truthound.stores.concurrency.locks import NoOpLockStrategy


@pytest.fixture
def lock_manager() -> FileLockManager:
    """Create a lock manager for tests."""
    return FileLockManager(strategy=NoOpLockStrategy())


class TestAtomicFileWriter:
    """Tests for AtomicFileWriter."""

    def test_basic_write(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test basic atomic write."""
        target = tmp_path / "test.txt"
        content = b"Hello, World!"

        with AtomicFileWriter(target, lock_manager=lock_manager, use_lock=False) as writer:
            writer.write(content)
            result = writer.commit()

        assert result.success
        assert target.exists()
        assert target.read_bytes() == content

    def test_write_creates_parent_dirs(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test that parent directories are created."""
        target = tmp_path / "subdir" / "deep" / "test.txt"

        with AtomicFileWriter(target, lock_manager=lock_manager, use_lock=False) as writer:
            writer.write(b"content")
            writer.commit()

        assert target.exists()

    def test_write_with_backup(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test creating backup before overwrite."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"original")

        with AtomicFileWriter(
            target,
            create_backup=True,
            lock_manager=lock_manager,
            use_lock=False,
        ) as writer:
            writer.write(b"new content")
            result = writer.commit()

        assert result.success
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.backup_path.read_bytes() == b"original"
        assert target.read_bytes() == b"new content"

    def test_write_with_checksum(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test computing checksum on write."""
        target = tmp_path / "test.txt"

        with AtomicFileWriter(
            target,
            compute_checksum=True,
            lock_manager=lock_manager,
            use_lock=False,
        ) as writer:
            writer.write(b"test")
            result = writer.commit()

        assert result.success
        assert result.checksum is not None
        assert len(result.checksum) == 64  # SHA-256 hex

    def test_no_commit_cleans_up(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test that temp file is cleaned up if not committed."""
        target = tmp_path / "test.txt"

        with AtomicFileWriter(target, lock_manager=lock_manager, use_lock=False) as writer:
            writer.write(b"content")
            # Don't commit

        assert not target.exists()
        # No temp files should remain
        assert len(list(tmp_path.glob(".*"))) == 0

    def test_exception_cleans_up(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test cleanup on exception."""
        target = tmp_path / "test.txt"

        with pytest.raises(ValueError):
            with AtomicFileWriter(
                target, lock_manager=lock_manager, use_lock=False
            ) as writer:
                writer.write(b"content")
                raise ValueError("Test error")

        assert not target.exists()

    def test_double_commit_raises(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test that double commit raises error."""
        target = tmp_path / "test.txt"

        with AtomicFileWriter(target, lock_manager=lock_manager, use_lock=False) as writer:
            writer.write(b"content")
            writer.commit()

            with pytest.raises(RuntimeError):
                writer.commit()

    def test_write_after_commit_raises(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test that write after commit raises error."""
        target = tmp_path / "test.txt"

        with AtomicFileWriter(target, lock_manager=lock_manager, use_lock=False) as writer:
            writer.write(b"content")
            writer.commit()

            with pytest.raises(RuntimeError):
                writer.write(b"more")


class TestAtomicFileReader:
    """Tests for AtomicFileReader."""

    def test_basic_read(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test basic atomic read."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"Hello, World!")

        with AtomicFileReader(target, lock_manager=lock_manager, use_lock=False) as reader:
            content = reader.read()

        assert content == b"Hello, World!"

    def test_read_lines(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test reading lines."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"line1\nline2\nline3")

        with AtomicFileReader(target, lock_manager=lock_manager, use_lock=False) as reader:
            lines = reader.readlines()

        assert lines == [b"line1\n", b"line2\n", b"line3"]

    def test_read_with_checksum_validation(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test checksum validation on read."""
        target = tmp_path / "test.txt"
        content = b"test content"
        target.write_bytes(content)

        # Calculate expected checksum
        import hashlib

        expected = hashlib.sha256(content).hexdigest()

        with AtomicFileReader(
            target,
            validate_checksum=expected,
            lock_manager=lock_manager,
            use_lock=False,
        ) as reader:
            result = reader.read()

        assert result == content

    def test_read_with_invalid_checksum(
        self, tmp_path: Path, lock_manager: FileLockManager
    ) -> None:
        """Test that invalid checksum raises error."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"test content")

        with pytest.raises(ValueError, match="Checksum mismatch"):
            with AtomicFileReader(
                target,
                validate_checksum="invalid",
                lock_manager=lock_manager,
                use_lock=False,
            ) as reader:
                reader.read()


class TestAtomicWriteFunction:
    """Tests for atomic_write convenience function."""

    def test_write_bytes(self, tmp_path: Path) -> None:
        """Test writing bytes."""
        target = tmp_path / "test.txt"
        result = atomic_write(target, b"content")

        assert result.success
        assert target.read_bytes() == b"content"

    def test_write_string(self, tmp_path: Path) -> None:
        """Test writing string."""
        target = tmp_path / "test.txt"
        result = atomic_write(target, "content")

        assert result.success
        assert target.read_bytes() == b"content"

    def test_write_with_compression(self, tmp_path: Path) -> None:
        """Test writing with compression."""
        import gzip

        target = tmp_path / "test.txt.gz"
        result = atomic_write(target, b"content", compress=True)

        assert result.success
        assert gzip.decompress(target.read_bytes()) == b"content"


class TestAtomicReadFunction:
    """Tests for atomic_read convenience function."""

    def test_read_bytes(self, tmp_path: Path) -> None:
        """Test reading bytes."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"content")

        result = atomic_read(target)
        assert result == b"content"

    def test_read_with_decompression(self, tmp_path: Path) -> None:
        """Test reading with decompression."""
        import gzip

        target = tmp_path / "test.txt.gz"
        target.write_bytes(gzip.compress(b"content"))

        result = atomic_read(target, decompress=True)
        assert result == b"content"


class TestSafeRename:
    """Tests for safe_rename function."""

    def test_basic_rename(self, tmp_path: Path) -> None:
        """Test basic file rename."""
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_bytes(b"content")

        result = safe_rename(source, target)

        assert result.success
        assert not source.exists()
        assert target.read_bytes() == b"content"

    def test_rename_with_backup(self, tmp_path: Path) -> None:
        """Test rename with backup of existing target."""
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_bytes(b"new content")
        target.write_bytes(b"old content")

        result = safe_rename(source, target, create_backup=True)

        assert result.success
        assert result.backup_path is not None
        assert result.backup_path.read_bytes() == b"old content"
        assert target.read_bytes() == b"new content"


class TestAtomicUpdate:
    """Tests for atomic_update context manager."""

    def test_update_existing_file(self, tmp_path: Path) -> None:
        """Test updating an existing file."""
        target = tmp_path / "counter.txt"
        target.write_bytes(b"5")

        with atomic_update(target) as (content, write):
            count = int(content or b"0")
            result = write(str(count + 1).encode())

        assert result.success
        assert target.read_bytes() == b"6"

    def test_update_new_file(self, tmp_path: Path) -> None:
        """Test creating new file with update."""
        target = tmp_path / "counter.txt"

        with atomic_update(target) as (content, write):
            assert content is None
            write(b"1")

        assert target.read_bytes() == b"1"


class TestConcurrentWrites:
    """Tests for concurrent write scenarios."""

    def test_concurrent_writes_to_same_file(self, tmp_path: Path) -> None:
        """Test multiple threads writing to the same file."""
        target = tmp_path / "test.txt"
        results: list[AtomicOperation] = []
        lock = threading.Lock()

        def write_content(i: int) -> None:
            result = atomic_write(target, f"content-{i}")
            with lock:
                results.append(result)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_content, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # All writes should succeed (last one wins)
        assert all(r.success for r in results)
        assert target.exists()

    def test_concurrent_writes_to_different_files(self, tmp_path: Path) -> None:
        """Test multiple threads writing to different files."""

        def write_content(i: int) -> AtomicOperation:
            target = tmp_path / f"test-{i}.txt"
            return atomic_write(target, f"content-{i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_content, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]

        assert all(r.success for r in results)
        assert len(list(tmp_path.glob("test-*.txt"))) == 20

    def test_read_write_concurrency(self, tmp_path: Path) -> None:
        """Test concurrent reads and writes to different files."""
        # Use different files for different operations to avoid lock contention
        read_results: list[dict] = []
        write_results: list[bool] = []
        lock = threading.Lock()

        def read_file(i: int) -> None:
            target = tmp_path / f"read_test_{i}.json"
            target.write_bytes(json.dumps({"count": i}).encode())
            content = atomic_read(target)
            data = json.loads(content)
            with lock:
                read_results.append(data)

        def write_file(i: int) -> None:
            target = tmp_path / f"write_test_{i}.json"
            result = atomic_write(target, json.dumps({"count": i}))
            with lock:
                write_results.append(result.success)

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Mix reads and writes to different files
            futures = []
            for i in range(50):
                if i % 3 == 0:
                    futures.append(executor.submit(write_file, i))
                else:
                    futures.append(executor.submit(read_file, i))

            for future in as_completed(futures):
                future.result()

        # All writes should succeed
        assert all(write_results)
        # All reads should return valid JSON
        assert all("count" in r for r in read_results)
