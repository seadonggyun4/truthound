"""Atomic file operations for safe concurrent access.

This module provides atomic file operations that ensure data integrity
even when multiple processes are accessing the same files.

Key concepts:
- Write-to-temp-then-rename pattern for atomic writes
- Read with retry for handling concurrent modifications
- Transaction support for multi-file operations

Example:
    >>> # Atomic write
    >>> with AtomicFileWriter("/path/to/file.json") as writer:
    ...     writer.write(json.dumps(data).encode())
    ...     writer.commit()
    >>>
    >>> # Simple atomic write
    >>> atomic_write(Path("/path/to/file.json"), content)
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from truthound.stores.concurrency.locks import LockMode
from truthound.stores.concurrency.manager import FileLockManager, get_default_manager


@dataclass
class AtomicOperation:
    """Represents an atomic operation result.

    Attributes:
        success: Whether the operation succeeded.
        path: Path that was operated on.
        backup_path: Path to backup file (if created).
        error: Error message if operation failed.
        bytes_written: Number of bytes written (for write ops).
        checksum: Checksum of written data (if computed).
    """

    success: bool
    path: Path
    backup_path: Path | None = None
    error: str | None = None
    bytes_written: int = 0
    checksum: str | None = None


class AtomicFileWriter:
    """Atomic file writer using write-to-temp-then-rename pattern.

    This class ensures that file writes are atomic by:
    1. Writing to a temporary file
    2. Syncing the temp file to disk
    3. Atomically renaming temp file to target

    If any step fails, the original file remains unchanged.

    Example:
        >>> with AtomicFileWriter("/path/to/file.json") as writer:
        ...     writer.write(b'{"key": "value"}')
        ...     writer.commit()
        >>>
        >>> # With automatic cleanup on failure
        >>> with AtomicFileWriter("/path/to/file.json") as writer:
        ...     writer.write(b'data')
        ...     raise ValueError("Something went wrong")
        ...     # Temp file is automatically cleaned up
    """

    def __init__(
        self,
        path: Path | str,
        mode: str = "wb",
        create_backup: bool = False,
        sync_on_commit: bool = True,
        lock_manager: FileLockManager | None = None,
        use_lock: bool = True,
        compute_checksum: bool = False,
    ) -> None:
        """Initialize the atomic writer.

        Args:
            path: Target file path.
            mode: File open mode ("wb" or "w").
            create_backup: Whether to backup existing file before overwrite.
            sync_on_commit: Whether to fsync before rename.
            lock_manager: Lock manager to use. None uses default.
            use_lock: Whether to acquire exclusive lock.
            compute_checksum: Whether to compute checksum of written data.
        """
        self._path = Path(path)
        self._mode = mode
        self._create_backup = create_backup
        self._sync_on_commit = sync_on_commit
        self._lock_manager = lock_manager or get_default_manager()
        self._use_lock = use_lock
        self._compute_checksum = compute_checksum

        self._temp_file: Any = None
        self._temp_path: Path | None = None
        self._backup_path: Path | None = None
        self._committed = False
        self._bytes_written = 0
        self._hasher = hashlib.sha256() if compute_checksum else None
        self._lock_handle = None

    def __enter__(self) -> "AtomicFileWriter":
        """Enter context and create temp file."""
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Acquire lock if enabled
        if self._use_lock:
            self._lock_handle = self._lock_manager.lock(
                self._path,
                LockMode.EXCLUSIVE,
            )

        # Create temp file in same directory for atomic rename
        fd, temp_path = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        self._temp_path = Path(temp_path)

        # Open with specified mode
        os.close(fd)
        self._temp_file = open(self._temp_path, self._mode)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and cleanup."""
        try:
            if self._temp_file and not self._temp_file.closed:
                self._temp_file.close()

            # Clean up temp file if not committed
            if not self._committed and self._temp_path and self._temp_path.exists():
                self._temp_path.unlink(missing_ok=True)

            # Clean up backup if commit failed
            if not self._committed and self._backup_path and self._backup_path.exists():
                # Restore from backup
                if self._path.exists():
                    self._path.unlink()
                self._backup_path.rename(self._path)

        finally:
            # Release lock
            if self._lock_handle:
                self._lock_manager.unlock(self._lock_handle)

    def write(self, data: bytes | str) -> int:
        """Write data to the temp file.

        Args:
            data: Data to write.

        Returns:
            Number of bytes written.
        """
        if self._committed:
            raise RuntimeError("Cannot write after commit")

        if isinstance(data, str) and "b" in self._mode:
            data = data.encode("utf-8")
        elif isinstance(data, bytes) and "b" not in self._mode:
            data = data.decode("utf-8")

        count = self._temp_file.write(data)
        self._bytes_written += count if isinstance(count, int) else len(data)

        if self._hasher and isinstance(data, bytes):
            self._hasher.update(data)
        elif self._hasher and isinstance(data, str):
            self._hasher.update(data.encode("utf-8"))

        return count

    def writelines(self, lines: list[bytes | str]) -> None:
        """Write multiple lines to the temp file.

        Args:
            lines: Lines to write.
        """
        for line in lines:
            self.write(line)

    def commit(self) -> AtomicOperation:
        """Commit the write by renaming temp file to target.

        Returns:
            AtomicOperation with result details.

        Raises:
            RuntimeError: If already committed or not in context.
        """
        if self._committed:
            raise RuntimeError("Already committed")

        if self._temp_file is None:
            raise RuntimeError("Must be used within context manager")

        try:
            # Close temp file
            self._temp_file.close()

            # Sync to disk if requested
            if self._sync_on_commit:
                self._sync_file(self._temp_path)

            # Create backup if requested and file exists
            if self._create_backup and self._path.exists():
                self._backup_path = self._path.with_suffix(
                    f"{self._path.suffix}.bak.{int(time.time())}"
                )
                shutil.copy2(self._path, self._backup_path)

            # Atomic rename
            self._temp_path.replace(self._path)

            self._committed = True

            return AtomicOperation(
                success=True,
                path=self._path,
                backup_path=self._backup_path,
                bytes_written=self._bytes_written,
                checksum=self._hasher.hexdigest() if self._hasher else None,
            )

        except Exception as e:
            return AtomicOperation(
                success=False,
                path=self._path,
                error=str(e),
            )

    def _sync_file(self, path: Path) -> None:
        """Sync file to disk."""
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @property
    def checksum(self) -> str | None:
        """Get checksum of written data."""
        if self._hasher:
            return self._hasher.hexdigest()
        return None


class AtomicFileReader:
    """Thread-safe file reader with retry logic.

    Handles reading files that may be modified by concurrent writers.
    Retries on temporary failures and validates data integrity.

    Example:
        >>> with AtomicFileReader("/path/to/file.json") as reader:
        ...     data = reader.read()
    """

    def __init__(
        self,
        path: Path | str,
        mode: str = "rb",
        lock_manager: FileLockManager | None = None,
        use_lock: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        validate_checksum: str | None = None,
    ) -> None:
        """Initialize the atomic reader.

        Args:
            path: File path to read.
            mode: File open mode ("rb" or "r").
            lock_manager: Lock manager to use.
            use_lock: Whether to acquire shared lock.
            max_retries: Maximum number of retries on failure.
            retry_delay: Delay between retries in seconds.
            validate_checksum: Expected SHA-256 checksum to validate.
        """
        self._path = Path(path)
        self._mode = mode
        self._lock_manager = lock_manager or get_default_manager()
        self._use_lock = use_lock
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._validate_checksum = validate_checksum

        self._file: Any = None
        self._lock_handle = None
        self._content: bytes | str | None = None

    def __enter__(self) -> "AtomicFileReader":
        """Enter context and open file."""
        if self._use_lock:
            self._lock_handle = self._lock_manager.lock(
                self._path,
                LockMode.SHARED,
            )

        for attempt in range(self._max_retries):
            try:
                self._file = open(self._path, self._mode)
                return self
            except (OSError, IOError) as e:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(self._retry_delay)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and cleanup."""
        if self._file and not self._file.closed:
            self._file.close()

        if self._lock_handle:
            self._lock_manager.unlock(self._lock_handle)

    def read(self, size: int = -1) -> bytes | str:
        """Read data from the file.

        Args:
            size: Number of bytes to read. -1 for all.

        Returns:
            File content.

        Raises:
            ValueError: If checksum validation fails.
        """
        if self._file is None:
            raise RuntimeError("Must be used within context manager")

        content = self._file.read(size)

        # Validate checksum if requested
        if self._validate_checksum and size == -1:
            if isinstance(content, str):
                actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
            else:
                actual = hashlib.sha256(content).hexdigest()

            if actual != self._validate_checksum:
                raise ValueError(
                    f"Checksum mismatch: expected {self._validate_checksum}, "
                    f"got {actual}"
                )

        return content

    def readline(self) -> bytes | str:
        """Read a line from the file."""
        if self._file is None:
            raise RuntimeError("Must be used within context manager")
        return self._file.readline()

    def readlines(self) -> list[bytes | str]:
        """Read all lines from the file."""
        if self._file is None:
            raise RuntimeError("Must be used within context manager")
        return self._file.readlines()

    def __iter__(self) -> Iterator[bytes | str]:
        """Iterate over lines in the file."""
        if self._file is None:
            raise RuntimeError("Must be used within context manager")
        return iter(self._file)


def atomic_write(
    path: Path | str,
    content: bytes | str,
    create_backup: bool = False,
    lock_manager: FileLockManager | None = None,
    compress: bool = False,
) -> AtomicOperation:
    """Convenience function for atomic file write.

    Args:
        path: Target file path.
        content: Content to write.
        create_backup: Whether to backup existing file.
        lock_manager: Lock manager to use.
        compress: Whether to gzip compress content.

    Returns:
        AtomicOperation with result.

    Example:
        >>> result = atomic_write(Path("data.json"), json.dumps(data))
        >>> if result.success:
        ...     print(f"Wrote {result.bytes_written} bytes")
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    if compress:
        content = gzip.compress(content)

    mode = "wb"

    with AtomicFileWriter(
        path,
        mode=mode,
        create_backup=create_backup,
        lock_manager=lock_manager,
    ) as writer:
        writer.write(content)
        return writer.commit()


def atomic_read(
    path: Path | str,
    lock_manager: FileLockManager | None = None,
    decompress: bool = False,
) -> bytes:
    """Convenience function for atomic file read.

    Args:
        path: File path to read.
        lock_manager: Lock manager to use.
        decompress: Whether to gzip decompress content.

    Returns:
        File content as bytes.

    Example:
        >>> content = atomic_read(Path("data.json"))
        >>> data = json.loads(content)
    """
    with AtomicFileReader(path, lock_manager=lock_manager) as reader:
        content = reader.read()

    if isinstance(content, str):
        content = content.encode("utf-8")

    if decompress:
        content = gzip.decompress(content)

    return content


def safe_rename(
    source: Path | str,
    target: Path | str,
    lock_manager: FileLockManager | None = None,
    create_backup: bool = False,
) -> AtomicOperation:
    """Safely rename a file with locking.

    Args:
        source: Source file path.
        target: Target file path.
        lock_manager: Lock manager to use.
        create_backup: Whether to backup existing target.

    Returns:
        AtomicOperation with result.
    """
    source = Path(source)
    target = Path(target)
    manager = lock_manager or get_default_manager()

    backup_path = None

    try:
        # Lock both source and target
        with manager.acquire(source, LockMode.EXCLUSIVE):
            with manager.acquire(target, LockMode.EXCLUSIVE):
                # Create backup if requested
                if create_backup and target.exists():
                    backup_path = target.with_suffix(
                        f"{target.suffix}.bak.{int(time.time())}"
                    )
                    shutil.copy2(target, backup_path)

                # Rename
                source.replace(target)

                return AtomicOperation(
                    success=True,
                    path=target,
                    backup_path=backup_path,
                )

    except Exception as e:
        return AtomicOperation(
            success=False,
            path=target,
            error=str(e),
            backup_path=backup_path,
        )


@contextmanager
def atomic_update(
    path: Path | str,
    lock_manager: FileLockManager | None = None,
    create_backup: bool = True,
) -> Iterator[tuple[bytes | None, Callable[[bytes], AtomicOperation]]]:
    """Context manager for read-modify-write operations.

    Provides atomic read-modify-write pattern for updating files.

    Args:
        path: File path to update.
        lock_manager: Lock manager to use.
        create_backup: Whether to create backup.

    Yields:
        Tuple of (current_content, write_function).

    Example:
        >>> with atomic_update(Path("counter.txt")) as (content, write):
        ...     count = int(content or b"0")
        ...     result = write(str(count + 1).encode())
    """
    path = Path(path)
    manager = lock_manager or get_default_manager()

    with manager.acquire(path, LockMode.EXCLUSIVE):
        # Read current content
        current_content: bytes | None = None
        if path.exists():
            with open(path, "rb") as f:
                current_content = f.read()

        def write_content(new_content: bytes) -> AtomicOperation:
            return atomic_write(
                path,
                new_content,
                create_backup=create_backup,
                lock_manager=manager,
            )

        yield current_content, write_content
