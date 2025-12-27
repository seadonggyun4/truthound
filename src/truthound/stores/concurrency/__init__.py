"""Concurrency control for filesystem stores.

This module provides thread-safe and process-safe concurrency primitives
for filesystem operations. It follows a layered architecture:

1. Lock Strategy Layer: Pluggable locking mechanisms (fcntl, portalocker, etc.)
2. Lock Manager Layer: High-level lock acquisition and management
3. Atomic Operations Layer: Safe file read/write operations
4. Transaction Layer: Multi-file atomic operations with rollback

Example:
    >>> from truthound.stores.concurrency import FileLockManager, AtomicFileWriter
    >>>
    >>> # Using lock manager directly
    >>> with FileLockManager.acquire("/path/to/file", LockMode.EXCLUSIVE):
    ...     # Safe to write
    ...     pass
    >>>
    >>> # Using atomic writer
    >>> with AtomicFileWriter("/path/to/file") as writer:
    ...     writer.write(b"content")
    ...     writer.commit()
"""

from truthound.stores.concurrency.locks import (
    LockMode,
    LockStrategy,
    LockHandle,
    FcntlLockStrategy,
    PortalockerStrategy,
    FileLockStrategy,
    NoOpLockStrategy,
    get_default_lock_strategy,
)
from truthound.stores.concurrency.manager import (
    FileLockManager,
    LockContext,
    LockTimeout,
    DeadlockError,
)
from truthound.stores.concurrency.atomic import (
    AtomicFileWriter,
    AtomicFileReader,
    AtomicOperation,
    atomic_write,
    atomic_read,
    safe_rename,
)
from truthound.stores.concurrency.index import (
    ConcurrentIndex,
    IndexEntry,
    IndexTransaction,
)

__all__ = [
    # Lock primitives
    "LockMode",
    "LockStrategy",
    "LockHandle",
    "FcntlLockStrategy",
    "PortalockerStrategy",
    "FileLockStrategy",
    "NoOpLockStrategy",
    "get_default_lock_strategy",
    # Lock management
    "FileLockManager",
    "LockContext",
    "LockTimeout",
    "DeadlockError",
    # Atomic operations
    "AtomicFileWriter",
    "AtomicFileReader",
    "AtomicOperation",
    "atomic_write",
    "atomic_read",
    "safe_rename",
    # Index management
    "ConcurrentIndex",
    "IndexEntry",
    "IndexTransaction",
]
