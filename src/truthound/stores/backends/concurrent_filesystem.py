"""Concurrent filesystem store with full concurrency control.

This module provides a thread-safe and process-safe filesystem store
implementation that builds on the concurrency primitives in the
concurrency submodule.

Features:
- Thread-safe operations via locking
- Process-safe operations via file locks
- Atomic writes with write-to-temp-then-rename
- Consistent index reads via snapshots
- Transaction support for batch operations
- Automatic recovery from failures

Example:
    >>> from truthound.stores.backends.concurrent_filesystem import (
    ...     ConcurrentFileSystemStore,
    ...     ConcurrencyConfig,
    ... )
    >>>
    >>> # Create store with concurrency enabled
    >>> store = ConcurrentFileSystemStore(
    ...     base_path=".truthound/results",
    ...     concurrency=ConcurrencyConfig(
    ...         lock_strategy="auto",
    ...         enable_wal=True,
    ...     ),
    ... )
    >>>
    >>> # Use with transaction for batch operations
    >>> with store.batch() as batch:
    ...     batch.save(result1)
    ...     batch.save(result2)
    ...     batch.commit()
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from truthound.stores.base import (
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
    ExpectationStore,
)
from truthound.stores.results import ValidationResult
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.concurrency.locks import (
    LockMode,
    LockStrategy,
    get_default_lock_strategy,
    NoOpLockStrategy,
    FcntlLockStrategy,
    FileLockStrategy,
    PortalockerStrategy,
)
from truthound.stores.concurrency.manager import FileLockManager, LockStatistics
from truthound.stores.concurrency.atomic import (
    AtomicFileWriter,
    AtomicFileReader,
    atomic_write,
    atomic_read,
)
from truthound.stores.concurrency.index import ConcurrentIndex, IndexTransaction


T = TypeVar("T")


class LockStrategyType(str, Enum):
    """Available lock strategy types."""

    AUTO = "auto"  # Auto-detect best strategy
    FCNTL = "fcntl"  # POSIX fcntl (Unix only)
    FILELOCK = "filelock"  # filelock library
    PORTALOCKER = "portalocker"  # portalocker library
    NONE = "none"  # No locking (single-threaded)


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency control.

    Attributes:
        lock_strategy: Which lock strategy to use.
        enable_wal: Whether to use write-ahead logging.
        lock_timeout: Default timeout for lock acquisition.
        enable_deadlock_detection: Whether to detect deadlocks.
        enable_statistics: Whether to collect lock statistics.
        create_backup: Whether to backup files before overwrite.
    """

    lock_strategy: LockStrategyType | str = LockStrategyType.AUTO
    enable_wal: bool = True
    lock_timeout: float = 30.0
    enable_deadlock_detection: bool = True
    enable_statistics: bool = True
    create_backup: bool = False


@dataclass
class ConcurrentFileSystemConfig(StoreConfig):
    """Configuration for concurrent filesystem store.

    Extends StoreConfig with concurrency-specific options.

    Attributes:
        base_path: Base directory for storing files.
        file_extension: File extension to use.
        create_dirs: Whether to create directories if missing.
        pretty_print: Whether to format JSON with indentation.
        use_compression: Whether to compress stored files.
        concurrency: Concurrency control configuration.
    """

    base_path: str = ".truthound/store"
    file_extension: str = ".json"
    create_dirs: bool = True
    pretty_print: bool = True
    use_compression: bool = False
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    def get_full_path(self) -> Path:
        """Get the full storage path including namespace and prefix."""
        path = Path(self.base_path)
        if self.namespace:
            path = path / self.namespace
        if self.prefix:
            path = path / self.prefix
        return path


class BatchContext:
    """Context for batch operations on the store.

    Provides transactional semantics for multiple store operations.
    All operations are buffered and applied atomically on commit.

    Example:
        >>> with store.batch() as batch:
        ...     batch.save(result1)
        ...     batch.save(result2)
        ...     batch.delete("old-result")
        ...     batch.commit()
    """

    def __init__(
        self,
        store: "ConcurrentFileSystemStore",
        index_txn: IndexTransaction,
    ) -> None:
        """Initialize batch context.

        Args:
            store: Parent store.
            index_txn: Index transaction for tracking changes.
        """
        self._store = store
        self._index_txn = index_txn
        self._pending_saves: list[ValidationResult] = []
        self._pending_deletes: list[str] = []
        self._committed = False

    def save(self, item: ValidationResult) -> str:
        """Add item to pending saves.

        Args:
            item: Item to save.

        Returns:
            The item's run ID.
        """
        if self._committed:
            raise RuntimeError("Batch already committed")

        self._pending_saves.append(item)
        return item.run_id

    def delete(self, item_id: str) -> None:
        """Add item to pending deletes.

        Args:
            item_id: Item to delete.
        """
        if self._committed:
            raise RuntimeError("Batch already committed")

        self._pending_deletes.append(item_id)

    def commit(self) -> int:
        """Commit all pending operations.

        Returns:
            Number of operations performed.

        Raises:
            RuntimeError: If already committed.
            StoreWriteError: If any operation fails.
        """
        if self._committed:
            raise RuntimeError("Batch already committed")

        count = 0
        errors = []

        # Perform deletes first
        for item_id in self._pending_deletes:
            try:
                self._store._do_delete(item_id, self._index_txn)
                count += 1
            except Exception as e:
                errors.append(f"Delete {item_id}: {e}")

        # Then saves
        for item in self._pending_saves:
            try:
                self._store._do_save(item, self._index_txn)
                count += 1
            except Exception as e:
                errors.append(f"Save {item.run_id}: {e}")

        # Commit index transaction
        self._index_txn.commit()
        self._committed = True

        if errors:
            raise StoreWriteError(f"Batch errors: {'; '.join(errors)}")

        return count

    def rollback(self) -> None:
        """Rollback all pending operations."""
        if self._committed:
            raise RuntimeError("Cannot rollback committed batch")

        self._pending_saves.clear()
        self._pending_deletes.clear()
        self._index_txn.rollback()
        self._committed = True

    @property
    def pending_count(self) -> int:
        """Number of pending operations."""
        return len(self._pending_saves) + len(self._pending_deletes)


class ConcurrentFileSystemStore(ValidationStore[ConcurrentFileSystemConfig]):
    """Thread-safe and process-safe filesystem store.

    This store implementation provides full concurrency control for
    multi-threaded and multi-process access to the filesystem.

    Features:
    - Pluggable lock strategies (fcntl, filelock, portalocker)
    - Atomic file writes using temp-and-rename pattern
    - Consistent index reads via MVCC-like snapshots
    - Transaction support for batch operations
    - Write-ahead logging for durability
    - Automatic recovery from failures
    - Lock statistics for debugging

    Example:
        >>> store = ConcurrentFileSystemStore(
        ...     base_path=".truthound/results",
        ...     concurrency=ConcurrencyConfig(lock_strategy="auto"),
        ... )
        >>>
        >>> # Simple operations
        >>> run_id = store.save(result)
        >>> retrieved = store.get(run_id)
        >>>
        >>> # Batch operations
        >>> with store.batch() as batch:
        ...     batch.save(result1)
        ...     batch.save(result2)
        ...     batch.commit()
    """

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "validations",
        compression: bool = False,
        concurrency: ConcurrencyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the concurrent filesystem store.

        Args:
            base_path: Base directory for storing files.
            namespace: Namespace for organizing data.
            prefix: Additional path prefix.
            compression: Whether to compress stored files.
            concurrency: Concurrency control configuration.
            **kwargs: Additional configuration options.
        """
        concurrency = concurrency or ConcurrencyConfig()

        config = ConcurrentFileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            use_compression=compression,
            concurrency=concurrency,
            **{k: v for k, v in kwargs.items() if hasattr(ConcurrentFileSystemConfig, k)},
        )
        super().__init__(config)

        self._lock_strategy: LockStrategy | None = None
        self._lock_manager: FileLockManager | None = None
        self._index: ConcurrentIndex | None = None

    @classmethod
    def _default_config(cls) -> ConcurrentFileSystemConfig:
        """Create default configuration."""
        return ConcurrentFileSystemConfig()

    def _create_lock_strategy(self) -> LockStrategy:
        """Create the appropriate lock strategy based on config."""
        strategy_type = self._config.concurrency.lock_strategy

        if isinstance(strategy_type, str):
            strategy_type = LockStrategyType(strategy_type)

        if strategy_type == LockStrategyType.AUTO:
            return get_default_lock_strategy()
        elif strategy_type == LockStrategyType.FCNTL:
            return FcntlLockStrategy()
        elif strategy_type == LockStrategyType.FILELOCK:
            return FileLockStrategy()
        elif strategy_type == LockStrategyType.PORTALOCKER:
            return PortalockerStrategy()
        elif strategy_type == LockStrategyType.NONE:
            return NoOpLockStrategy()
        else:
            return get_default_lock_strategy()

    def _do_initialize(self) -> None:
        """Initialize the store with concurrency primitives."""
        path = self._config.get_full_path()

        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        # Create lock strategy and manager
        self._lock_strategy = self._create_lock_strategy()
        self._lock_manager = FileLockManager(
            strategy=self._lock_strategy,
            enable_deadlock_detection=self._config.concurrency.enable_deadlock_detection,
            enable_statistics=self._config.concurrency.enable_statistics,
            default_timeout=self._config.concurrency.lock_timeout,
        )

        # Create concurrent index
        self._index = ConcurrentIndex(
            base_path=path,
            lock_manager=self._lock_manager,
            wal_enabled=self._config.concurrency.enable_wal,
        )
        self._index.initialize()

    def _get_file_path(self, item_id: str) -> Path:
        """Get the file path for an item."""
        ext = self._config.file_extension
        if self._config.use_compression:
            ext += ".gz"
        return self._config.get_full_path() / f"{item_id}{ext}"

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize data to bytes."""
        indent = 2 if self._config.pretty_print else None
        json_str = json.dumps(data, indent=indent, default=str)
        content = json_str.encode("utf-8")

        if self._config.use_compression:
            content = gzip.compress(content)

        return content

    def _deserialize(self, content: bytes) -> dict[str, Any]:
        """Deserialize bytes to data."""
        if self._config.use_compression:
            content = gzip.decompress(content)

        return json.loads(content.decode("utf-8"))

    def _do_save(
        self,
        item: ValidationResult,
        index_txn: IndexTransaction | None = None,
    ) -> str:
        """Internal save implementation.

        Args:
            item: Item to save.
            index_txn: Optional index transaction to use.

        Returns:
            The item's run ID.
        """
        item_id = item.run_id
        file_path = self._get_file_path(item_id)

        # Serialize content
        content = self._serialize(item.to_dict())

        # Write file atomically
        result = atomic_write(
            file_path,
            content,
            create_backup=self._config.concurrency.create_backup,
            lock_manager=self._lock_manager,
        )

        if not result.success:
            raise StoreWriteError(f"Failed to write {file_path}: {result.error}")

        # Update index
        metadata = {
            "data_asset": item.data_asset,
            "run_time": item.run_time.isoformat(),
            "status": item.status.value,
            "file": file_path.name,
            "tags": item.tags,
        }

        if index_txn:
            index_txn.upsert(item_id, metadata)
        else:
            self._index.upsert(item_id, metadata)

        return item_id

    def _do_delete(
        self,
        item_id: str,
        index_txn: IndexTransaction | None = None,
    ) -> bool:
        """Internal delete implementation.

        Args:
            item_id: Item to delete.
            index_txn: Optional index transaction to use.

        Returns:
            True if item existed.
        """
        file_path = self._get_file_path(item_id)

        # Check existence
        exists = file_path.exists()

        # Delete file with lock
        if exists:
            with self._lock_manager.write_lock(file_path):
                try:
                    file_path.unlink(missing_ok=True)
                except OSError as e:
                    raise StoreWriteError(f"Failed to delete {file_path}: {e}")

        # Update index
        if index_txn:
            index_txn.remove(item_id)
        else:
            self._index.remove(item_id)

        return exists

    def save(self, item: ValidationResult) -> str:
        """Save a validation result to the filesystem.

        Thread-safe and process-safe.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()
        return self._do_save(item)

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result by run ID.

        Thread-safe and process-safe.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            StoreReadError: If reading fails.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            raise StoreNotFoundError("ValidationResult", item_id)

        try:
            content = atomic_read(file_path, lock_manager=self._lock_manager)
            data = self._deserialize(content)
            return ValidationResult.from_dict(data)

        except (json.JSONDecodeError, KeyError) as e:
            raise StoreReadError(f"Failed to parse {file_path}: {e}")
        except OSError as e:
            raise StoreReadError(f"Failed to read {file_path}: {e}")

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Thread-safe using snapshot isolation.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()

        snapshot = self._index.snapshot()
        return snapshot.contains(item_id) or self._get_file_path(item_id).exists()

    def delete(self, item_id: str) -> bool:
        """Delete a validation result.

        Thread-safe and process-safe.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        self.initialize()
        return self._do_delete(item_id)

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Thread-safe using snapshot isolation.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching run IDs.
        """
        self.initialize()

        snapshot = self._index.snapshot()

        if not query:
            return snapshot.list_ids()

        # Filter by query
        matching_ids: list[tuple[str, datetime]] = []

        for entry in snapshot.entries.values():
            meta = entry.metadata
            if query.matches(meta):
                run_time_str = meta.get("run_time")
                if run_time_str:
                    run_time = datetime.fromisoformat(run_time_str)
                else:
                    run_time = entry.created_at
                matching_ids.append((entry.item_id, run_time))

        # Sort
        reverse = not query.ascending
        matching_ids.sort(key=lambda x: x[1], reverse=reverse)

        # Apply offset and limit
        ids = [item_id for item_id, _ in matching_ids]

        if query.offset:
            ids = ids[query.offset:]
        if query.limit:
            ids = ids[:query.limit]

        return ids

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results.

        Thread-safe using snapshot isolation.

        Args:
            query: Query parameters for filtering.

        Returns:
            List of matching validation results.
        """
        ids = self.list_ids(query)
        results: list[ValidationResult] = []

        for item_id in ids:
            try:
                result = self.get(item_id)
                results.append(result)
            except (StoreNotFoundError, StoreReadError):
                # Skip corrupted or deleted entries
                continue

        return results

    def batch(self) -> BatchContext:
        """Start a batch operation context.

        Returns:
            BatchContext for accumulating operations.

        Example:
            >>> with store.batch() as batch:
            ...     batch.save(result1)
            ...     batch.save(result2)
            ...     batch.commit()
        """
        self.initialize()

        # Create an index transaction (caller is responsible for commit/rollback)
        txn = self._index.begin_transaction()
        return BatchContext(self, txn)

    def rebuild_index(self) -> int:
        """Rebuild the index from stored files.

        Thread-safe with write lock during rebuild.

        Returns:
            Number of items indexed.
        """
        self.initialize()

        def extract_metadata(file_path: Path) -> tuple[str, dict[str, Any]] | None:
            try:
                with open(file_path, "rb") as f:
                    content = f.read()

                if self._config.use_compression:
                    content = gzip.decompress(content)

                data = json.loads(content.decode("utf-8"))
                item_id = data.get("run_id")

                if item_id:
                    return item_id, {
                        "data_asset": data.get("data_asset", "unknown"),
                        "run_time": data.get("run_time"),
                        "status": data.get("status"),
                        "file": file_path.name,
                        "tags": data.get("tags", {}),
                    }
            except (json.JSONDecodeError, OSError, gzip.BadGzipFile):
                pass
            return None

        pattern = f"*{self._config.file_extension}*"
        return self._index.rebuild_from_files(pattern, extract_metadata)

    @property
    def lock_statistics(self) -> LockStatistics | None:
        """Get lock statistics for debugging.

        Returns:
            LockStatistics if statistics enabled, None otherwise.
        """
        if self._lock_manager:
            return self._lock_manager.statistics
        return None

    def close(self) -> None:
        """Close the store and release resources."""
        if self._lock_manager:
            self._lock_manager.release_all()


class ConcurrentFileSystemExpectationStore(ExpectationStore[ConcurrentFileSystemConfig]):
    """Thread-safe filesystem expectation store.

    Similar to ConcurrentFileSystemStore but for expectation suites.
    """

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "expectations",
        concurrency: ConcurrencyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the concurrent expectation store.

        Args:
            base_path: Base directory for storing files.
            namespace: Namespace for organizing data.
            prefix: Additional path prefix.
            concurrency: Concurrency control configuration.
            **kwargs: Additional configuration options.
        """
        concurrency = concurrency or ConcurrencyConfig()

        config = ConcurrentFileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            concurrency=concurrency,
            **{k: v for k, v in kwargs.items() if hasattr(ConcurrentFileSystemConfig, k)},
        )
        super().__init__(config)

        self._lock_manager: FileLockManager | None = None
        self._index: ConcurrentIndex | None = None

    @classmethod
    def _default_config(cls) -> ConcurrentFileSystemConfig:
        """Create default configuration."""
        return ConcurrentFileSystemConfig(prefix="expectations")

    def _do_initialize(self) -> None:
        """Initialize the store with concurrency primitives."""
        path = self._config.get_full_path()

        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        # Create lock manager
        strategy = get_default_lock_strategy()
        self._lock_manager = FileLockManager(
            strategy=strategy,
            enable_deadlock_detection=self._config.concurrency.enable_deadlock_detection,
            enable_statistics=self._config.concurrency.enable_statistics,
            default_timeout=self._config.concurrency.lock_timeout,
        )

        # Create concurrent index
        self._index = ConcurrentIndex(
            base_path=path,
            index_filename="_expectations_index.json",
            lock_manager=self._lock_manager,
            wal_enabled=self._config.concurrency.enable_wal,
        )
        self._index.initialize()

    def _get_file_path(self, suite_name: str) -> Path:
        """Get the file path for a suite."""
        safe_name = suite_name.replace("/", "_").replace("\\", "_")
        return self._config.get_full_path() / f"{safe_name}{self._config.file_extension}"

    def save(self, item: ExpectationSuite) -> str:
        """Save an expectation suite.

        Thread-safe and process-safe.

        Args:
            item: The suite to save.

        Returns:
            The suite name.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        file_path = self._get_file_path(item.name)
        indent = 2 if self._config.pretty_print else None
        content = json.dumps(item.to_dict(), indent=indent, default=str)

        result = atomic_write(
            file_path,
            content,
            create_backup=self._config.concurrency.create_backup,
            lock_manager=self._lock_manager,
        )

        if not result.success:
            raise StoreWriteError(f"Failed to write {file_path}: {result.error}")

        # Update index
        self._index.upsert(item.name, {
            "data_asset": item.data_asset,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "file": file_path.name,
        })

        return item.name

    def get(self, item_id: str) -> ExpectationSuite:
        """Retrieve an expectation suite by name.

        Thread-safe and process-safe.

        Args:
            item_id: The suite name.

        Returns:
            The expectation suite.

        Raises:
            StoreNotFoundError: If the suite doesn't exist.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            raise StoreNotFoundError("ExpectationSuite", item_id)

        try:
            content = atomic_read(file_path, lock_manager=self._lock_manager)
            data = json.loads(content.decode("utf-8"))
            return ExpectationSuite.from_dict(data)

        except (json.JSONDecodeError, KeyError) as e:
            raise StoreReadError(f"Failed to parse {file_path}: {e}")
        except OSError as e:
            raise StoreReadError(f"Failed to read {file_path}: {e}")

    def exists(self, item_id: str) -> bool:
        """Check if a suite exists."""
        self.initialize()
        return self._index.contains(item_id) or self._get_file_path(item_id).exists()

    def delete(self, item_id: str) -> bool:
        """Delete an expectation suite.

        Thread-safe and process-safe.

        Args:
            item_id: The suite name.

        Returns:
            True if deleted, False if it didn't exist.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)
        exists = file_path.exists()

        if exists:
            with self._lock_manager.write_lock(file_path):
                try:
                    file_path.unlink(missing_ok=True)
                except OSError as e:
                    raise StoreWriteError(f"Failed to delete {file_path}: {e}")

        self._index.remove(item_id)
        return exists

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List all suite names."""
        self.initialize()

        snapshot = self._index.snapshot()

        if not query or not query.data_asset:
            return sorted(snapshot.list_ids())

        # Filter by data_asset
        return sorted([
            entry.item_id
            for entry in snapshot.entries.values()
            if entry.metadata.get("data_asset") == query.data_asset
        ])

    def query(self, query: StoreQuery) -> list[ExpectationSuite]:
        """Query expectation suites."""
        names = self.list_ids(query)
        suites: list[ExpectationSuite] = []

        for name in names:
            try:
                suite = self.get(name)
                suites.append(suite)
            except (StoreNotFoundError, StoreReadError):
                continue

        # Apply limit
        if query.limit:
            suites = suites[query.offset:query.offset + query.limit]
        elif query.offset:
            suites = suites[query.offset:]

        return suites

    def close(self) -> None:
        """Close the store and release resources."""
        if self._lock_manager:
            self._lock_manager.release_all()
