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

import hashlib
import json
import math
import shutil
import tempfile
import threading
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

from yaml import YAMLError

from truthound.stores._filesystem_io import contained_path
from truthound.stores.backends.filesystem import FileSystemConfig, _FileSystemCodec, _page
from truthound.stores.base import (
    ExpectationStore,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.compression.base import CompressionError
from truthound.stores.concurrency.atomic import (
    AtomicFileWriter,
    atomic_read,
    atomic_write,
)
from truthound.stores.concurrency.index import ConcurrentIndex, IndexTransaction
from truthound.stores.concurrency.locks import (
    FcntlLockStrategy,
    FileLockStrategy,
    LockMode,
    LockStrategy,
    NoOpLockStrategy,
    PortalockerStrategy,
    get_default_lock_strategy,
)
from truthound.stores.concurrency.manager import FileLockManager, LockStatistics
from truthound.stores.expectations import ExpectationSuite
from truthound.stores.results import ValidationResult

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

    def __post_init__(self):
        self.lock_strategy = LockStrategyType(self.lock_strategy)
        if (
            type(self.lock_timeout) not in (int, float)
            or not math.isfinite(self.lock_timeout)
            or self.lock_timeout <= 0
        ):
            raise ValueError("Lock timeout must be finite and positive")
        for name in (
            "enable_wal",
            "enable_deadlock_detection",
            "enable_statistics",
            "create_backup",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Concurrency flags must be boolean")


@dataclass
class ConcurrentFileSystemConfig(FileSystemConfig):
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

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.concurrency, ConcurrencyConfig):
            raise TypeError("concurrency must be ConcurrencyConfig")
        self.concurrency = deepcopy(self.concurrency)
        self.concurrency.__post_init__()
        self.metadata = deepcopy(self.metadata)

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
    Operations are buffered until commit. Pre-index failures restore touched
    files while their locks remain held. Index-commit failure is uncertain;
    retained recovery files are not a multi-file crash-atomic journal.

    Example:
        >>> with store.batch() as batch:
        ...     batch.save(result1)
        ...     batch.save(result2)
        ...     batch.delete("old-result")
        ...     batch.commit()
    """

    def __init__(
        self,
        store: ConcurrentFileSystemStore,
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
        if self._committed:
            raise RuntimeError("Batch already finished")
        store = self._store
        with store._lock_manager.write_lock(store.config.get_full_path() / "_batch-owner"):
            store._guard()
            return self._commit_owned()

    def _commit_owned(self) -> int:
        """Commit all pending operations.

        Returns:
            Number of operations performed.

        Raises:
            RuntimeError: If already committed.
            StoreWriteError: If any operation fails.
        """
        if self._committed:
            raise RuntimeError("Batch already committed")

        store = self._store
        store._guard()
        paths = {store._get_file_path(identity) for identity in self._pending_deletes}
        paths.update(store._get_file_path(item.id) for item in self._pending_saves)
        if any(path.resolve().parent != store.config.get_full_path().resolve() for path in paths):
            raise StoreWriteError("Batch item identity is outside the store")
        recovery = Path(
            tempfile.mkdtemp(prefix=".batch-recovery-", dir=store.config.get_full_path())
        )
        store._active_batches.add(recovery)
        retained = False
        count = 0
        try:
            with ExitStack() as locks:
                for path in sorted(paths):
                    locks.enter_context(store._lock_manager.write_lock(path))
                backups = {}
                touched = set()
                committing_index = False
                try:
                    manifest = {
                        "format_version": 1,
                        "state": "preparing",
                        "files": [
                            {
                                "name": path.name,
                                "backup": str(number) if path.exists() else None,
                                "sha256": None,
                            }
                            for number, path in enumerate(sorted(paths))
                        ],
                    }
                    self._manifest(recovery, manifest)
                    for path, row in zip(sorted(paths), manifest["files"], strict=True):
                        store._guard()
                        original = recovery / row["backup"] if row["backup"] is not None else None
                        if original is not None:
                            shutil.copyfile(path, original)
                            row["sha256"] = self._hash_file(original)
                        backups[path] = original
                    manifest["state"] = "prepared"
                    self._manifest(recovery, manifest)
                    for identity in self._pending_deletes:
                        store._guard()
                        touched.add(store._get_file_path(identity))
                        store._do_delete(identity, self._index_txn)
                        count += 1
                    for item in self._pending_saves:
                        store._guard()
                        touched.add(store._get_file_path(item.id))
                        store._do_save(item, self._index_txn)
                        count += 1
                    store._guard()
                    manifest["state"] = "index_attempted"
                    self._manifest(recovery, manifest)
                    committing_index = True
                    self._index_txn.commit()
                    self._committed = True
                    from truthound.stores.concurrency.batch_recovery import finish

                    completed = finish(store, recovery, manifest, "committed")
                    store._active_batches.discard(recovery)
                    recovery = completed
                    store._active_batches.add(recovery)
                except BaseException as error:
                    self._committed = True  # Never replay a failed batch object.
                    if committing_index:
                        retained = True
                        raise StoreWriteError(
                            "Batch index commit uncertain; recovery retained"
                        ) from error
                    failed_restore = False
                    for path in touched:
                        try:
                            if backups[path] is None:
                                path.unlink(missing_ok=True)
                            else:
                                with (
                                    backups[path].open("rb") as original,
                                    AtomicFileWriter(
                                        path,
                                        lock_manager=store._lock_manager,
                                    ) as output,
                                ):
                                    shutil.copyfileobj(original, output)
                                    if not output.commit().success:
                                        raise StoreWriteError("Batch restore failed")
                        except Exception:
                            failed_restore = True
                    if self._index_txn.is_active:
                        self._index_txn.rollback()
                    if failed_restore:
                        retained = True
                        raise StoreWriteError(
                            "Batch rollback incomplete; recovery retained"
                        ) from error
                    raise
        finally:
            store._active_batches.discard(recovery)
            if retained:
                store._batch_uncertain = True
            else:
                try:
                    shutil.rmtree(recovery)
                except BaseException:
                    store._batch_uncertain = True
                    raise
        return count

    @staticmethod
    def _hash_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _manifest(self, directory, payload):
        result = atomic_write(
            directory / "manifest.json",
            json.dumps(payload),
            lock_manager=self._store._lock_manager,
        )
        if not result.success:
            raise StoreWriteError("Batch recovery manifest could not be persisted")

    def __enter__(self):
        if self._committed:
            raise RuntimeError("Batch already finished")
        return self

    def __exit__(self, *error):
        if not self._committed:
            self.rollback()

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


class _ConcurrentInitialization(_FileSystemCodec):
    """Publish one initialized index/lock manager per store instance."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._initialization_lock = threading.RLock()
        self._io_guard = None
        self._batch_uncertain = False
        self._active_batches = set()
        super().__init__(*args, **kwargs)

    @property
    def config(self):
        return deepcopy(self._config)

    def set_io_guard(self, guard) -> None:
        """Set a trusted local authorization check before selected store effects."""
        if guard is not None and not callable(guard):
            raise TypeError("Store IO guard must be callable")
        self._io_guard = guard

    def _guard(self) -> None:
        if self._io_guard is not None:
            self._io_guard()
        if self._batch_uncertain:
            raise StoreReadError("Pending batch recovery requires explicit reconciliation")
        pending = set(self.config.get_full_path().glob(".batch-recovery-*")) - self._active_batches
        if pending:
            self._prepare_storage()
            handle = self._lock_manager.try_lock(
                self.config.get_full_path() / "_batch-owner", LockMode.EXCLUSIVE
            )
            if handle is None:
                return  # A live batch holds data locks until it finishes.
            try:
                if (
                    set(self.config.get_full_path().glob(".batch-recovery-*"))
                    - self._active_batches
                ):
                    self._batch_uncertain = True
                    raise StoreReadError("Pending batch recovery requires explicit reconciliation")
            finally:
                self._lock_manager.unlock(handle)

    def initialize(self) -> None:
        self._guard()
        # BaseStore's lazy flag alone is not a synchronization primitive.
        # Keep failed initialization retryable; only BaseStore sets the flag
        # after _do_initialize has completed successfully.
        with self._initialization_lock:
            super().initialize()


class ConcurrentFileSystemStore(
    _ConcurrentInitialization, ValidationStore[ConcurrentFileSystemConfig]
):
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
        compression: bool | str = False,
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
        if type(compression) not in (bool, str):
            raise ValueError("Compression must be boolean or a registered codec name")
        concurrency = concurrency if concurrency is not None else ConcurrencyConfig()
        unknown = set(kwargs) - {entry.name for entry in fields(ConcurrentFileSystemConfig)}
        if unknown:
            raise TypeError("Unknown concurrent filesystem setting: " + ", ".join(sorted(unknown)))

        config = ConcurrentFileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            compression=compression if isinstance(compression, str) else None,
            use_compression=compression if type(compression) is bool else bool(compression),
            concurrency=concurrency,
            **kwargs,
        )
        super().__init__(deepcopy(config))

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
        self._prepare_storage()
        self._index.initialize()

    def _prepare_storage(self) -> None:
        """Create native components without automatically recovering an index."""
        if self._lock_manager is not None:
            return
        path = self._config.get_full_path()

        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            raise StoreReadError("Selected store directory does not exist")

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

    def _get_file_path(self, item_id: str) -> Path:
        """Get the file path for an item."""
        if not isinstance(item_id, str) or not item_id or "\0" in item_id:
            raise ValueError("Invalid concurrent record identity")
        if item_id.startswith(
            ("_index", "_expectations_index", "_batch-owner", ".batch-recovery-")
        ):
            raise ValueError("Reserved concurrent store identity")
        return contained_path(
            self._config.get_full_path(), quote(item_id, safe="") + self._suffix()
        )

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
        document = item.to_dict()
        document["metadata"] = {**deepcopy(self._config.metadata), **document.get("metadata", {})}
        content = self._serialize(document)

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
        metadata = self._index_metadata(item, file_path)

        if index_txn:
            index_txn.upsert(item_id, metadata)
        else:
            self._index.upsert(item_id, metadata)

        return item_id

    @staticmethod
    def _index_metadata(item, file_path):
        return {
            "data_asset": item.data_asset,
            "run_time": item.run_time.isoformat(),
            "status": item.status.value,
            "file": file_path.name,
            "tags": item.tags,
        }

    def list_batch_recovery(self, limit=100, offset=0):
        from truthound.stores.concurrency.batch_recovery import list_recovery

        return list_recovery(self, limit, offset)

    def cleanup_batch_recovery(self, recovery_id, *, expected_fingerprint):
        from truthound.stores.concurrency.batch_recovery import cleanup

        return cleanup(self, recovery_id, expected_fingerprint)

    def reconcile_batch(self, recovery_id, *, resolution, expected_fingerprint):
        from truthound.stores.concurrency.batch_recovery import reconcile

        return reconcile(self, recovery_id, resolution, expected_fingerprint)

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
                    raise StoreWriteError(f"Failed to delete {file_path}: {e}") from e

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

        with self._lock_manager.write_lock(self._get_file_path(item.id)):
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

        with self._lock_manager.read_lock(self._get_file_path(item_id)):
            file_path = self._get_file_path(item_id)

            if not file_path.exists():
                raise StoreNotFoundError("ValidationResult", item_id)

            try:
                content = atomic_read(file_path, lock_manager=self._lock_manager)
                data = self._deserialize(content)
                item = ValidationResult.from_dict(data)
                if item.id != item_id:
                    raise ValueError("Stored record identity mismatch")
                return item

            except (
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                YAMLError,
                CompressionError,
            ) as e:
                raise StoreReadError(f"Failed to parse {file_path}: {e}") from e
            except OSError as e:
                raise StoreReadError(f"Failed to read {file_path}: {e}") from e

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Thread-safe using snapshot isolation.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()

        with self._lock_manager.read_lock(self._get_file_path(item_id)):
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

        with self._lock_manager.write_lock(self._get_file_path(item_id)):
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

        _page([], query)
        if query.order_by not in {"run_time", "data_asset", "status"}:
            raise ValueError("Unsupported concurrent result ordering")
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
                sort_value = (
                    run_time if query.order_by == "run_time" else meta.get(query.order_by, "")
                )
                matching_ids.append((entry.item_id, sort_value))

        # Sort
        reverse = not query.ascending
        matching_ids.sort(key=lambda x: x[1], reverse=reverse)

        # Apply offset and limit
        ids = [item_id for item_id, _ in matching_ids]

        return _page(ids, query)

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
            except StoreNotFoundError:
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
                contained_path(self._config.get_full_path(), file_path.absolute())
                with open(file_path, "rb") as f:
                    content = f.read()

                data = self._deserialize(content)
                item = ValidationResult.from_dict(data)
                item_id = item.id
                if self._get_file_path(item_id) != file_path.absolute():
                    raise ValueError("Record filename does not match identity")
                return item_id, self._index_metadata(item, file_path)
            except (
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                YAMLError,
                CompressionError,
                OSError,
            ) as error:
                raise StoreReadError("Cannot rebuild from an invalid concurrent record") from error
            return None

        pattern = "*" + self._suffix()
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


class ConcurrentFileSystemExpectationStore(
    _ConcurrentInitialization, ExpectationStore[ConcurrentFileSystemConfig]
):
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
        concurrency = concurrency if concurrency is not None else ConcurrencyConfig()
        unknown = set(kwargs) - {entry.name for entry in fields(ConcurrentFileSystemConfig)}
        if unknown:
            raise TypeError("Unknown concurrent filesystem setting: " + ", ".join(sorted(unknown)))

        config = ConcurrentFileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            concurrency=concurrency,
            **kwargs,
        )
        super().__init__(deepcopy(config))

        self._lock_manager: FileLockManager | None = None
        self._index: ConcurrentIndex | None = None

    @classmethod
    def _default_config(cls) -> ConcurrentFileSystemConfig:
        """Create default configuration."""
        return ConcurrentFileSystemConfig(prefix="expectations")

    def _do_initialize(self) -> None:
        self._prepare_storage()
        self._index.initialize()

    def _prepare_storage(self) -> None:
        """Initialize the store with concurrency primitives."""
        if self._lock_manager is not None:
            return
        path = self._config.get_full_path()

        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            raise StoreReadError("Selected store directory does not exist")

        # Use the same selected strategy as validation records.
        strategy = ConcurrentFileSystemStore._create_lock_strategy(self)
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

    def _get_file_path(self, suite_name: str) -> Path:
        """Get the file path for a suite."""
        if not isinstance(suite_name, str) or not suite_name or "\0" in suite_name:
            raise ValueError("Invalid expectation suite name")
        if suite_name.startswith(
            ("_index", "_expectations_index", "_batch-owner", ".batch-recovery-")
        ):
            raise ValueError("Reserved concurrent store identity")
        return contained_path(
            self._config.get_full_path(), quote(suite_name, safe="") + self._suffix()
        )

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

        with self._lock_manager.write_lock(self._get_file_path(item.name)):
            file_path = self._get_file_path(item.name)
            document = item.to_dict()
            document["meta"] = {**deepcopy(self._config.metadata), **document.get("meta", {})}
            content = self._serialize(document)

            result = atomic_write(
                file_path,
                content,
                create_backup=self._config.concurrency.create_backup,
                lock_manager=self._lock_manager,
            )

            if not result.success:
                raise StoreWriteError(f"Failed to write {file_path}: {result.error}")

            # Update index
            self._index.upsert(
                item.name,
                {
                    "data_asset": item.data_asset,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "run_time": item.updated_at.isoformat() if item.updated_at else None,
                    "tags": deepcopy(item.tags),
                    "file": file_path.name,
                },
            )

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

        with self._lock_manager.read_lock(self._get_file_path(item_id)):
            file_path = self._get_file_path(item_id)

            if not file_path.exists():
                raise StoreNotFoundError("ExpectationSuite", item_id)

            try:
                content = atomic_read(file_path, lock_manager=self._lock_manager)
                data = self._deserialize(content)
                item = ExpectationSuite.from_dict(data)
                if item.name != item_id:
                    raise ValueError("Stored suite identity mismatch")
                return item

            except (
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                YAMLError,
                CompressionError,
            ) as e:
                raise StoreReadError(f"Failed to parse {file_path}: {e}") from e
            except OSError as e:
                raise StoreReadError(f"Failed to read {file_path}: {e}") from e

    def exists(self, item_id: str) -> bool:
        """Check if a suite exists."""
        self.initialize()

        with self._lock_manager.read_lock(self._get_file_path(item_id)):
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

        with self._lock_manager.write_lock(self._get_file_path(item_id)):
            file_path = self._get_file_path(item_id)
            exists = file_path.exists()

            if exists:
                with self._lock_manager.write_lock(file_path):
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError as e:
                        raise StoreWriteError(f"Failed to delete {file_path}: {e}") from e

            self._index.remove(item_id)
            return exists

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List all suite names."""
        self.initialize()

        snapshot = self._index.snapshot()

        if query is None:
            return sorted(snapshot.list_ids())
        _page([], query)
        if query.order_by not in {"run_time", "data_asset", "name"}:
            raise ValueError("Unsupported concurrent suite ordering")
        entries = [entry for entry in snapshot.entries.values() if query.matches(entry.metadata)]

        def sort_key(entry):
            if query.order_by == "name":
                return entry.item_id
            if query.order_by == "run_time":
                return entry.metadata.get("run_time") or entry.metadata.get("created_at") or ""
            return entry.metadata.get("data_asset") or ""

        entries.sort(key=sort_key, reverse=not query.ascending)
        return _page([entry.item_id for entry in entries], query)

    def query(self, query: StoreQuery) -> list[ExpectationSuite]:
        """Query expectation suites."""
        names = self.list_ids(query)
        suites: list[ExpectationSuite] = []

        for name in names:
            try:
                suite = self.get(name)
                suites.append(suite)
            except StoreNotFoundError:
                continue

        return suites

    def close(self) -> None:
        """Close the store and release resources."""
        if self._lock_manager:
            self._lock_manager.release_all()
