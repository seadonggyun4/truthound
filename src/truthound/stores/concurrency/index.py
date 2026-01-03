"""Concurrent index management for filesystem stores.

This module provides thread-safe and process-safe index operations
for the filesystem store. The index maintains metadata about stored
items for fast lookups without reading the actual files.

Key features:
- MVCC-like reads (consistent snapshots)
- Write-ahead logging for durability
- Automatic index recovery
- Transaction support for batch updates

Example:
    >>> index = ConcurrentIndex(Path(".truthound/store"))
    >>> with index.transaction() as txn:
    ...     txn.add("item-1", {"data_asset": "customers.csv"})
    ...     txn.add("item-2", {"data_asset": "orders.csv"})
    ...     txn.commit()
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from truthound.stores.concurrency.locks import LockMode
from truthound.stores.concurrency.manager import FileLockManager, get_default_manager
from truthound.stores.concurrency.atomic import atomic_write, atomic_read, AtomicFileWriter


T = TypeVar("T")


@dataclass
class IndexEntry:
    """Represents a single entry in the index.

    Attributes:
        item_id: Unique identifier for the item.
        metadata: Metadata about the item.
        created_at: When the entry was created.
        updated_at: When the entry was last updated.
        version: Entry version for optimistic locking.
    """

    item_id: str
    metadata: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
            version=data.get("version", 1),
        )


@dataclass
class IndexSnapshot:
    """Immutable snapshot of the index at a point in time.

    Used for MVCC-like reads to provide consistent view of index.
    """

    entries: dict[str, IndexEntry]
    version: int
    timestamp: datetime = field(default_factory=datetime.now)

    def get(self, item_id: str) -> IndexEntry | None:
        """Get an entry by ID."""
        return self.entries.get(item_id)

    def contains(self, item_id: str) -> bool:
        """Check if an entry exists."""
        return item_id in self.entries

    def list_ids(self) -> list[str]:
        """List all item IDs."""
        return list(self.entries.keys())

    def filter(
        self,
        predicate: Callable[[IndexEntry], bool],
    ) -> list[IndexEntry]:
        """Filter entries by predicate."""
        return [e for e in self.entries.values() if predicate(e)]

    def __len__(self) -> int:
        return len(self.entries)


class IndexTransaction:
    """Transaction for batch index updates.

    Provides ACID-like semantics for index modifications:
    - Atomic: All changes applied or none
    - Consistent: Index remains valid after transaction
    - Isolated: Changes not visible until commit
    - Durable: Changes persisted after commit

    Example:
        >>> with index.transaction() as txn:
        ...     txn.add("item-1", {"key": "value"})
        ...     txn.update("item-2", {"key": "new-value"})
        ...     txn.remove("item-3")
        ...     txn.commit()
    """

    def __init__(
        self,
        index: "ConcurrentIndex",
        snapshot: IndexSnapshot,
    ) -> None:
        """Initialize the transaction.

        Args:
            index: The parent index.
            snapshot: Initial snapshot for the transaction.
        """
        self._index = index
        self._snapshot = snapshot
        self._pending_adds: dict[str, IndexEntry] = {}
        self._pending_updates: dict[str, IndexEntry] = {}
        self._pending_removes: set[str] = set()
        self._committed = False
        self._rolled_back = False

    def add(self, item_id: str, metadata: dict[str, Any]) -> IndexEntry:
        """Add a new entry to the index.

        Args:
            item_id: Unique identifier for the item.
            metadata: Metadata to store.

        Returns:
            The created entry.

        Raises:
            ValueError: If item already exists.
        """
        self._check_active()

        if self._snapshot.contains(item_id) or item_id in self._pending_adds:
            if item_id not in self._pending_removes:
                raise ValueError(f"Item already exists: {item_id}")

        entry = IndexEntry(item_id=item_id, metadata=metadata)
        self._pending_adds[item_id] = entry
        self._pending_removes.discard(item_id)
        return entry

    def update(
        self,
        item_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> IndexEntry:
        """Update an existing entry.

        Args:
            item_id: Item to update.
            metadata: New metadata.
            merge: If True, merge with existing metadata.

        Returns:
            The updated entry.

        Raises:
            KeyError: If item doesn't exist.
        """
        self._check_active()

        existing = self._get_current(item_id)
        if existing is None:
            raise KeyError(f"Item not found: {item_id}")

        if merge:
            new_metadata = {**existing.metadata, **metadata}
        else:
            new_metadata = metadata

        entry = IndexEntry(
            item_id=item_id,
            metadata=new_metadata,
            created_at=existing.created_at,
            updated_at=datetime.now(),
            version=existing.version + 1,
        )
        self._pending_updates[item_id] = entry
        return entry

    def upsert(self, item_id: str, metadata: dict[str, Any]) -> IndexEntry:
        """Add or update an entry.

        Args:
            item_id: Item identifier.
            metadata: Metadata to store.

        Returns:
            The created or updated entry.
        """
        self._check_active()

        existing = self._get_current(item_id)
        if existing is None:
            return self.add(item_id, metadata)
        else:
            return self.update(item_id, metadata)

    def remove(self, item_id: str) -> bool:
        """Remove an entry from the index.

        Args:
            item_id: Item to remove.

        Returns:
            True if item existed, False otherwise.
        """
        self._check_active()

        exists = self._get_current(item_id) is not None

        self._pending_removes.add(item_id)
        self._pending_adds.pop(item_id, None)
        self._pending_updates.pop(item_id, None)

        return exists

    def get(self, item_id: str) -> IndexEntry | None:
        """Get an entry, including pending changes.

        Args:
            item_id: Item to get.

        Returns:
            Entry if found, None otherwise.
        """
        return self._get_current(item_id)

    def _get_current(self, item_id: str) -> IndexEntry | None:
        """Get current state of an entry including pending changes."""
        if item_id in self._pending_removes:
            return None
        if item_id in self._pending_updates:
            return self._pending_updates[item_id]
        if item_id in self._pending_adds:
            return self._pending_adds[item_id]
        return self._snapshot.get(item_id)

    def _check_active(self) -> None:
        """Check that transaction is active."""
        if self._committed:
            raise RuntimeError("Transaction already committed")
        if self._rolled_back:
            raise RuntimeError("Transaction rolled back")

    def commit(self) -> int:
        """Commit the transaction.

        Returns:
            Number of changes applied.

        Raises:
            RuntimeError: If transaction is not active.
        """
        self._check_active()

        changes = (
            len(self._pending_adds)
            + len(self._pending_updates)
            + len(self._pending_removes)
        )

        if changes > 0:
            self._index._apply_transaction(self)

        self._committed = True
        return changes

    def rollback(self) -> None:
        """Rollback the transaction, discarding all changes."""
        self._check_active()
        self._pending_adds.clear()
        self._pending_updates.clear()
        self._pending_removes.clear()
        self._rolled_back = True

    @property
    def pending_changes(self) -> int:
        """Number of pending changes."""
        return (
            len(self._pending_adds)
            + len(self._pending_updates)
            + len(self._pending_removes)
        )

    @property
    def is_active(self) -> bool:
        """Whether transaction is active."""
        return not self._committed and not self._rolled_back


class ConcurrentIndex:
    """Thread-safe and process-safe index for filesystem stores.

    This class manages an index file that tracks metadata about stored
    items. It provides:
    - Consistent reads via snapshots
    - Atomic writes via transactions
    - Automatic recovery from corruption
    - Write-ahead logging for durability

    Example:
        >>> index = ConcurrentIndex(Path(".truthound/store"))
        >>>
        >>> # Read operations (use snapshot)
        >>> snapshot = index.snapshot()
        >>> for item_id in snapshot.list_ids():
        ...     entry = snapshot.get(item_id)
        >>>
        >>> # Write operations (use transaction)
        >>> with index.transaction() as txn:
        ...     txn.add("new-item", {"data_asset": "data.csv"})
        ...     txn.commit()
    """

    def __init__(
        self,
        base_path: Path | str,
        index_filename: str = "_index.json",
        lock_manager: FileLockManager | None = None,
        wal_enabled: bool = True,
    ) -> None:
        """Initialize the concurrent index.

        Args:
            base_path: Base directory for the index.
            index_filename: Name of the index file.
            lock_manager: Lock manager to use.
            wal_enabled: Whether to use write-ahead logging.
        """
        self._base_path = Path(base_path)
        self._index_path = self._base_path / index_filename
        self._wal_path = self._base_path / f"{index_filename}.wal"
        self._lock_manager = lock_manager or get_default_manager()
        self._wal_enabled = wal_enabled

        # In-memory cache
        self._cache: dict[str, IndexEntry] = {}
        self._cache_version: int = 0
        self._cache_lock = threading.RLock()
        self._loaded = False

    def initialize(self) -> None:
        """Initialize the index, loading from disk if exists."""
        with self._cache_lock:
            if self._loaded:
                return

            self._base_path.mkdir(parents=True, exist_ok=True)

            # Recover from WAL if needed
            if self._wal_enabled and self._wal_path.exists():
                self._recover_from_wal()

            # Load index from disk or create empty
            if self._index_path.exists():
                self._load_from_disk()
            else:
                # Create empty index file
                self._save_to_disk()

            self._loaded = True

    def _load_from_disk(self) -> None:
        """Load index from disk file."""
        try:
            content = atomic_read(self._index_path, lock_manager=self._lock_manager)
            data = json.loads(content.decode("utf-8"))

            self._cache.clear()
            for item_id, entry_data in data.get("entries", {}).items():
                # Handle both old format (dict) and new format (IndexEntry)
                if isinstance(entry_data, dict) and "item_id" not in entry_data:
                    # Old format: just metadata
                    entry = IndexEntry(item_id=item_id, metadata=entry_data)
                else:
                    entry = IndexEntry.from_dict(entry_data)
                self._cache[item_id] = entry

            self._cache_version = data.get("version", 0)

        except (json.JSONDecodeError, OSError):
            # Index corrupted or missing, start fresh
            self._cache.clear()
            self._cache_version = 0

    def _save_to_disk(self) -> None:
        """Save index to disk file."""
        data = {
            "version": self._cache_version,
            "updated_at": datetime.now().isoformat(),
            "entries": {
                item_id: entry.to_dict() for item_id, entry in self._cache.items()
            },
        }

        content = json.dumps(data, indent=2, default=str)
        atomic_write(
            self._index_path,
            content,
            lock_manager=self._lock_manager,
        )

    def _recover_from_wal(self) -> None:
        """Recover uncommitted changes from write-ahead log."""
        try:
            with open(self._wal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    operation = json.loads(line)
                    op_type = operation.get("type")
                    item_id = operation.get("item_id")

                    if op_type == "add" or op_type == "update":
                        entry_data = operation.get("entry", {})
                        self._cache[item_id] = IndexEntry.from_dict(entry_data)
                    elif op_type == "remove":
                        self._cache.pop(item_id, None)

            # Save recovered state and remove WAL
            self._cache_version += 1
            self._save_to_disk()
            self._wal_path.unlink(missing_ok=True)

        except (json.JSONDecodeError, OSError):
            # WAL corrupted, ignore
            self._wal_path.unlink(missing_ok=True)

    def _write_wal(self, operations: list[dict[str, Any]]) -> None:
        """Write operations to write-ahead log."""
        if not self._wal_enabled:
            return

        with open(self._wal_path, "a") as f:
            for op in operations:
                f.write(json.dumps(op, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _clear_wal(self) -> None:
        """Clear the write-ahead log."""
        if self._wal_path.exists():
            self._wal_path.unlink(missing_ok=True)

    def snapshot(self) -> IndexSnapshot:
        """Get an immutable snapshot of the index.

        The snapshot provides a consistent view of the index that
        won't change even if the index is modified.

        Returns:
            IndexSnapshot with current state.
        """
        self.initialize()

        with self._cache_lock:
            return IndexSnapshot(
                entries=dict(self._cache),
                version=self._cache_version,
            )

    def begin_transaction(self) -> IndexTransaction:
        """Start a transaction for batch updates.

        Returns:
            IndexTransaction for making changes.
            Caller is responsible for calling commit() or rollback().

        Example:
            >>> txn = index.begin_transaction()
            >>> try:
            ...     txn.add("item", {"key": "value"})
            ...     txn.commit()
            ... except Exception:
            ...     txn.rollback()
            ...     raise
        """
        self.initialize()
        snapshot = self.snapshot()
        return IndexTransaction(self, snapshot)

    @contextmanager
    def transaction(self) -> Iterator[IndexTransaction]:
        """Start a transaction for batch updates with context manager.

        Yields:
            IndexTransaction for making changes.

        Example:
            >>> with index.transaction() as txn:
            ...     txn.add("item", {"key": "value"})
            ...     txn.commit()
        """
        txn = self.begin_transaction()

        try:
            yield txn
        except Exception:
            if txn.is_active:
                txn.rollback()
            raise
        else:
            if txn.is_active and txn.pending_changes > 0:
                # Auto-commit if not explicitly committed/rolled back
                txn.commit()

    def _apply_transaction(self, txn: IndexTransaction) -> None:
        """Apply a transaction's changes to the index.

        Args:
            txn: The transaction to apply.
        """
        with self._lock_manager.write_lock(self._index_path):
            with self._cache_lock:
                # Check for conflicts (optimistic concurrency)
                current_snapshot = self.snapshot()
                if current_snapshot.version != txn._snapshot.version:
                    # Check if any modified entries have changed
                    for item_id in list(txn._pending_updates.keys()) + list(
                        txn._pending_removes
                    ):
                        old_entry = txn._snapshot.get(item_id)
                        new_entry = current_snapshot.get(item_id)

                        if old_entry is None and new_entry is not None:
                            raise RuntimeError(
                                f"Conflict: {item_id} was added concurrently"
                            )
                        if old_entry is not None and new_entry is None:
                            raise RuntimeError(
                                f"Conflict: {item_id} was removed concurrently"
                            )
                        if (
                            old_entry is not None
                            and new_entry is not None
                            and old_entry.version != new_entry.version
                        ):
                            raise RuntimeError(
                                f"Conflict: {item_id} was modified concurrently"
                            )

                # Write to WAL first
                wal_operations = []
                for item_id, entry in txn._pending_adds.items():
                    wal_operations.append(
                        {"type": "add", "item_id": item_id, "entry": entry.to_dict()}
                    )
                for item_id, entry in txn._pending_updates.items():
                    wal_operations.append(
                        {"type": "update", "item_id": item_id, "entry": entry.to_dict()}
                    )
                for item_id in txn._pending_removes:
                    wal_operations.append({"type": "remove", "item_id": item_id})

                if wal_operations:
                    self._write_wal(wal_operations)

                # Apply changes to cache
                for item_id, entry in txn._pending_adds.items():
                    self._cache[item_id] = entry
                for item_id, entry in txn._pending_updates.items():
                    self._cache[item_id] = entry
                for item_id in txn._pending_removes:
                    self._cache.pop(item_id, None)

                self._cache_version += 1

                # Persist to disk
                self._save_to_disk()

                # Clear WAL after successful write
                self._clear_wal()

    # Convenience methods for simple operations

    def add(self, item_id: str, metadata: dict[str, Any]) -> IndexEntry:
        """Add a single entry (convenience method).

        Args:
            item_id: Item identifier.
            metadata: Metadata to store.

        Returns:
            The created entry.
        """
        with self.transaction() as txn:
            entry = txn.add(item_id, metadata)
            txn.commit()
            return entry

    def update(
        self,
        item_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> IndexEntry:
        """Update a single entry (convenience method).

        Args:
            item_id: Item to update.
            metadata: New metadata.
            merge: Whether to merge with existing.

        Returns:
            The updated entry.
        """
        with self.transaction() as txn:
            entry = txn.update(item_id, metadata, merge=merge)
            txn.commit()
            return entry

    def upsert(self, item_id: str, metadata: dict[str, Any]) -> IndexEntry:
        """Add or update a single entry (convenience method).

        Args:
            item_id: Item identifier.
            metadata: Metadata to store.

        Returns:
            The created or updated entry.
        """
        with self.transaction() as txn:
            entry = txn.upsert(item_id, metadata)
            txn.commit()
            return entry

    def remove(self, item_id: str) -> bool:
        """Remove a single entry (convenience method).

        Args:
            item_id: Item to remove.

        Returns:
            True if item existed.
        """
        with self.transaction() as txn:
            result = txn.remove(item_id)
            txn.commit()
            return result

    def get(self, item_id: str) -> IndexEntry | None:
        """Get an entry by ID.

        Args:
            item_id: Item to get.

        Returns:
            Entry if found, None otherwise.
        """
        return self.snapshot().get(item_id)

    def contains(self, item_id: str) -> bool:
        """Check if an entry exists.

        Args:
            item_id: Item to check.

        Returns:
            True if exists.
        """
        return self.snapshot().contains(item_id)

    def list_ids(self) -> list[str]:
        """List all item IDs.

        Returns:
            List of item IDs.
        """
        return self.snapshot().list_ids()

    def count(self) -> int:
        """Get number of entries.

        Returns:
            Entry count.
        """
        return len(self.snapshot())

    def clear(self) -> int:
        """Remove all entries.

        Returns:
            Number of entries removed.
        """
        snapshot = self.snapshot()
        count = len(snapshot)

        if count > 0:
            with self.transaction() as txn:
                for item_id in snapshot.list_ids():
                    txn.remove(item_id)
                txn.commit()

        return count

    def rebuild_from_files(
        self,
        file_pattern: str,
        metadata_extractor: Callable[[Path], tuple[str, dict[str, Any]] | None],
    ) -> int:
        """Rebuild index from files in directory.

        Args:
            file_pattern: Glob pattern for files.
            metadata_extractor: Function to extract (item_id, metadata) from file.

        Returns:
            Number of entries rebuilt.
        """
        self.initialize()

        with self.transaction() as txn:
            # Clear existing entries
            for item_id in self.list_ids():
                txn.remove(item_id)

            # Scan files
            count = 0
            for file_path in self._base_path.glob(file_pattern):
                if file_path.name.startswith("_"):
                    continue

                try:
                    result = metadata_extractor(file_path)
                    if result:
                        item_id, metadata = result
                        txn.add(item_id, metadata)
                        count += 1
                except Exception:
                    continue

            txn.commit()

        return count
