"""Replication synchronization logic.

This module provides synchronization functionality for keeping
replicas in sync with the primary store.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.stores.replication.base import (
    ConflictResolution,
    ReplicaHealth,
    ReplicaState,
    ReplicaTarget,
    ReplicationMetrics,
)

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        target_name: Name of the target replica.
        success: Whether sync was successful.
        items_synced: Number of items synced.
        items_failed: Number of items that failed.
        duration_ms: Duration of sync in milliseconds.
        error: Error message if failed.
        conflicts: Number of conflicts encountered.
    """

    target_name: str
    success: bool
    items_synced: int = 0
    items_failed: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    conflicts: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def complete(self, success: bool = True, error: str | None = None) -> None:
        """Mark sync as complete."""
        self.success = success
        self.error = error
        self.completed_at = datetime.now()


class ReplicationSyncer(Generic[T, ConfigT]):
    """Synchronizer for replication operations.

    Handles the actual synchronization of data between primary
    and replica stores.

    Example:
        >>> syncer = ReplicationSyncer(primary_store, metrics)
        >>> result = await syncer.sync_item(item_id, replica_target)
    """

    def __init__(
        self,
        primary: "ValidationStore[ConfigT]",
        metrics: ReplicationMetrics | None = None,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    ) -> None:
        """Initialize syncer.

        Args:
            primary: Primary storage backend.
            metrics: Replication metrics.
            conflict_resolution: Conflict resolution strategy.
        """
        self._primary = primary
        self._metrics = metrics or ReplicationMetrics()
        self._conflict_resolution = conflict_resolution
        self._lock = asyncio.Lock()

    @property
    def metrics(self) -> ReplicationMetrics:
        """Get replication metrics."""
        return self._metrics

    async def sync_item(
        self,
        item_id: str,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> bool:
        """Sync a single item to a replica.

        Args:
            item_id: Item identifier.
            item: Item to sync.
            target: Target replica.

        Returns:
            True if successful.
        """
        start_time = time.monotonic()

        try:
            # Check for conflicts
            if await self._has_conflict(item_id, item, target):
                resolved = await self._resolve_conflict(item_id, item, target)
                if not resolved:
                    return False

            # Sync to replica
            await self._write_to_replica(item, target)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            target.mark_healthy()
            self._metrics.record_replication_success(target.name, elapsed_ms)
            self._metrics.update_lag(target.name, elapsed_ms)

            return True

        except Exception as e:
            target.mark_unhealthy()
            self._metrics.record_replication_failure(target.name)
            return False

    async def sync_item_with_retry(
        self,
        item_id: str,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> bool:
        """Sync item with retry logic.

        Args:
            item_id: Item identifier.
            item: Item to sync.
            target: Target replica.

        Returns:
            True if successful.
        """
        for attempt in range(target.max_retry_attempts):
            try:
                success = await asyncio.wait_for(
                    self.sync_item(item_id, item, target),
                    timeout=target.sync_timeout_seconds,
                )
                if success:
                    return True

            except asyncio.TimeoutError:
                target.mark_degraded(target.sync_timeout_seconds * 1000)

            except Exception:
                pass

            # Wait before retry
            if attempt < target.max_retry_attempts - 1:
                delay = target.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)

        return False

    async def sync_batch(
        self,
        items: dict[str, T],
        target: ReplicaTarget[ConfigT],
    ) -> SyncResult:
        """Sync a batch of items to a replica.

        Args:
            items: Dictionary of item_id to item.
            target: Target replica.

        Returns:
            Sync result.
        """
        result = SyncResult(target_name=target.name, success=True)
        start_time = time.monotonic()

        for item_id, item in items.items():
            try:
                success = await self.sync_item_with_retry(item_id, item, target)
                if success:
                    result.items_synced += 1
                else:
                    result.items_failed += 1
            except Exception as e:
                result.items_failed += 1

        result.duration_ms = (time.monotonic() - start_time) * 1000
        result.success = result.items_failed == 0
        result.complete()

        return result

    async def full_sync(
        self,
        target: ReplicaTarget[ConfigT],
        batch_size: int = 100,
    ) -> SyncResult:
        """Perform full synchronization to a replica.

        Args:
            target: Target replica.
            batch_size: Number of items per batch.

        Returns:
            Sync result.
        """
        result = SyncResult(target_name=target.name, success=True)
        start_time = time.monotonic()

        target.mark_syncing()

        try:
            # Get all item IDs from primary
            item_ids = self._primary.list_ids()

            # Sync in batches
            for i in range(0, len(item_ids), batch_size):
                batch_ids = item_ids[i : i + batch_size]
                batch_items = {}

                for item_id in batch_ids:
                    try:
                        item = self._primary.get(item_id)
                        if item is not None:
                            batch_items[item_id] = item
                    except Exception:
                        pass

                batch_result = await self.sync_batch(batch_items, target)
                result.items_synced += batch_result.items_synced
                result.items_failed += batch_result.items_failed

            target.mark_healthy()
            result.success = result.items_failed == 0

        except Exception as e:
            target.mark_unhealthy()
            result.success = False
            result.error = str(e)

        result.duration_ms = (time.monotonic() - start_time) * 1000
        result.complete()

        return result

    async def _has_conflict(
        self,
        item_id: str,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> bool:
        """Check if there's a conflict with the replica.

        Args:
            item_id: Item identifier.
            item: Item to sync.
            target: Target replica.

        Returns:
            True if conflict exists.
        """
        try:
            existing = target.store.get(item_id)
            if existing is None:
                return False

            # Simple conflict detection - compare timestamps if available
            if hasattr(item, "run_time") and hasattr(existing, "run_time"):
                return item.run_time != existing.run_time

            return False

        except Exception:
            return False

    async def _resolve_conflict(
        self,
        item_id: str,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> bool:
        """Resolve a conflict according to the resolution strategy.

        Args:
            item_id: Item identifier.
            item: Item from primary.
            target: Target replica.

        Returns:
            True if resolved (proceed with sync), False to skip.
        """
        self._metrics.record_conflict()

        if self._conflict_resolution == ConflictResolution.PRIMARY_WINS:
            return True  # Primary always wins, proceed with sync

        elif self._conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            try:
                existing = target.store.get(item_id)
                if existing is None:
                    return True

                if hasattr(item, "run_time") and hasattr(existing, "run_time"):
                    if item.run_time >= existing.run_time:
                        return True
                    return False

                return True  # Default to proceeding

            except Exception:
                return True

        elif self._conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
            try:
                existing = target.store.get(item_id)
                return existing is None  # Only proceed if doesn't exist
            except Exception:
                return True

        elif self._conflict_resolution == ConflictResolution.MANUAL:
            return False  # Don't auto-resolve

        return True

    async def _write_to_replica(
        self,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> None:
        """Write item to replica store.

        Args:
            item: Item to write.
            target: Target replica.
        """
        # Run in executor since stores may be synchronous
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, target.store.save, item)

    async def check_health(
        self,
        target: ReplicaTarget[ConfigT],
    ) -> ReplicaHealth:
        """Check health of a replica.

        Args:
            target: Target replica to check.

        Returns:
            Health status.
        """
        try:
            start_time = time.monotonic()

            # Try to list a few IDs as health check
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: target.store.list_ids()[:1]),
                timeout=5.0,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if latency_ms > target.sync_timeout_seconds * 1000:
                target.mark_degraded(latency_ms)
                return ReplicaHealth.DEGRADED
            else:
                target.mark_healthy()
                return ReplicaHealth.HEALTHY

        except asyncio.TimeoutError:
            target.mark_unhealthy()
            return ReplicaHealth.UNHEALTHY

        except Exception:
            target.mark_unhealthy()
            return ReplicaHealth.UNHEALTHY
