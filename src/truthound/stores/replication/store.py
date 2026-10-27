"""Replicated store wrapper implementation.

This module provides a wrapper around storage backends that
adds cross-region replication capabilities.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.stores.base import StoreWriteError
from truthound.stores.replication.base import (
    ReadPreference,
    ReplicaHealth,
    ReplicaState,
    ReplicaTarget,
    ReplicationConfig,
    ReplicationMetrics,
    ReplicationMode,
)
from truthound.stores.replication.monitor import ReplicationMonitor
from truthound.stores.replication.syncer import ReplicationSyncer

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class ReplicatedStore(Generic[T, ConfigT]):
    """Store wrapper with cross-region replication.

    Wraps a primary store and replicates writes to one or more
    replica stores for disaster recovery and read scalability.

    Example:
        >>> primary = S3Store(region="us-east-1")
        >>> replica_eu = S3Store(region="eu-west-1")
        >>>
        >>> config = ReplicationConfig(
        ...     mode=ReplicationMode.ASYNC,
        ...     targets=[
        ...         ReplicaTarget(name="eu", store=replica_eu, region="eu-west-1"),
        ...     ],
        ... )
        >>> store = ReplicatedStore(primary, config)
        >>>
        >>> await store.save(result)  # Writes to primary, replicates to EU
    """

    def __init__(
        self,
        primary: ValidationStore[ConfigT],
        config: ReplicationConfig | None = None,
    ) -> None:
        """Initialize replicated store.

        Args:
            primary: Primary storage backend.
            config: Replication configuration.
        """
        self._primary = primary
        self._config = config or ReplicationConfig()
        self._config.validate()

        self._metrics = ReplicationMetrics()
        self._syncer = ReplicationSyncer(
            primary,
            self._metrics,
            self._config.conflict_resolution,
        )
        self._monitor = ReplicationMonitor(self._config, self._syncer)

        self._replication_queue: asyncio.Queue = asyncio.Queue()
        self._replication_task: asyncio.Task | None = None
        self._started = False
        self._pending_replications: set[asyncio.Task] = set()
        self._background_failures = 0

    def _replication_finished(self, task) -> None:
        self._pending_replications.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._background_failures += 1

    def set_io_guard(self, guard) -> None:
        """Check an owner-supplied authorization guard immediately before I/O.

        Applies to background replication/health reads as well as writes. The
        guard is trusted local code, not a serialized callback.
        """
        self._syncer.io_guard = guard

    def _save_primary(self, item: T) -> str:
        item_id = self._syncer._io(self._primary.save, item)
        self._metrics.record_primary_write()
        return item_id

    @property
    def primary(self) -> ValidationStore[ConfigT]:
        """Get primary store."""
        return self._primary

    @property
    def config(self) -> ReplicationConfig:
        """Get replication configuration."""
        return self._config

    @property
    def metrics(self) -> ReplicationMetrics:
        """Get replication metrics."""
        return self._metrics

    @property
    def monitor(self) -> ReplicationMonitor:
        """Get replication monitor."""
        return self._monitor

    def _get_healthy_replicas(self) -> list[ReplicaTarget[ConfigT]]:
        """Get list of healthy replica targets."""
        return [
            target
            for target in self._config.targets
            if target.health in (ReplicaHealth.HEALTHY, ReplicaHealth.DEGRADED)
            and target.state == ReplicaState.ACTIVE
        ]

    def _select_read_replica(self) -> ValidationStore[ConfigT] | None:
        """Select a replica for reading based on read preference."""
        preference = self._config.read_preference

        if preference == ReadPreference.PRIMARY:
            return None  # Use primary

        healthy = self._get_healthy_replicas()
        if len(healthy) != sum(target.state == ReplicaState.ACTIVE for target in self._config.targets):
            return False
        if not healthy:
            return None  # Fallback to primary

        if preference == ReadPreference.SECONDARY:
            # Return first healthy read replica
            for target in healthy:
                if target.is_read_replica or not target.is_read_replica:
                    return target.store
            return None

        elif preference == ReadPreference.NEAREST:
            # Return replica with lowest lag
            sorted_targets = sorted(healthy, key=lambda t: t.replication_lag_ms)
            if sorted_targets:
                return sorted_targets[0].store
            return None

        elif preference == ReadPreference.ANY:
            # Return any healthy replica
            if healthy:
                return healthy[0].store
            return None

        return None

    async def _replicate_to_target(
        self,
        item_id: str,
        item: T,
        target: ReplicaTarget[ConfigT],
    ) -> bool:
        """Replicate item to a specific target."""
        return await self._syncer.sync_item_with_retry(item_id, item, target)

    async def _replicate_async(
        self,
        item_id: str,
        item: T,
    ) -> None:
        """Replicate item asynchronously to all targets."""
        tasks = [
            self._replicate_to_target(item_id, item, target)
            for target in self._get_healthy_replicas()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _replicate_sync(
        self,
        item_id: str,
        item: T,
    ) -> bool:
        """Replicate item synchronously to all targets."""
        healthy = self._get_healthy_replicas()
        if not healthy:
            return True  # No replicas to sync to

        tasks = [
            self._replicate_to_target(item_id, item, target)
            for target in healthy
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return all(r is True for r in results)

    async def _replicate_semi_sync(
        self,
        item_id: str,
        item: T,
    ) -> bool:
        """Replicate item to at least min_sync_replicas."""
        healthy = self._get_healthy_replicas()
        if len(healthy) < self._config.min_sync_replicas:
            return False

        tasks = [
            self._replicate_to_target(item_id, item, target)
            for target in healthy
        ]
        owned = [asyncio.create_task(t) for t in tasks]
        pending = set(owned)
        success_count = 0
        try:
            while success_count < self._config.min_sync_replicas and pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                success_count += sum(
                    not task.cancelled() and task.exception() is None and task.result() is True
                    for task in done
                )
            return success_count >= self._config.min_sync_replicas
        finally:
            for task in owned:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)

    async def _process_replication_queue(self) -> None:
        """Background task to process async replication queue."""
        while True:
            try:
                item_id, item = await self._replication_queue.get()
                try:
                    await self._replicate_async(item_id, item)
                finally:
                    self._replication_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def start(self) -> None:
        """Start replication services."""
        if self._started:
            return

        self._started = True

        # Start monitor
        await self._monitor.start()

        # Start async replication worker if in async mode
        if self._config.mode == ReplicationMode.ASYNC:
            self._replication_task = asyncio.create_task(
                self._process_replication_queue()
            )

    async def stop(self) -> None:
        """Stop replication services."""
        await self._monitor.stop()
        await self.flush()

        if self._replication_task:
            self._replication_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._replication_task
            self._replication_task = None

        self._started = False

    async def flush(self) -> None:
        """Drain work owned by this instance; inspect metrics for replica failures.

        This is process-local draining, not a durable delivery receipt.
        """
        if self._replication_task is not None and not self._replication_task.done():
            await self._replication_queue.join()
        else:
            while not self._replication_queue.empty():
                item_id, item = self._replication_queue.get_nowait()
                try:
                    await self._replicate_async(item_id, item)
                finally:
                    self._replication_queue.task_done()
        while self._pending_replications:
            pending = tuple(self._pending_replications)
            await asyncio.gather(*pending, return_exceptions=True)
            self._pending_replications.difference_update(pending)

    def save(self, item: T) -> str:
        """Save an item to primary and replicate.

        For sync replication, this blocks until replication completes.
        For async replication, replication happens in background.

        Args:
            item: Item to save.

        Returns:
            Item ID.
        """
        if self._config.mode != ReplicationMode.ASYNC:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Use save_async from a running event loop")
        item_id = self._save_primary(item)

        # Handle replication based on mode
        if self._config.mode == ReplicationMode.SYNC:
            # Run synchronously
            loop = asyncio.new_event_loop()
            try:
                if not loop.run_until_complete(self._replicate_sync(item_id, item)):
                    raise StoreWriteError("Primary saved but synchronous replication failed")
            finally:
                loop.close()

        elif self._config.mode == ReplicationMode.SEMI_SYNC:
            loop = asyncio.new_event_loop()
            try:
                if not loop.run_until_complete(self._replicate_semi_sync(item_id, item)):
                    raise StoreWriteError("Primary saved but required replica count was not reached")
            finally:
                loop.close()

        elif self._config.mode == ReplicationMode.ASYNC:
            # Queue for background replication
            self._replication_queue.put_nowait((item_id, item))

        return item_id

    async def save_async(self, item: T) -> str:
        """Save an item asynchronously.

        Args:
            item: Item to save.

        Returns:
            Item ID.
        """
        # Save to primary
        loop = asyncio.get_event_loop()
        item_id = await loop.run_in_executor(None, self._save_primary, item)

        # Replicate based on mode
        if self._config.mode == ReplicationMode.SYNC:
            if not await self._replicate_sync(item_id, item):
                raise StoreWriteError("Primary saved but synchronous replication failed")
        elif self._config.mode == ReplicationMode.SEMI_SYNC:
            if not await self._replicate_semi_sync(item_id, item):
                raise StoreWriteError("Primary saved but required replica count was not reached")
        else:  # ASYNC
            task = asyncio.create_task(self._replicate_async(item_id, item))
            self._pending_replications.add(task)
            task.add_done_callback(self._replication_finished)

        return item_id

    def get(self, item_id: str) -> T | None:
        """Get an item, using read preference.

        Args:
            item_id: Item identifier.

        Returns:
            The item or None.
        """
        replica = self._select_read_replica()

        if replica is not None:
            try:
                return self._syncer._io(replica.get, item_id)
            except Exception:
                pass  # Fallback to primary

        return self._syncer._io(self._primary.get, item_id)

    def delete(self, item_id: str) -> bool:
        """Delete an item from primary and replicas.

        Args:
            item_id: Item identifier.

        Returns:
            True if deleted from primary.
        """
        # Delete from primary
        deleted = self._syncer._io(self._primary.delete, item_id)

        # Delete from replicas (best effort)
        for target in self._config.targets:
            with suppress(Exception):
                self._syncer._io(target.store.delete, item_id)

        return deleted

    def exists(self, item_id: str) -> bool:
        """Check if an item exists.

        Args:
            item_id: Item identifier.

        Returns:
            True if exists.
        """
        return self._syncer._io(self._primary.exists, item_id)

    def list_ids(self, **kwargs: Any) -> list[str]:
        """List item IDs.

        Returns:
            List of item IDs.
        """
        return self._syncer._io(self._primary.list_ids, **kwargs)

    async def full_sync(self, target_name: str | None = None) -> dict[str, Any]:
        """Perform full synchronization to replicas.

        Args:
            target_name: Optional specific target to sync.

        Returns:
            Sync results per target.
        """
        results = {}

        targets = self._config.targets
        if target_name:
            targets = [t for t in targets if t.name == target_name]

        for target in targets:
            result = await self._syncer.full_sync(target)
            results[target.name] = {
                "success": result.success,
                "items_synced": result.items_synced,
                "items_failed": result.items_failed,
                "duration_ms": result.duration_ms,
            }

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get replicated store statistics."""
        return {
            "mode": self._config.mode.value,
            "read_preference": self._config.read_preference.value,
            "metrics": self._metrics.to_dict(),
            "health": self._monitor.get_health_summary(),
            "queue_size": self._replication_queue.qsize(),
            "pending_tasks": len(self._pending_replications),
            "background_failures": self._background_failures,
        }

    async def __aenter__(self) -> ReplicatedStore[T, ConfigT]:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
