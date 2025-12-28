"""In-memory metric store.

Provides in-memory storage for metrics with time-based querying.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
import asyncio
from collections import defaultdict

from truthound.ml.monitoring.protocols import IMetricStore, ModelMetrics


@dataclass
class InMemoryStoreConfig:
    """Configuration for in-memory store.

    Attributes:
        max_entries_per_model: Maximum entries to keep per model
        retention_hours: Hours to retain metrics
        cleanup_interval_seconds: Cleanup interval
    """

    max_entries_per_model: int = 10000
    retention_hours: int = 24
    cleanup_interval_seconds: int = 300


class InMemoryMetricStore(IMetricStore):
    """In-memory metric store.

    Stores metrics in memory with automatic cleanup.
    Suitable for development and testing.

    Example:
        >>> store = InMemoryMetricStore()
        >>> await store.store(metrics)
        >>> history = await store.query(model_id, start, end)
    """

    def __init__(self, config: InMemoryStoreConfig | None = None):
        self._config = config or InMemoryStoreConfig()
        self._store: dict[str, list[ModelMetrics]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            await asyncio.sleep(self._config.cleanup_interval_seconds)
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up old metrics."""
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self._config.retention_hours)

            for model_id in list(self._store.keys()):
                entries = self._store[model_id]

                # Remove old entries
                self._store[model_id] = [
                    m for m in entries
                    if m.timestamp >= cutoff
                ]

                # Enforce max entries
                if len(self._store[model_id]) > self._config.max_entries_per_model:
                    self._store[model_id] = self._store[model_id][-self._config.max_entries_per_model:]

    async def store(self, metrics: ModelMetrics) -> None:
        """Store metrics.

        Args:
            metrics: Metrics to store
        """
        async with self._lock:
            self._store[metrics.model_id].append(metrics)

            # Enforce max entries
            if len(self._store[metrics.model_id]) > self._config.max_entries_per_model:
                self._store[metrics.model_id].pop(0)

    async def query(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean",
    ) -> list[ModelMetrics]:
        """Query historical metrics.

        Args:
            model_id: Model identifier
            start_time: Query start time
            end_time: Query end time
            aggregation: Aggregation function (not used for in-memory)

        Returns:
            List of metrics in time range
        """
        async with self._lock:
            entries = self._store.get(model_id, [])

            return [
                m for m in entries
                if start_time <= m.timestamp <= end_time
            ]

    async def get_latest(self, model_id: str) -> ModelMetrics | None:
        """Get latest metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Latest metrics or None
        """
        async with self._lock:
            entries = self._store.get(model_id, [])
            return entries[-1] if entries else None

    async def delete(
        self,
        model_id: str,
        before: datetime | None = None,
    ) -> int:
        """Delete metrics.

        Args:
            model_id: Model identifier
            before: Delete metrics before this time

        Returns:
            Number of deleted records
        """
        async with self._lock:
            if model_id not in self._store:
                return 0

            original_count = len(self._store[model_id])

            if before:
                self._store[model_id] = [
                    m for m in self._store[model_id]
                    if m.timestamp >= before
                ]
            else:
                self._store[model_id] = []

            return original_count - len(self._store[model_id])

    async def get_model_ids(self) -> list[str]:
        """Get all model IDs with stored metrics.

        Returns:
            List of model IDs
        """
        async with self._lock:
            return list(self._store.keys())

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics.

        Returns:
            Store statistics
        """
        async with self._lock:
            return {
                "total_models": len(self._store),
                "total_entries": sum(len(entries) for entries in self._store.values()),
                "entries_per_model": {
                    model_id: len(entries)
                    for model_id, entries in self._store.items()
                },
            }

    async def __aenter__(self) -> "InMemoryMetricStore":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
