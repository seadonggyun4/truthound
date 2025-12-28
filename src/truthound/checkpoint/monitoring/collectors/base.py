"""Base collector implementation.

Provides abstract base class with common functionality for all collectors.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator

from truthound.checkpoint.monitoring.protocols import (
    MetricCollectorProtocol,
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    CollectorError,
)
from truthound.checkpoint.monitoring.events import EventEmitter

logger = logging.getLogger(__name__)


class BaseCollector(ABC, EventEmitter):
    """Abstract base class for metric collectors.

    Provides common functionality including:
    - Connection management
    - Event emission
    - Health checks
    - Subscription management

    Subclasses must implement the abstract collection methods.
    """

    def __init__(
        self,
        name: str | None = None,
        collect_interval_seconds: float = 5.0,
        cache_ttl_seconds: float = 1.0,
    ) -> None:
        """Initialize collector.

        Args:
            name: Collector name.
            collect_interval_seconds: Interval for automatic collection.
            cache_ttl_seconds: TTL for cached metrics.
        """
        super().__init__()
        self._name = name or self.__class__.__name__
        self._collect_interval = collect_interval_seconds
        self._cache_ttl = cache_ttl_seconds
        self._connected = False
        self._running = False
        self._collect_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[MonitoringEvent] = asyncio.Queue()

        # Cached metrics
        self._cached_queue_metrics: list[QueueMetrics] = []
        self._cached_worker_metrics: list[WorkerMetrics] = []
        self._cached_task_metrics: list[TaskMetrics] = []
        self._cache_timestamp: datetime | None = None

        self.set_event_source(self._name)

    @property
    def name(self) -> str:
        """Get collector name."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Check if collector is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the data source.

        Override in subclasses that require connection setup.
        """
        self._connected = True
        logger.info(f"{self._name} connected")

    async def disconnect(self) -> None:
        """Disconnect from the data source.

        Override in subclasses that require cleanup.
        """
        await self.stop_collection()
        self._connected = False
        logger.info(f"{self._name} disconnected")

    async def start_collection(self) -> None:
        """Start automatic metric collection loop."""
        if self._running:
            return

        self._running = True
        self._collect_task = asyncio.create_task(self._collection_loop())
        logger.info(f"{self._name} started collection loop")

    async def stop_collection(self) -> None:
        """Stop automatic metric collection loop."""
        self._running = False
        if self._collect_task:
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass
            self._collect_task = None
        logger.info(f"{self._name} stopped collection loop")

    async def _collection_loop(self) -> None:
        """Internal collection loop."""
        while self._running:
            try:
                # Collect all metrics
                queue_metrics = await self.collect_queue_metrics()
                worker_metrics = await self.collect_worker_metrics()

                # Cache results
                self._cached_queue_metrics = queue_metrics
                self._cached_worker_metrics = worker_metrics
                self._cache_timestamp = datetime.now()

                # Emit collection event
                await self.emit_async(
                    MonitoringEventType.METRICS_COLLECTED,
                    data={
                        "queue_count": len(queue_metrics),
                        "worker_count": len(worker_metrics),
                    },
                )

            except Exception as e:
                logger.error(f"Collection error in {self._name}: {e}")

            await asyncio.sleep(self._collect_interval)

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if self._cache_timestamp is None:
            return False
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_ttl

    async def get_cached_queue_metrics(self) -> list[QueueMetrics]:
        """Get cached queue metrics, collecting if needed."""
        if not self._is_cache_valid():
            self._cached_queue_metrics = await self.collect_queue_metrics()
            self._cache_timestamp = datetime.now()
        return self._cached_queue_metrics

    async def get_cached_worker_metrics(self) -> list[WorkerMetrics]:
        """Get cached worker metrics, collecting if needed."""
        if not self._is_cache_valid():
            self._cached_worker_metrics = await self.collect_worker_metrics()
            self._cache_timestamp = datetime.now()
        return self._cached_worker_metrics

    @abstractmethod
    async def collect_queue_metrics(self) -> list[QueueMetrics]:
        """Collect current queue metrics.

        Must be implemented by subclasses.

        Returns:
            List of queue metrics.

        Raises:
            CollectorError: If collection fails.
        """
        pass

    @abstractmethod
    async def collect_worker_metrics(self) -> list[WorkerMetrics]:
        """Collect current worker metrics.

        Must be implemented by subclasses.

        Returns:
            List of worker metrics.

        Raises:
            CollectorError: If collection fails.
        """
        pass

    @abstractmethod
    async def collect_task_metrics(
        self,
        task_ids: list[str] | None = None,
    ) -> list[TaskMetrics]:
        """Collect task metrics.

        Must be implemented by subclasses.

        Args:
            task_ids: Optional list of specific task IDs to collect.

        Returns:
            List of task metrics.

        Raises:
            CollectorError: If collection fails.
        """
        pass

    async def subscribe(self) -> AsyncIterator[MonitoringEvent]:
        """Subscribe to real-time metric updates.

        Yields:
            Monitoring events as they occur.
        """
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                yield event
            except asyncio.TimeoutError:
                # Check if still running
                if not self._running and self._event_queue.empty():
                    break
            except asyncio.CancelledError:
                break

    def _emit_event(self, event: MonitoringEvent) -> None:
        """Internal method to queue events for subscribers."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event if queue is full
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass

    async def health_check(self) -> bool:
        """Check collector health.

        Override in subclasses for custom health checks.

        Returns:
            True if healthy, False otherwise.
        """
        return self._connected

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, connected={self._connected})"
