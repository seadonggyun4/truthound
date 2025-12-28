"""Redis-based metric collector.

Collects metrics from Redis, compatible with Celery result backend
and other Redis-based task queues.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    CollectorError,
)
from truthound.checkpoint.monitoring.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class RedisCollector(BaseCollector):
    """Redis-based metric collector.

    Collects metrics from Redis, supporting both direct Redis
    task storage and Celery result backend patterns.

    Key patterns supported:
    - Celery: celery-task-meta-{task_id}
    - Custom: truthound:tasks:{task_id}, truthound:queues:{queue}

    Example:
        >>> collector = RedisCollector(
        ...     redis_url="redis://localhost:6379/0",
        ...     key_prefix="truthound:",
        ... )
        >>> await collector.connect()
        >>>
        >>> metrics = await collector.collect_queue_metrics()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "truthound:",
        celery_compatible: bool = True,
        name: str = "redis",
        collect_interval_seconds: float = 5.0,
        cache_ttl_seconds: float = 2.0,
        pool_size: int = 10,
    ) -> None:
        """Initialize Redis collector.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Key prefix for Truthound keys.
            celery_compatible: Enable Celery-compatible key patterns.
            name: Collector name.
            collect_interval_seconds: Collection interval.
            cache_ttl_seconds: Cache TTL.
            pool_size: Connection pool size.
        """
        super().__init__(
            name=name,
            collect_interval_seconds=collect_interval_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._celery_compatible = celery_compatible
        self._pool_size = pool_size
        self._redis: Any = None
        self._pubsub: Any = None
        self._listen_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise CollectorError(
                "redis package not installed",
                self.name,
                "Install with: pip install redis",
            )

        try:
            self._redis = await redis.from_url(
                self._redis_url,
                max_connections=self._pool_size,
            )

            # Test connection
            await self._redis.ping()

            await super().connect()
            logger.info(f"Connected to Redis at {self._redis_url}")

        except Exception as e:
            raise CollectorError(
                f"Failed to connect to Redis: {e}",
                self.name,
                str(e),
            )

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        await self._stop_pubsub()

        if self._redis:
            await self._redis.close()
            self._redis = None

        await super().disconnect()

    async def _start_pubsub(self) -> None:
        """Start listening to Redis pub/sub for events."""
        if self._pubsub is not None or self._redis is None:
            return

        self._pubsub = self._redis.pubsub()

        # Subscribe to relevant channels
        channels = [
            f"{self._key_prefix}events:*",
        ]
        if self._celery_compatible:
            channels.append("celery-task-meta-*")

        await self._pubsub.psubscribe(*channels)
        self._listen_task = asyncio.create_task(self._pubsub_listener())

    async def _stop_pubsub(self) -> None:
        """Stop pub/sub listener."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None

    async def _pubsub_listener(self) -> None:
        """Listen for pub/sub events."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "pmessage":
                    await self._handle_pubsub_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Pub/sub listener error: {e}")

    async def _handle_pubsub_message(self, message: dict) -> None:
        """Handle a pub/sub message."""
        try:
            channel = message["channel"]
            data = message["data"]

            if isinstance(data, bytes):
                data = data.decode("utf-8")
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            # Determine event type from channel
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            if "task:completed" in channel:
                event_type = MonitoringEventType.TASK_COMPLETED
            elif "task:failed" in channel:
                event_type = MonitoringEventType.TASK_FAILED
            elif "task:started" in channel:
                event_type = MonitoringEventType.TASK_STARTED
            elif "worker" in channel:
                event_type = MonitoringEventType.WORKER_STATE_CHANGED
            else:
                event_type = MonitoringEventType.METRICS_COLLECTED

            event = MonitoringEvent(
                event_type=event_type,
                source=self.name,
                data={"channel": channel, "message": data},
            )
            self._emit_event(event)

        except Exception as e:
            logger.error(f"Error handling pub/sub message: {e}")

    async def collect_queue_metrics(self) -> list[QueueMetrics]:
        """Collect queue metrics from Redis."""
        if self._redis is None:
            raise CollectorError("Not connected to Redis", self.name)

        metrics = []

        try:
            # Get all queue keys
            queue_pattern = f"{self._key_prefix}queues:*:meta"
            async for key in self._redis.scan_iter(match=queue_pattern):
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                queue_name = key.replace(f"{self._key_prefix}queues:", "").replace(":meta", "")
                queue_data = await self._redis.hgetall(key)

                if queue_data:
                    # Decode bytes
                    queue_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in queue_data.items()
                    }

                    metrics.append(QueueMetrics(
                        queue_name=queue_name,
                        pending_count=int(queue_data.get("pending", 0)),
                        running_count=int(queue_data.get("running", 0)),
                        completed_count=int(queue_data.get("completed", 0)),
                        failed_count=int(queue_data.get("failed", 0)),
                        avg_wait_time_ms=float(queue_data.get("avg_wait_ms", 0)),
                        avg_execution_time_ms=float(queue_data.get("avg_exec_ms", 0)),
                        throughput_per_second=float(queue_data.get("throughput", 0)),
                    ))

            # Celery-compatible: inspect Celery queues
            if self._celery_compatible:
                celery_metrics = await self._collect_celery_queue_metrics()
                metrics.extend(celery_metrics)

        except Exception as e:
            raise CollectorError(
                f"Failed to collect queue metrics: {e}",
                self.name,
                str(e),
            )

        return metrics

    async def _collect_celery_queue_metrics(self) -> list[QueueMetrics]:
        """Collect metrics from Celery-style queues."""
        metrics = []

        try:
            # Check common Celery queue names
            queue_names = ["celery", "truthound", "default"]

            for queue_name in queue_names:
                pending = await self._redis.llen(queue_name)

                if pending > 0:
                    metrics.append(QueueMetrics(
                        queue_name=f"celery:{queue_name}",
                        pending_count=pending,
                        labels={"type": "celery"},
                    ))

        except Exception as e:
            logger.warning(f"Error collecting Celery queue metrics: {e}")

        return metrics

    async def collect_worker_metrics(self) -> list[WorkerMetrics]:
        """Collect worker metrics from Redis."""
        if self._redis is None:
            raise CollectorError("Not connected to Redis", self.name)

        metrics = []

        try:
            # Get all worker keys
            worker_pattern = f"{self._key_prefix}workers:*:meta"
            async for key in self._redis.scan_iter(match=worker_pattern):
                if isinstance(key, bytes):
                    key = key.decode("utf-8")

                worker_id = key.replace(f"{self._key_prefix}workers:", "").replace(":meta", "")
                worker_data = await self._redis.hgetall(key)

                if worker_data:
                    # Decode bytes
                    worker_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in worker_data.items()
                    }

                    last_heartbeat_str = worker_data.get("last_heartbeat", "")
                    try:
                        last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
                    except (ValueError, TypeError):
                        last_heartbeat = datetime.now()

                    tags_str = worker_data.get("tags", "")
                    tags = frozenset(tags_str.split(",")) if tags_str else frozenset()

                    metrics.append(WorkerMetrics(
                        worker_id=worker_id,
                        state=worker_data.get("state", "unknown"),
                        current_tasks=int(worker_data.get("current_tasks", 0)),
                        completed_tasks=int(worker_data.get("completed", 0)),
                        failed_tasks=int(worker_data.get("failed", 0)),
                        cpu_percent=float(worker_data.get("cpu_percent", 0)),
                        memory_mb=float(worker_data.get("memory_mb", 0)),
                        uptime_seconds=float(worker_data.get("uptime", 0)),
                        last_heartbeat=last_heartbeat,
                        hostname=worker_data.get("hostname", ""),
                        max_concurrency=int(worker_data.get("max_concurrency", 1)),
                        tags=tags,
                    ))

        except Exception as e:
            raise CollectorError(
                f"Failed to collect worker metrics: {e}",
                self.name,
                str(e),
            )

        return metrics

    async def collect_task_metrics(
        self,
        task_ids: list[str] | None = None,
    ) -> list[TaskMetrics]:
        """Collect task metrics from Redis."""
        if self._redis is None:
            raise CollectorError("Not connected to Redis", self.name)

        metrics = []

        try:
            if task_ids is not None:
                # Get specific tasks
                for task_id in task_ids:
                    task_data = await self._get_task_data(task_id)
                    if task_data:
                        metrics.append(task_data)
            else:
                # Get active tasks
                task_pattern = f"{self._key_prefix}tasks:*:meta"
                async for key in self._redis.scan_iter(match=task_pattern):
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")

                    task_id = key.replace(f"{self._key_prefix}tasks:", "").replace(":meta", "")
                    task_data = await self._get_task_data(task_id)

                    if task_data and task_data.state in ("pending", "running", "retrying"):
                        metrics.append(task_data)

        except Exception as e:
            raise CollectorError(
                f"Failed to collect task metrics: {e}",
                self.name,
                str(e),
            )

        return metrics

    async def _get_task_data(self, task_id: str) -> TaskMetrics | None:
        """Get task data from Redis."""
        key = f"{self._key_prefix}tasks:{task_id}:meta"
        task_data = await self._redis.hgetall(key)

        if not task_data:
            # Try Celery format
            if self._celery_compatible:
                celery_key = f"celery-task-meta-{task_id}"
                celery_data = await self._redis.get(celery_key)
                if celery_data:
                    try:
                        if isinstance(celery_data, bytes):
                            celery_data = celery_data.decode("utf-8")
                        data = json.loads(celery_data)
                        return TaskMetrics(
                            task_id=task_id,
                            checkpoint_name=data.get("task", "unknown"),
                            state=data.get("status", "unknown").lower(),
                            error=data.get("traceback"),
                        )
                    except json.JSONDecodeError:
                        pass
            return None

        # Decode bytes
        task_data = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in task_data.items()
        }

        # Parse timestamps
        def parse_timestamp(ts_str: str) -> datetime | None:
            if not ts_str:
                return None
            try:
                return datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                return None

        return TaskMetrics(
            task_id=task_id,
            checkpoint_name=task_data.get("checkpoint_name", ""),
            state=task_data.get("state", "unknown"),
            queue_name=task_data.get("queue_name", "default"),
            worker_id=task_data.get("worker_id"),
            submitted_at=parse_timestamp(task_data.get("submitted_at", "")) or datetime.now(),
            started_at=parse_timestamp(task_data.get("started_at", "")),
            completed_at=parse_timestamp(task_data.get("completed_at", "")),
            retries=int(task_data.get("retries", 0)),
            error=task_data.get("error"),
        )

    async def subscribe(self) -> AsyncIterator[MonitoringEvent]:
        """Subscribe to Redis pub/sub events."""
        await self._start_pubsub()

        async for event in super().subscribe():
            yield event

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if self._redis is None:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    # Convenience methods for storing metrics

    async def record_queue_metrics(
        self,
        queue_name: str,
        pending: int,
        running: int,
        completed: int,
        failed: int,
        avg_wait_ms: float = 0.0,
        avg_exec_ms: float = 0.0,
    ) -> None:
        """Record queue metrics to Redis.

        Args:
            queue_name: Queue name.
            pending: Pending count.
            running: Running count.
            completed: Completed count.
            failed: Failed count.
            avg_wait_ms: Average wait time.
            avg_exec_ms: Average execution time.
        """
        if self._redis is None:
            return

        key = f"{self._key_prefix}queues:{queue_name}:meta"
        total = completed + failed
        throughput = total / 60.0  # Approximate per-second over last minute

        await self._redis.hset(key, mapping={
            "pending": str(pending),
            "running": str(running),
            "completed": str(completed),
            "failed": str(failed),
            "avg_wait_ms": str(avg_wait_ms),
            "avg_exec_ms": str(avg_exec_ms),
            "throughput": str(throughput),
            "updated_at": datetime.now().isoformat(),
        })

    async def record_worker_metrics(
        self,
        worker_id: str,
        hostname: str,
        state: str = "online",
        current_tasks: int = 0,
        completed: int = 0,
        failed: int = 0,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        uptime: float = 0.0,
        max_concurrency: int = 4,
        tags: list[str] | None = None,
    ) -> None:
        """Record worker metrics to Redis."""
        if self._redis is None:
            return

        key = f"{self._key_prefix}workers:{worker_id}:meta"

        await self._redis.hset(key, mapping={
            "hostname": hostname,
            "state": state,
            "current_tasks": str(current_tasks),
            "completed": str(completed),
            "failed": str(failed),
            "cpu_percent": str(cpu_percent),
            "memory_mb": str(memory_mb),
            "uptime": str(uptime),
            "max_concurrency": str(max_concurrency),
            "tags": ",".join(tags or []),
            "last_heartbeat": datetime.now().isoformat(),
        })
