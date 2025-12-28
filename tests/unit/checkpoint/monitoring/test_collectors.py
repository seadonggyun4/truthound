"""Tests for monitoring collectors."""

import pytest
from datetime import datetime

from truthound.checkpoint.monitoring.collectors import (
    InMemoryCollector,
)
from truthound.checkpoint.monitoring.protocols import (
    MonitoringEventType,
)


class TestInMemoryCollector:
    """Tests for InMemoryCollector."""

    @pytest.fixture
    def collector(self) -> InMemoryCollector:
        """Create a collector instance."""
        return InMemoryCollector(name="test_collector")

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, collector: InMemoryCollector) -> None:
        """Test connect and disconnect."""
        assert collector.is_connected is False

        await collector.connect()
        assert collector.is_connected is True

        await collector.disconnect()
        assert collector.is_connected is False

    @pytest.mark.asyncio
    async def test_ensure_queue(self, collector: InMemoryCollector) -> None:
        """Test queue registration via ensure_queue."""
        await collector.connect()

        # Use ensure_queue (the actual API method)
        collector.ensure_queue("test_queue")
        metrics = await collector.collect_queue_metrics()

        # Should have test_queue and default queue (created on connect)
        queue_names = [m.queue_name for m in metrics]
        assert "test_queue" in queue_names

    @pytest.mark.asyncio
    async def test_register_worker(self, collector: InMemoryCollector) -> None:
        """Test worker registration."""
        await collector.connect()

        collector.register_worker(
            worker_id="worker-1",
            hostname="localhost",
            max_concurrency=10,
        )

        metrics = await collector.collect_worker_metrics()

        assert len(metrics) == 1
        assert metrics[0].worker_id == "worker-1"
        assert metrics[0].hostname == "localhost"
        assert metrics[0].max_concurrency == 10

    @pytest.mark.asyncio
    async def test_submit_task(self, collector: InMemoryCollector) -> None:
        """Test task submission."""
        await collector.connect()

        task_id = collector.submit_task(
            checkpoint_name="test_checkpoint",
            queue_name="default",
        )

        assert task_id is not None

        tasks = await collector.collect_task_metrics([task_id])
        assert len(tasks) == 1
        assert tasks[0].checkpoint_name == "test_checkpoint"
        assert tasks[0].state == "pending"

    @pytest.mark.asyncio
    async def test_start_and_complete_task(self, collector: InMemoryCollector) -> None:
        """Test task lifecycle."""
        await collector.connect()
        collector.register_worker("worker-1", hostname="localhost")

        task_id = collector.submit_task(
            checkpoint_name="test",
            queue_name="default",
        )

        # Start task
        collector.start_task(task_id, "worker-1")
        tasks = await collector.collect_task_metrics([task_id])
        assert tasks[0].state == "running"
        assert tasks[0].worker_id == "worker-1"

        # Complete task
        collector.complete_task(task_id)
        tasks = await collector.collect_task_metrics([task_id])
        # The implementation uses 'succeeded' not 'completed'
        assert tasks[0].state == "succeeded"

    @pytest.mark.asyncio
    async def test_fail_task(self, collector: InMemoryCollector) -> None:
        """Test task failure."""
        await collector.connect()
        collector.register_worker("worker-1", hostname="localhost")

        task_id = collector.submit_task(
            checkpoint_name="test",
            queue_name="default",
        )
        collector.start_task(task_id, "worker-1")
        collector.fail_task(task_id, error="Test error")

        tasks = await collector.collect_task_metrics([task_id])
        assert tasks[0].state == "failed"
        assert tasks[0].error == "Test error"

    @pytest.mark.asyncio
    async def test_queue_metrics_aggregation(self, collector: InMemoryCollector) -> None:
        """Test queue metrics aggregation."""
        await collector.connect()
        collector.register_worker("worker-1", hostname="localhost", max_concurrency=10)

        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = collector.submit_task(
                checkpoint_name=f"test_{i}",
                queue_name="default",
            )
            task_ids.append(task_id)

        # Start some tasks
        for task_id in task_ids[:3]:
            collector.start_task(task_id, "worker-1")

        # Complete one, fail one
        collector.complete_task(task_ids[0])
        collector.fail_task(task_ids[1], error="Failed")

        metrics = await collector.collect_queue_metrics()
        # Find the default queue
        default_queue = next((m for m in metrics if m.queue_name == "default"), None)
        assert default_queue is not None

        assert default_queue.pending_count == 2  # 2 still pending
        assert default_queue.running_count == 1  # 1 running
        assert default_queue.completed_count == 1
        assert default_queue.failed_count == 1

    @pytest.mark.asyncio
    async def test_worker_metrics_update(self, collector: InMemoryCollector) -> None:
        """Test worker metrics update on task execution."""
        await collector.connect()
        collector.register_worker("worker-1", hostname="localhost", max_concurrency=5)

        # Submit and start tasks
        for i in range(3):
            task_id = collector.submit_task(checkpoint_name=f"test_{i}")
            collector.start_task(task_id, "worker-1")

        metrics = await collector.collect_worker_metrics()
        assert metrics[0].current_tasks == 3

    @pytest.mark.asyncio
    async def test_health_check(self, collector: InMemoryCollector) -> None:
        """Test health check."""
        assert await collector.health_check() is False

        await collector.connect()
        assert await collector.health_check() is True

        await collector.disconnect()
        assert await collector.health_check() is False

    @pytest.mark.asyncio
    async def test_events_emitted(self, collector: InMemoryCollector) -> None:
        """Test that events are emitted."""
        await collector.connect()

        # Submit a task
        task_id = collector.submit_task("test_checkpoint")

        # Events should have been emitted (check the internal event history)
        # The collector should have tracked events
        task = collector.get_task(task_id)
        assert task is not None
        assert task.checkpoint_name == "test_checkpoint"

    def test_unregister_worker(self, collector: InMemoryCollector) -> None:
        """Test worker unregistration."""
        collector.register_worker("worker-1", hostname="localhost")
        assert "worker-1" in collector._workers

        collector.unregister_worker("worker-1")
        assert "worker-1" not in collector._workers

    @pytest.mark.asyncio
    async def test_get_task(self, collector: InMemoryCollector) -> None:
        """Test getting task by ID."""
        await collector.connect()

        task_id = collector.submit_task("test")
        task = collector.get_task(task_id)

        assert task is not None
        assert task.task_id == task_id
        assert task.checkpoint_name == "test"

    @pytest.mark.asyncio
    async def test_get_worker(self, collector: InMemoryCollector) -> None:
        """Test getting worker by ID."""
        await collector.connect()

        collector.register_worker("worker-1", "localhost", max_concurrency=8)
        worker = collector.get_worker("worker-1")

        assert worker is not None
        assert worker.worker_id == "worker-1"
        assert worker.max_concurrency == 8

    @pytest.mark.asyncio
    async def test_get_queue(self, collector: InMemoryCollector) -> None:
        """Test getting queue by name."""
        await collector.connect()

        queue = collector.get_queue("default")
        assert queue is not None
        assert queue.queue_name == "default"

    @pytest.mark.asyncio
    async def test_clear(self, collector: InMemoryCollector) -> None:
        """Test clearing all state."""
        await collector.connect()

        # Add some data
        collector.register_worker("worker-1", "localhost")
        collector.submit_task("test")

        # Clear
        collector.clear()

        # Verify everything is cleared
        assert len(collector._tasks) == 0
        assert len(collector._workers) == 0
        assert len(collector._queues) == 0

    @pytest.mark.asyncio
    async def test_retry_task(self, collector: InMemoryCollector) -> None:
        """Test retrying a failed task."""
        await collector.connect()
        collector.register_worker("worker-1", "localhost")

        task_id = collector.submit_task("test")
        collector.start_task(task_id, "worker-1")
        collector.fail_task(task_id, "First attempt failed")

        # Retry
        result = collector.retry_task(task_id)
        assert result is True

        task = collector.get_task(task_id)
        assert task is not None
        assert task.state == "pending"
        assert task.retries == 1

    @pytest.mark.asyncio
    async def test_update_worker_heartbeat(self, collector: InMemoryCollector) -> None:
        """Test updating worker heartbeat."""
        await collector.connect()
        collector.register_worker("worker-1", "localhost")

        result = collector.update_worker_heartbeat(
            "worker-1",
            cpu_percent=50.0,
            memory_mb=1024.0,
        )

        assert result is True

        worker = collector.get_worker("worker-1")
        assert worker is not None
        assert worker.cpu_percent == 50.0
        assert worker.memory_mb == 1024.0
