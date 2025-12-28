"""Tests for Distributed Checkpoint Orchestration Framework.

This module provides comprehensive tests for the distributed checkpoint
orchestration system, including protocols, backends, and the orchestrator.
"""

from __future__ import annotations

import asyncio
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from truthound.checkpoint.distributed.protocols import (
    # Exceptions
    DistributedError,
    TaskSubmissionError,
    TaskTimeoutError,
    TaskCancelledError,
    WorkerNotAvailableError,
    BackendNotAvailableError,
    # Enums
    TaskState,
    TaskPriority,
    WorkerState,
    BackendCapability,
    # Data Classes
    DistributedConfig,
    WorkerInfo,
    ClusterState,
    DistributedTaskResult,
    DistributedTask,
)


# =============================================================================
# Test Exceptions
# =============================================================================


class TestDistributedExceptions:
    """Tests for distributed exception classes."""

    def test_distributed_error(self):
        """Test base distributed error."""
        error = DistributedError("test error", {"key": "value"})
        assert str(error) == "test error"
        assert error.message == "test error"
        assert error.details == {"key": "value"}

    def test_task_submission_error(self):
        """Test task submission error."""
        error = TaskSubmissionError(
            "submission failed",
            checkpoint_name="my_checkpoint",
            reason="queue full",
        )
        assert "submission failed" in str(error)
        assert error.checkpoint_name == "my_checkpoint"
        assert error.reason == "queue full"

    def test_task_timeout_error(self):
        """Test task timeout error."""
        error = TaskTimeoutError("timed out", "task-123", 30.0)
        assert error.task_id == "task-123"
        assert error.timeout_seconds == 30.0

    def test_task_cancelled_error(self):
        """Test task cancelled error."""
        error = TaskCancelledError("cancelled", "task-456", "user request")
        assert error.task_id == "task-456"
        assert error.reason == "user request"

    def test_worker_not_available_error(self):
        """Test worker not available error."""
        error = WorkerNotAvailableError(
            "no workers",
            required_workers=3,
            available_workers=1,
        )
        assert error.required_workers == 3
        assert error.available_workers == 1

    def test_backend_not_available_error(self):
        """Test backend not available error."""
        error = BackendNotAvailableError(
            "celery",
            reason="not installed",
            install_hint="pip install celery",
        )
        assert error.backend_name == "celery"
        assert error.reason == "not installed"
        assert error.install_hint == "pip install celery"


# =============================================================================
# Test Enums
# =============================================================================


class TestTaskState:
    """Tests for TaskState enum."""

    def test_terminal_states(self):
        """Test terminal state detection."""
        assert TaskState.SUCCEEDED.is_terminal
        assert TaskState.FAILED.is_terminal
        assert TaskState.CANCELLED.is_terminal
        assert TaskState.REVOKED.is_terminal
        assert TaskState.TIMEOUT.is_terminal

        assert not TaskState.PENDING.is_terminal
        assert not TaskState.RUNNING.is_terminal
        assert not TaskState.QUEUED.is_terminal

    def test_active_states(self):
        """Test active state detection."""
        assert TaskState.RUNNING.is_active
        assert TaskState.RETRYING.is_active

        assert not TaskState.PENDING.is_active
        assert not TaskState.SUCCEEDED.is_active


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_from_int(self):
        """Test priority from integer conversion."""
        assert TaskPriority.from_int(-1) == TaskPriority.LOWEST
        assert TaskPriority.from_int(0) == TaskPriority.LOWEST
        assert TaskPriority.from_int(3) == TaskPriority.LOW
        assert TaskPriority.from_int(5) == TaskPriority.NORMAL
        assert TaskPriority.from_int(7) == TaskPriority.HIGH
        assert TaskPriority.from_int(9) == TaskPriority.HIGHEST
        assert TaskPriority.from_int(10) == TaskPriority.CRITICAL
        assert TaskPriority.from_int(100) == TaskPriority.CRITICAL


class TestBackendCapability:
    """Tests for BackendCapability flags."""

    def test_capability_combinations(self):
        """Test capability flag combinations."""
        caps = BackendCapability.ASYNC_SUBMIT | BackendCapability.BATCH_SUBMIT

        assert caps & BackendCapability.ASYNC_SUBMIT
        assert caps & BackendCapability.BATCH_SUBMIT
        assert not (caps & BackendCapability.PRIORITY_QUEUE)

    def test_standard_capabilities(self):
        """Test standard capability set."""
        standard = BackendCapability.STANDARD

        assert standard & BackendCapability.ASYNC_SUBMIT
        assert standard & BackendCapability.BATCH_SUBMIT
        assert standard & BackendCapability.RETRY_POLICY
        assert standard & BackendCapability.TASK_REVOKE


# =============================================================================
# Test Data Classes
# =============================================================================


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DistributedConfig()

        assert config.backend == "local"
        assert config.max_workers == 4
        assert config.max_retries == 3
        assert config.task_timeout_seconds == 3600.0

    def test_with_backend(self):
        """Test config with different backend."""
        config = DistributedConfig(backend="local")
        new_config = config.with_backend("celery", broker_url="redis://localhost")

        assert new_config.backend == "celery"
        assert new_config.backend_options["broker_url"] == "redis://localhost"
        # Original unchanged
        assert config.backend == "local"


class TestWorkerInfo:
    """Tests for WorkerInfo."""

    def test_is_available(self):
        """Test worker availability check."""
        worker = WorkerInfo(
            worker_id="w1",
            hostname="host1",
            state=WorkerState.ONLINE,
            current_tasks=1,
            max_concurrency=4,
        )

        assert worker.is_available
        assert worker.available_slots == 3

    def test_not_available_when_offline(self):
        """Test worker unavailable when offline."""
        worker = WorkerInfo(
            worker_id="w1",
            hostname="host1",
            state=WorkerState.OFFLINE,
            current_tasks=0,
            max_concurrency=4,
        )

        assert not worker.is_available
        assert worker.available_slots == 0

    def test_not_available_when_full(self):
        """Test worker unavailable when at capacity."""
        worker = WorkerInfo(
            worker_id="w1",
            hostname="host1",
            state=WorkerState.ONLINE,
            current_tasks=4,
            max_concurrency=4,
        )

        assert not worker.is_available
        assert worker.available_slots == 0

    def test_load_factor(self):
        """Test worker load factor calculation."""
        worker = WorkerInfo(
            worker_id="w1",
            hostname="host1",
            state=WorkerState.ONLINE,
            current_tasks=2,
            max_concurrency=4,
        )

        assert worker.load_factor == 0.5


class TestClusterState:
    """Tests for ClusterState."""

    def test_cluster_metrics(self):
        """Test cluster state metrics."""
        workers = [
            WorkerInfo(
                worker_id="w1",
                hostname="host1",
                state=WorkerState.ONLINE,
                current_tasks=2,
                max_concurrency=4,
            ),
            WorkerInfo(
                worker_id="w2",
                hostname="host2",
                state=WorkerState.ONLINE,
                current_tasks=3,
                max_concurrency=4,
            ),
        ]

        state = ClusterState(
            workers=workers,
            total_capacity=8,
            current_load=5,
            pending_tasks=10,
        )

        assert state.available_capacity == 3
        assert len(state.online_workers) == 2
        assert state.utilization == 5 / 8


class TestDistributedTaskResult:
    """Tests for DistributedTaskResult."""

    def test_success_result(self):
        """Test successful task result."""
        result = DistributedTaskResult(
            task_id="task-123",
            checkpoint_name="my_checkpoint",
            state=TaskState.SUCCEEDED,
            submitted_at=datetime(2025, 1, 1, 10, 0, 0),
            started_at=datetime(2025, 1, 1, 10, 0, 5),
            completed_at=datetime(2025, 1, 1, 10, 1, 5),
        )

        assert result.success
        assert result.duration_ms == 60000  # 60 seconds
        assert result.queue_time_ms == 5000  # 5 seconds

    def test_failed_result(self):
        """Test failed task result."""
        result = DistributedTaskResult(
            task_id="task-456",
            checkpoint_name="my_checkpoint",
            state=TaskState.FAILED,
            error="validation error",
        )

        assert not result.success
        assert result.error == "validation error"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = DistributedTaskResult(
            task_id="task-789",
            checkpoint_name="test",
            state=TaskState.SUCCEEDED,
        )

        d = result.to_dict()
        assert d["task_id"] == "task-789"
        assert d["state"] == "succeeded"


# =============================================================================
# Test DistributedTask
# =============================================================================


class TestDistributedTask:
    """Tests for DistributedTask class."""

    def test_create_task(self):
        """Test task creation."""
        checkpoint = MagicMock()
        checkpoint.name = "test_checkpoint"

        task = DistributedTask.create(checkpoint)

        assert task.checkpoint_name == "test_checkpoint"
        assert task.task_id.startswith("task-")
        assert task.state == TaskState.PENDING

    def test_set_result(self):
        """Test setting task result."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)

        mock_result = MagicMock()
        task._set_result(mock_result)

        assert task.state == TaskState.SUCCEEDED
        assert task._result == mock_result

    def test_set_error(self):
        """Test setting task error."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)

        task._set_error("test error", ValueError("bad value"))

        assert task.state == TaskState.FAILED
        assert task._error == "test error"
        assert isinstance(task._exception, ValueError)

    def test_cancel(self):
        """Test task cancellation."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)

        assert task.cancel()
        assert task.state == TaskState.CANCELLED

    def test_cannot_cancel_completed(self):
        """Test cannot cancel completed task."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)

        task._set_result(MagicMock())

        assert not task.cancel()

    def test_callbacks(self):
        """Test completion callbacks."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)

        callback_called = []

        def callback(result, exc):
            callback_called.append((result, exc))

        task.add_callback(callback)
        mock_result = MagicMock()
        task._set_result(mock_result)

        assert len(callback_called) == 1
        assert callback_called[0][0] == mock_result

    def test_to_result(self):
        """Test conversion to DistributedTaskResult."""
        checkpoint = MagicMock()
        checkpoint.name = "test"
        task = DistributedTask.create(checkpoint)
        task._set_result(MagicMock())

        result = task.to_result()

        assert isinstance(result, DistributedTaskResult)
        assert result.task_id == task.task_id
        assert result.state == TaskState.SUCCEEDED


# =============================================================================
# Test Base Backend
# =============================================================================


class TestBaseDistributedBackend:
    """Tests for BaseDistributedBackend."""

    def test_metrics_collection(self):
        """Test task metrics collection."""
        from truthound.checkpoint.distributed.base import TaskMetrics

        metrics = TaskMetrics()

        metrics.record_submission()
        metrics.record_submission()

        result1 = DistributedTaskResult(
            task_id="t1",
            checkpoint_name="cp1",
            state=TaskState.SUCCEEDED,
            submitted_at=datetime.now(),
            started_at=datetime.now(),
            completed_at=datetime.now() + timedelta(seconds=1),
        )
        metrics.record_completion(result1)

        result2 = DistributedTaskResult(
            task_id="t2",
            checkpoint_name="cp2",
            state=TaskState.FAILED,
        )
        metrics.record_completion(result2)

        assert metrics.tasks_submitted == 2
        assert metrics.tasks_succeeded == 1
        assert metrics.tasks_failed == 1
        assert metrics.success_rate == 0.5

    def test_prometheus_export(self):
        """Test Prometheus metrics export."""
        from truthound.checkpoint.distributed.base import TaskMetrics

        metrics = TaskMetrics(tasks_submitted=10, tasks_succeeded=8)
        prometheus = metrics.to_prometheus()

        assert "truthound_distributed_tasks_submitted_total 10" in prometheus
        assert "truthound_distributed_tasks_succeeded_total 8" in prometheus


# =============================================================================
# Test Local Backend
# =============================================================================


class TestLocalBackend:
    """Tests for LocalBackend."""

    def test_create_backend(self):
        """Test local backend creation."""
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

        backend = LocalBackend(max_workers=2)

        assert backend.name == "local"
        assert not backend.is_connected

    def test_connect_disconnect(self):
        """Test connection lifecycle."""
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

        backend = LocalBackend(max_workers=2)

        backend.connect()
        assert backend.is_connected

        state = backend.get_cluster_state()
        assert len(state.workers) == 2
        assert state.total_capacity == 2

        backend.disconnect()
        assert not backend.is_connected

    def test_capabilities(self):
        """Test local backend capabilities."""
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

        backend = LocalBackend()

        assert backend.capabilities & BackendCapability.ASYNC_SUBMIT
        assert backend.capabilities & BackendCapability.BATCH_SUBMIT
        assert backend.capabilities & BackendCapability.RESULT_BACKEND


# =============================================================================
# Test Orchestrator
# =============================================================================


class TestDistributedCheckpointOrchestrator:
    """Tests for DistributedCheckpointOrchestrator."""

    def test_create_orchestrator(self):
        """Test orchestrator creation."""
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend
        from truthound.checkpoint.distributed.orchestrator import (
            DistributedCheckpointOrchestrator,
        )

        backend = LocalBackend()
        orchestrator = DistributedCheckpointOrchestrator(backend)

        assert orchestrator.backend == backend
        assert not orchestrator.is_connected

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from truthound.checkpoint.distributed.orchestrator import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitBreakerState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config)

        assert cb.is_closed

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert not cb.is_closed
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.can_execute()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.is_closed

    def test_rate_limiter(self):
        """Test rate limiter functionality."""
        from truthound.checkpoint.distributed.orchestrator import (
            RateLimiter,
            RateLimitConfig,
        )

        config = RateLimitConfig(
            max_tasks_per_second=10,
            max_concurrent_tasks=3,
            burst_limit=5,
        )
        limiter = RateLimiter(config)

        # Should allow burst
        for _ in range(3):
            assert limiter.acquire(timeout=0.1)

        # Release one
        limiter.release()

        # Should allow one more
        assert limiter.acquire(timeout=0.1)


# =============================================================================
# Test Registry
# =============================================================================


class TestBackendRegistry:
    """Tests for BackendRegistry."""

    def test_list_backends(self):
        """Test listing backends."""
        from truthound.checkpoint.distributed.registry import (
            BackendRegistry,
            list_backends,
        )

        backends = list_backends()
        assert "local" in backends

    def test_get_backend(self):
        """Test getting backend by name."""
        from truthound.checkpoint.distributed.registry import get_backend

        backend = get_backend("local")
        assert backend.name == "local"

    def test_get_backend_auto_select(self):
        """Test auto-selecting best backend."""
        from truthound.checkpoint.distributed.registry import get_backend

        # Should return local as it's always available
        backend = get_backend()
        assert backend is not None

    def test_is_backend_available(self):
        """Test backend availability check."""
        from truthound.checkpoint.distributed.registry import is_backend_available

        assert is_backend_available("local")

    def test_custom_backend_registration(self):
        """Test registering custom backend."""
        from truthound.checkpoint.distributed.registry import (
            BackendRegistry,
            register_backend,
        )
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

        register_backend(
            name="custom-test",
            factory=lambda **kw: LocalBackend(**kw),
            priority=1,
            description="Custom test backend",
            check_available=lambda: True,
        )

        info = BackendRegistry.get("custom-test")
        assert info is not None
        assert info.description == "Custom test backend"

        # Cleanup
        BackendRegistry.unregister("custom-test")


# =============================================================================
# Test Get Orchestrator
# =============================================================================


class TestGetOrchestrator:
    """Tests for get_orchestrator function."""

    def test_get_orchestrator_by_name(self):
        """Test getting orchestrator by backend name."""
        from truthound.checkpoint.distributed.registry import get_orchestrator

        orchestrator = get_orchestrator("local")
        assert orchestrator.backend.name == "local"

    def test_get_orchestrator_with_backend_instance(self):
        """Test getting orchestrator with existing backend."""
        from truthound.checkpoint.distributed.registry import get_orchestrator
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend

        backend = LocalBackend(max_workers=8)
        orchestrator = get_orchestrator(backend)

        assert orchestrator.backend is backend

    def test_get_orchestrator_with_config(self):
        """Test getting orchestrator with config."""
        from truthound.checkpoint.distributed.registry import get_orchestrator
        from truthound.checkpoint.distributed.protocols import DistributedConfig

        config = DistributedConfig(max_retries=5)
        orchestrator = get_orchestrator("local", config=config)

        assert orchestrator._config.max_retries == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the distributed orchestration system."""

    def test_full_workflow(self):
        """Test full workflow from orchestrator to result."""
        from truthound.checkpoint.distributed.registry import get_orchestrator
        from truthound.checkpoint.checkpoint import Checkpoint

        orchestrator = get_orchestrator("local", max_workers=2)

        with orchestrator:
            # Create a simple checkpoint
            checkpoint = Checkpoint(
                name="test_checkpoint",
                data_source="tests/fixtures/sample.csv",
                validators=["null"],
            )

            # Submit task
            task = orchestrator.submit(checkpoint, priority=5)

            assert task.task_id is not None
            assert task.checkpoint_name == "test_checkpoint"

    def test_group_submission(self):
        """Test group checkpoint submission."""
        from truthound.checkpoint.distributed.registry import get_orchestrator
        from truthound.checkpoint.checkpoint import Checkpoint

        orchestrator = get_orchestrator("local", max_workers=4)

        with orchestrator:
            checkpoints = [
                Checkpoint(name=f"cp_{i}", data_source=f"data_{i}.csv")
                for i in range(3)
            ]

            tasks = orchestrator.submit_group("test_group", checkpoints)

            assert len(tasks) == 3

            status = orchestrator.get_group_status("test_group")
            assert status["total_tasks"] == 3

    def test_scheduled_execution(self):
        """Test scheduled task execution."""
        from truthound.checkpoint.distributed.registry import get_orchestrator
        from truthound.checkpoint.checkpoint import Checkpoint

        orchestrator = get_orchestrator("local")

        with orchestrator:
            checkpoint = Checkpoint(
                name="scheduled_cp",
                data_source="data.csv",
            )

            # Schedule for 0.5 seconds from now
            task_id = orchestrator.schedule(
                checkpoint,
                delay_seconds=0.5,
            )

            assert task_id is not None

            # Verify it's in scheduled tasks
            scheduled = orchestrator.get_scheduled_tasks()
            assert len(scheduled) >= 1
