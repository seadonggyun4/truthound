"""Distributed Checkpoint Orchestrator.

This module provides the main orchestrator class for distributed
checkpoint execution with advanced features like scheduling,
rate limiting, and circuit breaker support.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterator
from queue import PriorityQueue, Empty
from uuid import uuid4

from truthound.checkpoint.distributed.base import (
    BaseDistributedBackend,
    BaseDistributedOrchestrator,
    TaskMetrics,
)
from truthound.checkpoint.distributed.protocols import (
    BackendCapability,
    ClusterState,
    DistributedConfig,
    DistributedError,
    DistributedTask,
    DistributedTaskProtocol,
    DistributedTaskResult,
    TaskCancelledError,
    TaskPriority,
    TaskState,
    TaskSubmissionError,
    TaskTimeoutError,
    WorkerInfo,
    WorkerNotAvailableError,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult


logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiting
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        max_tasks_per_second: Maximum tasks per second.
        max_tasks_per_minute: Maximum tasks per minute.
        max_concurrent_tasks: Maximum concurrent tasks.
        burst_limit: Allow burst up to this limit.
    """

    max_tasks_per_second: float = 10.0
    max_tasks_per_minute: float = 100.0
    max_concurrent_tasks: int = 50
    burst_limit: int = 20


class RateLimiter:
    """Token bucket rate limiter for task submission."""

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._tokens = config.burst_limit
        self._last_update = time.time()
        self._concurrent_count = 0
        self._minute_count = 0
        self._minute_start = time.time()

    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire permission to submit a task.

        Args:
            timeout: Maximum time to wait.

        Returns:
            True if acquired, False if timeout.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                self._refill_tokens()
                self._reset_minute_if_needed()

                if (
                    self._tokens >= 1
                    and self._concurrent_count < self._config.max_concurrent_tasks
                    and self._minute_count < self._config.max_tasks_per_minute
                ):
                    self._tokens -= 1
                    self._concurrent_count += 1
                    self._minute_count += 1
                    return True

            time.sleep(0.1)

        return False

    def release(self) -> None:
        """Release a concurrent task slot."""
        with self._lock:
            self._concurrent_count = max(0, self._concurrent_count - 1)

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        refill = elapsed * self._config.max_tasks_per_second
        self._tokens = min(self._config.burst_limit, self._tokens + refill)
        self._last_update = now

    def _reset_minute_if_needed(self) -> None:
        """Reset minute counter if a minute has passed."""
        now = time.time()
        if now - self._minute_start >= 60:
            self._minute_count = 0
            self._minute_start = now


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreakerState:
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures to open circuit.
        success_threshold: Number of successes to close circuit.
        timeout_seconds: Time before half-open from open.
        half_open_max_calls: Max calls in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker for distributed task submission."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self.state == CircuitBreakerState.CLOSED

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self._config.timeout_seconds:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        return True
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitBreakerState.OPEN
                self._success_count = 0
                logger.warning("Circuit breaker OPEN after failure in HALF_OPEN")

            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker OPEN after {self._failure_count} failures"
                    )

    def reset(self) -> None:
        """Reset the circuit breaker."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# =============================================================================
# Scheduled Task
# =============================================================================


@dataclass
class ScheduledTask:
    """A task scheduled for future execution.

    Attributes:
        task_id: Unique task identifier.
        checkpoint: The checkpoint to execute.
        scheduled_time: When to execute.
        priority: Task priority.
        context: Execution context.
        recurring: If True, reschedule after execution.
        interval_seconds: Interval for recurring tasks.
    """

    task_id: str
    checkpoint: "Checkpoint"
    scheduled_time: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    context: dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    interval_seconds: float = 0.0

    def __lt__(self, other: "ScheduledTask") -> bool:
        """Compare by scheduled time for priority queue."""
        return self.scheduled_time < other.scheduled_time


# =============================================================================
# Main Orchestrator
# =============================================================================


class DistributedCheckpointOrchestrator(BaseDistributedOrchestrator):
    """Advanced orchestrator for distributed checkpoint execution.

    This orchestrator provides additional features beyond the base class:
    - Rate limiting for task submission
    - Circuit breaker for failure protection
    - Task scheduling for future execution
    - Checkpoint groups for logical organization
    - Result aggregation and reporting

    Example:
        >>> from truthound.checkpoint.distributed import (
        ...     DistributedCheckpointOrchestrator,
        ...     get_backend,
        ... )
        >>>
        >>> # Create orchestrator with backend
        >>> backend = get_backend("celery", broker_url="redis://localhost:6379")
        >>> orchestrator = DistributedCheckpointOrchestrator(backend)
        >>>
        >>> with orchestrator:
        ...     # Submit tasks with rate limiting
        ...     task = orchestrator.submit(checkpoint, priority=7)
        ...
        ...     # Schedule for future execution
        ...     scheduled = orchestrator.schedule(
        ...         checkpoint,
        ...         delay_seconds=300,
        ...         recurring=True,
        ...         interval_seconds=3600,
        ...     )
        ...
        ...     # Submit a group of checkpoints
        ...     group_tasks = orchestrator.submit_group(
        ...         "daily_validations",
        ...         [cp1, cp2, cp3],
        ...     )
        ...
        ...     # Wait for results
        ...     results = orchestrator.gather(group_tasks)
    """

    def __init__(
        self,
        backend: BaseDistributedBackend,
        config: DistributedConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            backend: The distributed backend to use.
            config: Distributed configuration.
            rate_limit_config: Rate limiting configuration.
            circuit_breaker_config: Circuit breaker configuration.
        """
        super().__init__(backend, config)

        self._rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self._circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Scheduling
        self._scheduled_tasks: PriorityQueue[ScheduledTask] = PriorityQueue()
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_running = False

        # Groups
        self._groups: dict[str, list[str]] = defaultdict(list)  # group -> task_ids

        # Callbacks
        self._on_rate_limited: list[Callable[[str], None]] = []
        self._on_circuit_open: list[Callable[[], None]] = []

    # -------------------------------------------------------------------------
    # Enhanced Task Submission
    # -------------------------------------------------------------------------

    def submit(
        self,
        checkpoint: "Checkpoint",
        priority: int | TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        wait_for_rate_limit: bool = True,
        rate_limit_timeout: float = 30.0,
        **kwargs: Any,
    ) -> DistributedTaskProtocol["CheckpointResult"]:
        """Submit a checkpoint with rate limiting and circuit breaker.

        Args:
            checkpoint: Checkpoint to execute.
            priority: Task priority (0-10 or TaskPriority).
            timeout: Task timeout in seconds.
            context: Additional context for the run.
            wait_for_rate_limit: Wait for rate limit slot.
            rate_limit_timeout: Timeout for rate limit wait.
            **kwargs: Backend-specific options.

        Returns:
            Task handle for tracking execution.

        Raises:
            DistributedError: If circuit breaker is open.
            TaskSubmissionError: If rate limit timeout exceeded.
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            raise DistributedError(
                "Circuit breaker is open",
                {"state": self._circuit_breaker.state},
            )

        # Rate limiting
        if wait_for_rate_limit:
            if not self._rate_limiter.acquire(timeout=rate_limit_timeout):
                for callback in self._on_rate_limited:
                    callback(checkpoint.name)
                raise TaskSubmissionError(
                    "Rate limit timeout exceeded",
                    checkpoint_name=checkpoint.name,
                    reason="Could not acquire rate limit slot",
                )

        try:
            task = super().submit(checkpoint, priority, timeout, context, **kwargs)

            # Add completion callback for circuit breaker
            if hasattr(task, "add_callback"):
                task.add_callback(self._handle_task_completion)

            return task

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._rate_limiter.release()
            raise

    def _handle_task_completion(
        self,
        result: "CheckpointResult | None",
        exception: Exception | None,
    ) -> None:
        """Handle task completion for circuit breaker."""
        if exception:
            self._circuit_breaker.record_failure()
        else:
            self._circuit_breaker.record_success()
        self._rate_limiter.release()

    # -------------------------------------------------------------------------
    # Group Submission
    # -------------------------------------------------------------------------

    def submit_group(
        self,
        group_name: str,
        checkpoints: list["Checkpoint"],
        priority: int | TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[DistributedTaskProtocol["CheckpointResult"]]:
        """Submit a group of checkpoints.

        Groups allow logical organization of related checkpoints
        and aggregate result tracking.

        Args:
            group_name: Name for the group.
            checkpoints: List of checkpoints to execute.
            priority: Task priority.
            timeout: Task timeout in seconds.
            context: Additional context for all runs.
            **kwargs: Backend-specific options.

        Returns:
            List of task handles.
        """
        tasks = []
        group_context = {
            **(context or {}),
            "group_name": group_name,
            "group_size": len(checkpoints),
        }

        for i, checkpoint in enumerate(checkpoints):
            cp_context = {
                **group_context,
                "group_index": i,
            }
            try:
                task = self.submit(checkpoint, priority, timeout, cp_context, **kwargs)
                tasks.append(task)
                self._groups[group_name].append(task.task_id)
            except Exception as e:
                logger.error(f"Failed to submit {checkpoint.name} in group {group_name}: {e}")

        return tasks

    def get_group_tasks(
        self,
        group_name: str,
    ) -> list[DistributedTaskProtocol["CheckpointResult"]]:
        """Get all tasks in a group.

        Args:
            group_name: Name of the group.

        Returns:
            List of task handles.
        """
        tasks = []
        for task_id in self._groups.get(group_name, []):
            task = self._backend.get_task(task_id)
            if task:
                tasks.append(task)
        return tasks

    def get_group_status(self, group_name: str) -> dict[str, Any]:
        """Get status summary for a group.

        Args:
            group_name: Name of the group.

        Returns:
            Status summary dictionary.
        """
        tasks = self.get_group_tasks(group_name)

        states = defaultdict(int)
        for task in tasks:
            states[task.state.value] += 1

        return {
            "group_name": group_name,
            "total_tasks": len(tasks),
            "states": dict(states),
            "completed": sum(1 for t in tasks if t.is_ready()),
            "successful": sum(1 for t in tasks if t.is_successful()),
        }

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    def schedule(
        self,
        checkpoint: "Checkpoint",
        delay_seconds: float | None = None,
        scheduled_time: datetime | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        context: dict[str, Any] | None = None,
        recurring: bool = False,
        interval_seconds: float = 0.0,
    ) -> str:
        """Schedule a checkpoint for future execution.

        Args:
            checkpoint: Checkpoint to execute.
            delay_seconds: Delay from now (mutually exclusive with scheduled_time).
            scheduled_time: Specific time to execute.
            priority: Task priority.
            context: Execution context.
            recurring: If True, reschedule after execution.
            interval_seconds: Interval for recurring tasks.

        Returns:
            Scheduled task ID.

        Raises:
            ValueError: If neither delay nor scheduled_time provided.
        """
        if delay_seconds is not None:
            run_time = datetime.now() + timedelta(seconds=delay_seconds)
        elif scheduled_time is not None:
            run_time = scheduled_time
        else:
            raise ValueError("Either delay_seconds or scheduled_time must be provided")

        task = ScheduledTask(
            task_id=f"scheduled-{uuid4().hex[:12]}",
            checkpoint=checkpoint,
            scheduled_time=run_time,
            priority=priority,
            context=context or {},
            recurring=recurring,
            interval_seconds=interval_seconds,
        )

        self._scheduled_tasks.put(task)
        logger.info(
            f"Scheduled {checkpoint.name} for {run_time.isoformat()}"
            + (f" (recurring every {interval_seconds}s)" if recurring else "")
        )

        # Start scheduler if not running
        self._ensure_scheduler_running()

        return task.task_id

    def cancel_scheduled(self, task_id: str) -> bool:
        """Cancel a scheduled task.

        Note: This creates a new queue excluding the task.

        Args:
            task_id: ID of the scheduled task.

        Returns:
            True if found and cancelled.
        """
        new_queue: PriorityQueue[ScheduledTask] = PriorityQueue()
        found = False

        while not self._scheduled_tasks.empty():
            try:
                task = self._scheduled_tasks.get_nowait()
                if task.task_id == task_id:
                    found = True
                else:
                    new_queue.put(task)
            except Empty:
                break

        self._scheduled_tasks = new_queue
        return found

    def get_scheduled_tasks(self) -> list[ScheduledTask]:
        """Get all scheduled tasks.

        Returns:
            List of scheduled tasks (copy).
        """
        tasks = []
        temp_queue: PriorityQueue[ScheduledTask] = PriorityQueue()

        while not self._scheduled_tasks.empty():
            try:
                task = self._scheduled_tasks.get_nowait()
                tasks.append(task)
                temp_queue.put(task)
            except Empty:
                break

        self._scheduled_tasks = temp_queue
        return tasks

    def _ensure_scheduler_running(self) -> None:
        """Ensure the scheduler thread is running."""
        if self._scheduler_running:
            return

        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="distributed-checkpoint-scheduler",
        )
        self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        """Scheduler main loop."""
        while self._scheduler_running:
            try:
                # Check for due tasks
                while True:
                    try:
                        task = self._scheduled_tasks.get_nowait()
                    except Empty:
                        break

                    if datetime.now() >= task.scheduled_time:
                        # Execute task
                        self._execute_scheduled_task(task)
                    else:
                        # Put back and wait
                        self._scheduled_tasks.put(task)
                        break

                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5.0)

    def _execute_scheduled_task(self, scheduled: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            logger.info(f"Executing scheduled task: {scheduled.checkpoint.name}")
            self.submit(
                scheduled.checkpoint,
                priority=scheduled.priority,
                context=scheduled.context,
            )

            # Reschedule if recurring
            if scheduled.recurring and scheduled.interval_seconds > 0:
                next_run = datetime.now() + timedelta(seconds=scheduled.interval_seconds)
                new_task = ScheduledTask(
                    task_id=f"scheduled-{uuid4().hex[:12]}",
                    checkpoint=scheduled.checkpoint,
                    scheduled_time=next_run,
                    priority=scheduled.priority,
                    context=scheduled.context,
                    recurring=True,
                    interval_seconds=scheduled.interval_seconds,
                )
                self._scheduled_tasks.put(new_task)
                logger.info(f"Rescheduled {scheduled.checkpoint.name} for {next_run.isoformat()}")

        except Exception as e:
            logger.error(f"Failed to execute scheduled task {scheduled.task_id}: {e}")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect and stop scheduler."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None
        super().disconnect()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_rate_limited(self, callback: Callable[[str], None]) -> None:
        """Register a rate limit callback.

        Args:
            callback: Function called with checkpoint name when rate limited.
        """
        self._on_rate_limited.append(callback)

    def on_circuit_open(self, callback: Callable[[], None]) -> None:
        """Register a circuit open callback.

        Args:
            callback: Function called when circuit breaker opens.
        """
        self._on_circuit_open.append(callback)

    # -------------------------------------------------------------------------
    # Status and Monitoring
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status summary.

        Returns:
            Status dictionary with metrics and state info.
        """
        return {
            "backend": self._backend.name,
            "is_connected": self.is_connected,
            "circuit_breaker_state": self._circuit_breaker.state,
            "scheduled_tasks": self._scheduled_tasks.qsize(),
            "groups": {
                name: len(task_ids)
                for name, task_ids in self._groups.items()
            },
            "metrics": self.metrics.to_dict(),
            "cluster": self.get_cluster_state().__dict__ if self.is_connected else None,
        }

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._circuit_breaker.reset()
        logger.info("Circuit breaker reset")
