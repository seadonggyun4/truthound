"""Process-isolated timeout execution for reliable operation termination.

This module provides a robust timeout system that can reliably terminate
operations including native Rust code (like Polars) by using process isolation.

Key features:
- Process-based execution for reliable termination
- Pre-execution complexity estimation
- Circuit breaker pattern for repeated failures
- Resource monitoring (memory, CPU)
- Graceful degradation with multiple strategies

Problem Solved:
- Python threading cannot interrupt Polars Rust operations
- Process isolation ensures SIGTERM/SIGKILL always works

Design Principles:
- Strategy Pattern: Multiple execution backends (thread, process, async)
- Circuit Breaker: Prevent cascade failures
- Bulkhead: Isolate resources per operation
- Fail-Fast: Pre-check before expensive operations

Example:
    from truthound.profiler.process_timeout import (
        ProcessTimeoutExecutor,
        TimeoutConfig,
        with_process_timeout,
    )

    # Basic usage
    executor = ProcessTimeoutExecutor()
    result = executor.execute(
        expensive_polars_operation,
        timeout_seconds=30,
    )

    # With complexity estimation
    result = executor.execute_with_estimation(
        operation,
        data_size=1_000_000,
        timeout_seconds=60,
    )
"""

from __future__ import annotations

import functools
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from multiprocessing import Queue
from typing import (
    Any,
    Callable,
    Generic,
    Generator,
    Protocol,
    TypeVar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")


class ExecutionBackend(str, Enum):
    """Available execution backends."""

    THREAD = "thread"       # Fast but can't interrupt native code
    PROCESS = "process"     # Reliable but has serialization overhead
    ADAPTIVE = "adaptive"   # Auto-select based on operation type
    INLINE = "inline"       # No isolation (for debugging)


class TimeoutAction(str, Enum):
    """Actions to take when timeout occurs."""

    SKIP = "skip"           # Skip and continue
    PARTIAL = "partial"     # Return partial results
    FAIL = "fail"           # Raise exception
    RETRY = "retry"         # Retry with extended timeout
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker


class TerminationMethod(str, Enum):
    """Methods for terminating processes."""

    GRACEFUL = "graceful"   # SIGTERM, wait, then SIGKILL
    IMMEDIATE = "immediate"  # SIGKILL directly
    COOPERATIVE = "cooperative"  # Set flag and wait


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class ExecutionMetrics:
    """Metrics from an execution attempt."""

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    backend_used: ExecutionBackend = ExecutionBackend.THREAD
    was_terminated: bool = False
    termination_method: TerminationMethod | None = None
    retries: int = 0

    def complete(self) -> None:
        """Mark execution as complete."""
        self.completed_at = datetime.now()
        self.elapsed_seconds = (self.completed_at - self.started_at).total_seconds()


@dataclass
class ExecutionResult(Generic[T]):
    """Result of a timed execution.

    Attributes:
        success: Whether operation completed successfully
        value: Result value if successful
        error: Exception if failed
        timed_out: Whether operation was terminated due to timeout
        metrics: Execution metrics
        partial_result: Partial result if available
    """

    success: bool
    value: T | None = None
    error: Exception | None = None
    timed_out: bool = False
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    partial_result: Any = None

    @classmethod
    def ok(cls, value: T, metrics: ExecutionMetrics | None = None) -> "ExecutionResult[T]":
        """Create successful result."""
        m = metrics or ExecutionMetrics()
        m.complete()
        return cls(success=True, value=value, metrics=m)

    @classmethod
    def timeout(
        cls,
        metrics: ExecutionMetrics | None = None,
        partial: Any = None,
    ) -> "ExecutionResult[T]":
        """Create timeout result."""
        m = metrics or ExecutionMetrics()
        m.complete()
        m.was_terminated = True
        return cls(success=False, timed_out=True, metrics=m, partial_result=partial)

    @classmethod
    def failure(
        cls,
        error: Exception,
        metrics: ExecutionMetrics | None = None,
    ) -> "ExecutionResult[T]":
        """Create failure result."""
        m = metrics or ExecutionMetrics()
        m.complete()
        return cls(success=False, error=error, metrics=m)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "timed_out": self.timed_out,
            "elapsed_seconds": self.metrics.elapsed_seconds,
            "backend_used": self.metrics.backend_used.value,
            "was_terminated": self.metrics.was_terminated,
            "error": str(self.error) if self.error else None,
        }


# =============================================================================
# Complexity Estimator
# =============================================================================


@dataclass
class ComplexityEstimate:
    """Estimated complexity of an operation.

    Attributes:
        estimated_time_seconds: Expected execution time
        estimated_memory_mb: Expected memory usage
        confidence: Confidence in the estimate (0-1)
        recommendation: Recommended execution backend
        should_sample: Whether to sample data first
        max_safe_rows: Maximum rows to process safely
    """

    estimated_time_seconds: float
    estimated_memory_mb: float
    confidence: float = 0.5
    recommendation: ExecutionBackend = ExecutionBackend.ADAPTIVE
    should_sample: bool = False
    max_safe_rows: int = 0
    risk_level: str = "unknown"

    def exceeds_timeout(self, timeout_seconds: float) -> bool:
        """Check if estimated time exceeds timeout."""
        # Use confidence-adjusted estimate
        adjusted = self.estimated_time_seconds * (2 - self.confidence)
        return adjusted > timeout_seconds

    def exceeds_memory(self, max_memory_mb: float) -> bool:
        """Check if estimated memory exceeds limit."""
        adjusted = self.estimated_memory_mb * (2 - self.confidence)
        return adjusted > max_memory_mb


class ComplexityEstimator(Protocol):
    """Protocol for complexity estimation."""

    def estimate(
        self,
        operation_type: str,
        data_size: int,
        column_count: int = 1,
        **kwargs: Any,
    ) -> ComplexityEstimate:
        """Estimate operation complexity."""
        ...


class DefaultComplexityEstimator:
    """Default complexity estimator using heuristics.

    Uses empirical constants calibrated for common operations.
    """

    # Calibration constants (rows per second for different operations)
    OPERATION_SPEEDS: dict[str, float] = {
        "profile_column": 100_000,      # 100K rows/sec
        "pattern_match": 50_000,        # 50K rows/sec
        "distribution": 200_000,        # 200K rows/sec
        "correlation": 10_000,          # 10K rows/sec (O(n²))
        "unique_count": 150_000,        # 150K rows/sec
        "null_check": 500_000,          # 500K rows/sec
        "type_inference": 75_000,       # 75K rows/sec
        "default": 100_000,             # Default fallback
    }

    # Memory constants (bytes per row for different operations)
    MEMORY_PER_ROW: dict[str, float] = {
        "profile_column": 100,          # 100 bytes/row
        "pattern_match": 200,           # 200 bytes/row (regex buffers)
        "distribution": 50,             # 50 bytes/row
        "correlation": 300,             # 300 bytes/row (matrix)
        "unique_count": 150,            # 150 bytes/row (hash set)
        "default": 100,
    }

    def __init__(self, safety_factor: float = 1.5):
        """Initialize estimator.

        Args:
            safety_factor: Multiplier for conservative estimates
        """
        self.safety_factor = safety_factor
        self._history: list[tuple[str, int, float]] = []  # For calibration

    def estimate(
        self,
        operation_type: str,
        data_size: int,
        column_count: int = 1,
        **kwargs: Any,
    ) -> ComplexityEstimate:
        """Estimate operation complexity.

        Args:
            operation_type: Type of operation
            data_size: Number of rows
            column_count: Number of columns
            **kwargs: Additional hints

        Returns:
            Complexity estimate
        """
        # Get calibration constants
        speed = self.OPERATION_SPEEDS.get(
            operation_type,
            self.OPERATION_SPEEDS["default"],
        )
        memory_per_row = self.MEMORY_PER_ROW.get(
            operation_type,
            self.MEMORY_PER_ROW["default"],
        )

        # Calculate estimates
        time_estimate = (data_size / speed) * column_count * self.safety_factor
        memory_estimate = (data_size * memory_per_row * column_count) / (1024 * 1024)  # MB

        # Determine confidence based on data size
        if data_size < 10_000:
            confidence = 0.9  # High confidence for small data
        elif data_size < 100_000:
            confidence = 0.7
        elif data_size < 1_000_000:
            confidence = 0.5
        else:
            confidence = 0.3  # Low confidence for very large data

        # Determine recommendation
        if time_estimate > 60 or memory_estimate > 1000:
            recommendation = ExecutionBackend.PROCESS
            should_sample = True
            risk_level = "high"
        elif time_estimate > 10 or memory_estimate > 500:
            recommendation = ExecutionBackend.PROCESS
            should_sample = False
            risk_level = "medium"
        else:
            recommendation = ExecutionBackend.THREAD
            should_sample = False
            risk_level = "low"

        # Calculate safe row limit
        max_safe_rows = int(speed * 30)  # 30 seconds worth

        return ComplexityEstimate(
            estimated_time_seconds=time_estimate,
            estimated_memory_mb=memory_estimate,
            confidence=confidence,
            recommendation=recommendation,
            should_sample=should_sample,
            max_safe_rows=max_safe_rows,
            risk_level=risk_level,
        )

    def record_actual(
        self,
        operation_type: str,
        data_size: int,
        actual_seconds: float,
    ) -> None:
        """Record actual execution time for future calibration.

        Args:
            operation_type: Type of operation
            data_size: Number of rows
            actual_seconds: Actual execution time
        """
        self._history.append((operation_type, data_size, actual_seconds))

        # Keep last 100 records
        if len(self._history) > 100:
            self._history = self._history[-100:]


# Global estimator instance
default_complexity_estimator = DefaultComplexityEstimator()


# =============================================================================
# Circuit Breaker
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes before closing from half-open
        timeout_seconds: Time before trying half-open from open
        half_open_max_calls: Max calls in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, track failures
    - OPEN: Reject all requests, wait for timeout
    - HALF_OPEN: Allow limited requests to test recovery

    Example:
        breaker = CircuitBreaker()

        if breaker.can_execute():
            try:
                result = operation()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
        else:
            raise CircuitOpenError("Circuit is open")
    """

    def __init__(
        self,
        name: str = "default",
        config: CircuitBreakerConfig | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this breaker
            config: Configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting)."""
        return self.state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if execution is allowed.

        Returns:
            True if execution should proceed
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN state
            return False

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._open()
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._open()

    def reset(self) -> None:
        """Reset circuit to closed state."""
        with self._lock:
            self._close()

    def _check_state_transition(self) -> None:
        """Check and perform state transitions."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._half_open()

    def _open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._failure_count = 0
        self._success_count = 0
        logger.warning(f"Circuit breaker '{self.name}' opened")

    def _close(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' closed")

    def _half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' half-open")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time,
            }


class CircuitBreakerRegistry:
    """Registry for circuit breakers by operation type."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry
circuit_breaker_registry = CircuitBreakerRegistry()


# =============================================================================
# Execution Strategy Protocol
# =============================================================================


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies.

    Defines how an operation is executed with timeout control.
    """

    name: ExecutionBackend

    @abstractmethod
    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        **kwargs: Any,
    ) -> ExecutionResult[T]:
        """Execute function with timeout.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds
            **kwargs: Additional options

        Returns:
            Execution result
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this strategy is available."""
        pass


# =============================================================================
# Thread-Based Strategy
# =============================================================================


class ThreadExecutionStrategy(ExecutionStrategy):
    """Thread-based execution strategy.

    Fast with low overhead but cannot interrupt native code.
    Best for pure Python operations.
    """

    name = ExecutionBackend.THREAD

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers

    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        **kwargs: Any,
    ) -> ExecutionResult[T]:
        """Execute in thread with timeout."""
        metrics = ExecutionMetrics(backend_used=self.name)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future: Future[T] = executor.submit(func)

            try:
                value = future.result(timeout=timeout_seconds)
                return ExecutionResult.ok(value, metrics)

            except FuturesTimeoutError:
                metrics.was_terminated = True
                # Note: Cannot actually terminate the thread
                return ExecutionResult.timeout(metrics)

            except Exception as e:
                return ExecutionResult.failure(e, metrics)

    def is_available(self) -> bool:
        return True


# =============================================================================
# Process-Based Strategy
# =============================================================================


def _process_worker(
    func_pickle: bytes,
    result_queue: Queue,
    ready_event: mp.Event,
) -> None:
    """Worker function for process execution.

    This runs in a separate process and can be terminated.

    Args:
        func_pickle: Pickled function to execute
        result_queue: Queue to put result
        ready_event: Event to signal ready
    """
    try:
        # Signal ready
        ready_event.set()

        # Deserialize and execute
        func = pickle.loads(func_pickle)
        result = func()

        # Put success result
        result_queue.put(("success", result, None))

    except Exception as e:
        # Put error result
        tb = traceback.format_exc()
        result_queue.put(("error", None, (type(e).__name__, str(e), tb)))


class ProcessExecutionStrategy(ExecutionStrategy):
    """Process-based execution strategy.

    Uses separate process for reliable termination.
    Has serialization overhead but can terminate any code.
    """

    name = ExecutionBackend.PROCESS

    def __init__(
        self,
        graceful_timeout: float = 2.0,
        start_method: str | None = None,
    ):
        """Initialize process strategy.

        Args:
            graceful_timeout: Time to wait after SIGTERM before SIGKILL
            start_method: Process start method (spawn, fork, forkserver)
        """
        self.graceful_timeout = graceful_timeout
        self.start_method = start_method or self._get_default_start_method()

    def _get_default_start_method(self) -> str:
        """Get default start method for platform."""
        if sys.platform == "darwin":
            return "spawn"  # fork is problematic on macOS
        elif sys.platform == "win32":
            return "spawn"
        else:
            return "fork"  # Faster on Linux

    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        **kwargs: Any,
    ) -> ExecutionResult[T]:
        """Execute in separate process with timeout."""
        metrics = ExecutionMetrics(backend_used=self.name)

        # Serialize function
        try:
            func_pickle = pickle.dumps(func)
        except Exception as e:
            return ExecutionResult.failure(
                ValueError(f"Cannot serialize function: {e}"),
                metrics,
            )

        # Create communication primitives
        ctx = mp.get_context(self.start_method)
        result_queue: Queue = ctx.Queue()
        ready_event = ctx.Event()

        # Start process
        process = ctx.Process(
            target=_process_worker,
            args=(func_pickle, result_queue, ready_event),
        )
        process.start()

        try:
            # Wait for process to be ready
            if not ready_event.wait(timeout=5.0):
                self._terminate_process(process, metrics)
                return ExecutionResult.timeout(metrics)

            # Wait for result with timeout
            try:
                status, value, error_info = result_queue.get(timeout=timeout_seconds)

                if status == "success":
                    return ExecutionResult.ok(value, metrics)
                else:
                    error_type, error_msg, tb = error_info
                    error = RuntimeError(f"{error_type}: {error_msg}")
                    return ExecutionResult.failure(error, metrics)

            except queue.Empty:
                # Timeout
                self._terminate_process(process, metrics)
                return ExecutionResult.timeout(metrics)

        finally:
            # Ensure process is cleaned up
            if process.is_alive():
                self._terminate_process(process, metrics)
            process.join(timeout=1.0)

    def _terminate_process(
        self,
        process: mp.Process,
        metrics: ExecutionMetrics,
    ) -> None:
        """Terminate a process gracefully then forcefully."""
        metrics.was_terminated = True

        if not process.is_alive():
            return

        # Try graceful termination
        process.terminate()
        metrics.termination_method = TerminationMethod.GRACEFUL

        # Wait for graceful exit
        process.join(timeout=self.graceful_timeout)

        # Force kill if still alive
        if process.is_alive():
            try:
                os.kill(process.pid, signal.SIGKILL)
                metrics.termination_method = TerminationMethod.IMMEDIATE
            except (ProcessLookupError, OSError):
                pass

            process.join(timeout=1.0)

    def is_available(self) -> bool:
        """Check if multiprocessing is available."""
        try:
            ctx = mp.get_context(self.start_method)
            return True
        except Exception:
            return False


# =============================================================================
# Adaptive Strategy
# =============================================================================


class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Adaptive strategy that selects backend based on operation.

    Uses complexity estimation to choose between thread and process.
    """

    name = ExecutionBackend.ADAPTIVE

    def __init__(
        self,
        estimator: ComplexityEstimator | None = None,
        thread_threshold_seconds: float = 5.0,
    ):
        """Initialize adaptive strategy.

        Args:
            estimator: Complexity estimator
            thread_threshold_seconds: Use thread for operations under this
        """
        self.estimator = estimator or default_complexity_estimator
        self.thread_threshold = thread_threshold_seconds
        self._thread_strategy = ThreadExecutionStrategy()
        self._process_strategy = ProcessExecutionStrategy()

    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        operation_type: str = "default",
        data_size: int = 0,
        **kwargs: Any,
    ) -> ExecutionResult[T]:
        """Execute with adaptive backend selection."""
        # Estimate complexity
        estimate = self.estimator.estimate(
            operation_type=operation_type,
            data_size=data_size,
        )

        # Select strategy
        if estimate.recommendation == ExecutionBackend.PROCESS:
            strategy = self._process_strategy
        elif estimate.estimated_time_seconds > self.thread_threshold:
            strategy = self._process_strategy
        else:
            strategy = self._thread_strategy

        logger.debug(
            f"Adaptive strategy selected {strategy.name.value} for "
            f"{operation_type} ({data_size} rows, "
            f"est. {estimate.estimated_time_seconds:.2f}s)"
        )

        # Execute
        result = strategy.execute(func, timeout_seconds, **kwargs)

        # Record actual time for calibration
        if isinstance(self.estimator, DefaultComplexityEstimator):
            self.estimator.record_actual(
                operation_type,
                data_size,
                result.metrics.elapsed_seconds,
            )

        return result

    def is_available(self) -> bool:
        return self._thread_strategy.is_available()


# =============================================================================
# Inline Strategy (No Isolation)
# =============================================================================


class InlineExecutionStrategy(ExecutionStrategy):
    """Inline execution without isolation.

    Useful for debugging and trusted operations.
    No timeout protection.
    """

    name = ExecutionBackend.INLINE

    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        **kwargs: Any,
    ) -> ExecutionResult[T]:
        """Execute inline without isolation."""
        metrics = ExecutionMetrics(backend_used=self.name)

        try:
            value = func()
            return ExecutionResult.ok(value, metrics)
        except Exception as e:
            return ExecutionResult.failure(e, metrics)

    def is_available(self) -> bool:
        return True


# =============================================================================
# Strategy Registry
# =============================================================================


class ExecutionStrategyRegistry:
    """Registry for execution strategies."""

    def __init__(self) -> None:
        self._strategies: dict[ExecutionBackend, ExecutionStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default strategies."""
        self.register(ThreadExecutionStrategy())
        self.register(ProcessExecutionStrategy())
        self.register(AdaptiveExecutionStrategy())
        self.register(InlineExecutionStrategy())

    def register(self, strategy: ExecutionStrategy) -> None:
        """Register a strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, backend: ExecutionBackend) -> ExecutionStrategy:
        """Get strategy by backend type."""
        if backend not in self._strategies:
            raise KeyError(f"Unknown backend: {backend}")
        return self._strategies[backend]

    def get_available(self) -> list[ExecutionBackend]:
        """Get list of available backends."""
        return [
            backend
            for backend, strategy in self._strategies.items()
            if strategy.is_available()
        ]


# Global registry
execution_strategy_registry = ExecutionStrategyRegistry()


# =============================================================================
# Resource Monitor
# =============================================================================


@dataclass
class ResourceLimits:
    """Resource limits for execution.

    Attributes:
        max_memory_mb: Maximum memory usage
        max_cpu_percent: Maximum CPU usage
        max_open_files: Maximum open file descriptors
    """

    max_memory_mb: float = 0  # 0 = unlimited
    max_cpu_percent: float = 0  # 0 = unlimited
    max_open_files: int = 0  # 0 = unlimited


@dataclass
class ResourceUsage:
    """Current resource usage."""

    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    open_files: int = 0


class ResourceMonitor:
    """Monitors resource usage during execution.

    Uses psutil if available, otherwise provides estimates.
    """

    def __init__(self):
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False

    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        if not self._psutil_available:
            return ResourceUsage()

        try:
            import psutil
            process = psutil.Process()

            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            cpu_percent = process.cpu_percent()

            try:
                open_files = len(process.open_files())
            except Exception:
                open_files = 0

            return ResourceUsage(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                open_files=open_files,
            )
        except Exception:
            return ResourceUsage()

    def exceeds_limits(
        self,
        usage: ResourceUsage,
        limits: ResourceLimits,
    ) -> tuple[bool, str]:
        """Check if usage exceeds limits.

        Returns:
            Tuple of (exceeds, reason)
        """
        if limits.max_memory_mb > 0 and usage.memory_mb > limits.max_memory_mb:
            return True, f"Memory usage {usage.memory_mb:.1f}MB exceeds limit {limits.max_memory_mb:.1f}MB"

        if limits.max_cpu_percent > 0 and usage.cpu_percent > limits.max_cpu_percent:
            return True, f"CPU usage {usage.cpu_percent:.1f}% exceeds limit {limits.max_cpu_percent:.1f}%"

        if limits.max_open_files > 0 and usage.open_files > limits.max_open_files:
            return True, f"Open files {usage.open_files} exceeds limit {limits.max_open_files}"

        return False, ""


# Global monitor
resource_monitor = ResourceMonitor()


# =============================================================================
# Process Timeout Executor (Main Interface)
# =============================================================================


@dataclass
class ProcessTimeoutConfig:
    """Configuration for process timeout executor.

    Attributes:
        default_timeout_seconds: Default timeout
        default_backend: Default execution backend
        enable_circuit_breaker: Use circuit breaker
        enable_complexity_estimation: Pre-check complexity
        enable_resource_monitoring: Monitor resources
        resource_limits: Resource limits
        graceful_termination_seconds: Time before force kill
        max_retries: Maximum retry attempts
        retry_backoff_factor: Backoff multiplier for retries
    """

    default_timeout_seconds: float = 60.0
    default_backend: ExecutionBackend = ExecutionBackend.ADAPTIVE
    enable_circuit_breaker: bool = True
    enable_complexity_estimation: bool = True
    enable_resource_monitoring: bool = True
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    graceful_termination_seconds: float = 2.0
    max_retries: int = 0
    retry_backoff_factor: float = 2.0

    @classmethod
    def strict(cls) -> "ProcessTimeoutConfig":
        """Create strict configuration."""
        return cls(
            default_timeout_seconds=30.0,
            enable_circuit_breaker=True,
            resource_limits=ResourceLimits(max_memory_mb=1000),
            max_retries=0,
        )

    @classmethod
    def lenient(cls) -> "ProcessTimeoutConfig":
        """Create lenient configuration."""
        return cls(
            default_timeout_seconds=300.0,
            enable_circuit_breaker=False,
            max_retries=2,
        )

    @classmethod
    def fast(cls) -> "ProcessTimeoutConfig":
        """Create fast configuration (thread-based)."""
        return cls(
            default_timeout_seconds=10.0,
            default_backend=ExecutionBackend.THREAD,
            enable_complexity_estimation=False,
        )

    @classmethod
    def safe(cls) -> "ProcessTimeoutConfig":
        """Create safe configuration (process-based)."""
        return cls(
            default_timeout_seconds=60.0,
            default_backend=ExecutionBackend.PROCESS,
            enable_circuit_breaker=True,
            enable_complexity_estimation=True,
        )


class ProcessTimeoutExecutor:
    """Enterprise-grade timeout executor with process isolation.

    This is the main interface for executing operations with reliable
    timeout control, including native code like Polars.

    Features:
    - Process isolation for reliable termination
    - Pre-execution complexity estimation
    - Circuit breaker for cascade prevention
    - Resource monitoring
    - Retry with backoff

    Example:
        # Basic usage
        executor = ProcessTimeoutExecutor()
        result = executor.execute(
            lambda: expensive_operation(),
            timeout_seconds=30,
        )

        if result.success:
            print(result.value)
        elif result.timed_out:
            print("Operation timed out")
        else:
            print(f"Error: {result.error}")

        # With hints for better execution
        result = executor.execute_with_hints(
            lambda: profile_column(df, "email"),
            timeout_seconds=60,
            operation_type="profile_column",
            data_size=1_000_000,
        )
    """

    def __init__(
        self,
        config: ProcessTimeoutConfig | None = None,
        estimator: ComplexityEstimator | None = None,
    ):
        """Initialize executor.

        Args:
            config: Executor configuration
            estimator: Complexity estimator
        """
        self.config = config or ProcessTimeoutConfig()
        self.estimator = estimator or default_complexity_estimator
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def execute(
        self,
        func: Callable[[], T],
        timeout_seconds: float | None = None,
        backend: ExecutionBackend | None = None,
        operation_name: str = "operation",
    ) -> ExecutionResult[T]:
        """Execute function with timeout.

        Args:
            func: Function to execute
            timeout_seconds: Timeout (uses config default if None)
            backend: Execution backend (uses config default if None)
            operation_name: Name for logging and circuit breaker

        Returns:
            Execution result
        """
        timeout = timeout_seconds or self.config.default_timeout_seconds
        backend = backend or self.config.default_backend

        # Check circuit breaker
        if self.config.enable_circuit_breaker:
            breaker = self._get_circuit_breaker(operation_name)
            if not breaker.can_execute():
                return ExecutionResult.failure(
                    RuntimeError(f"Circuit breaker open for '{operation_name}'"),
                    ExecutionMetrics(),
                )

        # Check resources
        if self.config.enable_resource_monitoring:
            usage = resource_monitor.get_current_usage()
            exceeds, reason = resource_monitor.exceeds_limits(
                usage, self.config.resource_limits
            )
            if exceeds:
                return ExecutionResult.failure(
                    RuntimeError(f"Resource limit exceeded: {reason}"),
                    ExecutionMetrics(),
                )

        # Execute with retry
        result = self._execute_with_retry(func, timeout, backend, operation_name)

        # Update circuit breaker
        if self.config.enable_circuit_breaker:
            if result.success:
                breaker.record_success()
            else:
                breaker.record_failure()

        return result

    def execute_with_hints(
        self,
        func: Callable[[], T],
        timeout_seconds: float | None = None,
        operation_type: str = "default",
        data_size: int = 0,
        column_count: int = 1,
    ) -> ExecutionResult[T]:
        """Execute with complexity hints for better decisions.

        Args:
            func: Function to execute
            timeout_seconds: Timeout
            operation_type: Type of operation
            data_size: Number of rows
            column_count: Number of columns

        Returns:
            Execution result
        """
        timeout = timeout_seconds or self.config.default_timeout_seconds

        # Estimate complexity
        if self.config.enable_complexity_estimation:
            estimate = self.estimator.estimate(
                operation_type=operation_type,
                data_size=data_size,
                column_count=column_count,
            )

            # Check if operation will likely timeout
            if estimate.exceeds_timeout(timeout):
                logger.warning(
                    f"Operation '{operation_type}' estimated to take "
                    f"{estimate.estimated_time_seconds:.1f}s, exceeds timeout {timeout}s. "
                    f"Consider sampling to {estimate.max_safe_rows} rows."
                )

            # Use recommended backend
            backend = estimate.recommendation
        else:
            backend = self.config.default_backend

        return self.execute(
            func,
            timeout_seconds=timeout,
            backend=backend,
            operation_name=operation_type,
        )

    def execute_safe(
        self,
        func: Callable[[], T],
        timeout_seconds: float | None = None,
        default: T | None = None,
    ) -> T | None:
        """Execute and return default on failure.

        Args:
            func: Function to execute
            timeout_seconds: Timeout
            default: Default value on failure

        Returns:
            Result value or default
        """
        result = self.execute(func, timeout_seconds)
        if result.success:
            return result.value
        return default

    def _execute_with_retry(
        self,
        func: Callable[[], T],
        timeout: float,
        backend: ExecutionBackend,
        operation_name: str,
    ) -> ExecutionResult[T]:
        """Execute with retry logic."""
        strategy = execution_strategy_registry.get(backend)
        retries = 0
        current_timeout = timeout

        while True:
            result = strategy.execute(func, current_timeout)
            result.metrics.retries = retries

            if result.success or not result.timed_out:
                return result

            # Retry logic
            if retries >= self.config.max_retries:
                return result

            retries += 1
            current_timeout *= self.config.retry_backoff_factor

            logger.info(
                f"Retrying '{operation_name}' (attempt {retries + 1}/{self.config.max_retries + 1}), "
                f"timeout={current_timeout:.1f}s"
            )

    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name)
            return self._circuit_breakers[name]

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "config": {
                "default_timeout": self.config.default_timeout_seconds,
                "default_backend": self.config.default_backend.value,
            },
            "circuit_breakers": {
                name: breaker.get_stats()
                for name, breaker in self._circuit_breakers.items()
            },
        }

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._circuit_breakers.values():
                breaker.reset()


# =============================================================================
# Convenience Functions
# =============================================================================


def with_process_timeout(
    func: Callable[[], T],
    timeout_seconds: float,
    default: T | None = None,
) -> T | None:
    """Execute function with process-based timeout.

    Simple convenience function for one-off executions.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds
        default: Value to return on timeout/failure

    Returns:
        Function result or default

    Example:
        result = with_process_timeout(
            lambda: expensive_polars_operation(),
            timeout_seconds=30,
            default=None,
        )
    """
    executor = ProcessTimeoutExecutor()
    return executor.execute_safe(func, timeout_seconds, default)


def estimate_execution_time(
    operation_type: str,
    data_size: int,
    column_count: int = 1,
) -> ComplexityEstimate:
    """Estimate execution time for an operation.

    Args:
        operation_type: Type of operation
        data_size: Number of rows
        column_count: Number of columns

    Returns:
        Complexity estimate

    Example:
        estimate = estimate_execution_time("pattern_match", 1_000_000)
        print(f"Estimated time: {estimate.estimated_time_seconds:.1f}s")
        print(f"Recommended backend: {estimate.recommendation.value}")
    """
    return default_complexity_estimator.estimate(
        operation_type=operation_type,
        data_size=data_size,
        column_count=column_count,
    )


def create_timeout_executor(
    timeout_seconds: float = 60.0,
    backend: str = "adaptive",
    enable_circuit_breaker: bool = True,
) -> ProcessTimeoutExecutor:
    """Create a configured timeout executor.

    Args:
        timeout_seconds: Default timeout
        backend: Execution backend (thread, process, adaptive)
        enable_circuit_breaker: Enable circuit breaker

    Returns:
        Configured executor

    Example:
        executor = create_timeout_executor(
            timeout_seconds=30,
            backend="process",
        )
    """
    config = ProcessTimeoutConfig(
        default_timeout_seconds=timeout_seconds,
        default_backend=ExecutionBackend(backend),
        enable_circuit_breaker=enable_circuit_breaker,
    )
    return ProcessTimeoutExecutor(config)


# =============================================================================
# Context Manager
# =============================================================================


@contextmanager
def process_timeout_context(
    timeout_seconds: float,
    operation_name: str = "operation",
) -> Generator[ProcessTimeoutExecutor, None, None]:
    """Context manager for process timeout execution.

    Args:
        timeout_seconds: Timeout in seconds
        operation_name: Name for logging

    Yields:
        Executor instance

    Example:
        with process_timeout_context(30.0, "profiling") as executor:
            result = executor.execute(lambda: profile(data))
    """
    config = ProcessTimeoutConfig(default_timeout_seconds=timeout_seconds)
    executor = ProcessTimeoutExecutor(config)

    try:
        yield executor
    finally:
        # Cleanup if needed
        pass


# =============================================================================
# Decorator
# =============================================================================


def timeout_protected(
    timeout_seconds: float = 60.0,
    backend: ExecutionBackend = ExecutionBackend.ADAPTIVE,
    default: Any = None,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator to add timeout protection to a function.

    Args:
        timeout_seconds: Timeout in seconds
        backend: Execution backend
        default: Default value on timeout

    Returns:
        Decorated function

    Example:
        @timeout_protected(timeout_seconds=30)
        def expensive_operation(data):
            return process(data)

        result = expensive_operation(my_data)  # Will timeout after 30s
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        executor = ProcessTimeoutExecutor(
            ProcessTimeoutConfig(
                default_timeout_seconds=timeout_seconds,
                default_backend=backend,
            )
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            result = executor.execute(
                lambda: func(*args, **kwargs),
                operation_name=func.__name__,
            )
            if result.success:
                return result.value
            return default

        return wrapper

    return decorator
