"""Timeout management for profiling operations.

This module provides timeout controls at various levels:
- Table-level timeouts
- Column-level timeouts
- Analyzer-level timeouts

Key features:
- Graceful timeout handling with partial results
- Configurable retry policies
- Timeout inheritance (table -> column -> analyzer)
"""

from __future__ import annotations

import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Generator, Generic, TypeVar

from truthound.profiler.errors import (
    ErrorCollector,
    ErrorSeverity,
    ProfilerError,
    TimeoutError as ProfilerTimeoutError,
)


T = TypeVar("T")


# =============================================================================
# Timeout Configuration
# =============================================================================


class TimeoutAction(str, Enum):
    """Actions to take when timeout occurs."""

    SKIP = "skip"           # Skip the operation, continue with others
    PARTIAL = "partial"     # Return partial results if available
    FAIL = "fail"          # Fail the entire profiling
    RETRY = "retry"        # Retry with extended timeout


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior.

    Attributes:
        table_timeout: Maximum time for entire table profiling
        column_timeout: Maximum time per column
        analyzer_timeout: Maximum time per analyzer
        default_action: Action when timeout occurs
        retry_count: Number of retries before giving up
        retry_multiplier: Timeout multiplier for each retry
        grace_period: Extra time before hard kill
    """

    table_timeout: timedelta | None = None
    column_timeout: timedelta | None = timedelta(seconds=60)
    analyzer_timeout: timedelta | None = timedelta(seconds=10)
    default_action: TimeoutAction = TimeoutAction.SKIP
    retry_count: int = 0
    retry_multiplier: float = 1.5
    grace_period: timedelta = timedelta(seconds=1)

    @classmethod
    def strict(cls) -> "TimeoutConfig":
        """Create strict timeout configuration."""
        return cls(
            table_timeout=timedelta(minutes=5),
            column_timeout=timedelta(seconds=30),
            analyzer_timeout=timedelta(seconds=5),
            default_action=TimeoutAction.FAIL,
        )

    @classmethod
    def lenient(cls) -> "TimeoutConfig":
        """Create lenient timeout configuration."""
        return cls(
            table_timeout=timedelta(minutes=30),
            column_timeout=timedelta(minutes=5),
            analyzer_timeout=timedelta(seconds=60),
            default_action=TimeoutAction.SKIP,
            retry_count=2,
        )

    @classmethod
    def no_timeout(cls) -> "TimeoutConfig":
        """Create configuration with no timeouts."""
        return cls(
            table_timeout=None,
            column_timeout=None,
            analyzer_timeout=None,
        )

    def get_column_seconds(self) -> float | None:
        """Get column timeout in seconds."""
        return self.column_timeout.total_seconds() if self.column_timeout else None

    def get_analyzer_seconds(self) -> float | None:
        """Get analyzer timeout in seconds."""
        return self.analyzer_timeout.total_seconds() if self.analyzer_timeout else None

    def get_table_seconds(self) -> float | None:
        """Get table timeout in seconds."""
        return self.table_timeout.total_seconds() if self.table_timeout else None


# =============================================================================
# Timeout Results
# =============================================================================


@dataclass
class TimeoutResult(Generic[T]):
    """Result of a timed operation.

    Attributes:
        success: Whether operation completed successfully
        value: Result value (if success)
        timed_out: Whether operation timed out
        elapsed_seconds: Time taken
        retries: Number of retries attempted
        error: Exception if failed (not timeout)
    """

    success: bool
    value: T | None = None
    timed_out: bool = False
    elapsed_seconds: float = 0.0
    retries: int = 0
    error: Exception | None = None

    @classmethod
    def ok(cls, value: T, elapsed: float = 0.0) -> "TimeoutResult[T]":
        """Create successful result."""
        return cls(success=True, value=value, elapsed_seconds=elapsed)

    @classmethod
    def timeout(cls, elapsed: float, retries: int = 0) -> "TimeoutResult[T]":
        """Create timeout result."""
        return cls(
            success=False,
            timed_out=True,
            elapsed_seconds=elapsed,
            retries=retries,
        )

    @classmethod
    def failure(cls, error: Exception, elapsed: float = 0.0) -> "TimeoutResult[T]":
        """Create failure result."""
        return cls(success=False, error=error, elapsed_seconds=elapsed)


# =============================================================================
# Timeout Executor
# =============================================================================


class TimeoutExecutor:
    """Executes operations with timeout control.

    This executor provides reliable timeout functionality using
    thread-based execution with proper cleanup.

    Example:
        executor = TimeoutExecutor()

        result = executor.run(
            lambda: expensive_operation(),
            timeout=10.0,
        )

        if result.timed_out:
            print("Operation timed out")
        elif result.success:
            print(f"Result: {result.value}")
    """

    def __init__(
        self,
        config: TimeoutConfig | None = None,
        error_collector: ErrorCollector | None = None,
    ):
        """Initialize executor.

        Args:
            config: Timeout configuration
            error_collector: Error collector for timeout errors
        """
        self.config = config or TimeoutConfig()
        self.error_collector = error_collector

    def run(
        self,
        func: Callable[[], T],
        *,
        timeout: float | None = None,
        action: TimeoutAction | None = None,
        context: str = "",
    ) -> TimeoutResult[T]:
        """Run a function with timeout.

        Args:
            func: Function to execute
            timeout: Timeout in seconds (uses config default if None)
            action: Action on timeout (uses config default if None)
            context: Description for error messages

        Returns:
            Execution result
        """
        if timeout is None:
            timeout = self.config.get_analyzer_seconds()

        if timeout is None:
            # No timeout - run directly
            start = time.perf_counter()
            try:
                value = func()
                elapsed = time.perf_counter() - start
                return TimeoutResult.ok(value, elapsed)
            except Exception as e:
                elapsed = time.perf_counter() - start
                return TimeoutResult.failure(e, elapsed)

        action = action or self.config.default_action

        # Run with timeout using thread pool
        return self._run_with_timeout(func, timeout, action, context)

    def _run_with_timeout(
        self,
        func: Callable[[], T],
        timeout: float,
        action: TimeoutAction,
        context: str,
    ) -> TimeoutResult[T]:
        """Run function with timeout using thread pool."""
        start = time.perf_counter()
        retries = 0
        current_timeout = timeout

        while True:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future: Future[T] = executor.submit(func)

                try:
                    value = future.result(timeout=current_timeout)
                    elapsed = time.perf_counter() - start
                    return TimeoutResult.ok(value, elapsed)

                except FuturesTimeoutError:
                    elapsed = time.perf_counter() - start

                    # Record error
                    if self.error_collector:
                        self.error_collector.add(
                            ProfilerTimeoutError(
                                f"Operation timed out after {elapsed:.2f}s: {context}",
                                severity=ErrorSeverity.WARNING,
                            ),
                            recovered=True,
                        )

                    # Check for retry
                    if action == TimeoutAction.RETRY and retries < self.config.retry_count:
                        retries += 1
                        current_timeout *= self.config.retry_multiplier
                        continue

                    return TimeoutResult.timeout(elapsed, retries)

                except Exception as e:
                    elapsed = time.perf_counter() - start
                    return TimeoutResult.failure(e, elapsed)

    def run_column(
        self,
        func: Callable[[], T],
        column: str,
    ) -> TimeoutResult[T]:
        """Run a column-level operation with timeout.

        Args:
            func: Function to execute
            column: Column name for context

        Returns:
            Execution result
        """
        return self.run(
            func,
            timeout=self.config.get_column_seconds(),
            context=f"column: {column}",
        )

    def run_analyzer(
        self,
        func: Callable[[], T],
        analyzer: str,
        column: str,
    ) -> TimeoutResult[T]:
        """Run an analyzer-level operation with timeout.

        Args:
            func: Function to execute
            analyzer: Analyzer name
            column: Column name

        Returns:
            Execution result
        """
        return self.run(
            func,
            timeout=self.config.get_analyzer_seconds(),
            context=f"analyzer: {analyzer} on column: {column}",
        )


# =============================================================================
# Timeout Context Manager
# =============================================================================


@contextmanager
def timeout_context(
    seconds: float | None,
    *,
    on_timeout: Callable[[], None] | None = None,
    message: str = "Operation timed out",
) -> Generator[None, None, None]:
    """Context manager for timeout control.

    This uses signal-based timeout on Unix systems and is best
    for main thread operations.

    Args:
        seconds: Timeout in seconds (None = no timeout)
        on_timeout: Callback when timeout occurs
        message: Error message for timeout

    Yields:
        None

    Raises:
        ProfilerTimeoutError: If timeout occurs

    Example:
        with timeout_context(10.0, message="Query timed out"):
            result = run_expensive_query()
    """
    if seconds is None:
        yield
        return

    def handler(signum, frame):
        if on_timeout:
            on_timeout()
        raise ProfilerTimeoutError(message)

    # Only use signal on Unix and main thread
    use_signal = (
        hasattr(signal, 'SIGALRM') and
        threading.current_thread() is threading.main_thread()
    )

    if use_signal:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Fallback: no timeout protection in threads
        yield


# =============================================================================
# Timeout-Aware Profiler Mixin
# =============================================================================


class TimeoutAwareMixin:
    """Mixin class for adding timeout awareness to profilers.

    Add this mixin to profiler classes to enable timeout control.

    Example:
        class MyProfiler(TimeoutAwareMixin, Profiler):
            def profile_column(self, column, data):
                with self.column_timeout(column):
                    return super().profile_column(column, data)
    """

    timeout_config: TimeoutConfig
    _timeout_executor: TimeoutExecutor | None = None

    def get_timeout_executor(self) -> TimeoutExecutor:
        """Get or create timeout executor."""
        if self._timeout_executor is None:
            config = getattr(self, 'timeout_config', TimeoutConfig())
            collector = getattr(self, 'error_collector', None)
            self._timeout_executor = TimeoutExecutor(config, collector)
        return self._timeout_executor

    @contextmanager
    def table_timeout(self) -> Generator[None, None, None]:
        """Context manager for table-level timeout."""
        config = getattr(self, 'timeout_config', TimeoutConfig())
        seconds = config.get_table_seconds()
        with timeout_context(seconds, message="Table profiling timed out"):
            yield

    @contextmanager
    def column_timeout(self, column: str) -> Generator[None, None, None]:
        """Context manager for column-level timeout."""
        config = getattr(self, 'timeout_config', TimeoutConfig())
        seconds = config.get_column_seconds()
        with timeout_context(seconds, message=f"Column '{column}' profiling timed out"):
            yield

    def run_with_timeout(
        self,
        func: Callable[[], T],
        timeout: float | None = None,
        context: str = "",
    ) -> TimeoutResult[T]:
        """Run function with timeout protection."""
        executor = self.get_timeout_executor()
        return executor.run(func, timeout=timeout, context=context)


# =============================================================================
# Deadline Tracker
# =============================================================================


class DeadlineTracker:
    """Tracks deadline for complex operations.

    Useful when you need to track remaining time across
    multiple sub-operations.

    Example:
        tracker = DeadlineTracker(total_seconds=60)

        for item in items:
            if tracker.is_expired:
                break

            remaining = tracker.remaining_seconds
            result = process_with_timeout(item, timeout=remaining)
    """

    def __init__(self, total_seconds: float):
        """Initialize tracker.

        Args:
            total_seconds: Total time budget in seconds
        """
        self.total_seconds = total_seconds
        self.start_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.perf_counter() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.total_seconds - self.elapsed_seconds)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return self.elapsed_seconds >= self.total_seconds

    @property
    def progress(self) -> float:
        """Get progress as ratio (0.0 to 1.0)."""
        return min(1.0, self.elapsed_seconds / self.total_seconds)

    def check(self, message: str = "Deadline exceeded") -> None:
        """Check deadline and raise if expired.

        Args:
            message: Error message

        Raises:
            ProfilerTimeoutError: If deadline exceeded
        """
        if self.is_expired:
            raise ProfilerTimeoutError(message)


# =============================================================================
# Convenience Functions
# =============================================================================


def with_timeout(
    func: Callable[[], T],
    seconds: float,
    *,
    default: T | None = None,
) -> T | None:
    """Execute function with timeout, returning default on timeout.

    Args:
        func: Function to execute
        seconds: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Function result or default

    Example:
        result = with_timeout(
            lambda: expensive_operation(),
            seconds=10.0,
            default=None,
        )
    """
    executor = TimeoutExecutor()
    result = executor.run(func, timeout=seconds)

    if result.success:
        return result.value
    return default


def create_timeout_config(
    table_seconds: float | None = None,
    column_seconds: float | None = 60,
    analyzer_seconds: float | None = 10,
    action: TimeoutAction = TimeoutAction.SKIP,
) -> TimeoutConfig:
    """Create a timeout configuration.

    Args:
        table_seconds: Table timeout in seconds
        column_seconds: Column timeout in seconds
        analyzer_seconds: Analyzer timeout in seconds
        action: Default action on timeout

    Returns:
        Timeout configuration
    """
    return TimeoutConfig(
        table_timeout=timedelta(seconds=table_seconds) if table_seconds else None,
        column_timeout=timedelta(seconds=column_seconds) if column_seconds else None,
        analyzer_timeout=timedelta(seconds=analyzer_seconds) if analyzer_seconds else None,
        default_action=action,
    )
