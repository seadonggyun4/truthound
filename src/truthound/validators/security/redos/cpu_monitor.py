"""Real-time CPU Monitoring for Regex Execution.

This module provides real-time CPU monitoring during regex execution
to detect and abort operations that consume excessive resources.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CPU Monitor System                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│ Resource│   │ Monitor │    │ Threshold│   │  Abort   │    │ Report  │
│ Sampler │   │ Thread  │    │ Checker  │   │ Handler  │    │Generator│
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

Monitoring capabilities:
1. CPU usage percentage
2. Execution time
3. Memory usage (optional)
4. Thread count

Usage:
    from truthound.validators.security.redos.cpu_monitor import (
        CPUMonitor,
        execute_with_monitoring,
    )

    # Execute with monitoring
    result = execute_with_monitoring(
        pattern=r"(a+)+b",
        input_string="a" * 20,
        cpu_limit=50.0,  # Max 50% CPU
        time_limit=1.0,  # Max 1 second
    )

    if result.was_aborted:
        print(f"Aborted due to: {result.abort_reason}")
    else:
        print(f"Match result: {result.match_result}")
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

# Try to import psutil for accurate CPU monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


T = TypeVar("T")


class AbortReason(Enum):
    """Reasons for aborting regex execution."""

    NONE = auto()
    CPU_LIMIT_EXCEEDED = auto()
    TIME_LIMIT_EXCEEDED = auto()
    MEMORY_LIMIT_EXCEEDED = auto()
    USER_REQUESTED = auto()


@dataclass
class ResourceLimits:
    """Resource limits for monitored execution.

    Attributes:
        cpu_percent_limit: Max CPU usage percentage (0-100)
        time_limit_seconds: Max execution time
        memory_limit_mb: Max memory usage in MB
        sample_interval_ms: How often to sample (milliseconds)
        warmup_samples: Number of samples before checking limits
    """

    cpu_percent_limit: float = 80.0
    time_limit_seconds: float = 5.0
    memory_limit_mb: float = 500.0
    sample_interval_ms: int = 10
    warmup_samples: int = 3

    @classmethod
    def strict(cls) -> "ResourceLimits":
        """Create strict limits for untrusted patterns."""
        return cls(
            cpu_percent_limit=50.0,
            time_limit_seconds=1.0,
            memory_limit_mb=100.0,
            sample_interval_ms=5,
            warmup_samples=2,
        )

    @classmethod
    def lenient(cls) -> "ResourceLimits":
        """Create lenient limits for trusted patterns."""
        return cls(
            cpu_percent_limit=95.0,
            time_limit_seconds=30.0,
            memory_limit_mb=1000.0,
            sample_interval_ms=50,
            warmup_samples=5,
        )


@dataclass
class ResourceSample:
    """A single resource usage sample."""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    thread_count: int = 1


@dataclass
class CPUMonitorResult(Generic[T]):
    """Result of CPU-monitored execution.

    Attributes:
        result: The actual result (if completed)
        exception: Any exception that occurred
        was_aborted: Whether execution was aborted
        abort_reason: Reason for abort
        samples: Resource usage samples
        total_time_seconds: Total execution time
        peak_cpu_percent: Peak CPU usage observed
        peak_memory_mb: Peak memory usage observed
        average_cpu_percent: Average CPU usage
    """

    result: T | None = None
    exception: Exception | None = None
    was_aborted: bool = False
    abort_reason: AbortReason = AbortReason.NONE
    samples: list[ResourceSample] = field(default_factory=list)
    total_time_seconds: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0

    @property
    def success(self) -> bool:
        """Check if execution completed successfully."""
        return not self.was_aborted and self.exception is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "was_aborted": self.was_aborted,
            "abort_reason": self.abort_reason.name,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "average_cpu_percent": round(self.average_cpu_percent, 2),
            "sample_count": len(self.samples),
            "exception": str(self.exception) if self.exception else None,
        }


class ResourceSampler:
    """Samples system resource usage."""

    def __init__(self):
        """Initialize the sampler."""
        self._process = None
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            # Initialize CPU measurement
            self._process.cpu_percent()

    def sample(self) -> ResourceSample:
        """Take a resource usage sample."""
        timestamp = time.time()

        if HAS_PSUTIL and self._process:
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            thread_count = self._process.num_threads()
        else:
            # Fallback: estimate based on time
            cpu_percent = 0.0
            memory_mb = 0.0
            thread_count = threading.active_count()

        return ResourceSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            thread_count=thread_count,
        )


class CPUMonitor:
    """Monitor CPU usage during execution and abort if limits exceeded.

    This monitor runs in a separate thread and periodically checks
    resource usage. If limits are exceeded, it sets a flag that
    can be checked by the main execution.

    Example:
        monitor = CPUMonitor(limits=ResourceLimits.strict())

        # Start monitoring
        monitor.start()

        try:
            # Perform work
            result = some_regex_operation()
        finally:
            # Stop monitoring
            monitor.stop()

        # Check results
        print(f"Peak CPU: {monitor.get_result().peak_cpu_percent}%")
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        on_threshold_exceeded: Callable[[AbortReason, ResourceSample], None] | None = None,
    ):
        """Initialize the monitor.

        Args:
            limits: Resource limits to enforce
            on_threshold_exceeded: Callback when threshold is exceeded
        """
        self.limits = limits or ResourceLimits()
        self._on_threshold_exceeded = on_threshold_exceeded

        self._sampler = ResourceSampler()
        self._samples: list[ResourceSample] = []
        self._running = False
        self._abort_requested = False
        self._abort_reason = AbortReason.NONE
        self._monitor_thread: threading.Thread | None = None
        self._start_time = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring."""
        with self._lock:
            if self._running:
                return

            self._samples.clear()
            self._abort_requested = False
            self._abort_reason = AbortReason.NONE
            self._start_time = time.time()
            self._running = True

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        with self._lock:
            self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def should_abort(self) -> bool:
        """Check if execution should be aborted."""
        return self._abort_requested

    def request_abort(self, reason: AbortReason = AbortReason.USER_REQUESTED) -> None:
        """Request execution abort."""
        self._abort_requested = True
        self._abort_reason = reason

    def get_result(self) -> CPUMonitorResult[None]:
        """Get monitoring result."""
        total_time = time.time() - self._start_time if self._start_time else 0.0

        # Calculate statistics
        peak_cpu = 0.0
        peak_memory = 0.0
        cpu_sum = 0.0

        for sample in self._samples:
            peak_cpu = max(peak_cpu, sample.cpu_percent)
            peak_memory = max(peak_memory, sample.memory_mb)
            cpu_sum += sample.cpu_percent

        avg_cpu = cpu_sum / len(self._samples) if self._samples else 0.0

        return CPUMonitorResult(
            was_aborted=self._abort_requested,
            abort_reason=self._abort_reason,
            samples=list(self._samples),
            total_time_seconds=total_time,
            peak_cpu_percent=peak_cpu,
            peak_memory_mb=peak_memory,
            average_cpu_percent=avg_cpu,
        )

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        sample_count = 0
        interval = self.limits.sample_interval_ms / 1000.0

        while self._running and not self._abort_requested:
            # Take sample
            sample = self._sampler.sample()
            self._samples.append(sample)
            sample_count += 1

            # Check limits (after warmup)
            if sample_count > self.limits.warmup_samples:
                abort_reason = self._check_limits(sample)
                if abort_reason != AbortReason.NONE:
                    self._abort_requested = True
                    self._abort_reason = abort_reason

                    if self._on_threshold_exceeded:
                        self._on_threshold_exceeded(abort_reason, sample)

                    break

            # Sleep until next sample
            time.sleep(interval)

    def _check_limits(self, sample: ResourceSample) -> AbortReason:
        """Check if any limits are exceeded."""
        # Check CPU limit
        if sample.cpu_percent > self.limits.cpu_percent_limit:
            return AbortReason.CPU_LIMIT_EXCEEDED

        # Check time limit
        elapsed = sample.timestamp - self._start_time
        if elapsed > self.limits.time_limit_seconds:
            return AbortReason.TIME_LIMIT_EXCEEDED

        # Check memory limit
        if sample.memory_mb > self.limits.memory_limit_mb:
            return AbortReason.MEMORY_LIMIT_EXCEEDED

        return AbortReason.NONE


class MonitoredRegexExecutor:
    """Execute regex operations with CPU monitoring.

    This executor wraps regex operations and monitors resource usage,
    aborting if limits are exceeded.

    Example:
        executor = MonitoredRegexExecutor(
            limits=ResourceLimits(cpu_percent_limit=50.0)
        )

        result = executor.match(r"(a+)+b", "a" * 20)
        if result.was_aborted:
            print(f"Aborted: {result.abort_reason.name}")
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
    ):
        """Initialize the executor.

        Args:
            limits: Resource limits to enforce
        """
        self.limits = limits or ResourceLimits()

    def match(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> CPUMonitorResult[re.Match | None]:
        """Execute regex match with monitoring.

        Args:
            pattern: Regex pattern
            string: String to match
            flags: Regex flags

        Returns:
            CPUMonitorResult with match result or abort info
        """
        return self._execute(
            lambda compiled: compiled.match(string),
            pattern,
            flags,
        )

    def search(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> CPUMonitorResult[re.Match | None]:
        """Execute regex search with monitoring.

        Args:
            pattern: Regex pattern
            string: String to search
            flags: Regex flags

        Returns:
            CPUMonitorResult with search result
        """
        return self._execute(
            lambda compiled: compiled.search(string),
            pattern,
            flags,
        )

    def findall(
        self,
        pattern: str | re.Pattern,
        string: str,
        flags: int = 0,
    ) -> CPUMonitorResult[list[Any]]:
        """Execute regex findall with monitoring.

        Args:
            pattern: Regex pattern
            string: String to search
            flags: Regex flags

        Returns:
            CPUMonitorResult with findall result
        """
        return self._execute(
            lambda compiled: compiled.findall(string),
            pattern,
            flags,
        )

    def _execute(
        self,
        operation: Callable[[re.Pattern], T],
        pattern: str | re.Pattern,
        flags: int = 0,
    ) -> CPUMonitorResult[T]:
        """Execute an operation with monitoring."""
        # Compile pattern
        if isinstance(pattern, str):
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                return CPUMonitorResult(
                    exception=e,
                    was_aborted=False,
                )
        else:
            compiled = pattern

        # Set up monitoring
        monitor = CPUMonitor(limits=self.limits)
        result: list[T | None] = [None]
        exception: list[Exception | None] = [None]
        completed = threading.Event()

        def run_operation() -> None:
            try:
                result[0] = operation(compiled)
            except Exception as e:
                exception[0] = e
            finally:
                completed.set()

        # Start monitoring
        monitor.start()

        # Run operation in thread
        op_thread = threading.Thread(target=run_operation, daemon=True)
        op_thread.start()

        # Wait for completion or abort
        while not completed.is_set():
            if monitor.should_abort():
                # Operation should be aborted
                break
            completed.wait(timeout=0.01)

        # Stop monitoring
        monitor.stop()

        # Get monitor result
        monitor_result = monitor.get_result()

        # Combine with operation result
        return CPUMonitorResult(
            result=result[0],
            exception=exception[0],
            was_aborted=monitor_result.was_aborted,
            abort_reason=monitor_result.abort_reason,
            samples=monitor_result.samples,
            total_time_seconds=monitor_result.total_time_seconds,
            peak_cpu_percent=monitor_result.peak_cpu_percent,
            peak_memory_mb=monitor_result.peak_memory_mb,
            average_cpu_percent=monitor_result.average_cpu_percent,
        )


# ============================================================================
# Convenience functions
# ============================================================================


def execute_with_monitoring(
    pattern: str,
    input_string: str,
    operation: str = "match",
    cpu_limit: float = 80.0,
    time_limit: float = 5.0,
    memory_limit: float = 500.0,
) -> CPUMonitorResult[Any]:
    """Execute a regex operation with CPU monitoring.

    Args:
        pattern: Regex pattern
        input_string: String to match against
        operation: Operation type ("match", "search", "findall")
        cpu_limit: Max CPU usage percentage
        time_limit: Max execution time in seconds
        memory_limit: Max memory in MB

    Returns:
        CPUMonitorResult with operation result

    Example:
        result = execute_with_monitoring(
            r"(a+)+b",
            "a" * 20,
            cpu_limit=50.0,
            time_limit=1.0,
        )

        if result.was_aborted:
            print(f"Aborted due to: {result.abort_reason.name}")
        else:
            print(f"Match: {result.result}")
    """
    limits = ResourceLimits(
        cpu_percent_limit=cpu_limit,
        time_limit_seconds=time_limit,
        memory_limit_mb=memory_limit,
    )

    executor = MonitoredRegexExecutor(limits=limits)

    if operation == "match":
        return executor.match(pattern, input_string)
    elif operation == "search":
        return executor.search(pattern, input_string)
    elif operation == "findall":
        return executor.findall(pattern, input_string)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def monitored_match(
    pattern: str,
    string: str,
    limits: ResourceLimits | None = None,
) -> CPUMonitorResult[re.Match | None]:
    """Execute regex match with default monitoring.

    Args:
        pattern: Regex pattern
        string: String to match
        limits: Optional resource limits

    Returns:
        CPUMonitorResult with match result
    """
    executor = MonitoredRegexExecutor(limits=limits)
    return executor.match(pattern, string)
