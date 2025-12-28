"""Runtime resource limiting for validators.

This module provides mechanisms to limit and monitor resource usage:
- Memory limits with automatic termination
- CPU time limits
- Real-time resource monitoring
- Graceful degradation under resource pressure

Example:
    from truthound.validators.sdk.enterprise.resources import (
        ResourceLimits,
        ResourceLimiter,
    )

    limits = ResourceLimits(
        max_memory_mb=512,
        max_cpu_seconds=30,
        max_wall_time_seconds=60,
    )

    limiter = ResourceLimiter(limits)
    with limiter.enforce():
        result = validator.validate(data)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceExceededError(Exception):
    """Raised when resource limits are exceeded."""

    def __init__(
        self,
        resource_type: str,
        limit: float,
        actual: float,
        message: str = "",
    ):
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual
        super().__init__(
            message or f"{resource_type} limit exceeded: {actual} > {limit}"
        )


class ResourceType(Enum):
    """Types of resources that can be limited."""

    MEMORY = auto()
    CPU_TIME = auto()
    WALL_TIME = auto()
    FILE_DESCRIPTORS = auto()
    PROCESSES = auto()


@dataclass(frozen=True)
class ResourceLimits:
    """Resource limits configuration.

    Attributes:
        max_memory_mb: Maximum memory in megabytes
        max_cpu_seconds: Maximum CPU time in seconds
        max_wall_time_seconds: Maximum wall clock time
        max_file_descriptors: Maximum open file descriptors
        max_processes: Maximum child processes
        soft_memory_threshold: Warning threshold for memory (0.0-1.0)
        check_interval_seconds: How often to check resources
        graceful_degradation: Whether to degrade gracefully on limits
    """

    max_memory_mb: int = 512
    max_cpu_seconds: float = 60.0
    max_wall_time_seconds: float = 120.0
    max_file_descriptors: int = 256
    max_processes: int = 4
    soft_memory_threshold: float = 0.8
    check_interval_seconds: float = 0.5
    graceful_degradation: bool = True

    @classmethod
    def strict(cls) -> "ResourceLimits":
        """Create strict limits for untrusted code."""
        return cls(
            max_memory_mb=256,
            max_cpu_seconds=30.0,
            max_wall_time_seconds=60.0,
            max_file_descriptors=64,
            max_processes=1,
            graceful_degradation=False,
        )

    @classmethod
    def standard(cls) -> "ResourceLimits":
        """Create standard limits."""
        return cls()

    @classmethod
    def generous(cls) -> "ResourceLimits":
        """Create generous limits for trusted code."""
        return cls(
            max_memory_mb=4096,
            max_cpu_seconds=300.0,
            max_wall_time_seconds=600.0,
            max_file_descriptors=1024,
            max_processes=16,
            graceful_degradation=True,
        )


@dataclass
class ResourceUsage:
    """Current resource usage snapshot.

    Attributes:
        memory_mb: Current memory usage in MB
        memory_percent: Percentage of limit used
        cpu_seconds: CPU time consumed
        cpu_percent: Percentage of CPU limit used
        wall_seconds: Wall clock time elapsed
        wall_percent: Percentage of wall time limit used
        file_descriptors: Open file descriptors
        timestamp: When snapshot was taken
    """

    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_seconds: float = 0.0
    cpu_percent: float = 0.0
    wall_seconds: float = 0.0
    wall_percent: float = 0.0
    file_descriptors: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_within_limits(self) -> bool:
        """Check if usage is within limits."""
        return all([
            self.memory_percent < 100,
            self.cpu_percent < 100,
            self.wall_percent < 100,
        ])

    def is_near_limits(self, threshold: float = 0.8) -> bool:
        """Check if usage is approaching limits."""
        return any([
            self.memory_percent > threshold * 100,
            self.cpu_percent > threshold * 100,
            self.wall_percent > threshold * 100,
        ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "cpu_seconds": self.cpu_seconds,
            "cpu_percent": self.cpu_percent,
            "wall_seconds": self.wall_seconds,
            "wall_percent": self.wall_percent,
            "file_descriptors": self.file_descriptors,
            "timestamp": self.timestamp.isoformat(),
        }


class ResourceMonitor:
    """Monitors resource usage in real-time.

    Runs in a background thread and tracks:
    - Memory usage via psutil or /proc
    - CPU time via os.times()
    - Wall clock time
    - File descriptors
    """

    def __init__(
        self,
        limits: ResourceLimits,
        on_threshold: Callable[[ResourceUsage], None] | None = None,
        on_exceeded: Callable[[ResourceType, float, float], None] | None = None,
    ):
        """Initialize resource monitor.

        Args:
            limits: Resource limits to enforce
            on_threshold: Callback when soft threshold is reached
            on_exceeded: Callback when hard limit is exceeded
        """
        self.limits = limits
        self._on_threshold = on_threshold
        self._on_exceeded = on_exceeded
        self._start_time: float | None = None
        self._start_cpu: float | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._history: list[ResourceUsage] = []
        self._lock = threading.Lock()
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self._psutil_available:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        else:
            # Fallback to /proc on Linux
            try:
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # VmRSS is in KB
                            return int(line.split()[1]) / 1024
            except (FileNotFoundError, ValueError):
                pass
            return 0.0

    def _get_cpu_seconds(self) -> float:
        """Get CPU time consumed since start."""
        if self._start_cpu is None:
            return 0.0
        times = os.times()
        return (times.user + times.system) - self._start_cpu

    def _get_file_descriptors(self) -> int:
        """Get number of open file descriptors."""
        if self._psutil_available:
            import psutil
            process = psutil.Process()
            return process.num_fds() if hasattr(process, "num_fds") else 0
        else:
            try:
                return len(os.listdir("/proc/self/fd"))
            except (FileNotFoundError, PermissionError):
                return 0

    def get_usage(self) -> ResourceUsage:
        """Get current resource usage snapshot."""
        if self._start_time is None:
            return ResourceUsage()

        memory_mb = self._get_memory_mb()
        cpu_seconds = self._get_cpu_seconds()
        wall_seconds = time.perf_counter() - self._start_time
        fd_count = self._get_file_descriptors()

        return ResourceUsage(
            memory_mb=memory_mb,
            memory_percent=(memory_mb / self.limits.max_memory_mb * 100)
                if self.limits.max_memory_mb > 0 else 0,
            cpu_seconds=cpu_seconds,
            cpu_percent=(cpu_seconds / self.limits.max_cpu_seconds * 100)
                if self.limits.max_cpu_seconds > 0 else 0,
            wall_seconds=wall_seconds,
            wall_percent=(wall_seconds / self.limits.max_wall_time_seconds * 100)
                if self.limits.max_wall_time_seconds > 0 else 0,
            file_descriptors=fd_count,
        )

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            usage = self.get_usage()

            with self._lock:
                self._history.append(usage)
                # Keep last 1000 samples
                if len(self._history) > 1000:
                    self._history = self._history[-1000:]

            # Check thresholds
            if usage.is_near_limits(self.limits.soft_memory_threshold):
                if self._on_threshold:
                    self._on_threshold(usage)

            # Check hard limits
            if usage.memory_percent >= 100:
                if self._on_exceeded:
                    self._on_exceeded(
                        ResourceType.MEMORY,
                        self.limits.max_memory_mb,
                        usage.memory_mb,
                    )
            if usage.cpu_percent >= 100:
                if self._on_exceeded:
                    self._on_exceeded(
                        ResourceType.CPU_TIME,
                        self.limits.max_cpu_seconds,
                        usage.cpu_seconds,
                    )
            if usage.wall_percent >= 100:
                if self._on_exceeded:
                    self._on_exceeded(
                        ResourceType.WALL_TIME,
                        self.limits.max_wall_time_seconds,
                        usage.wall_seconds,
                    )

            time.sleep(self.limits.check_interval_seconds)

    def start(self) -> None:
        """Start monitoring."""
        times = os.times()
        self._start_time = time.perf_counter()
        self._start_cpu = times.user + times.system
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_history(self) -> list[ResourceUsage]:
        """Get usage history."""
        with self._lock:
            return self._history.copy()

    def get_peak_usage(self) -> ResourceUsage:
        """Get peak resource usage."""
        with self._lock:
            if not self._history:
                return ResourceUsage()

            return ResourceUsage(
                memory_mb=max(u.memory_mb for u in self._history),
                memory_percent=max(u.memory_percent for u in self._history),
                cpu_seconds=max(u.cpu_seconds for u in self._history),
                cpu_percent=max(u.cpu_percent for u in self._history),
                wall_seconds=max(u.wall_seconds for u in self._history),
                wall_percent=max(u.wall_percent for u in self._history),
                file_descriptors=max(u.file_descriptors for u in self._history),
            )


class ResourceLimiter(ABC):
    """Abstract base class for resource limiters."""

    @abstractmethod
    @contextmanager
    def enforce(self) -> Generator[ResourceMonitor, None, None]:
        """Context manager to enforce limits."""
        pass


class MemoryLimiter(ResourceLimiter):
    """Enforces memory limits using OS mechanisms.

    On Linux, uses resource.setrlimit for hard limits.
    On all platforms, monitors and raises exceptions.
    """

    def __init__(self, max_memory_mb: int, graceful: bool = True):
        """Initialize memory limiter.

        Args:
            max_memory_mb: Maximum memory in MB
            graceful: Whether to allow graceful degradation
        """
        self.max_memory_mb = max_memory_mb
        self.graceful = graceful
        self._original_limit: tuple[int, int] | None = None

    def _set_limit(self) -> None:
        """Set OS-level memory limit."""
        try:
            import resource

            max_bytes = self.max_memory_mb * 1024 * 1024
            self._original_limit = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except (ImportError, ValueError, OSError) as e:
            logger.warning(f"Could not set memory limit: {e}")

    def _restore_limit(self) -> None:
        """Restore original memory limit."""
        if self._original_limit is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, self._original_limit)
            except (ImportError, ValueError, OSError):
                pass

    @contextmanager
    def enforce(self) -> Generator[ResourceMonitor, None, None]:
        """Enforce memory limits."""
        limits = ResourceLimits(max_memory_mb=self.max_memory_mb)
        monitor = ResourceMonitor(limits)

        self._set_limit()
        monitor.start()

        try:
            yield monitor
        finally:
            monitor.stop()
            self._restore_limit()


class CPULimiter(ResourceLimiter):
    """Enforces CPU time limits.

    Uses SIGXCPU on Unix systems for hard limits.
    On all platforms, monitors and raises exceptions.
    """

    def __init__(self, max_cpu_seconds: float, graceful: bool = True):
        """Initialize CPU limiter.

        Args:
            max_cpu_seconds: Maximum CPU time in seconds
            graceful: Whether to allow graceful degradation
        """
        self.max_cpu_seconds = max_cpu_seconds
        self.graceful = graceful
        self._original_limit: tuple[int, int] | None = None
        self._original_handler: Any = None

    def _set_limit(self) -> None:
        """Set OS-level CPU limit."""
        try:
            import resource

            max_seconds = int(self.max_cpu_seconds)
            self._original_limit = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (max_seconds, max_seconds))

            # Set up signal handler
            def handler(signum: int, frame: Any) -> None:
                raise ResourceExceededError(
                    "CPU_TIME",
                    self.max_cpu_seconds,
                    self.max_cpu_seconds,
                    "CPU time limit exceeded",
                )

            self._original_handler = signal.signal(signal.SIGXCPU, handler)
        except (ImportError, ValueError, OSError, AttributeError) as e:
            logger.warning(f"Could not set CPU limit: {e}")

    def _restore_limit(self) -> None:
        """Restore original CPU limit."""
        if self._original_limit is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, self._original_limit)
            except (ImportError, ValueError, OSError):
                pass

        if self._original_handler is not None:
            try:
                signal.signal(signal.SIGXCPU, self._original_handler)
            except (ValueError, OSError):
                pass

    @contextmanager
    def enforce(self) -> Generator[ResourceMonitor, None, None]:
        """Enforce CPU limits."""
        limits = ResourceLimits(max_cpu_seconds=self.max_cpu_seconds)
        monitor = ResourceMonitor(limits)

        self._set_limit()
        monitor.start()

        try:
            yield monitor
        finally:
            monitor.stop()
            self._restore_limit()


class CombinedResourceLimiter(ResourceLimiter):
    """Combines multiple resource limiters."""

    def __init__(self, limits: ResourceLimits):
        """Initialize combined limiter.

        Args:
            limits: Resource limits configuration
        """
        self.limits = limits
        self._exceeded = False
        self._exceeded_type: ResourceType | None = None

    def _on_exceeded(
        self,
        resource_type: ResourceType,
        limit: float,
        actual: float,
    ) -> None:
        """Handle resource exceeded event."""
        self._exceeded = True
        self._exceeded_type = resource_type
        logger.warning(
            f"Resource limit exceeded: {resource_type.name} "
            f"(limit={limit}, actual={actual})"
        )

    @contextmanager
    def enforce(self) -> Generator[ResourceMonitor, None, None]:
        """Enforce all configured limits."""
        monitor = ResourceMonitor(
            self.limits,
            on_exceeded=self._on_exceeded,
        )

        # Set OS-level limits where possible
        try:
            import resource

            # Memory limit
            if self.limits.max_memory_mb > 0:
                max_bytes = self.limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

            # CPU limit
            if self.limits.max_cpu_seconds > 0:
                max_seconds = int(self.limits.max_cpu_seconds)
                resource.setrlimit(resource.RLIMIT_CPU, (max_seconds, max_seconds))

            # File descriptor limit
            if self.limits.max_file_descriptors > 0:
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (self.limits.max_file_descriptors, self.limits.max_file_descriptors),
                )
        except (ImportError, ValueError, OSError) as e:
            logger.debug(f"Could not set OS resource limits: {e}")

        monitor.start()
        self._exceeded = False

        try:
            yield monitor

            # Check if limits were exceeded during execution
            if self._exceeded and not self.limits.graceful_degradation:
                usage = monitor.get_peak_usage()
                raise ResourceExceededError(
                    self._exceeded_type.name if self._exceeded_type else "UNKNOWN",
                    getattr(self.limits, f"max_{self._exceeded_type.name.lower()}", 0)
                        if self._exceeded_type else 0,
                    getattr(usage, f"{self._exceeded_type.name.lower()}_mb", 0)
                        if self._exceeded_type else 0,
                )
        finally:
            monitor.stop()


def with_resource_limits(
    limits: ResourceLimits | None = None,
    max_memory_mb: int | None = None,
    max_cpu_seconds: float | None = None,
    max_wall_time_seconds: float | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to apply resource limits to a function.

    Args:
        limits: Full resource limits configuration
        max_memory_mb: Override max memory
        max_cpu_seconds: Override max CPU time
        max_wall_time_seconds: Override max wall time

    Returns:
        Decorated function with resource limits

    Example:
        @with_resource_limits(max_memory_mb=256, max_cpu_seconds=30)
        def expensive_validation(data):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            effective_limits = limits or ResourceLimits()

            if max_memory_mb is not None:
                effective_limits = ResourceLimits(
                    max_memory_mb=max_memory_mb,
                    max_cpu_seconds=effective_limits.max_cpu_seconds,
                    max_wall_time_seconds=effective_limits.max_wall_time_seconds,
                )
            if max_cpu_seconds is not None:
                effective_limits = ResourceLimits(
                    max_memory_mb=effective_limits.max_memory_mb,
                    max_cpu_seconds=max_cpu_seconds,
                    max_wall_time_seconds=effective_limits.max_wall_time_seconds,
                )
            if max_wall_time_seconds is not None:
                effective_limits = ResourceLimits(
                    max_memory_mb=effective_limits.max_memory_mb,
                    max_cpu_seconds=effective_limits.max_cpu_seconds,
                    max_wall_time_seconds=max_wall_time_seconds,
                )

            limiter = CombinedResourceLimiter(effective_limits)
            with limiter.enforce():
                return func(*args, **kwargs)

        return wrapper
    return decorator
