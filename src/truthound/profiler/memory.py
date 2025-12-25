"""Memory usage monitoring and OOM prevention.

This module provides comprehensive memory monitoring capabilities:
- Real-time memory usage tracking
- OOM (Out of Memory) risk detection and prevention
- Memory-aware batch processing
- Memory profiling for optimization
- Automatic memory cleanup triggers

Key features:
- psutil-based memory tracking
- Configurable thresholds and alerts
- Context managers for scoped monitoring
- Integration with profiling operations
- Memory leak detection

Example:
    from truthound.profiler.memory import (
        MemoryMonitor,
        memory_guard,
        MemoryTracker,
    )

    # Simple usage with context manager
    with memory_guard(max_memory_mb=1024):
        process_large_dataset(data)

    # Detailed monitoring
    monitor = MemoryMonitor(threshold_percent=80)
    monitor.start()

    for batch in data_batches:
        if monitor.is_critical():
            break
        process_batch(batch)

    report = monitor.stop()
    print(f"Peak memory: {report.peak_mb:.1f} MB")
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Protocol, TypeVar

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn(
        "psutil not installed. Memory monitoring will use fallback methods. "
        "Install with: pip install psutil",
        UserWarning,
    )


# =============================================================================
# Types and Enums
# =============================================================================


class MemoryUnit(str, Enum):
    """Memory size units."""

    BYTES = "bytes"
    KB = "kb"
    MB = "mb"
    GB = "gb"

    @classmethod
    def convert(
        cls,
        value: float,
        from_unit: "MemoryUnit",
        to_unit: "MemoryUnit",
    ) -> float:
        """Convert between memory units."""
        # Convert to bytes first
        multipliers = {
            cls.BYTES: 1,
            cls.KB: 1024,
            cls.MB: 1024 * 1024,
            cls.GB: 1024 * 1024 * 1024,
        }
        bytes_value = value * multipliers[from_unit]
        return bytes_value / multipliers[to_unit]


class MemoryStatus(str, Enum):
    """Memory usage status levels."""

    OK = "ok"  # Normal usage
    WARNING = "warning"  # Approaching threshold
    CRITICAL = "critical"  # Near limit, action needed
    OOM_RISK = "oom_risk"  # Immediate OOM risk


class MemoryAction(str, Enum):
    """Actions to take when memory is critical."""

    NONE = "none"
    WARN = "warn"
    GC_COLLECT = "gc_collect"
    RAISE_ERROR = "raise_error"
    CALLBACK = "callback"


# =============================================================================
# Exceptions
# =============================================================================


class MemoryLimitExceeded(Exception):
    """Raised when memory limit is exceeded."""

    def __init__(
        self,
        current_mb: float,
        limit_mb: float,
        message: str = "",
    ):
        self.current_mb = current_mb
        self.limit_mb = limit_mb
        super().__init__(
            message or f"Memory limit exceeded: {current_mb:.1f} MB > {limit_mb:.1f} MB"
        )


class OOMRiskDetected(Exception):
    """Raised when OOM risk is detected."""

    def __init__(
        self,
        available_mb: float,
        required_mb: float | None = None,
    ):
        self.available_mb = available_mb
        self.required_mb = required_mb
        msg = f"OOM risk: only {available_mb:.1f} MB available"
        if required_mb:
            msg += f", but {required_mb:.1f} MB required"
        super().__init__(msg)


# =============================================================================
# Memory Information
# =============================================================================


@dataclass(frozen=True)
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""

    timestamp: datetime
    process_rss_bytes: int  # Resident Set Size
    process_vms_bytes: int  # Virtual Memory Size
    process_percent: float  # Process memory as % of total
    system_total_bytes: int
    system_available_bytes: int
    system_used_bytes: int
    system_percent: float

    @property
    def process_rss_mb(self) -> float:
        """Process RSS in MB."""
        return self.process_rss_bytes / (1024 * 1024)

    @property
    def process_vms_mb(self) -> float:
        """Process VMS in MB."""
        return self.process_vms_bytes / (1024 * 1024)

    @property
    def system_available_mb(self) -> float:
        """System available memory in MB."""
        return self.system_available_bytes / (1024 * 1024)

    @property
    def system_total_mb(self) -> float:
        """System total memory in MB."""
        return self.system_total_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "process": {
                "rss_bytes": self.process_rss_bytes,
                "rss_mb": self.process_rss_mb,
                "vms_bytes": self.process_vms_bytes,
                "vms_mb": self.process_vms_mb,
                "percent": self.process_percent,
            },
            "system": {
                "total_bytes": self.system_total_bytes,
                "total_mb": self.system_total_mb,
                "available_bytes": self.system_available_bytes,
                "available_mb": self.system_available_mb,
                "used_bytes": self.system_used_bytes,
                "percent": self.system_percent,
            },
        }


@dataclass
class MemoryReport:
    """Complete memory usage report."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Process memory stats
    initial_rss_mb: float
    final_rss_mb: float
    peak_rss_mb: float
    min_rss_mb: float
    avg_rss_mb: float

    # System memory stats
    initial_system_percent: float
    final_system_percent: float
    peak_system_percent: float

    # Status tracking
    status_history: list[tuple[datetime, MemoryStatus]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gc_collections: int = 0

    # Snapshots (if detailed tracking enabled)
    snapshots: list[MemorySnapshot] = field(default_factory=list)

    @property
    def memory_growth_mb(self) -> float:
        """Memory growth during monitoring period."""
        return self.final_rss_mb - self.initial_rss_mb

    @property
    def memory_growth_percent(self) -> float:
        """Memory growth as percentage."""
        if self.initial_rss_mb == 0:
            return 0.0
        return (self.memory_growth_mb / self.initial_rss_mb) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "process": {
                "initial_rss_mb": self.initial_rss_mb,
                "final_rss_mb": self.final_rss_mb,
                "peak_rss_mb": self.peak_rss_mb,
                "min_rss_mb": self.min_rss_mb,
                "avg_rss_mb": self.avg_rss_mb,
                "growth_mb": self.memory_growth_mb,
                "growth_percent": self.memory_growth_percent,
            },
            "system": {
                "initial_percent": self.initial_system_percent,
                "final_percent": self.final_system_percent,
                "peak_percent": self.peak_system_percent,
            },
            "gc_collections": self.gc_collections,
            "warnings": self.warnings,
            "n_snapshots": len(self.snapshots),
        }


# =============================================================================
# Memory Reader (Platform Abstraction)
# =============================================================================


class MemoryReader(Protocol):
    """Protocol for reading memory information."""

    def get_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        ...


class PsutilMemoryReader:
    """Memory reader using psutil."""

    def __init__(self, pid: int | None = None):
        """Initialize reader.

        Args:
            pid: Process ID to monitor (None for current process)
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for PsutilMemoryReader")
        self._process = psutil.Process(pid)

    def get_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        proc_mem = self._process.memory_info()
        proc_percent = self._process.memory_percent()
        sys_mem = psutil.virtual_memory()

        return MemorySnapshot(
            timestamp=datetime.now(),
            process_rss_bytes=proc_mem.rss,
            process_vms_bytes=proc_mem.vms,
            process_percent=proc_percent,
            system_total_bytes=sys_mem.total,
            system_available_bytes=sys_mem.available,
            system_used_bytes=sys_mem.used,
            system_percent=sys_mem.percent,
        )


class FallbackMemoryReader:
    """Fallback memory reader when psutil is not available.

    Uses resource module on Unix or basic estimation on other platforms.
    """

    def __init__(self) -> None:
        self._has_resource = False
        try:
            import resource
            self._resource = resource
            self._has_resource = True
        except ImportError:
            pass

    def get_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot (limited without psutil)."""
        timestamp = datetime.now()

        if self._has_resource:
            # Unix systems
            usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
            rss_bytes = usage.ru_maxrss
            # On macOS, ru_maxrss is in bytes; on Linux, it's in KB
            if sys.platform == "darwin":
                pass  # Already in bytes
            else:
                rss_bytes *= 1024
        else:
            # Estimate from sys.getsizeof of globals
            rss_bytes = sum(sys.getsizeof(obj) for obj in gc.get_objects()[:1000])

        # Estimate total system memory (fallback)
        total_bytes = 8 * 1024 * 1024 * 1024  # Assume 8GB

        return MemorySnapshot(
            timestamp=timestamp,
            process_rss_bytes=rss_bytes,
            process_vms_bytes=rss_bytes,  # No VMS info
            process_percent=rss_bytes / total_bytes * 100,
            system_total_bytes=total_bytes,
            system_available_bytes=total_bytes - rss_bytes,
            system_used_bytes=rss_bytes,
            system_percent=rss_bytes / total_bytes * 100,
        )


def get_memory_reader(pid: int | None = None) -> MemoryReader:
    """Get the best available memory reader.

    Args:
        pid: Process ID to monitor (None for current process)

    Returns:
        Memory reader instance
    """
    if PSUTIL_AVAILABLE:
        return PsutilMemoryReader(pid)
    return FallbackMemoryReader()


# =============================================================================
# Memory Monitor Configuration
# =============================================================================


@dataclass
class MemoryConfig:
    """Configuration for memory monitoring.

    Attributes:
        warning_threshold_percent: System memory % to trigger warning
        critical_threshold_percent: System memory % to trigger critical
        oom_threshold_percent: System memory % to consider OOM risk
        max_process_memory_mb: Maximum process memory allowed (None = unlimited)
        check_interval_seconds: How often to check memory
        enable_gc_on_warning: Run GC when warning threshold hit
        enable_gc_on_critical: Run GC when critical threshold hit
        raise_on_oom_risk: Raise exception on OOM risk
        callback_on_warning: Callback when warning threshold hit
        callback_on_critical: Callback when critical threshold hit
        keep_snapshots: Whether to keep all snapshots
        max_snapshots: Maximum snapshots to keep (0 = unlimited)
    """

    warning_threshold_percent: float = 70.0
    critical_threshold_percent: float = 85.0
    oom_threshold_percent: float = 95.0
    max_process_memory_mb: float | None = None

    check_interval_seconds: float = 1.0
    enable_gc_on_warning: bool = False
    enable_gc_on_critical: bool = True
    raise_on_oom_risk: bool = True

    callback_on_warning: Callable[[MemorySnapshot], None] | None = None
    callback_on_critical: Callable[[MemorySnapshot], None] | None = None

    keep_snapshots: bool = False
    max_snapshots: int = 1000

    def get_status(self, snapshot: MemorySnapshot) -> MemoryStatus:
        """Determine memory status from snapshot."""
        # Check process limit
        if self.max_process_memory_mb is not None:
            if snapshot.process_rss_mb > self.max_process_memory_mb:
                return MemoryStatus.CRITICAL

        # Check system memory
        percent = snapshot.system_percent

        if percent >= self.oom_threshold_percent:
            return MemoryStatus.OOM_RISK
        elif percent >= self.critical_threshold_percent:
            return MemoryStatus.CRITICAL
        elif percent >= self.warning_threshold_percent:
            return MemoryStatus.WARNING
        else:
            return MemoryStatus.OK


# =============================================================================
# Memory Monitor
# =============================================================================


class MemoryMonitor:
    """Real-time memory usage monitor.

    Monitors memory usage and provides alerts when thresholds are exceeded.

    Example:
        monitor = MemoryMonitor(
            warning_threshold_percent=70,
            critical_threshold_percent=85,
        )

        monitor.start()

        # Do work...
        for batch in batches:
            if monitor.is_critical():
                print("Memory critical, stopping")
                break
            process(batch)

        report = monitor.stop()
        print(f"Peak memory: {report.peak_rss_mb:.1f} MB")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        warning_threshold_percent: float = 70.0,
        critical_threshold_percent: float = 85.0,
        max_process_memory_mb: float | None = None,
        check_interval_seconds: float = 1.0,
    ):
        """Initialize monitor.

        Args:
            config: Full configuration (overrides other params)
            warning_threshold_percent: Warning threshold
            critical_threshold_percent: Critical threshold
            max_process_memory_mb: Max process memory
            check_interval_seconds: Check interval
        """
        if config is not None:
            self._config = config
        else:
            self._config = MemoryConfig(
                warning_threshold_percent=warning_threshold_percent,
                critical_threshold_percent=critical_threshold_percent,
                max_process_memory_mb=max_process_memory_mb,
                check_interval_seconds=check_interval_seconds,
            )

        self._reader = get_memory_reader()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # State
        self._snapshots: deque[MemorySnapshot] = deque(
            maxlen=self._config.max_snapshots if self._config.max_snapshots > 0 else None
        )
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._initial_snapshot: MemorySnapshot | None = None
        self._peak_rss_bytes: int = 0
        self._min_rss_bytes: int = float("inf")  # type: ignore
        self._sum_rss_bytes: int = 0
        self._sample_count: int = 0
        self._gc_count: int = 0
        self._status_history: list[tuple[datetime, MemoryStatus]] = []
        self._current_status: MemoryStatus = MemoryStatus.OK
        self._warnings: list[str] = []

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def current_status(self) -> MemoryStatus:
        """Get current memory status."""
        return self._current_status

    def start(self) -> None:
        """Start monitoring in background thread."""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.now()
        self._initial_snapshot = self._reader.get_snapshot()
        self._peak_rss_bytes = self._initial_snapshot.process_rss_bytes
        self._min_rss_bytes = self._initial_snapshot.process_rss_bytes

        if self._config.keep_snapshots:
            self._snapshots.append(self._initial_snapshot)

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MemoryReport:
        """Stop monitoring and return report."""
        self._running = False
        self._end_time = datetime.now()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        return self._generate_report()

    def check(self) -> MemorySnapshot:
        """Take a manual memory check.

        Returns:
            Current memory snapshot
        """
        snapshot = self._reader.get_snapshot()
        self._process_snapshot(snapshot)
        return snapshot

    def is_ok(self) -> bool:
        """Check if memory status is OK."""
        return self._current_status == MemoryStatus.OK

    def is_warning(self) -> bool:
        """Check if memory is at warning level."""
        return self._current_status == MemoryStatus.WARNING

    def is_critical(self) -> bool:
        """Check if memory is at critical level."""
        return self._current_status in (
            MemoryStatus.CRITICAL,
            MemoryStatus.OOM_RISK,
        )

    def get_available_mb(self) -> float:
        """Get available system memory in MB."""
        snapshot = self._reader.get_snapshot()
        return snapshot.system_available_mb

    def get_process_memory_mb(self) -> float:
        """Get current process memory in MB."""
        snapshot = self._reader.get_snapshot()
        return snapshot.process_rss_mb

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                snapshot = self._reader.get_snapshot()
                self._process_snapshot(snapshot)
            except Exception as e:
                self._warnings.append(f"Monitor error: {e}")

            time.sleep(self._config.check_interval_seconds)

    def _process_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Process a memory snapshot."""
        with self._lock:
            # Update stats
            self._peak_rss_bytes = max(self._peak_rss_bytes, snapshot.process_rss_bytes)
            self._min_rss_bytes = min(self._min_rss_bytes, snapshot.process_rss_bytes)
            self._sum_rss_bytes += snapshot.process_rss_bytes
            self._sample_count += 1

            if self._config.keep_snapshots:
                self._snapshots.append(snapshot)

            # Check status
            new_status = self._config.get_status(snapshot)

            if new_status != self._current_status:
                self._status_history.append((snapshot.timestamp, new_status))
                self._current_status = new_status

            # Take actions based on status
            if new_status == MemoryStatus.WARNING:
                if self._config.callback_on_warning:
                    try:
                        self._config.callback_on_warning(snapshot)
                    except Exception:
                        pass

                if self._config.enable_gc_on_warning:
                    gc.collect()
                    self._gc_count += 1

            elif new_status == MemoryStatus.CRITICAL:
                if self._config.callback_on_critical:
                    try:
                        self._config.callback_on_critical(snapshot)
                    except Exception:
                        pass

                if self._config.enable_gc_on_critical:
                    gc.collect()
                    self._gc_count += 1

            elif new_status == MemoryStatus.OOM_RISK:
                if self._config.raise_on_oom_risk:
                    raise OOMRiskDetected(snapshot.system_available_mb)

    def _generate_report(self) -> MemoryReport:
        """Generate memory report."""
        with self._lock:
            final_snapshot = self._reader.get_snapshot()

            return MemoryReport(
                start_time=self._start_time or datetime.now(),
                end_time=self._end_time or datetime.now(),
                duration_seconds=(
                    (self._end_time - self._start_time).total_seconds()
                    if self._start_time and self._end_time
                    else 0.0
                ),
                initial_rss_mb=(
                    self._initial_snapshot.process_rss_mb
                    if self._initial_snapshot
                    else 0.0
                ),
                final_rss_mb=final_snapshot.process_rss_mb,
                peak_rss_mb=self._peak_rss_bytes / (1024 * 1024),
                min_rss_mb=self._min_rss_bytes / (1024 * 1024),
                avg_rss_mb=(
                    self._sum_rss_bytes / self._sample_count / (1024 * 1024)
                    if self._sample_count > 0
                    else 0.0
                ),
                initial_system_percent=(
                    self._initial_snapshot.system_percent
                    if self._initial_snapshot
                    else 0.0
                ),
                final_system_percent=final_snapshot.system_percent,
                peak_system_percent=max(
                    s.system_percent for s in self._snapshots
                ) if self._snapshots else final_snapshot.system_percent,
                status_history=list(self._status_history),
                warnings=list(self._warnings),
                gc_collections=self._gc_count,
                snapshots=list(self._snapshots) if self._config.keep_snapshots else [],
            )


# =============================================================================
# Memory Guard Context Manager
# =============================================================================


@contextmanager
def memory_guard(
    max_memory_mb: float | None = None,
    warning_threshold_percent: float = 70.0,
    critical_threshold_percent: float = 85.0,
    raise_on_critical: bool = False,
    callback: Callable[[MemorySnapshot], None] | None = None,
) -> Iterator[MemoryMonitor]:
    """Context manager for memory-guarded execution.

    Monitors memory usage during the context and optionally
    raises exceptions if limits are exceeded.

    Args:
        max_memory_mb: Maximum process memory allowed
        warning_threshold_percent: Warning threshold
        critical_threshold_percent: Critical threshold
        raise_on_critical: Whether to raise on critical status
        callback: Callback on warning/critical

    Yields:
        MemoryMonitor instance

    Example:
        with memory_guard(max_memory_mb=1024) as monitor:
            process_data(data)
            if monitor.is_critical():
                cleanup()
    """
    config = MemoryConfig(
        warning_threshold_percent=warning_threshold_percent,
        critical_threshold_percent=critical_threshold_percent,
        max_process_memory_mb=max_memory_mb,
        callback_on_warning=callback,
        callback_on_critical=callback,
    )

    monitor = MemoryMonitor(config=config)
    monitor.start()

    try:
        yield monitor
    finally:
        report = monitor.stop()

        if raise_on_critical and any(
            status in (MemoryStatus.CRITICAL, MemoryStatus.OOM_RISK)
            for _, status in report.status_history
        ):
            raise MemoryLimitExceeded(
                report.peak_rss_mb,
                max_memory_mb or float("inf"),
                f"Memory exceeded critical threshold (peak: {report.peak_rss_mb:.1f} MB)",
            )


# =============================================================================
# Memory Tracker (Lightweight)
# =============================================================================


class MemoryTracker:
    """Lightweight memory tracker for specific operations.

    Unlike MemoryMonitor, this doesn't run a background thread.
    Instead, it takes snapshots on demand.

    Example:
        tracker = MemoryTracker()

        tracker.checkpoint("start")
        do_operation()
        tracker.checkpoint("after_operation")

        print(tracker.get_delta("start", "after_operation"))
    """

    def __init__(self):
        """Initialize tracker."""
        self._reader = get_memory_reader()
        self._checkpoints: dict[str, MemorySnapshot] = {}

    def checkpoint(self, name: str) -> MemorySnapshot:
        """Take a memory checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Memory snapshot
        """
        snapshot = self._reader.get_snapshot()
        self._checkpoints[name] = snapshot
        return snapshot

    def get_checkpoint(self, name: str) -> MemorySnapshot | None:
        """Get a checkpoint by name."""
        return self._checkpoints.get(name)

    def get_delta(
        self,
        from_name: str,
        to_name: str,
    ) -> dict[str, float]:
        """Get memory delta between checkpoints.

        Args:
            from_name: Starting checkpoint
            to_name: Ending checkpoint

        Returns:
            Dictionary with memory deltas
        """
        from_snap = self._checkpoints.get(from_name)
        to_snap = self._checkpoints.get(to_name)

        if not from_snap or not to_snap:
            return {}

        return {
            "rss_delta_mb": to_snap.process_rss_mb - from_snap.process_rss_mb,
            "vms_delta_mb": to_snap.process_vms_mb - from_snap.process_vms_mb,
            "system_delta_percent": to_snap.system_percent - from_snap.system_percent,
            "duration_seconds": (to_snap.timestamp - from_snap.timestamp).total_seconds(),
        }

    def get_all_checkpoints(self) -> dict[str, MemorySnapshot]:
        """Get all checkpoints."""
        return dict(self._checkpoints)

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()

    def summary(self) -> dict[str, Any]:
        """Get summary of all checkpoints."""
        if not self._checkpoints:
            return {}

        snapshots = list(self._checkpoints.values())
        rss_values = [s.process_rss_mb for s in snapshots]

        return {
            "n_checkpoints": len(snapshots),
            "checkpoints": list(self._checkpoints.keys()),
            "min_rss_mb": min(rss_values),
            "max_rss_mb": max(rss_values),
            "first_checkpoint": min(self._checkpoints.keys()),
            "last_checkpoint": max(self._checkpoints.keys()),
        }


# =============================================================================
# Memory-Aware Batch Processor
# =============================================================================


T = TypeVar("T")
R = TypeVar("R")


class MemoryAwareBatchProcessor(Generic[T, R]):
    """Batch processor that adapts to memory constraints.

    Automatically adjusts batch size based on available memory.

    Example:
        processor = MemoryAwareBatchProcessor(
            process_fn=process_batch,
            max_memory_percent=80,
        )

        results = processor.process(all_items)
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], list[R]],
        initial_batch_size: int = 1000,
        min_batch_size: int = 10,
        max_batch_size: int = 100000,
        max_memory_percent: float = 80.0,
        memory_check_frequency: int = 1,
    ):
        """Initialize processor.

        Args:
            process_fn: Function to process a batch
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            max_memory_percent: Maximum memory usage percent
            memory_check_frequency: Check memory every N batches
        """
        self.process_fn = process_fn
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_memory_percent = max_memory_percent
        self.memory_check_frequency = memory_check_frequency

        self._reader = get_memory_reader()
        self._current_batch_size = initial_batch_size
        self._batch_count = 0

    def process(
        self,
        items: list[T],
        callback: Callable[[int, int], None] | None = None,
    ) -> list[R]:
        """Process all items in adaptive batches.

        Args:
            items: Items to process
            callback: Progress callback (processed, total)

        Returns:
            Combined results from all batches
        """
        results: list[R] = []
        total = len(items)
        processed = 0

        while processed < total:
            # Get current batch
            batch_end = min(processed + self._current_batch_size, total)
            batch = items[processed:batch_end]

            # Process batch
            batch_results = self.process_fn(batch)
            results.extend(batch_results)

            processed = batch_end
            self._batch_count += 1

            # Report progress
            if callback:
                callback(processed, total)

            # Check memory and adjust batch size
            if self._batch_count % self.memory_check_frequency == 0:
                self._adjust_batch_size()

            # Run GC periodically
            if self._batch_count % 10 == 0:
                gc.collect()

        return results

    def _adjust_batch_size(self) -> None:
        """Adjust batch size based on memory usage."""
        snapshot = self._reader.get_snapshot()
        current_percent = snapshot.system_percent

        if current_percent >= self.max_memory_percent:
            # Reduce batch size
            new_size = max(
                self.min_batch_size,
                int(self._current_batch_size * 0.7),
            )
            self._current_batch_size = new_size

        elif current_percent < self.max_memory_percent * 0.7:
            # Can increase batch size
            new_size = min(
                self.max_batch_size,
                int(self._current_batch_size * 1.3),
            )
            self._current_batch_size = new_size


# =============================================================================
# Memory Leak Detector
# =============================================================================


@dataclass
class LeakSuspect:
    """Potential memory leak information."""

    type_name: str
    count_initial: int
    count_final: int
    count_delta: int
    growth_percent: float
    sample_referrers: list[str] = field(default_factory=list)


class MemoryLeakDetector:
    """Detects potential memory leaks by tracking object counts.

    Example:
        detector = MemoryLeakDetector()

        detector.start()
        do_operations()
        suspects = detector.detect()

        for suspect in suspects:
            print(f"Possible leak: {suspect.type_name} +{suspect.count_delta}")
    """

    def __init__(
        self,
        min_growth_count: int = 100,
        min_growth_percent: float = 10.0,
    ):
        """Initialize detector.

        Args:
            min_growth_count: Minimum object count growth to flag
            min_growth_percent: Minimum growth percentage to flag
        """
        self.min_growth_count = min_growth_count
        self.min_growth_percent = min_growth_percent
        self._initial_counts: dict[str, int] = {}

    def start(self) -> None:
        """Start tracking object counts."""
        gc.collect()
        self._initial_counts = self._count_objects()

    def detect(self) -> list[LeakSuspect]:
        """Detect potential memory leaks.

        Returns:
            List of suspected leaks
        """
        gc.collect()
        final_counts = self._count_objects()

        suspects = []

        for type_name, final_count in final_counts.items():
            initial_count = self._initial_counts.get(type_name, 0)
            delta = final_count - initial_count

            if delta < self.min_growth_count:
                continue

            if initial_count > 0:
                growth_percent = (delta / initial_count) * 100
            else:
                growth_percent = 100.0

            if growth_percent < self.min_growth_percent:
                continue

            suspects.append(LeakSuspect(
                type_name=type_name,
                count_initial=initial_count,
                count_final=final_count,
                count_delta=delta,
                growth_percent=growth_percent,
            ))

        # Sort by delta (largest first)
        suspects.sort(key=lambda s: s.count_delta, reverse=True)
        return suspects

    def _count_objects(self) -> dict[str, int]:
        """Count objects by type."""
        counts: dict[str, int] = {}

        for obj in gc.get_objects():
            try:
                type_name = type(obj).__name__
                counts[type_name] = counts.get(type_name, 0) + 1
            except Exception:
                pass

        return counts


# =============================================================================
# Convenience Functions
# =============================================================================


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage.

    Returns:
        Dictionary with memory information
    """
    reader = get_memory_reader()
    snapshot = reader.get_snapshot()

    return {
        "process_rss_mb": snapshot.process_rss_mb,
        "process_vms_mb": snapshot.process_vms_mb,
        "process_percent": snapshot.process_percent,
        "system_available_mb": snapshot.system_available_mb,
        "system_total_mb": snapshot.system_total_mb,
        "system_percent": snapshot.system_percent,
    }


def check_memory_available(
    required_mb: float,
    safety_margin: float = 0.2,
) -> bool:
    """Check if enough memory is available.

    Args:
        required_mb: Required memory in MB
        safety_margin: Safety margin (0.2 = 20% buffer)

    Returns:
        True if enough memory is available
    """
    reader = get_memory_reader()
    snapshot = reader.get_snapshot()

    available = snapshot.system_available_mb
    required_with_margin = required_mb * (1 + safety_margin)

    return available >= required_with_margin


def estimate_batch_size(
    item_size_bytes: int,
    target_memory_mb: float = 100,
) -> int:
    """Estimate optimal batch size for given item size.

    Args:
        item_size_bytes: Size of each item in bytes
        target_memory_mb: Target memory usage per batch

    Returns:
        Recommended batch size
    """
    target_bytes = target_memory_mb * 1024 * 1024
    return max(1, int(target_bytes / item_size_bytes))


def force_gc() -> dict[str, int]:
    """Force garbage collection and return stats.

    Returns:
        GC collection statistics
    """
    before = get_memory_usage()

    collected = {
        "gen0": gc.collect(0),
        "gen1": gc.collect(1),
        "gen2": gc.collect(2),
    }

    after = get_memory_usage()

    collected["freed_mb"] = before["process_rss_mb"] - after["process_rss_mb"]

    return collected


def monitor_function(
    func: Callable[..., R],
    *args: Any,
    **kwargs: Any,
) -> tuple[R, MemoryReport]:
    """Execute a function with memory monitoring.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (function result, memory report)
    """
    monitor = MemoryMonitor()
    monitor.start()

    try:
        result = func(*args, **kwargs)
    finally:
        report = monitor.stop()

    return result, report
