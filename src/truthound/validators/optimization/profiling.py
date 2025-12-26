"""Validator performance profiling framework.

This module provides comprehensive profiling capabilities for validators:
    - Execution time measurement (per-validator, per-column)
    - Memory usage tracking (peak, delta, GC impact)
    - Statistical aggregation (mean, median, p95, p99)
    - Historical performance tracking
    - Regression detection
    - Export to various formats (JSON, Prometheus, HTML)

Design Principles:
    - Zero-overhead when disabled
    - Thread-safe for parallel execution
    - Extensible metric types
    - Integration with existing observability infrastructure

Usage:
    from truthound.validators.optimization.profiling import (
        ValidatorProfiler,
        ProfilerConfig,
        profile_validator,
    )

    # Simple profiling
    with profile_validator(my_validator) as profiler:
        issues = my_validator.validate(lf)

    print(profiler.metrics.to_dict())

    # Full profiling session
    profiler = ValidatorProfiler()
    profiler.start_session("my_validation_run")

    for validator in validators:
        with profiler.profile(validator):
            issues = validator.validate(lf)

    report = profiler.end_session()
    print(report.summary())
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Iterator, TypeVar, Generic

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Default histogram buckets for timing (in milliseconds)
DEFAULT_TIMING_BUCKETS_MS = [
    0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
]

# Memory tracking thresholds
MEMORY_WARNING_THRESHOLD_MB = 100
MEMORY_CRITICAL_THRESHOLD_MB = 500

# Maximum history entries to keep per validator
MAX_HISTORY_ENTRIES = 1000


class MetricType(Enum):
    """Types of metrics tracked for validators."""
    TIMING = auto()      # Execution time
    MEMORY = auto()      # Memory usage
    THROUGHPUT = auto()  # Rows per second
    ISSUE_COUNT = auto() # Validation issues found
    GC_IMPACT = auto()   # Garbage collection overhead


class ProfilerMode(Enum):
    """Profiler operating modes."""
    DISABLED = auto()    # No profiling
    BASIC = auto()       # Timing only
    STANDARD = auto()    # Timing + memory
    DETAILED = auto()    # All metrics + tracing
    DIAGNOSTIC = auto()  # Maximum detail for debugging


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class TimingMetrics:
    """Timing statistics for a validator."""
    durations_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.durations_ms)

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms) if self.durations_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.durations_ms) if self.durations_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.durations_ms) if self.durations_ms else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.durations_ms) < 2:
            return 0.0
        return statistics.stdev(self.durations_ms)

    @property
    def min_ms(self) -> float:
        return min(self.durations_ms) if self.durations_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.durations_ms) if self.durations_ms else 0.0

    def percentile(self, p: float) -> float:
        """Get percentile value (0-100)."""
        if not self.durations_ms:
            return 0.0
        sorted_durations = sorted(self.durations_ms)
        idx = int(len(sorted_durations) * p / 100)
        idx = min(idx, len(sorted_durations) - 1)
        return sorted_durations[idx]

    @property
    def p50_ms(self) -> float:
        return self.percentile(50)

    @property
    def p90_ms(self) -> float:
        return self.percentile(90)

    @property
    def p95_ms(self) -> float:
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        return self.percentile(99)

    def add(self, duration_ms: float) -> None:
        """Add a duration observation."""
        self.durations_ms.append(duration_ms)
        # Trim if too many entries
        if len(self.durations_ms) > MAX_HISTORY_ENTRIES:
            self.durations_ms = self.durations_ms[-MAX_HISTORY_ENTRIES:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


@dataclass
class MemoryMetrics:
    """Memory usage statistics for a validator."""
    peak_bytes: list[int] = field(default_factory=list)
    delta_bytes: list[int] = field(default_factory=list)
    gc_collections: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.peak_bytes)

    @property
    def mean_peak_mb(self) -> float:
        if not self.peak_bytes:
            return 0.0
        return statistics.mean(self.peak_bytes) / (1024 * 1024)

    @property
    def max_peak_mb(self) -> float:
        if not self.peak_bytes:
            return 0.0
        return max(self.peak_bytes) / (1024 * 1024)

    @property
    def mean_delta_mb(self) -> float:
        if not self.delta_bytes:
            return 0.0
        return statistics.mean(self.delta_bytes) / (1024 * 1024)

    @property
    def total_gc_collections(self) -> int:
        return sum(self.gc_collections)

    def add(self, peak: int, delta: int, gc_count: int = 0) -> None:
        """Add memory observation."""
        self.peak_bytes.append(peak)
        self.delta_bytes.append(delta)
        self.gc_collections.append(gc_count)
        # Trim if too many entries
        if len(self.peak_bytes) > MAX_HISTORY_ENTRIES:
            self.peak_bytes = self.peak_bytes[-MAX_HISTORY_ENTRIES:]
            self.delta_bytes = self.delta_bytes[-MAX_HISTORY_ENTRIES:]
            self.gc_collections = self.gc_collections[-MAX_HISTORY_ENTRIES:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_peak_mb": round(self.mean_peak_mb, 2),
            "max_peak_mb": round(self.max_peak_mb, 2),
            "mean_delta_mb": round(self.mean_delta_mb, 2),
            "total_gc_collections": self.total_gc_collections,
        }


@dataclass
class ThroughputMetrics:
    """Throughput statistics for a validator."""
    rows_processed: list[int] = field(default_factory=list)
    durations_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows_processed)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_processed)

    @property
    def mean_rows_per_sec(self) -> float:
        if not self.rows_processed or not self.durations_ms:
            return 0.0
        total_rows = sum(self.rows_processed)
        total_seconds = sum(self.durations_ms) / 1000
        return total_rows / total_seconds if total_seconds > 0 else 0.0

    def add(self, rows: int, duration_ms: float) -> None:
        """Add throughput observation."""
        self.rows_processed.append(rows)
        self.durations_ms.append(duration_ms)
        if len(self.rows_processed) > MAX_HISTORY_ENTRIES:
            self.rows_processed = self.rows_processed[-MAX_HISTORY_ENTRIES:]
            self.durations_ms = self.durations_ms[-MAX_HISTORY_ENTRIES:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_rows": self.total_rows,
            "mean_rows_per_sec": round(self.mean_rows_per_sec, 2),
        }


@dataclass
class ValidatorMetrics:
    """Complete metrics for a single validator."""
    validator_name: str
    validator_category: str
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    issue_counts: list[int] = field(default_factory=list)
    error_counts: int = 0
    last_execution: datetime | None = None

    @property
    def total_issues(self) -> int:
        return sum(self.issue_counts)

    @property
    def mean_issues(self) -> float:
        if not self.issue_counts:
            return 0.0
        return statistics.mean(self.issue_counts)

    @property
    def execution_count(self) -> int:
        return self.timing.count

    def record_execution(
        self,
        duration_ms: float,
        issue_count: int = 0,
        rows_processed: int = 0,
        peak_memory: int = 0,
        memory_delta: int = 0,
        gc_collections: int = 0,
        error: bool = False,
    ) -> None:
        """Record a complete execution observation."""
        self.timing.add(duration_ms)
        self.issue_counts.append(issue_count)

        if rows_processed > 0:
            self.throughput.add(rows_processed, duration_ms)

        if peak_memory > 0 or memory_delta != 0:
            self.memory.add(peak_memory, memory_delta, gc_collections)

        if error:
            self.error_counts += 1

        self.last_execution = datetime.now()

        # Trim issue counts
        if len(self.issue_counts) > MAX_HISTORY_ENTRIES:
            self.issue_counts = self.issue_counts[-MAX_HISTORY_ENTRIES:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "validator_category": self.validator_category,
            "execution_count": self.execution_count,
            "error_count": self.error_counts,
            "total_issues": self.total_issues,
            "mean_issues": round(self.mean_issues, 2),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "timing": self.timing.to_dict(),
            "memory": self.memory.to_dict(),
            "throughput": self.throughput.to_dict(),
        }


# =============================================================================
# Profiling Session and Context
# =============================================================================

@dataclass
class ExecutionSnapshot:
    """Snapshot of a single execution for detailed analysis."""
    validator_name: str
    timestamp: datetime
    duration_ms: float
    rows_processed: int
    issue_count: int
    peak_memory_bytes: int
    memory_delta_bytes: int
    gc_before: tuple[int, int, int]  # gen0, gen1, gen2 counts
    gc_after: tuple[int, int, int]
    success: bool
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "rows_processed": self.rows_processed,
            "issue_count": self.issue_count,
            "peak_memory_mb": round(self.peak_memory_bytes / (1024 * 1024), 2),
            "memory_delta_mb": round(self.memory_delta_bytes / (1024 * 1024), 2),
            "gc_collections": sum(self.gc_after) - sum(self.gc_before),
            "success": self.success,
            "error_message": self.error_message,
            "attributes": self.attributes,
        }


@dataclass
class ProfilingSession:
    """A profiling session containing multiple validator executions."""
    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    validator_metrics: dict[str, ValidatorMetrics] = field(default_factory=dict)
    snapshots: list[ExecutionSnapshot] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds() * 1000
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def total_validators(self) -> int:
        return len(self.validator_metrics)

    @property
    def total_executions(self) -> int:
        return sum(m.execution_count for m in self.validator_metrics.values())

    @property
    def total_issues(self) -> int:
        return sum(m.total_issues for m in self.validator_metrics.values())

    def get_or_create_metrics(
        self,
        validator_name: str,
        validator_category: str = "unknown",
    ) -> ValidatorMetrics:
        """Get or create metrics for a validator."""
        if validator_name not in self.validator_metrics:
            self.validator_metrics[validator_name] = ValidatorMetrics(
                validator_name=validator_name,
                validator_category=validator_category,
            )
        return self.validator_metrics[validator_name]

    def add_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        """Add an execution snapshot."""
        self.snapshots.append(snapshot)
        # Trim if too many
        if len(self.snapshots) > MAX_HISTORY_ENTRIES * 10:
            self.snapshots = self.snapshots[-MAX_HISTORY_ENTRIES * 10:]

    def summary(self) -> dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 2),
            "total_validators": self.total_validators,
            "total_executions": self.total_executions,
            "total_issues": self.total_issues,
            "attributes": self.attributes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "validators": {
                name: metrics.to_dict()
                for name, metrics in self.validator_metrics.items()
            },
            "snapshots_count": len(self.snapshots),
        }

    def to_json(self, include_snapshots: bool = False) -> str:
        """Export to JSON."""
        data = self.to_dict()
        if include_snapshots:
            data["snapshots"] = [s.to_dict() for s in self.snapshots]
        return json.dumps(data, indent=2, default=str)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProfilerConfig:
    """Configuration for validator profiling."""
    mode: ProfilerMode = ProfilerMode.STANDARD
    track_memory: bool = True
    track_gc: bool = True
    track_throughput: bool = True
    record_snapshots: bool = False
    max_snapshots: int = 1000
    memory_warning_mb: float = MEMORY_WARNING_THRESHOLD_MB
    memory_critical_mb: float = MEMORY_CRITICAL_THRESHOLD_MB
    timing_buckets_ms: list[float] = field(
        default_factory=lambda: DEFAULT_TIMING_BUCKETS_MS.copy()
    )

    @classmethod
    def disabled(cls) -> "ProfilerConfig":
        """Create a disabled configuration."""
        return cls(mode=ProfilerMode.DISABLED)

    @classmethod
    def basic(cls) -> "ProfilerConfig":
        """Create a basic configuration (timing only)."""
        return cls(
            mode=ProfilerMode.BASIC,
            track_memory=False,
            track_gc=False,
            track_throughput=False,
            record_snapshots=False,
        )

    @classmethod
    def detailed(cls) -> "ProfilerConfig":
        """Create a detailed configuration."""
        return cls(
            mode=ProfilerMode.DETAILED,
            record_snapshots=True,
        )

    @classmethod
    def diagnostic(cls) -> "ProfilerConfig":
        """Create a diagnostic configuration (maximum detail)."""
        return cls(
            mode=ProfilerMode.DIAGNOSTIC,
            record_snapshots=True,
            max_snapshots=10000,
        )


# =============================================================================
# Memory Tracker
# =============================================================================

class MemoryTracker:
    """Tracks memory usage during validator execution."""

    _psutil_available: bool | None = None

    def __init__(self):
        self._start_memory: int = 0
        self._peak_memory: int = 0
        self._gc_before: tuple[int, int, int] = (0, 0, 0)
        self._tracking: bool = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if memory tracking is available (psutil installed)."""
        if cls._psutil_available is None:
            try:
                import psutil
                cls._psutil_available = True
            except ImportError:
                cls._psutil_available = False
        return cls._psutil_available

    def _get_current_memory(self) -> int:
        """Get current process memory in bytes."""
        if not self.is_available():
            return 0
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    def _get_gc_counts(self) -> tuple[int, int, int]:
        """Get GC collection counts for all generations."""
        stats = gc.get_stats()
        return (
            stats[0].get("collections", 0),
            stats[1].get("collections", 0),
            stats[2].get("collections", 0),
        )

    def start(self) -> None:
        """Start memory tracking."""
        self._tracking = True
        self._start_memory = self._get_current_memory()
        self._peak_memory = self._start_memory
        self._gc_before = self._get_gc_counts()

    def update_peak(self) -> None:
        """Update peak memory if current is higher."""
        if self._tracking:
            current = self._get_current_memory()
            if current > self._peak_memory:
                self._peak_memory = current

    def stop(self) -> tuple[int, int, int, tuple[int, int, int], tuple[int, int, int]]:
        """Stop tracking and return (peak, delta, end_memory, gc_before, gc_after)."""
        if not self._tracking:
            return 0, 0, 0, (0, 0, 0), (0, 0, 0)

        self._tracking = False
        end_memory = self._get_current_memory()
        self.update_peak()
        gc_after = self._get_gc_counts()

        delta = end_memory - self._start_memory
        return (
            self._peak_memory,
            delta,
            end_memory,
            self._gc_before,
            gc_after,
        )


# =============================================================================
# Profiling Context Manager
# =============================================================================

@dataclass
class ProfileContext:
    """Context for a single validator profiling operation."""
    validator_name: str
    validator_category: str
    config: ProfilerConfig
    session: ProfilingSession | None = None
    rows_processed: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)

    # Internal state
    _start_time: float = field(default=0.0, init=False)
    _memory_tracker: MemoryTracker = field(default_factory=MemoryTracker, init=False)
    _completed: bool = field(default=False, init=False)
    _metrics: ValidatorMetrics | None = field(default=None, init=False)
    _snapshot: ExecutionSnapshot | None = field(default=None, init=False)
    _issue_count: int = field(default=0, init=False)
    _error: bool = field(default=False, init=False)
    _error_message: str | None = field(default=None, init=False)

    @property
    def metrics(self) -> ValidatorMetrics | None:
        """Get the metrics after completion."""
        return self._metrics

    @property
    def snapshot(self) -> ExecutionSnapshot | None:
        """Get the execution snapshot after completion."""
        return self._snapshot

    @property
    def duration_ms(self) -> float:
        """Get the execution duration."""
        if self._completed and self._metrics:
            return self._metrics.timing.durations_ms[-1] if self._metrics.timing.durations_ms else 0.0
        return (time.time() - self._start_time) * 1000 if self._start_time > 0 else 0.0

    def start(self) -> None:
        """Start profiling."""
        if self.config.mode == ProfilerMode.DISABLED:
            return

        self._start_time = time.time()

        if self.config.track_memory:
            self._memory_tracker.start()

    def set_issue_count(self, count: int) -> None:
        """Set the number of issues found."""
        self._issue_count = count

    def set_rows_processed(self, rows: int) -> None:
        """Set the number of rows processed."""
        self.rows_processed = rows

    def set_error(self, error_message: str | None = None) -> None:
        """Mark as failed with optional error message."""
        self._error = True
        self._error_message = error_message

    def add_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the profile."""
        self.attributes[key] = value

    def stop(self) -> None:
        """Stop profiling and record results."""
        if self.config.mode == ProfilerMode.DISABLED:
            self._completed = True
            return

        if self._completed:
            return

        end_time = time.time()
        duration_ms = (end_time - self._start_time) * 1000

        # Get memory stats
        peak_memory = 0
        memory_delta = 0
        gc_before = (0, 0, 0)
        gc_after = (0, 0, 0)
        gc_collections = 0

        if self.config.track_memory:
            peak_memory, memory_delta, _, gc_before, gc_after = self._memory_tracker.stop()
            gc_collections = sum(gc_after) - sum(gc_before)

        # Get or create metrics
        if self.session:
            self._metrics = self.session.get_or_create_metrics(
                self.validator_name,
                self.validator_category,
            )
        else:
            self._metrics = ValidatorMetrics(
                validator_name=self.validator_name,
                validator_category=self.validator_category,
            )

        # Record execution
        self._metrics.record_execution(
            duration_ms=duration_ms,
            issue_count=self._issue_count,
            rows_processed=self.rows_processed,
            peak_memory=peak_memory,
            memory_delta=memory_delta,
            gc_collections=gc_collections,
            error=self._error,
        )

        # Create snapshot if configured
        if self.config.record_snapshots:
            self._snapshot = ExecutionSnapshot(
                validator_name=self.validator_name,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                rows_processed=self.rows_processed,
                issue_count=self._issue_count,
                peak_memory_bytes=peak_memory,
                memory_delta_bytes=memory_delta,
                gc_before=gc_before,
                gc_after=gc_after,
                success=not self._error,
                error_message=self._error_message,
                attributes=self.attributes.copy(),
            )
            if self.session:
                self.session.add_snapshot(self._snapshot)

        self._completed = True


# =============================================================================
# Main Profiler Class
# =============================================================================

class ValidatorProfiler:
    """Main profiler for validator performance tracking.

    Thread-safe profiler that tracks execution metrics across multiple
    validators and sessions.

    Example:
        profiler = ValidatorProfiler()

        # Start a session
        profiler.start_session("validation_run_1")

        for validator in validators:
            with profiler.profile(validator) as ctx:
                issues = validator.validate(lf)
                ctx.set_issue_count(len(issues))

        # Get session results
        session = profiler.end_session()
        print(session.to_json())
    """

    def __init__(self, config: ProfilerConfig | None = None):
        """Initialize profiler.

        Args:
            config: Profiler configuration. Defaults to STANDARD mode.
        """
        self.config = config or ProfilerConfig()
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls
        self._current_session: ProfilingSession | None = None
        self._completed_sessions: list[ProfilingSession] = []
        self._global_metrics: dict[str, ValidatorMetrics] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.config.mode != ProfilerMode.DISABLED

    @property
    def current_session(self) -> ProfilingSession | None:
        """Get the current active session."""
        return self._current_session

    @property
    def global_metrics(self) -> dict[str, ValidatorMetrics]:
        """Get global metrics across all sessions."""
        return self._global_metrics.copy()

    def start_session(
        self,
        session_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> ProfilingSession:
        """Start a new profiling session.

        Args:
            session_id: Optional session identifier
            attributes: Optional session attributes

        Returns:
            The new profiling session
        """
        with self._lock:
            if self._current_session is not None:
                # End previous session
                self._end_session_internal()

            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self._current_session = ProfilingSession(
                session_id=session_id,
                start_time=datetime.now(),
                attributes=attributes or {},
            )
            return self._current_session

    def end_session(self) -> ProfilingSession | None:
        """End the current profiling session.

        Returns:
            The completed session, or None if no session was active
        """
        with self._lock:
            return self._end_session_internal()

    def _end_session_internal(self) -> ProfilingSession | None:
        """Internal session end (must hold lock)."""
        if self._current_session is None:
            return None

        session = self._current_session
        session.end_time = datetime.now()
        self._completed_sessions.append(session)
        self._current_session = None

        # Merge into global metrics
        for name, metrics in session.validator_metrics.items():
            if name not in self._global_metrics:
                self._global_metrics[name] = ValidatorMetrics(
                    validator_name=metrics.validator_name,
                    validator_category=metrics.validator_category,
                )
            global_m = self._global_metrics[name]
            # Merge timing
            global_m.timing.durations_ms.extend(metrics.timing.durations_ms)
            # Merge memory
            global_m.memory.peak_bytes.extend(metrics.memory.peak_bytes)
            global_m.memory.delta_bytes.extend(metrics.memory.delta_bytes)
            global_m.memory.gc_collections.extend(metrics.memory.gc_collections)
            # Merge throughput
            global_m.throughput.rows_processed.extend(metrics.throughput.rows_processed)
            global_m.throughput.durations_ms.extend(metrics.throughput.durations_ms)
            # Merge issues
            global_m.issue_counts.extend(metrics.issue_counts)
            global_m.error_counts += metrics.error_counts
            global_m.last_execution = metrics.last_execution

        return session

    @contextmanager
    def profile(
        self,
        validator: Any,
        rows_processed: int = 0,
        **attributes: Any,
    ) -> Iterator[ProfileContext]:
        """Profile a validator execution.

        Args:
            validator: The validator being profiled (must have name/category)
            rows_processed: Number of rows being validated
            **attributes: Additional attributes to record

        Yields:
            ProfileContext for recording metrics

        Example:
            with profiler.profile(my_validator, rows_processed=10000) as ctx:
                issues = my_validator.validate(lf)
                ctx.set_issue_count(len(issues))
        """
        # Extract validator info
        validator_name = getattr(validator, "name", type(validator).__name__)
        validator_category = getattr(validator, "category", "unknown")

        ctx = ProfileContext(
            validator_name=validator_name,
            validator_category=validator_category,
            config=self.config,
            session=self._current_session,
            rows_processed=rows_processed,
            attributes=dict(attributes),
        )

        ctx.start()
        try:
            yield ctx
        except Exception as e:
            ctx.set_error(str(e))
            raise
        finally:
            ctx.stop()

    def get_metrics(self, validator_name: str) -> ValidatorMetrics | None:
        """Get metrics for a specific validator.

        Args:
            validator_name: Name of the validator

        Returns:
            ValidatorMetrics or None if not found
        """
        with self._lock:
            # Check current session first
            if self._current_session and validator_name in self._current_session.validator_metrics:
                return self._current_session.validator_metrics[validator_name]
            # Fall back to global
            return self._global_metrics.get(validator_name)

    def get_slowest_validators(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the N slowest validators by mean execution time.

        Args:
            n: Number of validators to return

        Returns:
            List of (validator_name, mean_ms) tuples
        """
        with self._lock:
            all_metrics = {**self._global_metrics}
            if self._current_session:
                all_metrics.update(self._current_session.validator_metrics)

            sorted_validators = sorted(
                all_metrics.items(),
                key=lambda x: x[1].timing.mean_ms,
                reverse=True,
            )
            return [(name, m.timing.mean_ms) for name, m in sorted_validators[:n]]

    def get_memory_intensive_validators(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the N most memory-intensive validators.

        Args:
            n: Number of validators to return

        Returns:
            List of (validator_name, max_peak_mb) tuples
        """
        with self._lock:
            all_metrics = {**self._global_metrics}
            if self._current_session:
                all_metrics.update(self._current_session.validator_metrics)

            sorted_validators = sorted(
                all_metrics.items(),
                key=lambda x: x[1].memory.max_peak_mb,
                reverse=True,
            )
            return [(name, m.memory.max_peak_mb) for name, m in sorted_validators[:n]]

    def summary(self) -> dict[str, Any]:
        """Get a summary of all profiling data."""
        with self._lock:
            all_metrics = {**self._global_metrics}
            if self._current_session:
                all_metrics.update(self._current_session.validator_metrics)

            total_executions = sum(m.execution_count for m in all_metrics.values())
            total_issues = sum(m.total_issues for m in all_metrics.values())
            total_time_ms = sum(m.timing.total_ms for m in all_metrics.values())

            return {
                "total_validators": len(all_metrics),
                "total_executions": total_executions,
                "total_issues": total_issues,
                "total_time_ms": round(total_time_ms, 2),
                "completed_sessions": len(self._completed_sessions),
                "current_session_active": self._current_session is not None,
                "memory_tracking_available": MemoryTracker.is_available(),
            }

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._current_session = None
            self._completed_sessions.clear()
            self._global_metrics.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export all profiling data to a dictionary."""
        with self._lock:
            return {
                "summary": self.summary(),
                "global_metrics": {
                    name: m.to_dict() for name, m in self._global_metrics.items()
                },
                "completed_sessions": [
                    s.summary() for s in self._completed_sessions
                ],
                "current_session": (
                    self._current_session.summary()
                    if self._current_session else None
                ),
            }

    def to_json(self) -> str:
        """Export all profiling data to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            all_metrics = {**self._global_metrics}
            if self._current_session:
                all_metrics.update(self._current_session.validator_metrics)

        # Timing metrics
        lines.append("# HELP validator_execution_duration_ms Validator execution duration in milliseconds")
        lines.append("# TYPE validator_execution_duration_ms gauge")
        for name, m in all_metrics.items():
            labels = f'validator="{name}",category="{m.validator_category}"'
            lines.append(f"validator_execution_duration_ms_mean{{{labels}}} {m.timing.mean_ms:.3f}")
            lines.append(f"validator_execution_duration_ms_p95{{{labels}}} {m.timing.p95_ms:.3f}")
            lines.append(f"validator_execution_duration_ms_p99{{{labels}}} {m.timing.p99_ms:.3f}")

        # Execution count
        lines.append("")
        lines.append("# HELP validator_execution_count Total validator executions")
        lines.append("# TYPE validator_execution_count counter")
        for name, m in all_metrics.items():
            labels = f'validator="{name}",category="{m.validator_category}"'
            lines.append(f"validator_execution_count{{{labels}}} {m.execution_count}")

        # Memory metrics
        lines.append("")
        lines.append("# HELP validator_memory_peak_mb Peak memory usage in MB")
        lines.append("# TYPE validator_memory_peak_mb gauge")
        for name, m in all_metrics.items():
            if m.memory.count > 0:
                labels = f'validator="{name}",category="{m.validator_category}"'
                lines.append(f"validator_memory_peak_mb{{{labels}}} {m.memory.max_peak_mb:.2f}")

        # Issue count
        lines.append("")
        lines.append("# HELP validator_issues_total Total validation issues found")
        lines.append("# TYPE validator_issues_total counter")
        for name, m in all_metrics.items():
            labels = f'validator="{name}",category="{m.validator_category}"'
            lines.append(f"validator_issues_total{{{labels}}} {m.total_issues}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions and Decorators
# =============================================================================

# Global default profiler
_default_profiler: ValidatorProfiler | None = None
_profiler_lock = threading.Lock()


def get_default_profiler() -> ValidatorProfiler:
    """Get the default global profiler."""
    global _default_profiler
    with _profiler_lock:
        if _default_profiler is None:
            _default_profiler = ValidatorProfiler()
        return _default_profiler


def set_default_profiler(profiler: ValidatorProfiler) -> None:
    """Set the default global profiler."""
    global _default_profiler
    with _profiler_lock:
        _default_profiler = profiler


def reset_default_profiler() -> None:
    """Reset the default global profiler."""
    global _default_profiler
    with _profiler_lock:
        if _default_profiler:
            _default_profiler.reset()
        _default_profiler = None


@contextmanager
def profile_validator(
    validator: Any,
    rows_processed: int = 0,
    profiler: ValidatorProfiler | None = None,
    **attributes: Any,
) -> Iterator[ProfileContext]:
    """Profile a validator execution using the global or provided profiler.

    Args:
        validator: The validator to profile
        rows_processed: Number of rows being processed
        profiler: Optional profiler (uses global if not provided)
        **attributes: Additional attributes

    Yields:
        ProfileContext

    Example:
        with profile_validator(my_validator) as ctx:
            issues = my_validator.validate(lf)
            ctx.set_issue_count(len(issues))
    """
    if profiler is None:
        profiler = get_default_profiler()

    with profiler.profile(validator, rows_processed, **attributes) as ctx:
        yield ctx


def profiled(
    profiler: ValidatorProfiler | None = None,
    track_issues: bool = True,
) -> Callable:
    """Decorator for profiling validator methods.

    Args:
        profiler: Optional profiler (uses global if not provided)
        track_issues: Whether to track issue count from return value

    Returns:
        Decorated function

    Example:
        class MyValidator(Validator):
            @profiled()
            def validate(self, lf):
                return [issue1, issue2]
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            nonlocal profiler
            if profiler is None:
                profiler = get_default_profiler()

            # Try to get row count from LazyFrame
            rows = 0
            if args and hasattr(args[0], "select"):
                try:
                    import polars as pl
                    rows = args[0].select(pl.len()).collect().item()
                except Exception:
                    pass

            with profiler.profile(self, rows_processed=rows) as ctx:
                result = func(self, *args, **kwargs)
                if track_issues and isinstance(result, list):
                    ctx.set_issue_count(len(result))
                return result

        return wrapper
    return decorator


# =============================================================================
# Report Generation
# =============================================================================

class ProfilingReport:
    """Generates human-readable reports from profiling data."""

    def __init__(self, profiler: ValidatorProfiler):
        self.profiler = profiler

    def text_summary(self) -> str:
        """Generate a text summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATOR PROFILING REPORT")
        lines.append("=" * 60)

        summary = self.profiler.summary()
        lines.append(f"Total Validators: {summary['total_validators']}")
        lines.append(f"Total Executions: {summary['total_executions']}")
        lines.append(f"Total Issues Found: {summary['total_issues']}")
        lines.append(f"Total Time: {summary['total_time_ms']:.2f}ms")
        lines.append("")

        # Slowest validators
        lines.append("-" * 60)
        lines.append("TOP 10 SLOWEST VALIDATORS (by mean execution time)")
        lines.append("-" * 60)
        slowest = self.profiler.get_slowest_validators(10)
        for i, (name, mean_ms) in enumerate(slowest, 1):
            lines.append(f"{i:2}. {name}: {mean_ms:.2f}ms")
        lines.append("")

        # Memory intensive
        if MemoryTracker.is_available():
            lines.append("-" * 60)
            lines.append("TOP 10 MEMORY INTENSIVE VALIDATORS")
            lines.append("-" * 60)
            memory_heavy = self.profiler.get_memory_intensive_validators(10)
            for i, (name, peak_mb) in enumerate(memory_heavy, 1):
                lines.append(f"{i:2}. {name}: {peak_mb:.2f}MB peak")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def html_report(self) -> str:
        """Generate an HTML report."""
        data = self.profiler.to_dict()

        html = """<!DOCTYPE html>
<html>
<head>
    <title>Validator Profiling Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #ccc; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2196F3; }
        .warning { color: #ff9800; }
        .error { color: #f44336; }
    </style>
</head>
<body>
    <h1>Validator Profiling Report</h1>
"""

        # Summary section
        summary = data["summary"]
        html += f"""
    <h2>Summary</h2>
    <table>
        <tr><td>Total Validators</td><td class="metric">{summary['total_validators']}</td></tr>
        <tr><td>Total Executions</td><td class="metric">{summary['total_executions']}</td></tr>
        <tr><td>Total Issues</td><td class="metric">{summary['total_issues']}</td></tr>
        <tr><td>Total Time</td><td class="metric">{summary['total_time_ms']:.2f}ms</td></tr>
    </table>
"""

        # Validator details
        html += """
    <h2>Validator Performance</h2>
    <table>
        <tr>
            <th>Validator</th>
            <th>Category</th>
            <th>Executions</th>
            <th>Mean (ms)</th>
            <th>P95 (ms)</th>
            <th>P99 (ms)</th>
            <th>Max Peak (MB)</th>
            <th>Issues</th>
        </tr>
"""
        for name, metrics in data["global_metrics"].items():
            timing = metrics["timing"]
            memory = metrics["memory"]
            html += f"""
        <tr>
            <td>{name}</td>
            <td>{metrics['validator_category']}</td>
            <td>{metrics['execution_count']}</td>
            <td>{timing['mean_ms']:.2f}</td>
            <td>{timing['p95_ms']:.2f}</td>
            <td>{timing['p99_ms']:.2f}</td>
            <td>{memory['max_peak_mb']:.2f}</td>
            <td>{metrics['total_issues']}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "MetricType",
    "ProfilerMode",
    # Data classes
    "TimingMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "ValidatorMetrics",
    "ExecutionSnapshot",
    "ProfilingSession",
    "ProfilerConfig",
    "ProfileContext",
    # Main classes
    "ValidatorProfiler",
    "MemoryTracker",
    "ProfilingReport",
    # Convenience functions
    "get_default_profiler",
    "set_default_profiler",
    "reset_default_profiler",
    "profile_validator",
    "profiled",
    # Constants
    "DEFAULT_TIMING_BUCKETS_MS",
    "MEMORY_WARNING_THRESHOLD_MB",
    "MEMORY_CRITICAL_THRESHOLD_MB",
    "MAX_HISTORY_ENTRIES",
]
