"""Base classes and protocols for the benchmarking system.

This module provides the foundational abstractions for:
- Benchmark definition and execution
- Result collection and metrics
- Registry for benchmark discovery
"""

from __future__ import annotations

import gc
import hashlib
import platform
import statistics
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import polars as pl


# =============================================================================
# Types and Enums
# =============================================================================


class BenchmarkCategory(str, Enum):
    """Categories for benchmark classification."""

    PROFILING = "profiling"
    VALIDATION = "validation"
    DRIFT = "drift"
    LEARNING = "learning"
    SCANNING = "scanning"
    E2E = "e2e"
    MEMORY = "memory"
    THROUGHPUT = "throughput"


class BenchmarkSize(str, Enum):
    """Standard benchmark sizes for consistent comparison."""

    TINY = "tiny"           # 1K rows
    SMALL = "small"         # 10K rows
    MEDIUM = "medium"       # 100K rows
    LARGE = "large"         # 1M rows
    XLARGE = "xlarge"       # 10M rows
    STRESS = "stress"       # 100M rows

    @property
    def row_count(self) -> int:
        """Get the row count for this size."""
        sizes = {
            "tiny": 1_000,
            "small": 10_000,
            "medium": 100_000,
            "large": 1_000_000,
            "xlarge": 10_000_000,
            "stress": 100_000_000,
        }
        return sizes[self.value]


class MetricUnit(str, Enum):
    """Units for benchmark metrics."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    BYTES = "bytes"
    MEGABYTES = "megabytes"
    ROWS_PER_SECOND = "rows/s"
    MB_PER_SECOND = "MB/s"
    OPERATIONS_PER_SECOND = "ops/s"
    COUNT = "count"
    RATIO = "ratio"
    PERCENT = "percent"


# =============================================================================
# Metrics and Results
# =============================================================================


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: float
    unit: MetricUnit
    is_lower_better: bool = True
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value,
            "is_lower_better": self.is_lower_better,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class BenchmarkMetrics:
    """Collection of metrics from a benchmark run.

    Provides statistical analysis of timing measurements.
    """

    # Timing metrics (in seconds)
    durations: list[float] = field(default_factory=list)

    # Memory metrics (in bytes)
    peak_memory_bytes: int = 0
    memory_delta_bytes: int = 0

    # Throughput metrics
    rows_processed: int = 0
    bytes_processed: int = 0

    # Custom metrics
    custom_metrics: list[MetricValue] = field(default_factory=list)

    @property
    def mean_duration(self) -> float:
        """Mean execution time in seconds."""
        if not self.durations:
            return 0.0
        return statistics.mean(self.durations)

    @property
    def median_duration(self) -> float:
        """Median execution time in seconds."""
        if not self.durations:
            return 0.0
        return statistics.median(self.durations)

    @property
    def std_duration(self) -> float:
        """Standard deviation of execution time."""
        if len(self.durations) < 2:
            return 0.0
        return statistics.stdev(self.durations)

    @property
    def min_duration(self) -> float:
        """Minimum execution time."""
        if not self.durations:
            return 0.0
        return min(self.durations)

    @property
    def max_duration(self) -> float:
        """Maximum execution time."""
        if not self.durations:
            return 0.0
        return max(self.durations)

    @property
    def p95_duration(self) -> float:
        """95th percentile execution time."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]

    @property
    def p99_duration(self) -> float:
        """99th percentile execution time."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]

    @property
    def rows_per_second(self) -> float:
        """Throughput in rows per second."""
        if not self.mean_duration or not self.rows_processed:
            return 0.0
        return self.rows_processed / self.mean_duration

    @property
    def mb_per_second(self) -> float:
        """Throughput in megabytes per second."""
        if not self.mean_duration or not self.bytes_processed:
            return 0.0
        return (self.bytes_processed / (1024 * 1024)) / self.mean_duration

    def add_duration(self, duration: float) -> None:
        """Add a duration measurement."""
        self.durations.append(duration)

    def add_custom_metric(self, metric: MetricValue) -> None:
        """Add a custom metric."""
        self.custom_metrics.append(metric)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timing": {
                "mean_seconds": self.mean_duration,
                "median_seconds": self.median_duration,
                "std_seconds": self.std_duration,
                "min_seconds": self.min_duration,
                "max_seconds": self.max_duration,
                "p95_seconds": self.p95_duration,
                "p99_seconds": self.p99_duration,
                "iterations": len(self.durations),
            },
            "memory": {
                "peak_bytes": self.peak_memory_bytes,
                "peak_mb": self.peak_memory_bytes / (1024 * 1024),
                "delta_bytes": self.memory_delta_bytes,
            },
            "throughput": {
                "rows_processed": self.rows_processed,
                "bytes_processed": self.bytes_processed,
                "rows_per_second": self.rows_per_second,
                "mb_per_second": self.mb_per_second,
            },
            "custom": [m.to_dict() for m in self.custom_metrics],
        }


@dataclass
class EnvironmentInfo:
    """Information about the execution environment."""

    python_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    cpu_count: int = 0
    polars_version: str = ""
    truthound_version: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Capture current environment information."""
        import truthound

        return cls(
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            cpu_count=platform.os.cpu_count() or 0,
            polars_version=pl.__version__,
            truthound_version=getattr(truthound, "__version__", "unknown"),
            timestamp=datetime.now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "cpu_count": self.cpu_count,
            "polars_version": self.polars_version,
            "truthound_version": self.truthound_version,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkResult:
    """Result of a single benchmark execution.

    Contains all metrics, environment info, and execution details.
    """

    benchmark_name: str
    category: BenchmarkCategory
    success: bool
    metrics: BenchmarkMetrics
    environment: EnvironmentInfo
    parameters: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_traceback: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Total benchmark duration including setup/teardown."""
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def result_id(self) -> str:
        """Unique identifier for this result."""
        content = f"{self.benchmark_name}:{self.started_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.result_id,
            "benchmark_name": self.benchmark_name,
            "category": self.category.value,
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "environment": self.environment.to_dict(),
            "parameters": self.parameters,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "tags": self.tags,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Controls warmup, iterations, timeouts, and other execution parameters.

    Performance Notes:
        - Default settings optimized for quick feedback (<30 seconds total)
        - Use quick() for development (~5 seconds)
        - Use standard() for regular benchmarking (~15 seconds)
        - Use thorough() only for release validation (~60 seconds)
    """

    # Iteration control - reduced defaults for faster feedback
    warmup_iterations: int = 1
    measure_iterations: int = 3
    cooldown_seconds: float = 0.01  # Reduced from 0.1s

    # Timeout and limits
    timeout_seconds: float = 120.0  # Reduced from 300s
    max_memory_mb: float = 0  # 0 = unlimited

    # Data parameters - default to SMALL for speed
    default_size: BenchmarkSize = BenchmarkSize.SMALL
    gc_between_iterations: bool = False  # Disabled by default for speed

    # Output control
    verbose: bool = False
    capture_traceback: bool = True

    # Tags for result filtering
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Quick benchmark configuration for fast feedback (~5 seconds)."""
        return cls(
            warmup_iterations=1,
            measure_iterations=2,
            cooldown_seconds=0.0,
            default_size=BenchmarkSize.TINY,
            gc_between_iterations=False,
        )

    @classmethod
    def standard(cls) -> "BenchmarkConfig":
        """Standard benchmark configuration (~15 seconds)."""
        return cls(
            warmup_iterations=1,
            measure_iterations=3,
            cooldown_seconds=0.01,
            default_size=BenchmarkSize.SMALL,
            gc_between_iterations=False,
        )

    @classmethod
    def thorough(cls) -> "BenchmarkConfig":
        """Thorough benchmark configuration for CI/CD (~60 seconds)."""
        return cls(
            warmup_iterations=2,
            measure_iterations=5,
            cooldown_seconds=0.05,
            default_size=BenchmarkSize.MEDIUM,
            gc_between_iterations=True,
        )

    @classmethod
    def stress(cls) -> "BenchmarkConfig":
        """Stress test configuration (several minutes)."""
        return cls(
            warmup_iterations=1,
            measure_iterations=3,
            timeout_seconds=600.0,
            default_size=BenchmarkSize.LARGE,
            gc_between_iterations=True,
        )


# =============================================================================
# Benchmark Protocol and Base Class
# =============================================================================


@runtime_checkable
class BenchmarkProtocol(Protocol):
    """Protocol defining the benchmark interface."""

    name: str
    category: BenchmarkCategory
    description: str

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        """Prepare benchmark resources."""
        ...

    def run_iteration(self) -> Any:
        """Run a single benchmark iteration."""
        ...

    def teardown(self) -> None:
        """Clean up benchmark resources."""
        ...

    def get_metrics(self) -> BenchmarkMetrics:
        """Get collected metrics."""
        ...


class Benchmark(ABC):
    """Abstract base class for benchmarks.

    Provides the framework for defining benchmarks with:
    - Setup and teardown phases
    - Configurable warmup and measurement iterations
    - Automatic metric collection
    - Memory tracking

    Example:
        class MyBenchmark(Benchmark):
            name = "my_benchmark"
            category = BenchmarkCategory.PROFILING
            description = "Benchmark for my operation"

            def setup(self, config, **kwargs):
                self.data = generate_data(kwargs.get("rows", 1000))

            def run_iteration(self):
                return process(self.data)

            def teardown(self):
                del self.data
    """

    name: str = "base"
    category: BenchmarkCategory = BenchmarkCategory.E2E
    description: str = ""

    def __init__(self):
        self._metrics = BenchmarkMetrics()
        self._config: BenchmarkConfig | None = None
        self._parameters: dict[str, Any] = {}
        self._lock = threading.Lock()

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        """Prepare benchmark resources.

        Override this method to set up any data or resources needed.

        Args:
            config: Benchmark configuration
            **kwargs: Benchmark-specific parameters
        """
        self._config = config
        self._parameters = kwargs

    @abstractmethod
    def run_iteration(self) -> Any:
        """Run a single benchmark iteration.

        This is the core operation being measured. It should be
        deterministic and isolated from setup/teardown.

        Returns:
            Result of the iteration (for validation)
        """
        pass

    def teardown(self) -> None:
        """Clean up benchmark resources.

        Override this method to release any resources.
        """
        pass

    def get_metrics(self) -> BenchmarkMetrics:
        """Get the collected metrics."""
        return self._metrics

    def validate_result(self, result: Any) -> bool:
        """Validate the result of an iteration.

        Override to add custom validation logic.

        Args:
            result: Result from run_iteration()

        Returns:
            True if result is valid
        """
        return result is not None

    def execute(
        self,
        config: BenchmarkConfig | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Execute the benchmark with full lifecycle.

        Args:
            config: Benchmark configuration (uses default if None)
            **kwargs: Parameters passed to setup()

        Returns:
            BenchmarkResult with all metrics and metadata
        """
        config = config or BenchmarkConfig.standard()
        environment = EnvironmentInfo.capture()
        started_at = datetime.now()

        try:
            # Setup phase
            self.setup(config, **kwargs)

            # Track memory if psutil available
            memory_before = self._get_memory_usage()

            # Warmup iterations
            for _ in range(config.warmup_iterations):
                self.run_iteration()
                if config.gc_between_iterations:
                    gc.collect()
                time.sleep(config.cooldown_seconds)

            # Measurement iterations
            self._metrics = BenchmarkMetrics()
            peak_memory = memory_before

            for _ in range(config.measure_iterations):
                start_time = time.perf_counter()
                result = self.run_iteration()
                duration = time.perf_counter() - start_time

                self._metrics.add_duration(duration)

                # Validate result
                if not self.validate_result(result):
                    raise ValueError("Benchmark result validation failed")

                # Track peak memory
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)

                if config.gc_between_iterations:
                    gc.collect()
                time.sleep(config.cooldown_seconds)

            # Record memory metrics
            memory_after = self._get_memory_usage()
            self._metrics.peak_memory_bytes = peak_memory
            self._metrics.memory_delta_bytes = memory_after - memory_before

            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                success=True,
                metrics=self._metrics,
                environment=environment,
                parameters=kwargs,
                started_at=started_at,
                completed_at=datetime.now(),
                tags=config.tags,
            )

        except Exception as e:
            return BenchmarkResult(
                benchmark_name=self.name,
                category=self.category,
                success=False,
                metrics=self._metrics,
                environment=environment,
                parameters=kwargs,
                error=str(e),
                error_traceback=traceback.format_exc() if config.capture_traceback else None,
                started_at=started_at,
                completed_at=datetime.now(),
                tags=config.tags,
            )

        finally:
            self.teardown()

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            return 0


# =============================================================================
# Registry
# =============================================================================


class BenchmarkRegistry:
    """Registry for benchmark discovery and management.

    Allows benchmarks to be registered and discovered by name or category.

    Example:
        registry = BenchmarkRegistry()

        @registry.register
        class MyBenchmark(Benchmark):
            name = "my_benchmark"
            ...

        # Get benchmark by name
        benchmark = registry.get("my_benchmark")

        # Get all benchmarks in category
        benchmarks = registry.get_by_category(BenchmarkCategory.PROFILING)
    """

    def __init__(self) -> None:
        self._benchmarks: dict[str, type[Benchmark]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        benchmark_class: type[Benchmark],
    ) -> type[Benchmark]:
        """Register a benchmark class.

        Can be used as a decorator.

        Args:
            benchmark_class: Benchmark class to register

        Returns:
            The registered class (for decorator use)
        """
        with self._lock:
            name = benchmark_class.name
            if name in self._benchmarks:
                raise ValueError(f"Benchmark '{name}' is already registered")
            self._benchmarks[name] = benchmark_class
        return benchmark_class

    def unregister(self, name: str) -> None:
        """Unregister a benchmark by name."""
        with self._lock:
            if name in self._benchmarks:
                del self._benchmarks[name]

    def get(self, name: str) -> type[Benchmark]:
        """Get a benchmark class by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark class

        Raises:
            KeyError: If benchmark not found
        """
        with self._lock:
            if name not in self._benchmarks:
                available = list(self._benchmarks.keys())
                raise KeyError(
                    f"Benchmark '{name}' not found. Available: {available}"
                )
            return self._benchmarks[name]

    def get_by_category(
        self,
        category: BenchmarkCategory,
    ) -> list[type[Benchmark]]:
        """Get all benchmarks in a category.

        Args:
            category: Benchmark category

        Returns:
            List of benchmark classes
        """
        with self._lock:
            return [
                cls for cls in self._benchmarks.values()
                if cls.category == category
            ]

    def list_names(self) -> list[str]:
        """List all registered benchmark names."""
        with self._lock:
            return list(self._benchmarks.keys())

    def list_all(self) -> list[type[Benchmark]]:
        """List all registered benchmark classes."""
        with self._lock:
            return list(self._benchmarks.values())

    def create(self, name: str) -> Benchmark:
        """Create a benchmark instance by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark instance
        """
        benchmark_class = self.get(name)
        return benchmark_class()


# Global registry instance
benchmark_registry = BenchmarkRegistry()


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    """Decorator to register a benchmark with the global registry.

    Example:
        @register_benchmark
        class MyBenchmark(Benchmark):
            name = "my_benchmark"
            ...
    """
    return benchmark_registry.register(cls)
