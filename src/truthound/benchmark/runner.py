"""Benchmark runner and suite management.

This module provides the orchestration layer for running benchmarks:
- BenchmarkRunner: Runs individual or groups of benchmarks
- BenchmarkSuite: Collection of benchmarks to run together
- RunnerConfig: Configuration for the runner

Features:
- Parallel execution support
- Progress tracking
- Result aggregation
- CI/CD integration
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from truthound.benchmark.base import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSize,
    EnvironmentInfo,
    benchmark_registry,
)


# =============================================================================
# Configuration
# =============================================================================


class ParallelMode(str, Enum):
    """Parallelization mode for benchmark execution."""

    SEQUENTIAL = "sequential"  # Run one at a time
    PARALLEL = "parallel"      # Run in parallel
    AUTO = "auto"              # Decide based on benchmark type


@dataclass
class RunnerConfig:
    """Configuration for the benchmark runner.

    Controls execution behavior, parallelization, and output.
    """

    # Execution
    parallel_mode: ParallelMode = ParallelMode.SEQUENTIAL
    max_workers: int = 4
    fail_fast: bool = False  # Stop on first failure

    # Filtering
    categories: list[BenchmarkCategory] | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None

    # Size override
    size_override: BenchmarkSize | None = None

    # Output
    output_dir: Path | None = None
    save_results: bool = True
    verbose: bool = False

    # Callback
    on_benchmark_complete: Callable[[BenchmarkResult], None] | None = None

    @classmethod
    def quick(cls) -> "RunnerConfig":
        """Quick configuration for fast feedback."""
        return cls(
            parallel_mode=ParallelMode.PARALLEL,
            size_override=BenchmarkSize.SMALL,
            verbose=False,
        )

    @classmethod
    def ci(cls) -> "RunnerConfig":
        """Configuration for CI/CD environments."""
        return cls(
            parallel_mode=ParallelMode.SEQUENTIAL,
            size_override=BenchmarkSize.MEDIUM,
            fail_fast=True,
            save_results=True,
            verbose=True,
        )


# =============================================================================
# Suite Results
# =============================================================================


@dataclass
class SuiteResult:
    """Aggregated results from running a benchmark suite.

    Contains all individual results plus summary statistics.
    """

    suite_name: str
    results: list[BenchmarkResult]
    environment: EnvironmentInfo
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_benchmarks(self) -> int:
        """Total number of benchmarks run."""
        return len(self.results)

    @property
    def successful_benchmarks(self) -> int:
        """Number of successful benchmarks."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_benchmarks(self) -> int:
        """Number of failed benchmarks."""
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        """Success rate as a ratio."""
        if not self.results:
            return 0.0
        return self.successful_benchmarks / self.total_benchmarks

    @property
    def total_duration_seconds(self) -> float:
        """Total duration of all benchmarks."""
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def by_category(self) -> dict[BenchmarkCategory, list[BenchmarkResult]]:
        """Group results by category."""
        groups: dict[BenchmarkCategory, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.category not in groups:
                groups[result.category] = []
            groups[result.category].append(result)
        return groups

    def get_result(self, benchmark_name: str) -> BenchmarkResult | None:
        """Get result for a specific benchmark."""
        for result in self.results:
            if result.benchmark_name == benchmark_name:
                return result
        return None

    def filter_successful(self) -> list[BenchmarkResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    def filter_failed(self) -> list[BenchmarkResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "summary": {
                "total_benchmarks": self.total_benchmarks,
                "successful": self.successful_benchmarks,
                "failed": self.failed_benchmarks,
                "success_rate": self.success_rate,
                "total_duration_seconds": self.total_duration_seconds,
            },
            "environment": self.environment.to_dict(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "SuiteResult":
        """Load results from a JSON file."""
        data = json.loads(Path(path).read_text())
        # Reconstruct objects from dict
        env = EnvironmentInfo(
            python_version=data["environment"]["python_version"],
            platform_system=data["environment"]["platform"]["system"],
            platform_release=data["environment"]["platform"]["release"],
            platform_machine=data["environment"]["platform"]["machine"],
            cpu_count=data["environment"]["cpu_count"],
            polars_version=data["environment"]["polars_version"],
            truthound_version=data["environment"]["truthound_version"],
        )
        return cls(
            suite_name=data["suite_name"],
            results=[],  # Results would need more reconstruction
            environment=env,
            config=data.get("config", {}),
        )


# =============================================================================
# Benchmark Suite
# =============================================================================


class BenchmarkSuite:
    """Collection of benchmarks to run together.

    Provides a way to organize and configure groups of benchmarks.

    Example:
        suite = BenchmarkSuite("profiling")
        suite.add("profile")
        suite.add("profile_financial")
        suite.add("profile_wide")

        runner = BenchmarkRunner()
        results = runner.run_suite(suite)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        benchmark_config: BenchmarkConfig | None = None,
    ):
        self.name = name
        self.description = description
        self.benchmark_config = benchmark_config or BenchmarkConfig.standard()
        self._benchmarks: list[tuple[str, dict[str, Any]]] = []

    def add(
        self,
        benchmark_name: str,
        **kwargs: Any,
    ) -> "BenchmarkSuite":
        """Add a benchmark to the suite.

        Args:
            benchmark_name: Name of registered benchmark
            **kwargs: Parameters for the benchmark

        Returns:
            Self for chaining
        """
        self._benchmarks.append((benchmark_name, kwargs))
        return self

    def add_all_in_category(
        self,
        category: BenchmarkCategory,
        **kwargs: Any,
    ) -> "BenchmarkSuite":
        """Add all benchmarks in a category.

        Args:
            category: Category to add
            **kwargs: Parameters for all benchmarks

        Returns:
            Self for chaining
        """
        for benchmark_cls in benchmark_registry.get_by_category(category):
            self._benchmarks.append((benchmark_cls.name, kwargs))
        return self

    def add_all(self, **kwargs: Any) -> "BenchmarkSuite":
        """Add all registered benchmarks.

        Args:
            **kwargs: Parameters for all benchmarks

        Returns:
            Self for chaining
        """
        for name in benchmark_registry.list_names():
            self._benchmarks.append((name, kwargs))
        return self

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Iterate over benchmarks."""
        return iter(self._benchmarks)

    def __len__(self) -> int:
        """Number of benchmarks in suite."""
        return len(self._benchmarks)

    @classmethod
    def profiling(cls, size: BenchmarkSize = BenchmarkSize.SMALL) -> "BenchmarkSuite":
        """Create a profiling-focused suite (~10 seconds)."""
        suite = cls(
            "profiling",
            "Profiling performance benchmarks",
            benchmark_config=BenchmarkConfig.standard(),
        )
        suite.add_all_in_category(BenchmarkCategory.PROFILING, row_count=size.row_count)
        return suite

    @classmethod
    def validation(cls, size: BenchmarkSize = BenchmarkSize.SMALL) -> "BenchmarkSuite":
        """Create a validation-focused suite (~10 seconds)."""
        suite = cls(
            "validation",
            "Validation performance benchmarks",
            benchmark_config=BenchmarkConfig.standard(),
        )
        suite.add_all_in_category(BenchmarkCategory.VALIDATION, row_count=size.row_count)
        return suite

    @classmethod
    def full(cls, size: BenchmarkSize = BenchmarkSize.SMALL) -> "BenchmarkSuite":
        """Create a comprehensive suite with core benchmarks (~30 seconds).

        Note:
            Uses SMALL size by default for reasonable execution time.
            For thorough testing, use size=BenchmarkSize.MEDIUM.
        """
        suite = cls(
            "full",
            "Full benchmark suite",
            benchmark_config=BenchmarkConfig.standard(),
        )
        # Core benchmarks only - excludes heavy e2e and memory benchmarks
        suite.add("profile", row_count=size.row_count)
        suite.add("check", row_count=size.row_count)
        suite.add("learn", row_count=size.row_count)
        suite.add("compare", row_count=size.row_count)
        suite.add("scan", row_count=size.row_count)
        suite.add("throughput", row_count=size.row_count)
        return suite

    @classmethod
    def quick(cls) -> "BenchmarkSuite":
        """Create a quick suite for fast feedback (~5 seconds)."""
        suite = cls(
            "quick",
            "Quick benchmarks for fast feedback",
            benchmark_config=BenchmarkConfig.quick(),
        )
        suite.add("profile", row_count=BenchmarkSize.TINY.row_count)
        suite.add("check", row_count=BenchmarkSize.TINY.row_count)
        suite.add("learn", row_count=BenchmarkSize.TINY.row_count)
        return suite

    @classmethod
    def ci(cls) -> "BenchmarkSuite":
        """Create a CI-appropriate suite (~15 seconds)."""
        suite = cls(
            "ci",
            "CI/CD benchmark suite",
            benchmark_config=BenchmarkConfig.standard(),
        )
        suite.add("profile", row_count=BenchmarkSize.SMALL.row_count)
        suite.add("check", row_count=BenchmarkSize.SMALL.row_count)
        suite.add("learn", row_count=BenchmarkSize.SMALL.row_count)
        suite.add("compare", row_count=BenchmarkSize.TINY.row_count)
        suite.add("scan", row_count=BenchmarkSize.TINY.row_count)
        return suite


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """Orchestrates benchmark execution.

    Provides methods to run individual benchmarks, suites, or
    all registered benchmarks with configurable parallelization.

    Example:
        runner = BenchmarkRunner()

        # Run single benchmark
        result = runner.run("profile", row_count=1_000_000)

        # Run suite
        suite = BenchmarkSuite.profiling()
        results = runner.run_suite(suite)

        # Run all
        results = runner.run_all()
    """

    def __init__(
        self,
        config: RunnerConfig | None = None,
        benchmark_config: BenchmarkConfig | None = None,
    ):
        self.config = config or RunnerConfig()
        self.benchmark_config = benchmark_config or BenchmarkConfig.standard()
        self._lock = threading.Lock()

    def run(
        self,
        benchmark_name: str,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run a single benchmark by name.

        Args:
            benchmark_name: Name of the benchmark to run
            **kwargs: Parameters for the benchmark

        Returns:
            Benchmark result
        """
        benchmark = benchmark_registry.create(benchmark_name)

        # Apply size override if configured
        if self.config.size_override and "row_count" not in kwargs:
            kwargs["row_count"] = self.config.size_override.row_count

        result = benchmark.execute(self.benchmark_config, **kwargs)

        # Callback
        if self.config.on_benchmark_complete:
            self.config.on_benchmark_complete(result)

        # Save if configured
        if self.config.save_results and self.config.output_dir:
            self._save_result(result)

        return result

    def run_suite(
        self,
        suite: BenchmarkSuite,
    ) -> SuiteResult:
        """Run a benchmark suite.

        Args:
            suite: Suite to run

        Returns:
            Aggregated suite results
        """
        environment = EnvironmentInfo.capture()
        started_at = datetime.now()
        results: list[BenchmarkResult] = []

        # Runner's config takes precedence over suite's config
        # This allows CLI flags to override suite defaults
        benchmark_config = self.benchmark_config

        if self.config.verbose:
            print(f"\nRunning suite: {suite.name}")
            print(f"Benchmarks: {len(suite)}")
            print("-" * 40)

        if self.config.parallel_mode == ParallelMode.PARALLEL:
            results = self._run_parallel(suite, benchmark_config)
        else:
            results = self._run_sequential(suite, benchmark_config)

        suite_result = SuiteResult(
            suite_name=suite.name,
            results=results,
            environment=environment,
            started_at=started_at,
            completed_at=datetime.now(),
            config={
                "benchmark_config": {
                    "warmup_iterations": benchmark_config.warmup_iterations,
                    "measure_iterations": benchmark_config.measure_iterations,
                },
                "runner_config": {
                    "parallel_mode": self.config.parallel_mode.value,
                },
            },
        )

        if self.config.verbose:
            print("-" * 40)
            print(f"Completed: {suite_result.successful_benchmarks}/{suite_result.total_benchmarks}")
            print(f"Duration: {suite_result.total_duration_seconds:.2f}s")

        # Save suite results
        if self.config.save_results and self.config.output_dir:
            output_file = self.config.output_dir / f"{suite.name}_results.json"
            suite_result.save(output_file)

        return suite_result

    def run_all(
        self,
        **kwargs: Any,
    ) -> SuiteResult:
        """Run all registered benchmarks.

        Args:
            **kwargs: Parameters for all benchmarks

        Returns:
            Aggregated results
        """
        suite = BenchmarkSuite("all", "All registered benchmarks")

        # Apply filters
        for name in benchmark_registry.list_names():
            # Check category filter
            if self.config.categories:
                benchmark_cls = benchmark_registry.get(name)
                if benchmark_cls.category not in self.config.categories:
                    continue

            # Check include patterns
            if self.config.include_patterns:
                if not any(p in name for p in self.config.include_patterns):
                    continue

            # Check exclude patterns
            if self.config.exclude_patterns:
                if any(p in name for p in self.config.exclude_patterns):
                    continue

            suite.add(name, **kwargs)

        return self.run_suite(suite)

    def run_category(
        self,
        category: BenchmarkCategory,
        **kwargs: Any,
    ) -> SuiteResult:
        """Run all benchmarks in a category.

        Args:
            category: Category to run
            **kwargs: Parameters for benchmarks

        Returns:
            Aggregated results
        """
        suite = BenchmarkSuite(
            f"category_{category.value}",
            f"Benchmarks in {category.value} category",
        )
        suite.add_all_in_category(category, **kwargs)
        return self.run_suite(suite)

    def _run_sequential(
        self,
        suite: BenchmarkSuite,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        """Run benchmarks sequentially."""
        results = []

        for name, kwargs in suite:
            # Apply size override
            if self.config.size_override and "row_count" not in kwargs:
                kwargs["row_count"] = self.config.size_override.row_count

            if self.config.verbose:
                print(f"  Running: {name}...", end=" ", flush=True)

            try:
                benchmark = benchmark_registry.create(name)
                result = benchmark.execute(config, **kwargs)
                results.append(result)

                if self.config.verbose:
                    status = "OK" if result.success else "FAILED"
                    print(f"{status} ({result.metrics.mean_duration:.3f}s)")

                if self.config.on_benchmark_complete:
                    self.config.on_benchmark_complete(result)

                # Fail fast
                if not result.success and self.config.fail_fast:
                    break

            except Exception as e:
                if self.config.verbose:
                    print(f"ERROR: {e}")
                if self.config.fail_fast:
                    break

        return results

    def _run_parallel(
        self,
        suite: BenchmarkSuite,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        """Run benchmarks in parallel."""
        results: list[BenchmarkResult] = []

        def run_one(item: tuple[str, dict[str, Any]]) -> BenchmarkResult:
            name, kwargs = item
            if self.config.size_override and "row_count" not in kwargs:
                kwargs["row_count"] = self.config.size_override.row_count

            benchmark = benchmark_registry.create(name)
            return benchmark.execute(config, **kwargs)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(run_one, item): item[0]
                for item in suite
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    with self._lock:
                        results.append(result)

                    if self.config.verbose:
                        status = "OK" if result.success else "FAILED"
                        print(f"  {name}: {status} ({result.metrics.mean_duration:.3f}s)")

                    if self.config.on_benchmark_complete:
                        self.config.on_benchmark_complete(result)

                except Exception as e:
                    if self.config.verbose:
                        print(f"  {name}: ERROR - {e}")

        return results

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save individual result to file."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result.benchmark_name}_{result.result_id}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
