"""Pattern Performance Profiler - Regex Execution Timing and Analysis.

This module provides profiling capabilities for regex patterns to
measure execution time across different input types and sizes.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pattern Profiler                              │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│ Timing  │   │  Input  │    │Statistical│   │ Scaling  │    │ Report  │
│ Engine  │   │Generator│    │ Analysis │   │ Analysis │    │Generator│
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

Profiling capabilities:
1. Execution time measurement (mean, median, std)
2. Input size scaling analysis
3. Best/worst case detection
4. Backtracking detection via time patterns
5. Memory usage profiling (optional)

Usage:
    from truthound.validators.security.redos.profiler import (
        PatternProfiler,
        profile_pattern,
    )

    # Quick profile
    result = profile_pattern(r"^[a-z]+$")
    print(result.mean_time_ms)  # Average execution time
    print(result.scaling_complexity)  # "O(n)", "O(n²)", etc.

    # Detailed profiling
    profiler = PatternProfiler()
    result = profiler.profile(
        pattern=r"(a+)+b",
        test_inputs=["a" * n for n in range(1, 20)],
    )
    print(result.detect_exponential_behavior())  # True
"""

from __future__ import annotations

import gc
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Sequence


class ScalingComplexity(Enum):
    """Time complexity classifications."""

    CONSTANT = "O(1)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    POLYNOMIAL = "O(n^k)"
    EXPONENTIAL = "O(2^n)"
    UNKNOWN = "Unknown"


@dataclass
class TimingMeasurement:
    """A single timing measurement."""

    input_string: str
    input_length: int
    execution_time_ns: int
    matched: bool
    match_length: int = 0
    iteration: int = 0

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.execution_time_ns / 1_000_000

    @property
    def execution_time_us(self) -> float:
        """Get execution time in microseconds."""
        return self.execution_time_ns / 1_000


@dataclass
class BenchmarkConfig:
    """Configuration for pattern benchmarking.

    Attributes:
        iterations: Number of iterations per input
        warmup_iterations: Warmup iterations (not counted)
        input_sizes: List of input sizes to test
        timeout_per_iteration_ms: Max time per iteration
        gc_before_each: Run GC before each measurement
        test_matching_inputs: Generate matching test inputs
        test_non_matching_inputs: Generate non-matching inputs
    """

    iterations: int = 100
    warmup_iterations: int = 5
    input_sizes: list[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    timeout_per_iteration_ms: float = 5000.0
    gc_before_each: bool = False
    test_matching_inputs: bool = True
    test_non_matching_inputs: bool = True

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Quick profiling configuration."""
        return cls(
            iterations=10,
            warmup_iterations=2,
            input_sizes=[10, 50, 100],
            timeout_per_iteration_ms=1000.0,
        )

    @classmethod
    def thorough(cls) -> "BenchmarkConfig":
        """Thorough profiling configuration."""
        return cls(
            iterations=500,
            warmup_iterations=20,
            input_sizes=[10, 50, 100, 200, 500, 1000, 2000],
            timeout_per_iteration_ms=10000.0,
            gc_before_each=True,
        )


@dataclass
class ProfileResult:
    """Result of pattern profiling.

    Attributes:
        pattern: The profiled pattern
        measurements: All timing measurements
        mean_time_ns: Mean execution time
        median_time_ns: Median execution time
        std_dev_ns: Standard deviation
        min_time_ns: Minimum time
        max_time_ns: Maximum time
        scaling_complexity: Detected time complexity
        scaling_factor: Scaling factor (for polynomial)
        is_linear: Whether pattern exhibits linear scaling
        is_exponential: Whether pattern shows exponential behavior
        backtracking_detected: Whether backtracking was detected
        warnings: Any profiling warnings
    """

    pattern: str
    measurements: list[TimingMeasurement] = field(default_factory=list)
    mean_time_ns: float = 0.0
    median_time_ns: float = 0.0
    std_dev_ns: float = 0.0
    min_time_ns: float = 0.0
    max_time_ns: float = 0.0
    scaling_complexity: ScalingComplexity = ScalingComplexity.UNKNOWN
    scaling_factor: float = 1.0
    is_linear: bool = True
    is_exponential: bool = False
    backtracking_detected: bool = False
    warnings: list[str] = field(default_factory=list)
    size_to_time: dict[int, float] = field(default_factory=dict)

    @property
    def mean_time_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_time_ns / 1_000_000

    @property
    def mean_time_us(self) -> float:
        """Mean time in microseconds."""
        return self.mean_time_ns / 1_000

    def detect_exponential_behavior(self) -> bool:
        """Check if timing suggests exponential backtracking."""
        if len(self.size_to_time) < 3:
            return False

        sizes = sorted(self.size_to_time.keys())
        times = [self.size_to_time[s] for s in sizes]

        # Check if time roughly doubles as size increases
        # This is a simplified heuristic
        for i in range(2, len(times)):
            if times[i-1] > 0:
                ratio = times[i] / times[i-1]
                # If time more than doubles for small size increase
                size_ratio = sizes[i] / sizes[i-1]
                if ratio > 2.0 and size_ratio < 2.0:
                    return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "mean_time_ms": round(self.mean_time_ms, 4),
            "median_time_ms": round(self.median_time_ns / 1_000_000, 4),
            "std_dev_ms": round(self.std_dev_ns / 1_000_000, 4),
            "min_time_ms": round(self.min_time_ns / 1_000_000, 4),
            "max_time_ms": round(self.max_time_ns / 1_000_000, 4),
            "scaling_complexity": self.scaling_complexity.value,
            "scaling_factor": round(self.scaling_factor, 4),
            "is_linear": self.is_linear,
            "is_exponential": self.is_exponential,
            "backtracking_detected": self.backtracking_detected,
            "measurement_count": len(self.measurements),
            "warnings": self.warnings,
            "size_to_time_ms": {
                k: round(v / 1_000_000, 4)
                for k, v in self.size_to_time.items()
            },
        }


class InputGenerator:
    """Generate test inputs for pattern profiling."""

    @staticmethod
    def generate_matching(pattern: str, size: int) -> list[str]:
        """Generate inputs likely to match the pattern.

        Args:
            pattern: Regex pattern
            size: Approximate input size

        Returns:
            List of test inputs
        """
        inputs = []

        # Analyze pattern for likely matching content
        if re.search(r"\[a-z\]", pattern) or re.search(r"\\w", pattern):
            inputs.append("a" * size)
            inputs.append("".join(chr((i % 26) + ord('a')) for i in range(size)))

        if re.search(r"\[0-9\]", pattern) or re.search(r"\\d", pattern):
            inputs.append("1" * size)
            inputs.append("".join(str(i % 10) for i in range(size)))

        if re.search(r"\.", pattern):
            inputs.append("x" * size)

        # Default: alphanumeric
        if not inputs:
            inputs.append("a" * size)
            inputs.append("x1" * (size // 2))

        return inputs

    @staticmethod
    def generate_non_matching(pattern: str, size: int) -> list[str]:
        """Generate inputs likely to NOT match the pattern.

        These are important for detecting ReDoS - non-matching
        inputs that cause backtracking are dangerous.
        """
        inputs = []

        # Common non-matching patterns
        if pattern.endswith("$"):
            # End anchor - add something that doesn't match at end
            if "[a-z]" in pattern:
                inputs.append("a" * size + "!")
            else:
                inputs.append("x" * size + "!")

        if pattern.startswith("^"):
            # Start anchor - add something at start
            inputs.append("!" + "a" * size)

        # For quantified patterns, create inputs that almost match
        if "+" in pattern or "*" in pattern:
            base = "a" * size
            inputs.append(base + "!")
            inputs.append("!" + base)

        # Default
        if not inputs:
            inputs.append("@#$%" * (size // 4))

        return inputs

    @staticmethod
    def generate_adversarial(pattern: str, size: int) -> list[str]:
        """Generate adversarial inputs designed to trigger backtracking."""
        inputs = []

        # For nested quantifiers like (a+)+
        if re.search(r"\([^)]*[+*]\)[+*]", pattern):
            # Create input that almost matches but fails at end
            inputs.append("a" * size + "!")

        # For alternation with quantifier
        if re.search(r"\([^)]*\|[^)]*\)[+*]", pattern):
            inputs.append("a" * size + "!")

        # For backreferences
        if re.search(r"\\[1-9]", pattern):
            inputs.append("a" * (size // 2) + "b" * (size // 2))

        return inputs


class PatternProfiler:
    """Profile regex pattern performance.

    This profiler measures execution time of regex patterns across
    different input sizes and types to detect performance issues.

    Example:
        profiler = PatternProfiler()

        # Profile with custom inputs
        result = profiler.profile(
            r"^[a-z]+$",
            test_inputs=["hello", "world" * 100],
        )

        # Profile with automatic input generation
        result = profiler.profile_scaling(
            r"(a+)+b",
            max_size=30,
        )

        # Check for issues
        if result.is_exponential:
            print("WARNING: Exponential backtracking detected!")
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize the profiler.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()

    def profile(
        self,
        pattern: str,
        test_inputs: Sequence[str] | None = None,
        flags: int = 0,
    ) -> ProfileResult:
        """Profile a pattern with specific test inputs.

        Args:
            pattern: Regex pattern to profile
            test_inputs: Test inputs (auto-generated if None)
            flags: Regex flags

        Returns:
            ProfileResult with timing data
        """
        # Compile pattern
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return ProfileResult(
                pattern=pattern,
                warnings=[f"Invalid pattern: {e}"],
            )

        # Generate inputs if not provided
        if test_inputs is None:
            test_inputs = []
            for size in self.config.input_sizes:
                if self.config.test_matching_inputs:
                    test_inputs.extend(InputGenerator.generate_matching(pattern, size))
                if self.config.test_non_matching_inputs:
                    test_inputs.extend(InputGenerator.generate_non_matching(pattern, size))

        if not test_inputs:
            return ProfileResult(
                pattern=pattern,
                warnings=["No test inputs available"],
            )

        # Run measurements
        measurements: list[TimingMeasurement] = []
        size_to_times: dict[int, list[float]] = {}

        for test_input in test_inputs:
            input_len = len(test_input)

            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                try:
                    compiled.match(test_input)
                except Exception:
                    pass

            # Timed iterations
            for iteration in range(self.config.iterations):
                if self.config.gc_before_each:
                    gc.collect()

                # Measure
                start = time.perf_counter_ns()
                try:
                    match = compiled.match(test_input)
                    matched = match is not None
                    match_len = match.end() if match else 0
                except Exception:
                    matched = False
                    match_len = 0
                elapsed = time.perf_counter_ns() - start

                measurement = TimingMeasurement(
                    input_string=test_input[:100],  # Truncate for storage
                    input_length=input_len,
                    execution_time_ns=elapsed,
                    matched=matched,
                    match_length=match_len,
                    iteration=iteration,
                )
                measurements.append(measurement)

                # Track by size
                if input_len not in size_to_times:
                    size_to_times[input_len] = []
                size_to_times[input_len].append(elapsed)

                # Check timeout
                if elapsed > self.config.timeout_per_iteration_ms * 1_000_000:
                    break

        # Calculate statistics
        return self._analyze_measurements(pattern, measurements, size_to_times)

    def profile_scaling(
        self,
        pattern: str,
        min_size: int = 5,
        max_size: int = 100,
        step: int = 5,
        flags: int = 0,
    ) -> ProfileResult:
        """Profile pattern scaling behavior.

        Args:
            pattern: Pattern to profile
            min_size: Minimum input size
            max_size: Maximum input size
            step: Size increment
            flags: Regex flags

        Returns:
            ProfileResult with scaling analysis
        """
        # Generate inputs of increasing size
        test_inputs = []
        for size in range(min_size, max_size + 1, step):
            test_inputs.extend(InputGenerator.generate_matching(pattern, size))
            test_inputs.extend(InputGenerator.generate_adversarial(pattern, size))

        return self.profile(pattern, test_inputs, flags)

    def compare_patterns(
        self,
        patterns: Sequence[str],
        test_input: str,
    ) -> dict[str, ProfileResult]:
        """Compare performance of multiple patterns.

        Args:
            patterns: Patterns to compare
            test_input: Test input to use

        Returns:
            Dictionary mapping pattern to result
        """
        results = {}
        for pattern in patterns:
            results[pattern] = self.profile(pattern, [test_input])
        return results

    def _analyze_measurements(
        self,
        pattern: str,
        measurements: list[TimingMeasurement],
        size_to_times: dict[int, list[float]],
    ) -> ProfileResult:
        """Analyze timing measurements."""
        if not measurements:
            return ProfileResult(
                pattern=pattern,
                warnings=["No measurements collected"],
            )

        times = [m.execution_time_ns for m in measurements]

        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)

        # Calculate average time per size
        size_to_avg: dict[int, float] = {}
        for size, size_times in size_to_times.items():
            size_to_avg[size] = statistics.mean(size_times)

        # Analyze scaling
        scaling_info = self._analyze_scaling(size_to_avg)

        return ProfileResult(
            pattern=pattern,
            measurements=measurements,
            mean_time_ns=mean_time,
            median_time_ns=median_time,
            std_dev_ns=std_dev,
            min_time_ns=min_time,
            max_time_ns=max_time,
            scaling_complexity=scaling_info["complexity"],
            scaling_factor=scaling_info["factor"],
            is_linear=scaling_info["is_linear"],
            is_exponential=scaling_info["is_exponential"],
            backtracking_detected=scaling_info["is_exponential"],
            size_to_time=size_to_avg,
        )

    def _analyze_scaling(
        self,
        size_to_time: dict[int, float],
    ) -> dict[str, Any]:
        """Analyze timing scaling behavior."""
        result = {
            "complexity": ScalingComplexity.UNKNOWN,
            "factor": 1.0,
            "is_linear": True,
            "is_exponential": False,
        }

        if len(size_to_time) < 2:
            return result

        sizes = sorted(size_to_time.keys())
        times = [size_to_time[s] for s in sizes]

        # Normalize times (avoid division issues)
        min_time = min(t for t in times if t > 0) if any(t > 0 for t in times) else 1
        norm_times = [max(t / min_time, 1.0) for t in times]
        norm_sizes = [s / sizes[0] for s in sizes]

        # Check for exponential growth
        if len(sizes) >= 3:
            # Check if time doubles faster than size
            growth_rates = []
            for i in range(1, len(sizes)):
                if norm_times[i-1] > 0 and norm_sizes[i-1] > 0:
                    time_growth = norm_times[i] / norm_times[i-1]
                    size_growth = norm_sizes[i] / norm_sizes[i-1]
                    if size_growth > 0:
                        growth_rates.append(time_growth / size_growth)

            if growth_rates:
                avg_growth = statistics.mean(growth_rates)
                if avg_growth > 1.5:  # Time growing faster than linear
                    if avg_growth > 3.0:
                        result["complexity"] = ScalingComplexity.EXPONENTIAL
                        result["is_exponential"] = True
                        result["is_linear"] = False
                    elif avg_growth > 2.0:
                        result["complexity"] = ScalingComplexity.QUADRATIC
                        result["is_linear"] = False
                        result["factor"] = 2.0
                    else:
                        result["complexity"] = ScalingComplexity.POLYNOMIAL
                        result["is_linear"] = False
                elif avg_growth < 0.5:
                    result["complexity"] = ScalingComplexity.CONSTANT
                else:
                    result["complexity"] = ScalingComplexity.LINEAR
                    result["is_linear"] = True

        return result


# ============================================================================
# Convenience functions
# ============================================================================


def profile_pattern(
    pattern: str,
    test_inputs: Sequence[str] | None = None,
    config: BenchmarkConfig | None = None,
) -> ProfileResult:
    """Profile a regex pattern.

    Args:
        pattern: Pattern to profile
        test_inputs: Optional test inputs
        config: Optional benchmark config

    Returns:
        ProfileResult with timing data

    Example:
        result = profile_pattern(r"^[a-z]+$")
        print(f"Mean time: {result.mean_time_ms:.4f}ms")
        print(f"Complexity: {result.scaling_complexity.value}")
    """
    profiler = PatternProfiler(config or BenchmarkConfig.quick())
    return profiler.profile(pattern, test_inputs)


def quick_benchmark(
    pattern: str,
    test_input: str,
    iterations: int = 100,
) -> dict[str, float]:
    """Quick benchmark of a pattern with single input.

    Args:
        pattern: Pattern to benchmark
        test_input: Input to match
        iterations: Number of iterations

    Returns:
        Dictionary with timing statistics

    Example:
        stats = quick_benchmark(r"^[a-z]+$", "hello")
        print(f"Mean: {stats['mean_ms']:.4f}ms")
    """
    config = BenchmarkConfig(
        iterations=iterations,
        warmup_iterations=5,
        input_sizes=[],  # Not used
    )
    profiler = PatternProfiler(config)
    result = profiler.profile(pattern, [test_input])

    return {
        "mean_ms": result.mean_time_ms,
        "median_ms": result.median_time_ns / 1_000_000,
        "std_dev_ms": result.std_dev_ns / 1_000_000,
        "min_ms": result.min_time_ns / 1_000_000,
        "max_ms": result.max_time_ns / 1_000_000,
    }


def detect_exponential_pattern(
    pattern: str,
    max_input_size: int = 25,
) -> bool:
    """Detect if a pattern exhibits exponential backtracking.

    Args:
        pattern: Pattern to test
        max_input_size: Maximum input size to test

    Returns:
        True if exponential behavior detected

    Example:
        if detect_exponential_pattern(r"(a+)+b"):
            print("WARNING: Exponential backtracking!")
    """
    config = BenchmarkConfig(
        iterations=5,
        warmup_iterations=1,
        input_sizes=list(range(5, max_input_size + 1, 2)),
        timeout_per_iteration_ms=2000.0,
    )

    profiler = PatternProfiler(config)
    result = profiler.profile_scaling(pattern, min_size=5, max_size=max_input_size, step=2)

    return result.is_exponential or result.detect_exponential_behavior()
