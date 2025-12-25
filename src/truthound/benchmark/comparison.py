"""Benchmark comparison and regression detection.

This module provides tools for:
- Comparing benchmark results across runs
- Detecting performance regressions
- Setting and checking performance thresholds
- Historical trend analysis

Use for CI/CD integration to catch performance regressions.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from truthound.benchmark.base import BenchmarkResult, BenchmarkMetrics
from truthound.benchmark.runner import SuiteResult


# =============================================================================
# Types and Enums
# =============================================================================


class ComparisonResult(str, Enum):
    """Result of a benchmark comparison."""

    BETTER = "better"       # Performance improved
    SAME = "same"           # No significant change
    WORSE = "worse"         # Performance degraded
    REGRESSION = "regression"  # Significant regression detected


class ThresholdType(str, Enum):
    """Type of performance threshold."""

    ABSOLUTE = "absolute"   # Fixed value (e.g., < 1.0s)
    RELATIVE = "relative"   # Percentage change (e.g., < 10% slower)
    PERCENTILE = "percentile"  # Based on historical percentile


# =============================================================================
# Performance Threshold
# =============================================================================


@dataclass
class PerformanceThreshold:
    """Defines acceptable performance bounds for a benchmark.

    Used to detect regressions and enforce performance requirements.

    Example:
        # Must complete in under 1 second
        threshold = PerformanceThreshold(
            benchmark_name="profile",
            max_duration_seconds=1.0,
        )

        # No more than 10% slower than baseline
        threshold = PerformanceThreshold(
            benchmark_name="check",
            max_regression_percent=10.0,
        )
    """

    benchmark_name: str
    max_duration_seconds: float | None = None
    min_throughput_rows_per_sec: float | None = None
    max_memory_mb: float | None = None
    max_regression_percent: float | None = None
    max_p99_duration_seconds: float | None = None

    def check(
        self,
        result: BenchmarkResult,
        baseline: BenchmarkResult | None = None,
    ) -> tuple[bool, list[str]]:
        """Check if a result meets this threshold.

        Args:
            result: Result to check
            baseline: Optional baseline for regression comparison

        Returns:
            Tuple of (passed, list of violations)
        """
        violations = []

        # Check absolute duration
        if self.max_duration_seconds is not None:
            if result.metrics.mean_duration > self.max_duration_seconds:
                violations.append(
                    f"Duration {result.metrics.mean_duration:.3f}s exceeds "
                    f"max {self.max_duration_seconds:.3f}s"
                )

        # Check throughput
        if self.min_throughput_rows_per_sec is not None:
            if result.metrics.rows_per_second < self.min_throughput_rows_per_sec:
                violations.append(
                    f"Throughput {result.metrics.rows_per_second:.0f} rows/s "
                    f"below min {self.min_throughput_rows_per_sec:.0f} rows/s"
                )

        # Check memory
        if self.max_memory_mb is not None:
            peak_mb = result.metrics.peak_memory_bytes / (1024 * 1024)
            if peak_mb > self.max_memory_mb:
                violations.append(
                    f"Memory {peak_mb:.1f}MB exceeds max {self.max_memory_mb:.1f}MB"
                )

        # Check P99
        if self.max_p99_duration_seconds is not None:
            if result.metrics.p99_duration > self.max_p99_duration_seconds:
                violations.append(
                    f"P99 {result.metrics.p99_duration:.3f}s exceeds "
                    f"max {self.max_p99_duration_seconds:.3f}s"
                )

        # Check regression against baseline
        if self.max_regression_percent is not None and baseline is not None:
            if baseline.metrics.mean_duration > 0:
                pct_change = (
                    (result.metrics.mean_duration - baseline.metrics.mean_duration)
                    / baseline.metrics.mean_duration * 100
                )
                if pct_change > self.max_regression_percent:
                    violations.append(
                        f"Regression of {pct_change:.1f}% exceeds "
                        f"max {self.max_regression_percent:.1f}%"
                    )

        return len(violations) == 0, violations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "max_duration_seconds": self.max_duration_seconds,
            "min_throughput_rows_per_sec": self.min_throughput_rows_per_sec,
            "max_memory_mb": self.max_memory_mb,
            "max_regression_percent": self.max_regression_percent,
            "max_p99_duration_seconds": self.max_p99_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceThreshold":
        """Create from dictionary."""
        return cls(
            benchmark_name=data["benchmark_name"],
            max_duration_seconds=data.get("max_duration_seconds"),
            min_throughput_rows_per_sec=data.get("min_throughput_rows_per_sec"),
            max_memory_mb=data.get("max_memory_mb"),
            max_regression_percent=data.get("max_regression_percent"),
            max_p99_duration_seconds=data.get("max_p99_duration_seconds"),
        )


# =============================================================================
# Comparison Result Details
# =============================================================================


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs."""

    name: str
    baseline_value: float
    current_value: float
    absolute_change: float
    percent_change: float
    result: ComparisonResult
    is_improvement: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "absolute_change": self.absolute_change,
            "percent_change": self.percent_change,
            "result": self.result.value,
            "is_improvement": self.is_improvement,
        }


@dataclass
class BenchmarkComparison:
    """Detailed comparison between two benchmark runs."""

    benchmark_name: str
    baseline_result: BenchmarkResult
    current_result: BenchmarkResult
    metric_comparisons: list[MetricComparison]
    overall_result: ComparisonResult
    threshold_violations: list[str] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        """Check if there's a regression."""
        return self.overall_result == ComparisonResult.REGRESSION

    @property
    def has_improvement(self) -> bool:
        """Check if there's an improvement."""
        return self.overall_result == ComparisonResult.BETTER

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "overall_result": self.overall_result.value,
            "has_regression": self.has_regression,
            "has_improvement": self.has_improvement,
            "metric_comparisons": [m.to_dict() for m in self.metric_comparisons],
            "threshold_violations": self.threshold_violations,
        }


# =============================================================================
# Benchmark Comparator
# =============================================================================


class BenchmarkComparator:
    """Compares benchmark results between runs.

    Provides detailed analysis of performance changes and
    regression detection.

    Example:
        comparator = BenchmarkComparator(
            significance_threshold=0.05,
            regression_threshold=0.10,
        )

        comparison = comparator.compare(baseline_result, current_result)
        if comparison.has_regression:
            print("Performance regression detected!")
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,  # 5% change is significant
        regression_threshold: float = 0.10,    # 10% slower is regression
    ):
        self.significance_threshold = significance_threshold
        self.regression_threshold = regression_threshold

    def compare(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult,
        threshold: PerformanceThreshold | None = None,
    ) -> BenchmarkComparison:
        """Compare two benchmark results.

        Args:
            baseline: Previous/baseline result
            current: Current result to compare
            threshold: Optional threshold to check

        Returns:
            Detailed comparison
        """
        if baseline.benchmark_name != current.benchmark_name:
            raise ValueError(
                f"Cannot compare different benchmarks: "
                f"{baseline.benchmark_name} vs {current.benchmark_name}"
            )

        comparisons = []

        # Compare mean duration
        comparisons.append(self._compare_metric(
            "mean_duration",
            baseline.metrics.mean_duration,
            current.metrics.mean_duration,
            is_lower_better=True,
        ))

        # Compare P95 duration
        comparisons.append(self._compare_metric(
            "p95_duration",
            baseline.metrics.p95_duration,
            current.metrics.p95_duration,
            is_lower_better=True,
        ))

        # Compare throughput
        if baseline.metrics.rows_processed > 0 and current.metrics.rows_processed > 0:
            comparisons.append(self._compare_metric(
                "rows_per_second",
                baseline.metrics.rows_per_second,
                current.metrics.rows_per_second,
                is_lower_better=False,
            ))

        # Compare memory
        if baseline.metrics.peak_memory_bytes > 0 and current.metrics.peak_memory_bytes > 0:
            comparisons.append(self._compare_metric(
                "peak_memory_bytes",
                baseline.metrics.peak_memory_bytes,
                current.metrics.peak_memory_bytes,
                is_lower_better=True,
            ))

        # Determine overall result
        overall = self._determine_overall_result(comparisons)

        # Check threshold violations
        violations = []
        if threshold:
            passed, violation_list = threshold.check(current, baseline)
            violations = violation_list
            if not passed and overall != ComparisonResult.REGRESSION:
                overall = ComparisonResult.REGRESSION

        return BenchmarkComparison(
            benchmark_name=baseline.benchmark_name,
            baseline_result=baseline,
            current_result=current,
            metric_comparisons=comparisons,
            overall_result=overall,
            threshold_violations=violations,
        )

    def compare_suites(
        self,
        baseline: SuiteResult,
        current: SuiteResult,
        thresholds: dict[str, PerformanceThreshold] | None = None,
    ) -> list[BenchmarkComparison]:
        """Compare two suite results.

        Args:
            baseline: Previous suite result
            current: Current suite result
            thresholds: Optional thresholds by benchmark name

        Returns:
            List of comparisons
        """
        thresholds = thresholds or {}
        comparisons = []

        # Create lookup for baseline results
        baseline_lookup = {r.benchmark_name: r for r in baseline.results}

        for current_result in current.results:
            baseline_result = baseline_lookup.get(current_result.benchmark_name)
            if baseline_result is None:
                continue

            threshold = thresholds.get(current_result.benchmark_name)
            comparison = self.compare(baseline_result, current_result, threshold)
            comparisons.append(comparison)

        return comparisons

    def _compare_metric(
        self,
        name: str,
        baseline_value: float,
        current_value: float,
        is_lower_better: bool,
    ) -> MetricComparison:
        """Compare a single metric."""
        if baseline_value == 0:
            percent_change = 0.0 if current_value == 0 else float('inf')
        else:
            percent_change = (current_value - baseline_value) / baseline_value

        absolute_change = current_value - baseline_value

        # Determine if this is an improvement
        if is_lower_better:
            is_improvement = current_value < baseline_value
        else:
            is_improvement = current_value > baseline_value

        # Determine result category
        if abs(percent_change) < self.significance_threshold:
            result = ComparisonResult.SAME
        elif is_improvement:
            result = ComparisonResult.BETTER
        elif abs(percent_change) > self.regression_threshold:
            result = ComparisonResult.REGRESSION
        else:
            result = ComparisonResult.WORSE

        return MetricComparison(
            name=name,
            baseline_value=baseline_value,
            current_value=current_value,
            absolute_change=absolute_change,
            percent_change=percent_change * 100,  # Convert to percentage
            result=result,
            is_improvement=is_improvement,
        )

    def _determine_overall_result(
        self,
        comparisons: list[MetricComparison],
    ) -> ComparisonResult:
        """Determine overall result from metric comparisons."""
        if not comparisons:
            return ComparisonResult.SAME

        # Check for any regressions
        has_regression = any(c.result == ComparisonResult.REGRESSION for c in comparisons)
        if has_regression:
            return ComparisonResult.REGRESSION

        # Check for improvements
        has_improvement = any(c.result == ComparisonResult.BETTER for c in comparisons)
        has_worse = any(c.result == ComparisonResult.WORSE for c in comparisons)

        if has_improvement and not has_worse:
            return ComparisonResult.BETTER
        elif has_worse and not has_improvement:
            return ComparisonResult.WORSE
        else:
            return ComparisonResult.SAME


# =============================================================================
# Regression Detector
# =============================================================================


class RegressionDetector:
    """Detects performance regressions across benchmark runs.

    Maintains history and provides CI/CD integration.

    Example:
        detector = RegressionDetector(history_path=Path("benchmarks/history"))

        # Check for regressions
        regressions = detector.check(current_results)
        if regressions:
            print("Regressions detected!")
            for r in regressions:
                print(f"  {r.benchmark_name}: {r.overall_result}")

        # Save for future comparison
        detector.save_baseline(current_results)
    """

    def __init__(
        self,
        history_path: Path | None = None,
        comparator: BenchmarkComparator | None = None,
        thresholds: list[PerformanceThreshold] | None = None,
    ):
        self.history_path = history_path or Path(".benchmarks")
        self.comparator = comparator or BenchmarkComparator()
        self.thresholds = {t.benchmark_name: t for t in (thresholds or [])}
        self._history: list[SuiteResult] = []

    def load_history(self) -> None:
        """Load historical results from disk."""
        if not self.history_path.exists():
            return

        for file in sorted(self.history_path.glob("*.json")):
            try:
                data = json.loads(file.read_text())
                # Simplified loading - in production would fully reconstruct
                self._history.append(data)
            except Exception:
                pass

    def get_baseline(self) -> SuiteResult | None:
        """Get the most recent baseline result."""
        baseline_path = self.history_path / "baseline.json"
        if not baseline_path.exists():
            return None

        try:
            return SuiteResult.load(baseline_path)
        except Exception:
            return None

    def save_baseline(self, results: SuiteResult) -> None:
        """Save results as the new baseline."""
        self.history_path.mkdir(parents=True, exist_ok=True)
        baseline_path = self.history_path / "baseline.json"
        results.save(baseline_path)

        # Also save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.history_path / f"results_{timestamp}.json"
        results.save(history_path)

    def check(
        self,
        current: SuiteResult,
        baseline: SuiteResult | None = None,
    ) -> list[BenchmarkComparison]:
        """Check for regressions against baseline.

        Args:
            current: Current results to check
            baseline: Optional baseline (loads from disk if not provided)

        Returns:
            List of comparisons with regressions
        """
        if baseline is None:
            baseline = self.get_baseline()

        if baseline is None:
            return []  # No baseline to compare against

        comparisons = self.comparator.compare_suites(
            baseline,
            current,
            self.thresholds,
        )

        # Return only regressions
        return [c for c in comparisons if c.has_regression]

    def check_thresholds(
        self,
        results: SuiteResult,
    ) -> dict[str, tuple[bool, list[str]]]:
        """Check results against defined thresholds.

        Args:
            results: Results to check

        Returns:
            Dict of benchmark_name -> (passed, violations)
        """
        checks = {}

        for result in results.results:
            threshold = self.thresholds.get(result.benchmark_name)
            if threshold:
                passed, violations = threshold.check(result)
                checks[result.benchmark_name] = (passed, violations)

        return checks

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold."""
        self.thresholds[threshold.benchmark_name] = threshold

    def generate_report(
        self,
        current: SuiteResult,
        baseline: SuiteResult | None = None,
    ) -> str:
        """Generate a regression report.

        Args:
            current: Current results
            baseline: Optional baseline

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("REGRESSION DETECTION REPORT")
        lines.append("=" * 60)
        lines.append("")

        if baseline is None:
            baseline = self.get_baseline()

        if baseline is None:
            lines.append("No baseline available for comparison.")
            lines.append("Run with --save-baseline to establish a baseline.")
            return "\n".join(lines)

        comparisons = self.comparator.compare_suites(
            baseline,
            current,
            self.thresholds,
        )

        regressions = [c for c in comparisons if c.has_regression]
        improvements = [c for c in comparisons if c.has_improvement]

        lines.append(f"Compared against baseline from: {baseline.started_at}")
        lines.append(f"Total benchmarks compared: {len(comparisons)}")
        lines.append(f"Regressions: {len(regressions)}")
        lines.append(f"Improvements: {len(improvements)}")
        lines.append("")

        if regressions:
            lines.append("REGRESSIONS DETECTED:")
            lines.append("-" * 40)
            for comparison in regressions:
                lines.append(f"\n  {comparison.benchmark_name}")
                for mc in comparison.metric_comparisons:
                    if mc.result == ComparisonResult.REGRESSION:
                        lines.append(
                            f"    {mc.name}: {mc.baseline_value:.4f} -> "
                            f"{mc.current_value:.4f} ({mc.percent_change:+.1f}%)"
                        )
                for violation in comparison.threshold_violations:
                    lines.append(f"    ⚠ {violation}")
            lines.append("")

        if improvements:
            lines.append("IMPROVEMENTS:")
            lines.append("-" * 40)
            for comparison in improvements:
                lines.append(f"\n  {comparison.benchmark_name}")
                for mc in comparison.metric_comparisons:
                    if mc.result == ComparisonResult.BETTER:
                        lines.append(
                            f"    {mc.name}: {mc.baseline_value:.4f} -> "
                            f"{mc.current_value:.4f} ({mc.percent_change:+.1f}%)"
                        )
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
