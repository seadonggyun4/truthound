"""Native benchmark controls must affect execution and preserve failed measurements."""

import time

import pytest

from truthound.benchmark import Benchmark, BenchmarkConfig


class SlowBenchmark(Benchmark):
    name = "synthetic-slow"

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.closed = False

    def run_iteration(self):
        self.calls += 1
        time.sleep(0.02)
        return 1

    def teardown(self):
        self.closed = True


def test_native_benchmark_deadline_stops_later_iterations_and_runs_teardown():
    benchmark = SlowBenchmark()
    result = benchmark.execute(
        BenchmarkConfig(
            warmup_iterations=0, measure_iterations=3, cooldown_seconds=0, timeout_seconds=0.01
        )
    )
    assert not result.success
    assert benchmark.calls == 1 and benchmark.closed
    assert "time" in result.error.lower()


def test_native_benchmark_memory_budget_is_enforced_before_work():
    benchmark = SlowBenchmark()
    result = benchmark.execute(
        BenchmarkConfig(
            warmup_iterations=0, measure_iterations=1, cooldown_seconds=0, max_memory_mb=1
        )
    )
    assert not result.success
    assert benchmark.calls == 0 and benchmark.closed
    assert "memory" in result.error.lower()


@pytest.mark.parametrize(
    "settings",
    [
        {"measure_iterations": 0},
        {"warmup_iterations": -1},
        {"timeout_seconds": float("nan")},
        {"max_memory_mb": -1},
        {"cooldown_seconds": True},
    ],
)
def test_invalid_benchmark_configuration_rejected_before_execution(settings):
    with pytest.raises(ValueError):
        BenchmarkConfig(**settings)


def test_failed_native_measurement_cannot_pass_a_performance_threshold():
    from truthound.benchmark.comparison import PerformanceThreshold

    result = SlowBenchmark().execute(
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, max_memory_mb=1)
    )
    passed, violations = PerformanceThreshold(benchmark_name=result.benchmark_name).check(result)
    assert not passed and violations


def test_failed_measurement_cannot_be_reported_as_performance_improvement():
    from truthound.benchmark.comparison import BenchmarkComparator

    benchmark = SlowBenchmark()
    good = benchmark.execute(
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0)
    )
    failed = benchmark.execute(
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, max_memory_mb=1)
    )
    with pytest.raises(ValueError, match="successful measurements"):
        BenchmarkComparator().compare(good, failed)


def test_relative_threshold_requires_a_successful_matching_baseline():
    from truthound.benchmark.comparison import PerformanceThreshold

    result = SlowBenchmark().execute(
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0)
    )
    threshold = PerformanceThreshold(result.benchmark_name, max_regression_percent=10)
    passed, violations = threshold.check(result)
    assert not passed and violations
    assert threshold.check(result, result) == (True, [])


def test_suite_settings_apply_without_explicit_runner_override_and_are_captured():
    from truthound.benchmark import (
        BenchmarkRunner,
        BenchmarkSuite,
        RunnerConfig,
        benchmark_registry,
    )

    benchmark_registry.register(SlowBenchmark)
    try:
        settings = BenchmarkConfig(warmup_iterations=0, measure_iterations=2, cooldown_seconds=0)
        suite = BenchmarkSuite("synthetic", benchmark_config=settings).add(SlowBenchmark.name)
        settings.measure_iterations = 9
        result = BenchmarkRunner(RunnerConfig(save_results=False)).run_suite(suite)
        assert len(result.results[0].metrics.durations) == 2
        override = BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0)
        runner = BenchmarkRunner(RunnerConfig(save_results=False), override)
        override.measure_iterations = 7
        assert len(runner.run_suite(suite).results[0].metrics.durations) == 1
    finally:
        benchmark_registry.unregister(SlowBenchmark.name)


@pytest.mark.parametrize(
    "settings",
    [
        {"max_duration_seconds": float("nan")},
        {"max_regression_percent": -1},
        {"max_memory_mb": True},
    ],
)
def test_invalid_threshold_configuration_cannot_disable_a_gate(settings):
    from truthound.benchmark.comparison import PerformanceThreshold

    with pytest.raises(ValueError):
        PerformanceThreshold("synthetic", **settings)


def test_positive_duration_cannot_pass_a_zero_duration_relative_baseline():
    from dataclasses import replace

    from truthound.benchmark import BenchmarkMetrics
    from truthound.benchmark.comparison import PerformanceThreshold

    current = SlowBenchmark().execute(
        BenchmarkConfig(warmup_iterations=0, measure_iterations=1, cooldown_seconds=0)
    )
    baseline = replace(current, metrics=BenchmarkMetrics(durations=[0.0]))
    assert (
        PerformanceThreshold(current.benchmark_name, max_regression_percent=10).check(
            current, baseline
        )[0]
        is False
    )


@pytest.mark.parametrize(
    "options",
    [
        {"significance_threshold": float("nan")},
        {"regression_threshold": float("inf")},
        {"significance_threshold": -0.1},
        {"regression_threshold": True},
    ],
)
def test_comparator_rejects_invalid_decision_bounds(options):
    from truthound.benchmark.comparison import BenchmarkComparator

    with pytest.raises(ValueError):
        BenchmarkComparator(**options)


@pytest.mark.parametrize("names", [("missing",), ("same", "same"), ()])
def test_suite_comparison_cannot_silently_drop_or_duplicate_measurements(names):
    from truthound.benchmark.base import BenchmarkMetrics, BenchmarkResult
    from truthound.benchmark.comparison import BenchmarkComparator
    from truthound.benchmark.runner import SuiteResult

    baseline = SuiteResult(
        suite_name="baseline",
        environment=None,
        results=[
            BenchmarkResult(
                benchmark_name="same",
                category="synthetic",
                success=True,
                environment=None,
                metrics=BenchmarkMetrics(durations=[1.0]),
            )
        ],
    )
    current = SuiteResult(
        suite_name="current",
        environment=None,
        results=[
            BenchmarkResult(
                benchmark_name=name,
                category="synthetic",
                success=True,
                environment=None,
                metrics=BenchmarkMetrics(durations=[1.1]),
            )
            for name in names
        ],
    )
    with pytest.raises(ValueError):
        BenchmarkComparator().compare_suites(baseline, current)
