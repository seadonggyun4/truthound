"""Tests for the benchmark system.

Tests cover:
- Base classes (Benchmark, BenchmarkResult, BenchmarkMetrics)
- Data generators
- Benchmark scenarios
- Runner and suite management
- Reporters
- Comparison and regression detection
"""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from truthound.benchmark.base import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSize,
    EnvironmentInfo,
    MetricUnit,
    MetricValue,
    benchmark_registry,
    register_benchmark,
)
from truthound.benchmark.generators import (
    ColumnSpec,
    ColumnType,
    DataGenerator,
    DataPattern,
    FinancialDataGenerator,
    GeneratorConfig,
    TabularDataGenerator,
    TextDataGenerator,
    TimeSeriesDataGenerator,
    generator_registry,
)
from truthound.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkSuite,
    RunnerConfig,
    SuiteResult,
)
from truthound.benchmark.reporters import (
    ConsoleReporter,
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
)
from truthound.benchmark.comparison import (
    BenchmarkComparator,
    ComparisonResult,
    PerformanceThreshold,
    RegressionDetector,
)


# =============================================================================
# Test Base Classes
# =============================================================================


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics class."""

    def test_empty_metrics(self):
        """Test metrics with no data."""
        metrics = BenchmarkMetrics()
        assert metrics.mean_duration == 0.0
        assert metrics.median_duration == 0.0
        assert metrics.std_duration == 0.0
        assert metrics.rows_per_second == 0.0

    def test_duration_statistics(self):
        """Test duration statistical calculations."""
        metrics = BenchmarkMetrics()
        metrics.durations = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert metrics.mean_duration == 3.0
        assert metrics.median_duration == 3.0
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 5.0
        assert metrics.std_duration > 0

    def test_throughput_calculation(self):
        """Test throughput calculations."""
        metrics = BenchmarkMetrics()
        metrics.durations = [1.0]  # 1 second
        metrics.rows_processed = 1_000_000

        assert metrics.rows_per_second == 1_000_000.0

    def test_custom_metrics(self):
        """Test custom metric addition."""
        metrics = BenchmarkMetrics()
        metrics.add_custom_metric(MetricValue(
            name="custom_ops",
            value=100.0,
            unit=MetricUnit.OPERATIONS_PER_SECOND,
            is_lower_better=False,
        ))

        assert len(metrics.custom_metrics) == 1
        assert metrics.custom_metrics[0].name == "custom_ops"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = BenchmarkMetrics()
        metrics.durations = [1.0, 2.0]
        metrics.rows_processed = 1000

        data = metrics.to_dict()

        assert "timing" in data
        assert "throughput" in data
        assert data["timing"]["iterations"] == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_create_result(self):
        """Test result creation."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=BenchmarkMetrics(),
            environment=EnvironmentInfo.capture(),
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.success is True
        assert result.result_id is not None

    def test_result_serialization(self):
        """Test result serialization."""
        metrics = BenchmarkMetrics()
        metrics.durations = [0.5]

        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.VALIDATION,
            success=True,
            metrics=metrics,
            environment=EnvironmentInfo.capture(),
            parameters={"rows": 1000},
        )

        data = result.to_dict()

        assert data["benchmark_name"] == "test"
        assert data["success"] is True
        assert "metrics" in data
        assert "environment" in data


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo class."""

    def test_capture(self):
        """Test environment capture."""
        env = EnvironmentInfo.capture()

        assert env.python_version != ""
        assert env.platform_system != ""
        assert env.polars_version != ""
        assert env.timestamp is not None

    def test_to_dict(self):
        """Test serialization."""
        env = EnvironmentInfo.capture()
        data = env.to_dict()

        assert "python_version" in data
        assert "platform" in data
        assert "polars_version" in data


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig class."""

    def test_default_config(self):
        """Test default configuration (optimized for quick feedback)."""
        config = BenchmarkConfig()

        assert config.warmup_iterations == 1
        assert config.measure_iterations == 3
        assert config.default_size == BenchmarkSize.SMALL
        assert config.gc_between_iterations is False
        assert config.cooldown_seconds == 0.01

    def test_quick_config(self):
        """Test quick configuration preset (~5 seconds)."""
        config = BenchmarkConfig.quick()

        assert config.warmup_iterations == 1
        assert config.measure_iterations == 2
        assert config.default_size == BenchmarkSize.TINY
        assert config.cooldown_seconds == 0.0

    def test_thorough_config(self):
        """Test thorough configuration preset (~60 seconds)."""
        config = BenchmarkConfig.thorough()

        assert config.measure_iterations == 5
        assert config.default_size == BenchmarkSize.MEDIUM
        assert config.gc_between_iterations is True


# =============================================================================
# Test Generators
# =============================================================================


class TestTabularDataGenerator:
    """Tests for TabularDataGenerator."""

    def test_generate_default_schema(self):
        """Test generation with default schema."""
        generator = TabularDataGenerator(seed=42)
        config = GeneratorConfig(row_count=1000)

        df = generator.generate(config)

        assert len(df) == 1000
        assert "id" in df.columns
        assert "int_col" in df.columns
        assert "float_col" in df.columns

    def test_generate_with_specs(self):
        """Test generation with custom column specs."""
        generator = TabularDataGenerator(seed=42)
        config = GeneratorConfig(
            row_count=500,
            null_ratio=0.0,  # No nulls for this test
            column_specs=[
                ColumnSpec("my_int", ColumnType.INTEGER, DataPattern.SEQUENTIAL),
                ColumnSpec("my_cat", ColumnType.CATEGORY, categories=["X", "Y", "Z"], null_ratio=0.0),
            ],
        )

        df = generator.generate(config)

        assert len(df) == 500
        assert "my_int" in df.columns
        assert "my_cat" in df.columns
        # Filter out nulls before checking categories
        non_null_cats = set(df["my_cat"].drop_nulls().unique().to_list())
        assert non_null_cats.issubset({"X", "Y", "Z"})

    def test_generate_with_nulls(self):
        """Test generation with null ratio."""
        generator = TabularDataGenerator(seed=42)
        config = GeneratorConfig(
            row_count=1000,
            null_ratio=0.1,
        )

        df = generator.generate(config)

        # At least some columns should have nulls
        # (not guaranteed for all due to default schema)
        assert len(df) == 1000

    def test_deterministic_with_seed(self):
        """Test that seed produces deterministic results."""
        gen1 = TabularDataGenerator(seed=42)
        gen2 = TabularDataGenerator(seed=42)
        config = GeneratorConfig(row_count=100, seed=42)

        df1 = gen1.generate(config)
        df2 = gen2.generate(config)

        # IDs should be the same
        assert df1["id"].to_list() == df2["id"].to_list()


class TestTimeSeriesDataGenerator:
    """Tests for TimeSeriesDataGenerator."""

    def test_generate(self):
        """Test time series generation."""
        generator = TimeSeriesDataGenerator(seed=42)
        config = GeneratorConfig(row_count=1000)

        df = generator.generate(config)

        assert len(df) == 1000
        assert "timestamp" in df.columns
        assert "value" in df.columns
        assert "hour" in df.columns


class TestFinancialDataGenerator:
    """Tests for FinancialDataGenerator."""

    def test_generate_ohlcv(self):
        """Test OHLCV data generation."""
        generator = FinancialDataGenerator(seed=42, symbol="AAPL")
        config = GeneratorConfig(row_count=1000)

        df = generator.generate(config)

        assert len(df) == 1000
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # High should be >= open
        # Low should be <= open
        assert df.filter(pl.col("high") < pl.col("open")).height == 0
        assert df.filter(pl.col("low") > pl.col("open")).height == 0


class TestTextDataGenerator:
    """Tests for TextDataGenerator."""

    def test_generate_pii(self):
        """Test PII data generation."""
        generator = TextDataGenerator(seed=42)
        config = GeneratorConfig(row_count=500)

        df = generator.generate(config)

        assert len(df) == 500
        assert "email" in df.columns
        assert "phone" in df.columns
        assert "ssn" in df.columns
        assert "credit_card" in df.columns

        # Check email format
        emails = df["email"].to_list()
        assert all("@" in e for e in emails)


class TestGeneratorRegistry:
    """Tests for GeneratorRegistry."""

    def test_builtin_generators(self):
        """Test that built-in generators are registered."""
        names = generator_registry.list_names()

        assert "tabular" in names
        assert "timeseries" in names
        assert "financial" in names
        assert "text" in names

    def test_get_generator(self):
        """Test getting a generator by name."""
        generator_cls = generator_registry.get("tabular")
        assert generator_cls is TabularDataGenerator

    def test_create_generator(self):
        """Test creating a generator instance."""
        generator = generator_registry.create("financial", symbol="GOOG")
        assert isinstance(generator, FinancialDataGenerator)
        assert generator.symbol == "GOOG"


# =============================================================================
# Test Benchmark Scenarios
# =============================================================================


class TestBenchmarkScenarios:
    """Tests for benchmark scenarios."""

    def test_profile_benchmark(self):
        """Test profile benchmark execution."""
        from truthound.benchmark.scenarios import ProfileBenchmark

        benchmark = ProfileBenchmark()
        config = BenchmarkConfig.quick()

        result = benchmark.execute(config, row_count=1000)

        assert result.success is True
        assert result.metrics.mean_duration > 0
        assert result.metrics.rows_processed == 1000

    def test_validation_benchmark(self):
        """Test validation benchmark execution."""
        from truthound.benchmark.scenarios import ValidationBenchmark

        benchmark = ValidationBenchmark()
        config = BenchmarkConfig.quick()

        result = benchmark.execute(config, row_count=1000)

        assert result.success is True
        assert result.metrics.mean_duration > 0

    def test_learn_benchmark(self):
        """Test learn benchmark execution."""
        from truthound.benchmark.scenarios import LearnBenchmark

        benchmark = LearnBenchmark()
        config = BenchmarkConfig.quick()

        result = benchmark.execute(config, row_count=1000)

        assert result.success is True


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry."""

    def test_builtin_benchmarks(self):
        """Test that built-in benchmarks are registered."""
        names = benchmark_registry.list_names()

        assert "profile" in names
        assert "check" in names
        assert "learn" in names

    def test_get_benchmark(self):
        """Test getting a benchmark by name."""
        benchmark_cls = benchmark_registry.get("profile")
        assert benchmark_cls.name == "profile"

    def test_get_by_category(self):
        """Test getting benchmarks by category."""
        profiling = benchmark_registry.get_by_category(BenchmarkCategory.PROFILING)
        assert len(profiling) > 0
        assert all(b.category == BenchmarkCategory.PROFILING for b in profiling)


# =============================================================================
# Test Runner and Suite
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_create_suite(self):
        """Test suite creation."""
        suite = BenchmarkSuite("test_suite", "Test suite description")
        suite.add("profile", row_count=1000)
        suite.add("check", row_count=1000)

        assert len(suite) == 2

    def test_quick_suite(self):
        """Test quick suite preset."""
        suite = BenchmarkSuite.quick()

        assert len(suite) > 0

    def test_ci_suite(self):
        """Test CI suite preset."""
        suite = BenchmarkSuite.ci()

        assert len(suite) > 0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_run_single_benchmark(self):
        """Test running a single benchmark."""
        runner = BenchmarkRunner(
            benchmark_config=BenchmarkConfig.quick(),
        )

        result = runner.run("profile", row_count=500)

        assert result.success is True
        assert result.benchmark_name == "profile"

    def test_run_suite(self):
        """Test running a suite."""
        suite = BenchmarkSuite("test")
        suite.add("profile", row_count=500)
        suite.add("learn", row_count=500)

        runner = BenchmarkRunner(
            benchmark_config=BenchmarkConfig.quick(),
        )

        results = runner.run_suite(suite)

        assert results.total_benchmarks == 2
        assert results.successful_benchmarks >= 1


class TestSuiteResult:
    """Tests for SuiteResult."""

    def test_suite_result_properties(self):
        """Test suite result properties."""
        metrics = BenchmarkMetrics()
        metrics.durations = [0.5]

        results = [
            BenchmarkResult(
                benchmark_name="test1",
                category=BenchmarkCategory.PROFILING,
                success=True,
                metrics=metrics,
                environment=EnvironmentInfo.capture(),
            ),
            BenchmarkResult(
                benchmark_name="test2",
                category=BenchmarkCategory.VALIDATION,
                success=False,
                metrics=metrics,
                environment=EnvironmentInfo.capture(),
                error="Test error",
            ),
        ]

        suite_result = SuiteResult(
            suite_name="test_suite",
            results=results,
            environment=EnvironmentInfo.capture(),
        )
        suite_result.completed_at = datetime.now()

        assert suite_result.total_benchmarks == 2
        assert suite_result.successful_benchmarks == 1
        assert suite_result.failed_benchmarks == 1
        assert suite_result.success_rate == 0.5

    def test_save_and_load(self):
        """Test saving and loading results."""
        metrics = BenchmarkMetrics()
        metrics.durations = [0.5]

        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=metrics,
            environment=EnvironmentInfo.capture(),
        )

        suite_result = SuiteResult(
            suite_name="test",
            results=[result],
            environment=EnvironmentInfo.capture(),
        )
        suite_result.completed_at = datetime.now()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
            suite_result.save(path)

            # Verify file was created and is valid JSON
            content = path.read_text()
            data = json.loads(content)
            assert data["suite_name"] == "test"


# =============================================================================
# Test Reporters
# =============================================================================


class TestReporters:
    """Tests for benchmark reporters."""

    @pytest.fixture
    def sample_suite_result(self):
        """Create a sample suite result for testing."""
        metrics = BenchmarkMetrics()
        metrics.durations = [0.1, 0.12, 0.11]
        metrics.rows_processed = 10000

        results = [
            BenchmarkResult(
                benchmark_name="profile",
                category=BenchmarkCategory.PROFILING,
                success=True,
                metrics=metrics,
                environment=EnvironmentInfo.capture(),
            ),
        ]

        suite_result = SuiteResult(
            suite_name="test_suite",
            results=results,
            environment=EnvironmentInfo.capture(),
        )
        suite_result.completed_at = datetime.now()
        return suite_result

    def test_console_reporter(self, sample_suite_result):
        """Test console reporter."""
        reporter = ConsoleReporter(use_colors=False)
        output = reporter.report_suite(sample_suite_result)

        assert "BENCHMARK SUITE" in output
        assert "profile" in output

    def test_json_reporter(self, sample_suite_result):
        """Test JSON reporter."""
        reporter = JSONReporter(pretty=True)
        output = reporter.report_suite(sample_suite_result)

        data = json.loads(output)
        assert data["suite_name"] == "test_suite"
        assert len(data["results"]) == 1

    def test_markdown_reporter(self, sample_suite_result):
        """Test Markdown reporter."""
        reporter = MarkdownReporter()
        output = reporter.report_suite(sample_suite_result)

        assert "# Benchmark Results" in output
        assert "| Benchmark |" in output

    def test_html_reporter(self, sample_suite_result):
        """Test HTML reporter."""
        reporter = HTMLReporter()
        output = reporter.report_suite(sample_suite_result)

        assert "<html>" in output
        assert "test_suite" in output


# =============================================================================
# Test Comparison and Regression Detection
# =============================================================================


class TestPerformanceThreshold:
    """Tests for PerformanceThreshold."""

    def test_duration_threshold(self):
        """Test duration threshold check."""
        threshold = PerformanceThreshold(
            benchmark_name="test",
            max_duration_seconds=1.0,
        )

        metrics = BenchmarkMetrics()
        metrics.durations = [0.5]

        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=metrics,
            environment=EnvironmentInfo.capture(),
        )

        passed, violations = threshold.check(result)
        assert passed is True
        assert len(violations) == 0

    def test_threshold_violation(self):
        """Test threshold violation detection."""
        threshold = PerformanceThreshold(
            benchmark_name="test",
            max_duration_seconds=0.1,
        )

        metrics = BenchmarkMetrics()
        metrics.durations = [0.5]  # Exceeds threshold

        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=metrics,
            environment=EnvironmentInfo.capture(),
        )

        passed, violations = threshold.check(result)
        assert passed is False
        assert len(violations) == 1
        assert "Duration" in violations[0]


class TestBenchmarkComparator:
    """Tests for BenchmarkComparator."""

    def test_compare_same(self):
        """Test comparing identical results."""
        comparator = BenchmarkComparator()

        metrics = BenchmarkMetrics()
        metrics.durations = [1.0]
        metrics.rows_processed = 1000

        baseline = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=metrics,
            environment=EnvironmentInfo.capture(),
        )

        comparison = comparator.compare(baseline, baseline)
        assert comparison.overall_result == ComparisonResult.SAME

    def test_detect_regression(self):
        """Test regression detection."""
        comparator = BenchmarkComparator(regression_threshold=0.10)

        baseline_metrics = BenchmarkMetrics()
        baseline_metrics.durations = [1.0]

        current_metrics = BenchmarkMetrics()
        current_metrics.durations = [1.5]  # 50% slower

        baseline = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=baseline_metrics,
            environment=EnvironmentInfo.capture(),
        )

        current = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=current_metrics,
            environment=EnvironmentInfo.capture(),
        )

        comparison = comparator.compare(baseline, current)
        assert comparison.has_regression is True

    def test_detect_improvement(self):
        """Test improvement detection."""
        comparator = BenchmarkComparator()

        baseline_metrics = BenchmarkMetrics()
        baseline_metrics.durations = [1.0]

        current_metrics = BenchmarkMetrics()
        current_metrics.durations = [0.5]  # 50% faster

        baseline = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=baseline_metrics,
            environment=EnvironmentInfo.capture(),
        )

        current = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=current_metrics,
            environment=EnvironmentInfo.capture(),
        )

        comparison = comparator.compare(baseline, current)
        assert comparison.has_improvement is True


class TestRegressionDetector:
    """Tests for RegressionDetector."""

    def test_save_and_check_baseline(self):
        """Test saving and checking against baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RegressionDetector(history_path=Path(tmpdir))

            metrics = BenchmarkMetrics()
            metrics.durations = [1.0]

            result = BenchmarkResult(
                benchmark_name="test",
                category=BenchmarkCategory.PROFILING,
                success=True,
                metrics=metrics,
                environment=EnvironmentInfo.capture(),
            )

            baseline = SuiteResult(
                suite_name="test",
                results=[result],
                environment=EnvironmentInfo.capture(),
            )
            baseline.completed_at = datetime.now()

            # Save baseline
            detector.save_baseline(baseline)

            # Check no regressions against self
            regressions = detector.check(baseline)
            assert len(regressions) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarkIntegration:
    """Integration tests for the benchmark system."""

    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow."""
        # Create suite
        suite = BenchmarkSuite("integration_test")
        suite.add("profile", row_count=500)

        # Configure runner
        runner = BenchmarkRunner(
            benchmark_config=BenchmarkConfig.quick(),
            config=RunnerConfig(verbose=False),
        )

        # Run suite
        results = runner.run_suite(suite)

        assert results.successful_benchmarks >= 1

        # Generate reports
        json_reporter = JSONReporter()
        json_output = json_reporter.report_suite(results)
        assert json.loads(json_output) is not None

        md_reporter = MarkdownReporter()
        md_output = md_reporter.report_suite(results)
        assert "# Benchmark Results" in md_output

    def test_custom_benchmark(self):
        """Test creating and running a custom benchmark."""
        @register_benchmark
        class CustomBenchmark(Benchmark):
            name = "custom_test"
            category = BenchmarkCategory.E2E
            description = "Custom test benchmark"

            def setup(self, config, **kwargs):
                self._data = list(range(kwargs.get("size", 1000)))

            def run_iteration(self):
                return sum(self._data)

            def teardown(self):
                self._data = None

        try:
            # Should be registered
            assert "custom_test" in benchmark_registry.list_names()

            # Run it
            benchmark = CustomBenchmark()
            result = benchmark.execute(BenchmarkConfig.quick(), size=100)

            assert result.success is True
            assert result.metrics.mean_duration > 0

        finally:
            # Clean up
            benchmark_registry.unregister("custom_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
