"""Benchmark scenarios for Truthound operations.

This module provides concrete benchmark implementations for:
- Profiling operations
- Validation (check) operations
- Drift detection
- Schema learning
- PII scanning
- End-to-end workflows

Each benchmark is designed to be configurable and produce
consistent, comparable results.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import polars as pl

import truthound as th
from truthound.benchmark.base import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkSize,
    MetricUnit,
    MetricValue,
    register_benchmark,
)
from truthound.benchmark.generators import (
    DataGenerator,
    GeneratorConfig,
    TabularDataGenerator,
    TimeSeriesDataGenerator,
    FinancialDataGenerator,
    TextDataGenerator,
)


# =============================================================================
# Profiling Benchmark
# =============================================================================


@register_benchmark
class ProfileBenchmark(Benchmark):
    """Benchmark for th.profile() operation.

    Measures the performance of statistical profiling including:
    - Column type inference
    - Distribution statistics
    - Pattern detection
    - Memory estimation
    """

    name = "profile"
    category = BenchmarkCategory.PROFILING
    description = "Statistical profiling performance"

    def __init__(
        self,
        generator: DataGenerator | None = None,
    ):
        super().__init__()
        self._generator = generator or TabularDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0
        self._bytes_processed: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        # Get row count from kwargs or config
        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        # Generate data
        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)
        self._bytes_processed = self._df.estimated_size()

    def run_iteration(self) -> Any:
        """Profile the generated data."""
        return th.profile(self._df)

    def validate_result(self, result: Any) -> bool:
        """Validate that profiling produced valid output."""
        return result is not None and hasattr(result, "columns")

    def teardown(self) -> None:
        """Clean up data."""
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def execute(
        self,
        config: BenchmarkConfig | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Execute with proper metrics capture."""
        result = super().execute(config, **kwargs)
        # Set throughput metrics after execution
        result.metrics.rows_processed = self._row_count
        result.metrics.bytes_processed = self._bytes_processed
        return result

    def get_metrics(self) -> BenchmarkMetrics:
        """Get metrics with throughput info."""
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        metrics.bytes_processed = self._bytes_processed

        # Add custom metrics
        metrics.add_custom_metric(MetricValue(
            name="rows_per_second",
            value=metrics.rows_per_second,
            unit=MetricUnit.ROWS_PER_SECOND,
            is_lower_better=False,
            description="Profiling throughput",
        ))

        return metrics


# =============================================================================
# Validation Benchmark
# =============================================================================


@register_benchmark
class ValidationBenchmark(Benchmark):
    """Benchmark for th.check() operation.

    Measures validation performance with a representative set of validators.
    Uses core validators by default for reasonable benchmark times.

    Note:
        For full validator suite benchmarking, use validators=None explicitly.
        Default uses 5 core validators for ~10x faster execution.
    """

    name = "check"
    category = BenchmarkCategory.VALIDATION
    description = "Data quality validation performance"

    # Core validators that cover main categories (fast execution)
    # Note: regex excluded as it requires a pattern parameter
    CORE_VALIDATORS = ["null", "duplicate", "range", "type", "unique"]

    def __init__(
        self,
        generator: DataGenerator | None = None,
    ):
        super().__init__()
        self._generator = generator or TabularDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0
        self._validators: list[str] | None = None

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        # Use core validators by default for speed
        # Pass validators=None explicitly for full suite
        self._validators = kwargs.get("validators", self.CORE_VALIDATORS)

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
            null_ratio=kwargs.get("null_ratio", 0.05),
            duplicate_ratio=kwargs.get("duplicate_ratio", 0.01),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        """Run validation on the data."""
        return th.check(self._df, validators=self._validators)

    def validate_result(self, result: Any) -> bool:
        """Validate that check produced valid output."""
        return result is not None and hasattr(result, "issues")

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        return metrics


# =============================================================================
# Drift Detection Benchmark
# =============================================================================


@register_benchmark
class DriftBenchmark(Benchmark):
    """Benchmark for th.compare() drift detection.

    Measures drift detection performance including:
    - Distribution comparison
    - Statistical tests
    - Sampling effectiveness
    """

    name = "compare"
    category = BenchmarkCategory.DRIFT
    description = "Drift detection performance"

    def __init__(
        self,
        generator: DataGenerator | None = None,
    ):
        super().__init__()
        self._generator = generator or TabularDataGenerator()
        self._baseline: pl.DataFrame | None = None
        self._current: pl.DataFrame | None = None
        self._row_count: int = 0
        self._sample_size: int | None = None

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count
        self._sample_size = kwargs.get("sample_size")
        drift_magnitude = kwargs.get("drift_magnitude", 0.1)

        # Generate baseline
        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=42,
        )
        self._baseline = self._generator.generate(gen_config)

        # Generate current with drift
        gen_config_drifted = GeneratorConfig(
            row_count=row_count,
            seed=43,  # Different seed
        )
        self._current = self._generator.generate(gen_config_drifted)

    def run_iteration(self) -> Any:
        """Run drift detection."""
        return th.compare(
            self._baseline,
            self._current,
            sample_size=self._sample_size,
        )

    def validate_result(self, result: Any) -> bool:
        return result is not None

    def teardown(self) -> None:
        if self._baseline is not None:
            del self._baseline
            self._baseline = None
        if self._current is not None:
            del self._current
            self._current = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count * 2  # Both datasets
        return metrics


# =============================================================================
# Schema Learning Benchmark
# =============================================================================


@register_benchmark
class LearnBenchmark(Benchmark):
    """Benchmark for th.learn() schema learning.

    Measures schema inference performance including:
    - Type detection
    - Constraint learning
    - Value set extraction
    """

    name = "learn"
    category = BenchmarkCategory.LEARNING
    description = "Schema learning performance"

    def __init__(
        self,
        generator: DataGenerator | None = None,
    ):
        super().__init__()
        self._generator = generator or TabularDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        """Learn schema from data."""
        return th.learn(self._df)

    def validate_result(self, result: Any) -> bool:
        return result is not None and hasattr(result, "columns")

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        return metrics


# =============================================================================
# PII Scanning Benchmark
# =============================================================================


@register_benchmark
class ScanBenchmark(Benchmark):
    """Benchmark for th.scan() PII detection.

    Measures PII scanning performance including:
    - Pattern matching
    - Classification
    - Reporting
    """

    name = "scan"
    category = BenchmarkCategory.SCANNING
    description = "PII scanning performance"

    def __init__(self):
        super().__init__()
        self._generator = TextDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        """Scan for PII."""
        return th.scan(self._df)

    def validate_result(self, result: Any) -> bool:
        return result is not None and hasattr(result, "findings")

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        return metrics


# =============================================================================
# End-to-End Benchmark
# =============================================================================


@register_benchmark
class E2EBenchmark(Benchmark):
    """End-to-end benchmark running full workflow.

    Measures combined performance of:
    - Learn schema
    - Check with schema
    - Profile
    - Scan for PII
    """

    name = "e2e"
    category = BenchmarkCategory.E2E
    description = "End-to-end workflow performance"

    def __init__(self):
        super().__init__()
        self._generator = TextDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        """Run full workflow."""
        # Learn schema
        schema = th.learn(self._df)

        # Validate with schema
        report = th.check(self._df, schema=schema)

        # Profile
        profile = th.profile(self._df)

        # Scan
        pii_report = th.scan(self._df)

        return {
            "schema": schema,
            "report": report,
            "profile": profile,
            "pii_report": pii_report,
        }

    def validate_result(self, result: Any) -> bool:
        return (
            result is not None
            and all(k in result for k in ["schema", "report", "profile", "pii_report"])
        )

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        return metrics


# =============================================================================
# Specialized Benchmarks
# =============================================================================


@register_benchmark
class FinancialProfileBenchmark(Benchmark):
    """Benchmark profiling financial/OHLCV data.

    Tests performance on time-series financial data with
    specific patterns and distributions.
    """

    name = "profile_financial"
    category = BenchmarkCategory.PROFILING
    description = "Financial data profiling performance"

    def __init__(self):
        super().__init__()
        self._generator = FinancialDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        return th.profile(self._df)

    def validate_result(self, result: Any) -> bool:
        return result is not None

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count
        return metrics


@register_benchmark
class WideTableBenchmark(Benchmark):
    """Benchmark for wide tables (many columns).

    Tests scaling behavior with increasing column count.
    """

    name = "profile_wide"
    category = BenchmarkCategory.PROFILING
    description = "Wide table (many columns) performance"

    def __init__(self):
        super().__init__()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0
        self._column_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", 10_000)
        column_count = kwargs.get("column_count", 100)
        self._row_count = row_count
        self._column_count = column_count

        # Generate wide table
        data = {
            f"col_{i}": pl.arange(0, row_count, eager=True)
            for i in range(column_count)
        }
        self._df = pl.DataFrame(data)

    def run_iteration(self) -> Any:
        return th.profile(self._df)

    def validate_result(self, result: Any) -> bool:
        return result is not None

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count

        # Add column-specific metric
        metrics.add_custom_metric(MetricValue(
            name="column_count",
            value=self._column_count,
            unit=MetricUnit.COUNT,
            is_lower_better=False,
            description="Number of columns processed",
        ))

        return metrics


@register_benchmark
class ThroughputBenchmark(Benchmark):
    """Benchmark measuring maximum throughput.

    Measures single-operation throughput with lightweight validators
    to estimate sustained throughput capacity.

    Note:
        Previous implementation ran 100 operations per iteration which
        caused extremely long benchmark times. Now measures single operation
        throughput with minimal validators for realistic performance metrics.
    """

    name = "throughput"
    category = BenchmarkCategory.THROUGHPUT
    description = "Maximum throughput measurement"

    def __init__(self):
        super().__init__()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0
        # Use lightweight validators for throughput measurement
        self._validators: list[str] = ["null", "duplicate"]

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", 10_000)
        self._row_count = row_count

        # Use Polars native generation for speed
        self._df = pl.DataFrame({
            "id": pl.arange(0, row_count, eager=True),
            "value": pl.arange(0, row_count, eager=True).cast(pl.Float64),
            "category": pl.Series(["A", "B", "C"] * (row_count // 3 + 1)).head(row_count),
        })

    def run_iteration(self) -> Any:
        """Run single validation with minimal validators."""
        return th.check(self._df, validators=self._validators)

    def validate_result(self, result: Any) -> bool:
        return result is not None and hasattr(result, "issues")

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count

        # Calculate rows per second as throughput metric
        if metrics.mean_duration > 0:
            rows_per_sec = self._row_count / metrics.mean_duration
            metrics.add_custom_metric(MetricValue(
                name="rows_per_second",
                value=rows_per_sec,
                unit=MetricUnit.ROWS_PER_SECOND,
                is_lower_better=False,
                description="Validation throughput (rows/s)",
            ))

        return metrics


@register_benchmark
class MemoryBenchmark(Benchmark):
    """Benchmark measuring memory efficiency.

    Measures memory usage patterns during processing.
    """

    name = "memory"
    category = BenchmarkCategory.MEMORY
    description = "Memory efficiency measurement"

    def __init__(self):
        super().__init__()
        self._generator = TabularDataGenerator()
        self._df: pl.DataFrame | None = None
        self._row_count: int = 0

    def setup(self, config: BenchmarkConfig, **kwargs: Any) -> None:
        super().setup(config, **kwargs)

        row_count = kwargs.get("row_count", config.default_size.row_count)
        self._row_count = row_count

        gen_config = GeneratorConfig(
            row_count=row_count,
            seed=kwargs.get("seed", 42),
        )
        self._df = self._generator.generate(gen_config)

    def run_iteration(self) -> Any:
        """Run memory-intensive operations."""
        # Profile (creates intermediate structures)
        profile = th.profile(self._df)

        # Learn (creates schema objects)
        schema = th.learn(self._df)

        # Check (creates report objects)
        report = th.check(self._df, schema=schema)

        return {"profile": profile, "schema": schema, "report": report}

    def validate_result(self, result: Any) -> bool:
        return result is not None and all(
            k in result for k in ["profile", "schema", "report"]
        )

    def teardown(self) -> None:
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()

    def get_metrics(self) -> BenchmarkMetrics:
        metrics = super().get_metrics()
        metrics.rows_processed = self._row_count

        # Add memory efficiency metric
        if metrics.peak_memory_bytes > 0 and self._row_count > 0:
            bytes_per_row = metrics.peak_memory_bytes / self._row_count
            metrics.add_custom_metric(MetricValue(
                name="peak_bytes_per_row",
                value=bytes_per_row,
                unit=MetricUnit.BYTES,
                is_lower_better=True,
                description="Peak memory per row",
            ))

        return metrics
