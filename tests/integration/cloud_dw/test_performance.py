"""Performance benchmark tests for Cloud DW backends.

These tests measure and validate the performance of Truthound operations
on cloud data warehouses, including:
    - Query execution time
    - Data transfer performance
    - Validator execution time
    - Pushdown optimization benefits
    - Scalability with data size

All tests are marked as @pytest.mark.expensive as they may incur costs.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import pytest

from tests.integration.cloud_dw.fixtures import SQLDialect, StandardTestData

if TYPE_CHECKING:
    from tests.integration.cloud_dw.base import CloudDWTestBackend, TestDataset


# =============================================================================
# Benchmark Infrastructure
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        name: Benchmark name.
        duration_ms: Total duration in milliseconds.
        iterations: Number of iterations.
        min_ms: Minimum duration.
        max_ms: Maximum duration.
        mean_ms: Mean duration.
        std_ms: Standard deviation.
        p50_ms: 50th percentile (median).
        p90_ms: 90th percentile.
        p99_ms: 99th percentile.
        metadata: Additional benchmark metadata.
    """

    name: str
    duration_ms: float
    iterations: int = 1
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_timings(
        cls,
        name: str,
        timings: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> "BenchmarkResult":
        """Create result from list of timing measurements."""
        if not timings:
            return cls(name=name, duration_ms=0.0, metadata=metadata or {})

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        return cls(
            name=name,
            duration_ms=sum(timings),
            iterations=n,
            min_ms=min(timings),
            max_ms=max(timings),
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if n > 1 else 0.0,
            p50_ms=sorted_timings[int(n * 0.5)],
            p90_ms=sorted_timings[int(n * 0.9)] if n > 1 else sorted_timings[-1],
            p99_ms=sorted_timings[int(n * 0.99)] if n > 1 else sorted_timings[-1],
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "iterations": self.iterations,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "metadata": self.metadata,
        }


class Benchmark:
    """Context manager for benchmarking operations."""

    def __init__(self, name: str, iterations: int = 1):
        self.name = name
        self.iterations = iterations
        self.timings: list[float] = []
        self._start_time: float = 0.0

    def __enter__(self) -> "Benchmark":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self.timings.append(elapsed)

    def run(self, func: Callable[[], Any]) -> Any:
        """Run a function and record timing."""
        start = time.perf_counter()
        result = func()
        elapsed = (time.perf_counter() - start) * 1000
        self.timings.append(elapsed)
        return result

    def get_result(self, metadata: dict[str, Any] | None = None) -> BenchmarkResult:
        """Get benchmark result."""
        return BenchmarkResult.from_timings(
            self.name,
            self.timings,
            metadata,
        )


def get_dialect_for_backend(backend: "CloudDWTestBackend") -> SQLDialect:
    """Get the SQL dialect for a backend."""
    dialect_map = {
        "bigquery": SQLDialect.BIGQUERY,
        "snowflake": SQLDialect.SNOWFLAKE,
        "redshift": SQLDialect.REDSHIFT,
        "databricks": SQLDialect.DATABRICKS,
    }
    return dialect_map.get(backend.platform_name, SQLDialect.BIGQUERY)


# =============================================================================
# Query Performance Tests
# =============================================================================


@pytest.mark.expensive
class TestQueryPerformance:
    """Performance tests for query execution."""

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_simple_query_latency(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Measure latency of simple queries."""
        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "query_perf_test",
            schema,
            data,
        )

        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)

        # Warm up
        any_backend.execute_query(f"SELECT 1")

        # Measure simple SELECT
        benchmark = Benchmark("simple_select", iterations=5)
        for _ in range(5):
            benchmark.run(lambda: any_backend.execute_query(
                f"SELECT COUNT(*) AS cnt FROM {full_name}"
            ))

        result = benchmark.get_result({
            "backend": any_backend.platform_name,
            "row_count": 100,
        })

        # Assert reasonable latency (less than 5 seconds for simple query)
        assert result.mean_ms < 5000, f"Query too slow: {result.mean_ms}ms"

        # Log result for analysis
        print(f"\nBenchmark: {result.name}")
        print(f"  Mean: {result.mean_ms:.2f}ms")
        print(f"  P50: {result.p50_ms:.2f}ms")
        print(f"  P90: {result.p90_ms:.2f}ms")

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_aggregation_query_performance(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Measure performance of aggregation queries."""
        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=500)

        table = any_backend.create_test_table(
            any_dataset,
            "agg_perf_test",
            schema,
            data,
        )

        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)

        # Measure aggregation query
        benchmark = Benchmark("aggregation_query", iterations=3)
        for _ in range(3):
            benchmark.run(lambda: any_backend.execute_query(f"""
                SELECT
                    COUNT(*) AS total_count,
                    AVG(age) AS avg_age,
                    MIN(age) AS min_age,
                    MAX(age) AS max_age,
                    COUNT(DISTINCT email) AS unique_emails
                FROM {full_name}
            """))

        result = benchmark.get_result({
            "backend": any_backend.platform_name,
            "row_count": 500,
        })

        assert result.mean_ms < 10000, f"Aggregation too slow: {result.mean_ms}ms"

        print(f"\nBenchmark: {result.name}")
        print(f"  Mean: {result.mean_ms:.2f}ms")
        print(f"  P90: {result.p90_ms:.2f}ms")


# =============================================================================
# Validator Performance Tests
# =============================================================================


@pytest.mark.expensive
class TestValidatorPerformance:
    """Performance tests for Truthound validators."""

    @pytest.mark.performance
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_not_null_validator_performance(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Measure NotNullValidator performance."""
        import truthound as th
        from truthound.validators import NotNullValidator

        dialect = get_dialect_for_backend(any_backend)

        # Create dataset of varying sizes
        for row_count in [100, 500, 1000]:
            schema = StandardTestData.users_schema(dialect)
            data = StandardTestData.users_data(n=row_count)

            table = any_backend.create_test_table(
                any_dataset,
                f"notnull_perf_{row_count}",
                schema,
                data,
            )

            datasource = any_backend.create_datasource(any_dataset.name, table.name)

            benchmark = Benchmark(f"not_null_{row_count}")
            result = benchmark.run(lambda: th.check(
                datasource,
                validators=[NotNullValidator("id")],
            ))

            perf_result = benchmark.get_result({
                "backend": any_backend.platform_name,
                "row_count": row_count,
                "validator": "NotNullValidator",
            })

            print(f"\nNotNullValidator ({row_count} rows): {perf_result.mean_ms:.2f}ms")

            # Verify validator worked correctly
            assert result.success

    @pytest.mark.performance
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_multiple_validators_performance(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Measure performance of running multiple validators."""
        import truthound as th
        from truthound.validators import (
            NotNullValidator,
            UniqueValidator,
            RangeValidator,
        )

        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=500)

        table = any_backend.create_test_table(
            any_dataset,
            "multi_perf_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        validators = [
            NotNullValidator("id"),
            NotNullValidator("email"),
            NotNullValidator("name"),
            UniqueValidator("id"),
            UniqueValidator("email"),
            RangeValidator("age", min_value=0, max_value=100),
        ]

        benchmark = Benchmark("multiple_validators")
        result = benchmark.run(lambda: th.check(
            datasource,
            validators=validators,
        ))

        perf_result = benchmark.get_result({
            "backend": any_backend.platform_name,
            "row_count": 500,
            "validator_count": len(validators),
        })

        print(f"\n{len(validators)} Validators (500 rows): {perf_result.mean_ms:.2f}ms")
        print(f"  Per validator: {perf_result.mean_ms / len(validators):.2f}ms")

        assert result is not None


# =============================================================================
# Pushdown Performance Tests
# =============================================================================


@pytest.mark.expensive
class TestPushdownPerformance:
    """Performance tests comparing pushdown vs non-pushdown execution."""

    @pytest.mark.performance
    @pytest.mark.truthound
    @pytest.mark.requires_data
    def test_pushdown_speedup(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Compare performance with and without pushdown."""
        import truthound as th
        from truthound.validators import NotNullValidator, UniqueValidator

        dialect = get_dialect_for_backend(any_backend)

        # Use larger dataset to see pushdown benefits
        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=1000)

        table = any_backend.create_test_table(
            any_dataset,
            "pushdown_speedup_test",
            schema,
            data,
        )

        datasource = any_backend.create_datasource(any_dataset.name, table.name)

        validators = [
            NotNullValidator("id"),
            NotNullValidator("email"),
            UniqueValidator("id"),
        ]

        # Run without pushdown
        benchmark_no_pushdown = Benchmark("without_pushdown")
        benchmark_no_pushdown.run(lambda: th.check(
            datasource,
            validators=validators,
            pushdown=False,
        ))
        no_pushdown_result = benchmark_no_pushdown.get_result()

        # Run with pushdown
        benchmark_pushdown = Benchmark("with_pushdown")
        benchmark_pushdown.run(lambda: th.check(
            datasource,
            validators=validators,
            pushdown=True,
        ))
        pushdown_result = benchmark_pushdown.get_result()

        print(f"\nPushdown Performance Comparison (1000 rows):")
        print(f"  Without pushdown: {no_pushdown_result.mean_ms:.2f}ms")
        print(f"  With pushdown: {pushdown_result.mean_ms:.2f}ms")

        if no_pushdown_result.mean_ms > 0:
            speedup = no_pushdown_result.mean_ms / pushdown_result.mean_ms
            print(f"  Speedup: {speedup:.2f}x")

        # Note: Pushdown may not always be faster for small datasets
        # We just verify both modes work


# =============================================================================
# Scalability Tests
# =============================================================================


@pytest.mark.expensive
class TestScalability:
    """Scalability tests for different data sizes."""

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_linear_scalability(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Test that query time scales reasonably with data size."""
        dialect = get_dialect_for_backend(any_backend)

        row_counts = [100, 500, 1000]
        times_ms: list[tuple[int, float]] = []

        for row_count in row_counts:
            schema = StandardTestData.users_schema(dialect)
            data = StandardTestData.users_data(n=row_count)

            table = any_backend.create_test_table(
                any_dataset,
                f"scale_test_{row_count}",
                schema,
                data,
            )

            full_name = any_backend.get_full_table_name(any_dataset.name, table.name)

            # Measure query time
            benchmark = Benchmark(f"scale_{row_count}")
            for _ in range(3):
                benchmark.run(lambda: any_backend.execute_query(f"""
                    SELECT COUNT(*) AS cnt,
                           COUNT(DISTINCT email) AS unique_emails,
                           AVG(age) AS avg_age
                    FROM {full_name}
                """))

            result = benchmark.get_result()
            times_ms.append((row_count, result.mean_ms))

        print("\nScalability Results:")
        for row_count, time_ms in times_ms:
            print(f"  {row_count} rows: {time_ms:.2f}ms")

        # Verify times are reasonable
        # (not asserting linear scaling as cloud DW has fixed overhead)
        for row_count, time_ms in times_ms:
            assert time_ms < 30000, f"Query too slow for {row_count} rows"


# =============================================================================
# Cost Tracking Tests
# =============================================================================


@pytest.mark.expensive
class TestCostTracking:
    """Tests for cost tracking during performance tests."""

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_bytes_processed_tracking(
        self,
        any_backend: "CloudDWTestBackend",
        any_dataset: "TestDataset",
    ):
        """Track bytes processed for cost estimation."""
        dialect = get_dialect_for_backend(any_backend)

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = any_backend.create_test_table(
            any_dataset,
            "cost_tracking_test",
            schema,
            data,
        )

        # Get initial metrics
        initial_bytes = any_backend.metrics.total_bytes_processed

        full_name = any_backend.get_full_table_name(any_dataset.name, table.name)

        # Execute query
        any_backend.execute_query(f"SELECT * FROM {full_name}")

        # Check bytes processed was tracked
        final_bytes = any_backend.metrics.total_bytes_processed

        print(f"\nBytes Processed:")
        print(f"  Initial: {initial_bytes}")
        print(f"  Final: {final_bytes}")
        print(f"  Delta: {final_bytes - initial_bytes}")

        # Verify metrics were updated
        assert any_backend.metrics.total_queries > 0


# =============================================================================
# Backend-Specific Performance Tests
# =============================================================================


@pytest.mark.bigquery
@pytest.mark.expensive
class TestBigQueryPerformance:
    """BigQuery-specific performance tests."""

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_dry_run_performance(
        self,
        bigquery_backend: "CloudDWTestBackend",
        bigquery_dataset: "TestDataset",
    ):
        """Test dry run query performance (no actual execution)."""
        dialect = SQLDialect.BIGQUERY

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=100)

        table = bigquery_backend.create_test_table(
            bigquery_dataset,
            "dry_run_perf_test",
            schema,
            data,
        )

        full_name = bigquery_backend.get_full_table_name(
            bigquery_dataset.name, table.name
        )

        # Measure dry run query (should be fast)
        benchmark = Benchmark("dry_run_query", iterations=5)
        for _ in range(5):
            benchmark.run(lambda: bigquery_backend.execute_dry_run(
                f"SELECT * FROM {full_name}"
            ))

        result = benchmark.get_result({
            "backend": "bigquery",
            "mode": "dry_run",
        })

        print(f"\nBigQuery Dry Run: {result.mean_ms:.2f}ms")

        # Dry run should be very fast
        assert result.mean_ms < 2000


@pytest.mark.snowflake
@pytest.mark.expensive
class TestSnowflakePerformance:
    """Snowflake-specific performance tests."""

    @pytest.mark.performance
    @pytest.mark.requires_data
    def test_warehouse_performance(
        self,
        snowflake_backend: "CloudDWTestBackend",
        snowflake_dataset: "TestDataset",
    ):
        """Test query performance on Snowflake warehouse."""
        dialect = SQLDialect.SNOWFLAKE

        schema = StandardTestData.users_schema(dialect)
        data = StandardTestData.users_data(n=500)

        table = snowflake_backend.create_test_table(
            snowflake_dataset,
            "warehouse_perf_test",
            schema,
            data,
        )

        full_name = snowflake_backend.get_full_table_name(
            snowflake_dataset.name, table.name
        )

        benchmark = Benchmark("snowflake_query", iterations=3)
        for _ in range(3):
            benchmark.run(lambda: snowflake_backend.execute_query(f"""
                SELECT COUNT(*) AS cnt,
                       AVG(age) AS avg_age
                FROM {full_name}
            """))

        result = benchmark.get_result({
            "backend": "snowflake",
            "row_count": 500,
        })

        print(f"\nSnowflake Query: {result.mean_ms:.2f}ms")


# =============================================================================
# Benchmark Summary
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def print_benchmark_summary(request: pytest.FixtureRequest):
    """Print summary of all benchmarks at end of session."""
    yield

    # This runs after all tests complete
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    print("See individual test output for detailed results.")
    print("=" * 60)
