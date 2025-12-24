"""Tests for execution engines module.

This module tests the execution engine implementations:
- PolarsExecutionEngine
- PandasExecutionEngine
"""

import pytest
import polars as pl
import numpy as np

from truthound.execution import (
    PolarsExecutionEngine,
    PandasExecutionEngine,
    AggregationType,
    ExecutionSizeError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_polars_df():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "David", "Eve"],
        "age": [25, 30, 35, None, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "department": ["IT", "HR", "IT", "HR", "IT"],
    })


@pytest.fixture
def polars_engine(sample_polars_df):
    """Create a PolarsExecutionEngine for testing."""
    return PolarsExecutionEngine(sample_polars_df)


@pytest.fixture
def pandas_engine(sample_polars_df):
    """Create a PandasExecutionEngine for testing."""
    pandas_df = sample_polars_df.to_pandas()
    return PandasExecutionEngine(pandas_df)


# =============================================================================
# PolarsExecutionEngine Tests
# =============================================================================


class TestPolarsExecutionEngine:
    """Tests for PolarsExecutionEngine."""

    def test_engine_type(self, polars_engine):
        """Test engine type property."""
        assert polars_engine.engine_type == "polars"

    def test_count_rows(self, polars_engine):
        """Test counting rows."""
        assert polars_engine.count_rows() == 5

    def test_get_columns(self, polars_engine):
        """Test getting column names."""
        columns = polars_engine.get_columns()
        assert len(columns) == 5
        assert "id" in columns
        assert "name" in columns

    def test_count_nulls(self, polars_engine):
        """Test counting nulls in a column."""
        assert polars_engine.count_nulls("name") == 1
        assert polars_engine.count_nulls("age") == 1
        assert polars_engine.count_nulls("id") == 0

    def test_count_nulls_all(self, polars_engine):
        """Test counting nulls for all columns."""
        null_counts = polars_engine.count_nulls_all()
        assert null_counts["name"] == 1
        assert null_counts["age"] == 1
        assert null_counts["id"] == 0

    def test_count_distinct(self, polars_engine):
        """Test counting distinct values."""
        assert polars_engine.count_distinct("department") == 2
        assert polars_engine.count_distinct("id") == 5

    def test_get_stats(self, polars_engine):
        """Test getting statistics for a column."""
        stats = polars_engine.get_stats("salary")

        assert stats["count"] == 5
        assert stats["null_count"] == 0
        assert stats["mean"] == 70000.0
        assert stats["min"] == 50000.0
        assert stats["max"] == 90000.0

    def test_get_stats_with_nulls(self, polars_engine):
        """Test statistics for column with nulls."""
        stats = polars_engine.get_stats("age")

        assert stats["count"] == 4  # Non-null count
        assert stats["null_count"] == 1

    def test_get_quantiles(self, polars_engine):
        """Test getting quantiles."""
        quantiles = polars_engine.get_quantiles("salary", [0.25, 0.5, 0.75])

        assert len(quantiles) == 3
        assert quantiles[1] == 70000.0  # Median

    def test_get_value_counts(self, polars_engine):
        """Test getting value counts."""
        counts = polars_engine.get_value_counts("department")

        assert counts["IT"] == 3
        assert counts["HR"] == 2

    def test_get_value_counts_with_limit(self, polars_engine):
        """Test value counts with limit."""
        counts = polars_engine.get_value_counts("department", limit=1)
        assert len(counts) == 1

    def test_aggregate(self, polars_engine):
        """Test multiple aggregations."""
        result = polars_engine.aggregate({
            "salary": AggregationType.MEAN,
            "age": AggregationType.MAX,
        })

        assert result["salary_mean"] == 70000.0
        assert result["age_max"] == 45

    def test_get_distinct_values(self, polars_engine):
        """Test getting distinct values."""
        values = polars_engine.get_distinct_values("department")
        assert set(values) == {"IT", "HR"}

    def test_get_column_values(self, polars_engine):
        """Test getting column values."""
        values = polars_engine.get_column_values("id", limit=3)
        assert values == [1, 2, 3]

    def test_get_sample_values(self, polars_engine):
        """Test getting sample values."""
        values = polars_engine.get_sample_values("name", n=3)
        assert len(values) <= 3
        assert None not in values  # Should exclude nulls

    def test_count_matching_regex(self, polars_engine):
        """Test counting regex matches."""
        count = polars_engine.count_matching_regex("name", "^A")
        assert count == 1  # Only "Alice"

    def test_count_in_range(self, polars_engine):
        """Test counting values in range."""
        count = polars_engine.count_in_range("age", min_value=30, max_value=40)
        assert count == 2  # 30 and 35

    def test_count_in_set(self, polars_engine):
        """Test counting values in set."""
        count = polars_engine.count_in_set("department", {"IT"})
        assert count == 3

    def test_count_duplicates(self, polars_engine):
        """Test counting duplicates."""
        # No duplicates on unique columns
        assert polars_engine.count_duplicates(["id"]) == 0

        # Duplicates on department
        assert polars_engine.count_duplicates(["department"]) == 3  # 5 - 2 unique

    def test_get_duplicate_values(self, polars_engine):
        """Test getting duplicate values."""
        dupes = polars_engine.get_duplicate_values(["department"])
        assert len(dupes) == 2  # IT and HR both have duplicates

    def test_to_polars_lazyframe(self, polars_engine):
        """Test converting to LazyFrame."""
        lf = polars_engine.to_polars_lazyframe()
        assert isinstance(lf, pl.LazyFrame)

    def test_to_numpy(self, polars_engine):
        """Test converting to numpy."""
        arr = polars_engine.to_numpy(columns=["id", "salary"])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 2)

    def test_sample(self, polars_engine):
        """Test sampling."""
        sampled = polars_engine.sample(n=3, seed=42)
        assert sampled.count_rows() <= 3

    def test_caching(self, polars_engine):
        """Test result caching."""
        # First call
        count1 = polars_engine.count_rows()
        # Second call (should be cached)
        count2 = polars_engine.count_rows()
        assert count1 == count2

        # Clear cache
        polars_engine.clear_cache()

    def test_filter_polars_expression(self, polars_engine):
        """Test filtering with Polars expression."""
        filtered = polars_engine.filter(pl.col("age") > 30)
        assert filtered.count_rows() == 2

    def test_select(self, polars_engine):
        """Test selecting columns."""
        selected = polars_engine.select("id", "name")
        assert selected.get_columns() == ["id", "name"]


# =============================================================================
# PandasExecutionEngine Tests
# =============================================================================


class TestPandasExecutionEngine:
    """Tests for PandasExecutionEngine."""

    def test_engine_type(self, pandas_engine):
        """Test engine type property."""
        assert pandas_engine.engine_type == "pandas"

    def test_count_rows(self, pandas_engine):
        """Test counting rows."""
        assert pandas_engine.count_rows() == 5

    def test_get_columns(self, pandas_engine):
        """Test getting column names."""
        columns = pandas_engine.get_columns()
        assert len(columns) == 5

    def test_count_nulls(self, pandas_engine):
        """Test counting nulls."""
        assert pandas_engine.count_nulls("name") == 1
        assert pandas_engine.count_nulls("age") == 1

    def test_count_nulls_all(self, pandas_engine):
        """Test counting nulls for all columns."""
        null_counts = pandas_engine.count_nulls_all()
        assert null_counts["name"] == 1

    def test_count_distinct(self, pandas_engine):
        """Test counting distinct values."""
        assert pandas_engine.count_distinct("department") == 2

    def test_get_stats(self, pandas_engine):
        """Test getting statistics."""
        stats = pandas_engine.get_stats("salary")

        assert stats["count"] == 5
        assert stats["mean"] == 70000.0

    def test_get_value_counts(self, pandas_engine):
        """Test getting value counts."""
        counts = pandas_engine.get_value_counts("department")
        assert counts["IT"] == 3
        assert counts["HR"] == 2

    def test_aggregate(self, pandas_engine):
        """Test aggregations."""
        result = pandas_engine.aggregate({
            "salary": AggregationType.SUM,
            "id": AggregationType.COUNT,
        })

        assert result["salary_sum"] == 350000.0
        assert result["id_count"] == 5

    def test_get_distinct_values(self, pandas_engine):
        """Test getting distinct values."""
        values = pandas_engine.get_distinct_values("department")
        assert set(values) == {"IT", "HR"}

    def test_count_matching_regex(self, pandas_engine):
        """Test regex matching."""
        count = pandas_engine.count_matching_regex("name", "^A")
        assert count == 1

    def test_count_in_range(self, pandas_engine):
        """Test counting in range."""
        count = pandas_engine.count_in_range("salary", min_value=60000, max_value=80000)
        assert count == 3

    def test_count_in_set(self, pandas_engine):
        """Test counting in set."""
        count = pandas_engine.count_in_set("id", {1, 2, 3})
        assert count == 3

    def test_count_duplicates(self, pandas_engine):
        """Test counting duplicates."""
        assert pandas_engine.count_duplicates(["id"]) == 0
        assert pandas_engine.count_duplicates(["department"]) == 3

    def test_to_polars_lazyframe(self, pandas_engine):
        """Test converting to Polars LazyFrame."""
        lf = pandas_engine.to_polars_lazyframe()
        assert isinstance(lf, pl.LazyFrame)

    def test_to_numpy(self, pandas_engine):
        """Test converting to numpy."""
        arr = pandas_engine.to_numpy(columns=["id", "salary"])
        assert isinstance(arr, np.ndarray)

    def test_sample(self, pandas_engine):
        """Test sampling."""
        sampled = pandas_engine.sample(n=3, seed=42)
        assert sampled.count_rows() == 3


# =============================================================================
# Engine Comparison Tests
# =============================================================================


class TestEngineComparison:
    """Tests comparing Polars and Pandas engines produce same results."""

    def test_count_rows_match(self, polars_engine, pandas_engine):
        """Test row counts match."""
        assert polars_engine.count_rows() == pandas_engine.count_rows()

    def test_count_nulls_match(self, polars_engine, pandas_engine):
        """Test null counts match."""
        assert polars_engine.count_nulls("name") == pandas_engine.count_nulls("name")

    def test_count_distinct_match(self, polars_engine, pandas_engine):
        """Test distinct counts match."""
        assert polars_engine.count_distinct("department") == pandas_engine.count_distinct("department")

    def test_stats_match(self, polars_engine, pandas_engine):
        """Test statistics match (approximately)."""
        polars_stats = polars_engine.get_stats("salary")
        pandas_stats = pandas_engine.get_stats("salary")

        assert polars_stats["count"] == pandas_stats["count"]
        assert polars_stats["mean"] == pytest.approx(pandas_stats["mean"], rel=1e-6)
        assert polars_stats["min"] == pandas_stats["min"]
        assert polars_stats["max"] == pandas_stats["max"]

    def test_value_counts_match(self, polars_engine, pandas_engine):
        """Test value counts match."""
        polars_counts = polars_engine.get_value_counts("department")
        pandas_counts = pandas_engine.get_value_counts("department")

        assert polars_counts == pandas_counts


# =============================================================================
# Size Limit Tests
# =============================================================================


class TestSizeLimits:
    """Tests for size limit handling in engines."""

    def test_to_numpy_size_limit(self, sample_polars_df):
        """Test numpy conversion respects size limits."""
        from truthound.execution.base import ExecutionConfig

        config = ExecutionConfig(max_rows_for_numpy=3)
        engine = PolarsExecutionEngine(sample_polars_df, config)

        with pytest.raises(ExecutionSizeError):
            engine.to_numpy()

    def test_to_numpy_safe(self, sample_polars_df):
        """Test safe numpy conversion with sampling."""
        engine = PolarsExecutionEngine(sample_polars_df)
        arr = engine.to_numpy_safe(max_rows=3, seed=42)

        assert arr.shape[0] <= 3


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for using engines as context managers."""

    def test_polars_engine_context(self, sample_polars_df):
        """Test Polars engine as context manager."""
        with PolarsExecutionEngine(sample_polars_df) as engine:
            assert engine.count_rows() == 5

    def test_pandas_engine_context(self, sample_polars_df):
        """Test Pandas engine as context manager."""
        pandas_df = sample_polars_df.to_pandas()
        with PandasExecutionEngine(pandas_df) as engine:
            assert engine.count_rows() == 5
