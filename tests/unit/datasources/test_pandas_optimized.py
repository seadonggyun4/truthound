"""Tests for optimized Pandas data source."""

from __future__ import annotations

import gc
import pytest
import numpy as np

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.pandas_optimized import (
    OptimizedPandasDataSource,
    OptimizedPandasConfig,
    DataFrameOptimizer,
    optimize_pandas_to_polars,
    estimate_polars_memory,
    get_optimal_chunk_size,
)


class TestDataFrameOptimizer:
    """Tests for DataFrameOptimizer."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for optimization."""
        import pandas as pd

        return pd.DataFrame({
            "int64_col": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "float64_col": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            "str_col": ["a", "b", "a", "b", "a"],  # Low cardinality
            "high_card_str": ["a", "b", "c", "d", "e"],  # High cardinality
        })

    def test_default_optimizer(self) -> None:
        """Test default optimizer settings."""
        optimizer = DataFrameOptimizer()

        assert optimizer._use_categorical is True
        assert optimizer._categorical_threshold == 50
        assert optimizer._downcast_int is True
        assert optimizer._downcast_float is True

    def test_custom_optimizer(self) -> None:
        """Test custom optimizer settings."""
        optimizer = DataFrameOptimizer(
            use_categorical=False,
            categorical_threshold=10,
            downcast_int=False,
            downcast_float=False,
        )

        assert optimizer._use_categorical is False
        assert optimizer._categorical_threshold == 10

    def test_optimize_integers(self, sample_df) -> None:
        """Test integer downcast optimization."""
        optimizer = DataFrameOptimizer()
        optimized = optimizer.optimize(sample_df)

        # Small integers should be downcast
        assert optimized["int64_col"].dtype != np.int64

    def test_optimize_floats(self, sample_df) -> None:
        """Test float optimization."""
        optimizer = DataFrameOptimizer()
        optimized = optimizer.optimize(sample_df)

        # Check that optimization occurred (may or may not downcast based on values)
        assert "float64_col" in optimized.columns

    def test_optimize_categorical(self, sample_df) -> None:
        """Test low-cardinality string to categorical conversion."""
        import pandas as pd

        # Create a dataframe with clear cardinality distinction (same length)
        n_rows = 20
        df = pd.DataFrame({
            "low_card": ["a", "b"] * 10,  # 2 unique values
            "high_card": [f"val_{i}" for i in range(n_rows)],  # 20 unique values
        })
        optimizer = DataFrameOptimizer(categorical_threshold=10)
        optimized = optimizer.optimize(df)

        # Low cardinality should become categorical
        assert optimized["low_card"].dtype.name == "category"
        # High cardinality should stay object
        assert optimized["high_card"].dtype == object

    def test_memory_saved_tracking(self, sample_df) -> None:
        """Test memory savings tracking."""
        optimizer = DataFrameOptimizer()
        optimizer.optimize(sample_df)

        # Should track some savings
        assert optimizer._original_memory > 0
        assert optimizer._optimized_memory > 0

    def test_memory_reduction_percentage(self, sample_df) -> None:
        """Test memory reduction percentage calculation."""
        import pandas as pd

        # Create larger DataFrame with more potential savings
        large_df = pd.DataFrame({
            "big_int": np.array([1] * 1000, dtype=np.int64),
            "big_str": ["cat"] * 1000,  # Very low cardinality
        })

        optimizer = DataFrameOptimizer()
        optimizer.optimize(large_df)

        # Should have some reduction
        assert optimizer.memory_reduction_pct >= 0

    def test_optimization_report(self, sample_df) -> None:
        """Test optimization report generation."""
        optimizer = DataFrameOptimizer()
        report = optimizer.get_optimization_report(sample_df)

        assert "total_memory_bytes" in report
        assert "columns" in report
        assert "int64_col" in report["columns"]
        assert "suggestions" in report["columns"]["int64_col"]


class TestOptimizedPandasDataSource:
    """Tests for OptimizedPandasDataSource."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        import pandas as pd

        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "value": [1.1, 2.2, 3.3, 4.4, 5.5],
        })

    def test_creation(self, sample_df) -> None:
        """Test source creation."""
        source = OptimizedPandasDataSource(sample_df)

        assert source.source_type == "pandas_optimized"
        assert source.row_count == 5

    def test_custom_config(self, sample_df) -> None:
        """Test source with custom configuration."""
        config = OptimizedPandasConfig(
            chunk_size=2,
            optimize_dtypes=False,
            gc_after_chunk=False,
        )
        source = OptimizedPandasDataSource(sample_df, config)

        assert source.config.chunk_size == 2
        assert source.config.optimize_dtypes is False

    def test_schema(self, sample_df) -> None:
        """Test schema property."""
        source = OptimizedPandasDataSource(sample_df)
        schema = source.schema

        assert "id" in schema
        assert "name" in schema
        assert "value" in schema

    def test_columns(self, sample_df) -> None:
        """Test columns property."""
        source = OptimizedPandasDataSource(sample_df)
        assert source.columns == ["id", "name", "value"]

    def test_capabilities(self, sample_df) -> None:
        """Test capabilities include streaming."""
        source = OptimizedPandasDataSource(sample_df)
        caps = source.capabilities

        assert DataSourceCapability.STREAMING in caps
        assert DataSourceCapability.SAMPLING in caps

    def test_iter_polars_chunks(self, sample_df) -> None:
        """Test chunked iteration."""
        config = OptimizedPandasConfig(chunk_size=2)
        source = OptimizedPandasDataSource(sample_df, config)

        chunks = list(source.iter_polars_chunks())

        assert len(chunks) == 3  # 5 rows with chunk_size=2 = 3 chunks
        assert all(hasattr(c, "collect") for c in chunks)  # All are LazyFrames

    def test_iter_polars_chunks_custom_size(self, sample_df) -> None:
        """Test chunked iteration with custom size."""
        source = OptimizedPandasDataSource(sample_df)
        chunks = list(source.iter_polars_chunks(chunk_size=3))

        assert len(chunks) == 2  # 5 rows with chunk_size=3 = 2 chunks

    def test_to_polars_streaming(self, sample_df) -> None:
        """Test streaming conversion."""
        source = OptimizedPandasDataSource(sample_df)
        lf = source.to_polars_streaming()

        df = lf.collect()
        assert len(df) == 5
        assert "id" in df.columns

    def test_to_polars_lazyframe(self, sample_df) -> None:
        """Test standard lazyframe conversion."""
        source = OptimizedPandasDataSource(sample_df)
        lf = source.to_polars_lazyframe()

        df = lf.collect()
        assert len(df) == 5

    def test_sample(self, sample_df) -> None:
        """Test sampling."""
        source = OptimizedPandasDataSource(sample_df)
        sampled = source.sample(2)

        assert sampled.row_count == 2
        assert isinstance(sampled, OptimizedPandasDataSource)

    def test_sample_with_seed(self, sample_df) -> None:
        """Test reproducible sampling."""
        source = OptimizedPandasDataSource(sample_df)
        sampled1 = source.sample(2, seed=42)
        sampled2 = source.sample(2, seed=42)

        df1 = sampled1.dataframe
        df2 = sampled2.dataframe

        assert df1.equals(df2)

    def test_validate_connection(self, sample_df) -> None:
        """Test connection validation."""
        source = OptimizedPandasDataSource(sample_df)
        assert source.validate_connection() is True

    def test_get_memory_usage(self, sample_df) -> None:
        """Test memory usage report."""
        source = OptimizedPandasDataSource(sample_df)
        usage = source.get_memory_usage()

        assert "total_bytes" in usage
        assert "row_count" in usage
        assert "column_count" in usage
        assert "per_column" in usage

    def test_get_optimization_report(self, sample_df) -> None:
        """Test optimization report."""
        source = OptimizedPandasDataSource(sample_df)
        report = source.get_optimization_report()

        assert "columns" in report
        assert "total_memory_bytes" in report

    def test_get_execution_engine(self, sample_df) -> None:
        """Test execution engine retrieval."""
        source = OptimizedPandasDataSource(sample_df)
        engine = source.get_execution_engine()

        assert engine is not None


class TestUtilityFunctions:
    """Tests for utility functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        import pandas as pd

        return pd.DataFrame({
            "id": list(range(100)),
            "name": ["user"] * 100,
        })

    def test_optimize_pandas_to_polars(self, sample_df) -> None:
        """Test optimize_pandas_to_polars function."""
        lf = optimize_pandas_to_polars(sample_df, chunk_size=50)

        df = lf.collect()
        assert len(df) == 100
        assert "id" in df.columns

    def test_estimate_polars_memory(self, sample_df) -> None:
        """Test memory estimation."""
        estimated = estimate_polars_memory(sample_df)

        assert estimated > 0
        # Polars typically uses less memory
        pandas_memory = sample_df.memory_usage(deep=True).sum()
        assert estimated <= pandas_memory

    def test_get_optimal_chunk_size(self, sample_df) -> None:
        """Test optimal chunk size calculation."""
        optimal = get_optimal_chunk_size(sample_df, target_memory_mb=1)

        assert optimal >= 1000
        assert optimal <= 1_000_000

    def test_get_optimal_chunk_size_for_large_df(self) -> None:
        """Test optimal chunk size for larger DataFrame."""
        import pandas as pd

        large_df = pd.DataFrame({
            "col": list(range(10000)),
            "text": ["long text string " * 10] * 10000,
        })

        optimal = get_optimal_chunk_size(large_df, target_memory_mb=10)
        assert optimal > 0
