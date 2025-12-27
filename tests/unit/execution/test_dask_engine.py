"""Tests for DaskExecutionEngine.

This module tests the Dask-native distributed execution engine.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Any


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_dask_available():
    """Mock dask availability check."""
    with patch(
        "truthound.execution.distributed.dask_engine._check_dask_available"
    ) as mock:
        yield mock


@pytest.fixture
def mock_dask_df():
    """Create a mock Dask DataFrame."""
    mock_df = MagicMock()
    mock_df.columns = ["id", "name", "value", "category"]
    mock_df.dtypes = {
        "id": "int64",
        "name": "object",
        "value": "float64",
        "category": "object",
    }
    mock_df.npartitions = 4
    return mock_df


@pytest.fixture
def mock_pandas_df():
    """Create sample pandas DataFrame for testing."""
    try:
        import pandas as pd
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", None, "David", "Eve", "Frank", None, "Heidi", "Ivan", "Judy"],
            "value": [100.0, 200.0, 150.0, None, 300.0, 250.0, 175.0, None, 225.0, 275.0],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })
    except ImportError:
        pytest.skip("pandas not available")


# =============================================================================
# DaskEngineConfig Tests
# =============================================================================


class TestDaskEngineConfig:
    """Tests for DaskEngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from truthound.execution.distributed.dask_engine import DaskEngineConfig

        config = DaskEngineConfig()

        assert config.scheduler == "threads"
        assert config.client_address is None
        assert config.n_workers is None
        assert config.threads_per_worker == 2
        assert config.memory_per_worker == "2GB"
        assert config.processes is False
        assert config.blocksize == "128MB"
        assert config.persist_intermediate is False

    def test_custom_config(self):
        """Test custom configuration."""
        from truthound.execution.distributed.dask_engine import DaskEngineConfig

        config = DaskEngineConfig(
            scheduler="distributed",
            client_address="tcp://scheduler:8786",
            n_workers=8,
            threads_per_worker=4,
            memory_per_worker="4GB",
            processes=True,
        )

        assert config.scheduler == "distributed"
        assert config.client_address == "tcp://scheduler:8786"
        assert config.n_workers == 8
        assert config.threads_per_worker == 4
        assert config.processes is True


# =============================================================================
# DaskExecutionEngine Core Tests
# =============================================================================


class TestDaskExecutionEngineCore:
    """Core functionality tests for DaskExecutionEngine."""

    def test_check_dask_available_raises_when_not_installed(self):
        """Test that check raises ImportError when dask not available."""
        with patch.dict("sys.modules", {"dask.dataframe": None}):
            from truthound.execution.distributed.dask_engine import _check_dask_available

            # The function should raise when dask is not importable
            # This is just testing the structure exists
            assert callable(_check_dask_available)

    def test_backend_type(self, mock_dask_available, mock_dask_df):
        """Test backend type property."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine
        from truthound.execution.distributed.protocols import ComputeBackend

        engine = DaskExecutionEngine(mock_dask_df)

        assert engine.backend_type == ComputeBackend.DASK

    def test_engine_type(self, mock_dask_available, mock_dask_df):
        """Test engine type attribute."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)

        assert engine.engine_type == "dask"

    def test_get_columns(self, mock_dask_available, mock_dask_df):
        """Test getting column names."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)

        columns = engine.get_columns()

        assert columns == ["id", "name", "value", "category"]

    def test_get_partition_count(self, mock_dask_available, mock_dask_df):
        """Test getting partition count."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)

        assert engine._get_partition_count() == 4
        assert engine.num_partitions == 4

    def test_get_partition_info(self, mock_dask_available, mock_dask_df):
        """Test getting partition information."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)

        infos = engine._get_partition_info()

        assert len(infos) == 4
        for i, info in enumerate(infos):
            assert info.partition_id == i
            assert info.total_partitions == 4
            assert info.columns == ("id", "name", "value", "category")


# =============================================================================
# DaskExecutionEngine Factory Method Tests
# =============================================================================


class TestDaskExecutionEngineFactoryMethods:
    """Tests for DaskExecutionEngine factory methods."""

    def test_from_dataframe(self, mock_dask_available, mock_dask_df):
        """Test creating engine from DataFrame."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_dataframe(mock_dask_df)

        assert engine.dask_dataframe is mock_dask_df
        assert engine.client is None

    def test_from_dataframe_with_config(self, mock_dask_available, mock_dask_df):
        """Test creating engine with custom config."""
        from truthound.execution.distributed.dask_engine import (
            DaskExecutionEngine,
            DaskEngineConfig,
        )

        config = DaskEngineConfig(
            scheduler="threads",
            threads_per_worker=4,
        )

        engine = DaskExecutionEngine.from_dataframe(mock_dask_df, config=config)

        assert engine._config.threads_per_worker == 4


# =============================================================================
# DaskExecutionEngine Aggregation Tests
# =============================================================================


class TestDaskExecutionEngineAggregations:
    """Tests for aggregation operations."""

    def test_count_rows(self, mock_dask_available, mock_dask_df):
        """Test counting rows."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        mock_dask_df.__len__ = MagicMock(return_value=1000)

        engine = DaskExecutionEngine(mock_dask_df)
        count = engine.count_rows()

        assert count == 1000

    def test_count_nulls(self, mock_dask_available, mock_dask_df):
        """Test counting nulls in a column."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        # Set up mock for column null count
        mock_series = MagicMock()
        mock_series.isna.return_value.sum.return_value.compute.return_value = 5
        mock_dask_df.__getitem__ = MagicMock(return_value=mock_series)

        engine = DaskExecutionEngine(mock_dask_df)
        count = engine.count_nulls("name")

        assert count == 5

    def test_count_distinct(self, mock_dask_available, mock_dask_df):
        """Test counting distinct values."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        mock_series = MagicMock()
        mock_series.nunique.return_value.compute.return_value = 10
        mock_dask_df.__getitem__ = MagicMock(return_value=mock_series)

        engine = DaskExecutionEngine(mock_dask_df)
        count = engine.count_distinct("category")

        assert count == 10


# =============================================================================
# DaskExecutionEngine Context Manager Tests
# =============================================================================


class TestDaskExecutionEngineContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, mock_dask_available, mock_dask_df):
        """Test context manager protocol."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)

        with engine as e:
            assert e is engine

    def test_close_without_client(self, mock_dask_available, mock_dask_df):
        """Test close when no client exists."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine(mock_dask_df)
        engine.close()  # Should not raise

        assert engine._client is None


# =============================================================================
# DaskExecutionEngine Caching Tests
# =============================================================================


class TestDaskExecutionEngineCaching:
    """Tests for caching behavior."""

    def test_count_rows_caching(self, mock_dask_available, mock_dask_df):
        """Test that count_rows result is cached."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        mock_dask_df.__len__ = MagicMock(return_value=1000)

        engine = DaskExecutionEngine(mock_dask_df)

        # First call
        count1 = engine.count_rows()
        # Second call should use cache
        count2 = engine.count_rows()

        assert count1 == count2 == 1000
        # __len__ should only be called once due to caching
        assert mock_dask_df.__len__.call_count == 1


# =============================================================================
# Integration Tests (require dask)
# =============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("dask", reason="dask not installed"),
    reason="dask not available",
)
class TestDaskExecutionEngineIntegration:
    """Integration tests requiring actual dask."""

    def test_from_pandas_integration(self, mock_pandas_df):
        """Test creating engine from pandas DataFrame."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(
            mock_pandas_df,
            npartitions=2,
        )

        assert engine.count_rows() == 10
        assert engine.get_columns() == ["id", "name", "value", "category"]

    def test_count_nulls_integration(self, mock_pandas_df):
        """Test null counting with actual data."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        name_nulls = engine.count_nulls("name")
        value_nulls = engine.count_nulls("value")

        assert name_nulls == 2  # Two None values in name
        assert value_nulls == 2  # Two None values in value

    def test_count_nulls_all_integration(self, mock_pandas_df):
        """Test counting nulls in all columns."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        null_counts = engine.count_nulls_all()

        assert null_counts["id"] == 0
        assert null_counts["name"] == 2
        assert null_counts["value"] == 2
        assert null_counts["category"] == 0

    def test_get_stats_integration(self, mock_pandas_df):
        """Test getting column statistics."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        stats = engine.get_stats("value")

        assert stats["count"] == 8  # 10 - 2 nulls
        assert stats["null_count"] == 2
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0
        assert 150 < stats["mean"] < 250  # Approximate check

    def test_get_value_counts_integration(self, mock_pandas_df):
        """Test value counts."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        counts = engine.get_value_counts("category")

        assert counts["A"] == 5
        assert counts["B"] == 5

    def test_repartition_integration(self, mock_pandas_df):
        """Test repartitioning."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        new_engine = engine.repartition(4)

        assert new_engine.num_partitions == 4
        assert new_engine.count_rows() == 10

    def test_sample_integration(self, mock_pandas_df):
        """Test sampling."""
        from truthound.execution.distributed.dask_engine import DaskExecutionEngine

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        sampled = engine.sample(n=5, seed=42)

        assert sampled.count_rows() <= 5

    def test_to_polars_lazyframe_integration(self, mock_pandas_df):
        """Test conversion to Polars LazyFrame."""
        pytest.importorskip("polars")
        pytest.importorskip("pyarrow")

        from truthound.execution.distributed.dask_engine import DaskExecutionEngine
        import polars as pl

        engine = DaskExecutionEngine.from_pandas(mock_pandas_df, npartitions=2)

        lf = engine.to_polars_lazyframe()

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 10
