"""Tests for RayExecutionEngine.

This module tests the Ray-native distributed execution engine.
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_ray_available():
    """Mock ray availability check."""
    with (
        patch("truthound.execution.distributed.ray_engine._check_ray_available") as check_mock,
        patch("truthound.execution.distributed.ray_engine._ensure_ray_initialized") as init_mock,
    ):
        yield check_mock, init_mock


@pytest.fixture
def mock_ray_dataset():
    """Create a mock Ray Dataset."""
    mock_ds = MagicMock()

    # Schema mock
    mock_schema = MagicMock()
    mock_schema.names = ["id", "name", "value", "category"]
    mock_ds.schema.return_value = mock_schema

    # Basic operations
    mock_ds.count.return_value = 1000
    mock_ds.num_blocks.return_value = 4

    return mock_ds


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {"id": 1, "name": "Alice", "value": 100.0, "category": "A"},
        {"id": 2, "name": "Bob", "value": 200.0, "category": "B"},
        {"id": 3, "name": None, "value": 150.0, "category": "A"},
        {"id": 4, "name": "David", "value": None, "category": "B"},
        {"id": 5, "name": "Eve", "value": 300.0, "category": "A"},
        {"id": 6, "name": "Frank", "value": 250.0, "category": "B"},
        {"id": 7, "name": None, "value": 175.0, "category": "A"},
        {"id": 8, "name": "Heidi", "value": None, "category": "B"},
        {"id": 9, "name": "Ivan", "value": 225.0, "category": "A"},
        {"id": 10, "name": "Judy", "value": 275.0, "category": "B"},
    ]


# =============================================================================
# RayEngineConfig Tests
# =============================================================================


class TestRayEngineConfig:
    """Tests for RayEngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from truthound.execution.distributed.ray_engine import RayEngineConfig

        config = RayEngineConfig()

        assert config.ray_address is None
        assert config.num_cpus is None
        assert config.num_gpus is None
        assert config.batch_size == 4096
        assert config.prefetch_batches == 2
        assert config.concurrency is None
        assert config.use_actors is False
        assert config.actor_pool_size == 4
        assert config.target_max_block_size == 128 * 1024 * 1024

    def test_custom_config(self):
        """Test custom configuration."""
        from truthound.execution.distributed.ray_engine import RayEngineConfig

        config = RayEngineConfig(
            ray_address="ray://cluster:10001",
            num_cpus=16,
            num_gpus=2,
            batch_size=8192,
            use_actors=True,
            actor_pool_size=8,
        )

        assert config.ray_address == "ray://cluster:10001"
        assert config.num_cpus == 16
        assert config.num_gpus == 2
        assert config.batch_size == 8192
        assert config.use_actors is True
        assert config.actor_pool_size == 8


# =============================================================================
# RayExecutionEngine Core Tests
# =============================================================================


class TestRayExecutionEngineCore:
    """Core functionality tests for RayExecutionEngine."""

    def test_backend_type(self, mock_ray_available, mock_ray_dataset):
        """Test backend type property."""
        from truthound.execution.distributed.protocols import ComputeBackend
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        assert engine.backend_type == ComputeBackend.RAY

    def test_engine_type(self, mock_ray_available, mock_ray_dataset):
        """Test engine type attribute."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        assert engine.engine_type == "ray"

    def test_get_columns(self, mock_ray_available, mock_ray_dataset):
        """Test getting column names."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        columns = engine.get_columns()

        assert columns == ["id", "name", "value", "category"]

    def test_get_partition_count(self, mock_ray_available, mock_ray_dataset):
        """Test getting partition (block) count."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        assert engine._get_partition_count() == 4
        assert engine.num_partitions == 4

    def test_get_partition_info(self, mock_ray_available, mock_ray_dataset):
        """Test getting partition information."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        infos = engine._get_partition_info()

        assert len(infos) == 4
        for i, info in enumerate(infos):
            assert info.partition_id == i
            assert info.total_partitions == 4
            assert info.columns == ("id", "name", "value", "category")


# =============================================================================
# RayExecutionEngine Factory Method Tests
# =============================================================================


class TestRayExecutionEngineFactoryMethods:
    """Tests for RayExecutionEngine factory methods."""

    def test_from_dataset(self, mock_ray_available, mock_ray_dataset):
        """Test creating engine from Dataset."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_dataset(mock_ray_dataset)

        assert engine.dataset is mock_ray_dataset

    def test_from_dataset_with_config(self, mock_ray_available, mock_ray_dataset):
        """Test creating engine with custom config."""
        from truthound.execution.distributed.ray_engine import (
            RayEngineConfig,
            RayExecutionEngine,
        )

        config = RayEngineConfig(
            batch_size=8192,
            prefetch_batches=4,
        )

        engine = RayExecutionEngine.from_dataset(mock_ray_dataset, config=config)

        assert engine._config.batch_size == 8192
        assert engine._config.prefetch_batches == 4


# =============================================================================
# RayExecutionEngine Aggregation Tests
# =============================================================================


class TestRayExecutionEngineAggregations:
    """Tests for aggregation operations."""

    def test_count_rows(self, mock_ray_available, mock_ray_dataset):
        """Test counting rows."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)
        count = engine.count_rows()

        assert count == 1000

    def test_count_rows_caching(self, mock_ray_available, mock_ray_dataset):
        """Test that count_rows is cached."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        count1 = engine.count_rows()
        count2 = engine.count_rows()

        assert count1 == count2
        # count() should only be called once due to caching
        assert mock_ray_dataset.count.call_count == 1


# =============================================================================
# RayExecutionEngine Context Manager Tests
# =============================================================================


class TestRayExecutionEngineContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, mock_ray_available, mock_ray_dataset):
        """Test context manager protocol."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine(mock_ray_dataset)

        with engine as e:
            assert e is engine


# =============================================================================
# Integration Tests (require ray)
# =============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("ray", reason="ray not installed"),
    reason="ray not available",
)
class TestRayExecutionEngineIntegration:
    """Integration tests requiring actual ray."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Own one local Ray lifetime and close it after this test class."""
        import os
        import tempfile

        import ray

        if ray.is_initialized():
            raise RuntimeError("Ray unit contracts require their own local lifetime")
        with (
            patch.dict(os.environ, {"RAY_USAGE_STATS_ENABLED": "0", "RAY_ADDRESS": "local"}),
            tempfile.TemporaryDirectory(
                prefix="th-ray-unit-", dir="/tmp" if os.name == "posix" else tempfile.gettempdir()
            ) as directory,
        ):
            try:
                ray.init(
                    address="local",
                    num_cpus=4,
                    num_gpus=0,
                    include_dashboard=False,
                    object_store_memory=80 * 1024**2,
                    _node_ip_address="127.0.0.1",
                    _temp_dir=directory,
                )
                yield
            finally:
                ray.shutdown()

    def test_from_items_integration(self, sample_data):
        """Test creating engine from list of items."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        assert engine.count_rows() == 10
        assert set(engine.get_columns()) == {"id", "name", "value", "category"}

    def test_count_rows_integration(self, sample_data):
        """Test counting rows with actual data."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        assert engine.count_rows() == 10

    def test_sum_integration(self, sample_data):
        """Test sum aggregation."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        total = engine.dataset.sum("id")

        assert total == 55  # 1 + 2 + ... + 10

    def test_mean_integration(self, sample_data):
        """Test mean aggregation."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        mean = engine.dataset.mean("id")

        assert mean == 5.5

    def test_min_max_integration(self, sample_data):
        """Test min/max aggregation."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        min_val = engine.dataset.min("id")
        max_val = engine.dataset.max("id")

        assert min_val == 1
        assert max_val == 10

    def test_sample_integration(self, sample_data):
        """Test sampling."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        sampled = engine.sample(n=5, seed=42)

        assert sampled.count_rows() <= 5

    def test_repartition_integration(self, sample_data):
        """Test repartitioning."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)
        original_blocks = engine.num_partitions
        assert original_blocks > 0

        new_engine = engine.repartition(4)

        assert new_engine.num_partitions == 4
        assert new_engine.count_rows() == 10

    def test_configured_statistics_preserve_population_ddof(self, sample_data):
        from truthound.execution.distributed.protocols import AggregationSpec
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)
        spec = AggregationSpec().add("id", "std", ddof=0).add("id", "var", ddof=0)
        result = engine._aggregate_distributed(spec)
        assert result["id_std"] == pytest.approx(8.25**0.5)
        assert result["id_var"] == pytest.approx(8.25)

    def test_take_integration(self, sample_data):
        """Test take operation."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        rows = engine.take(3)

        assert len(rows) == 3
        assert all(isinstance(r, dict) for r in rows)

    def test_to_pandas_integration(self, sample_data):
        """Test conversion to pandas."""
        pytest.importorskip("pandas")
        import pandas as pd

        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        df = engine.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_to_arrow_integration(self, sample_data):
        """Test conversion to Arrow."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        table = engine.to_arrow()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 10

    def test_to_polars_lazyframe_integration(self, sample_data):
        """Test conversion to Polars LazyFrame."""
        pytest.importorskip("polars")
        pytest.importorskip("pyarrow")

        import polars as pl

        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        lf = engine.to_polars_lazyframe()

        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 10

    def test_materialize_integration(self, sample_data):
        """Test materializing the dataset."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        materialized = engine.materialize()

        assert materialized.count_rows() == 10

    def test_select_columns_integration(self, sample_data):
        """Test column selection."""
        from truthound.execution.distributed.ray_engine import RayExecutionEngine

        engine = RayExecutionEngine.from_items(sample_data)

        selected = engine.select_columns(["id", "name"])

        assert set(selected.get_columns()) == {"id", "name"}


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRayEngineRegistry:
    """Tests for ray engine registration in the global registry."""

    def test_ray_engine_registered(self):
        """Test that ray engine is registered."""
        from truthound.execution.distributed.registry import list_distributed_engines

        engines = list_distributed_engines()

        assert "ray" in engines

    def test_auto_detect_ray_dataset(self, mock_ray_available, mock_ray_dataset):
        """Test auto-detection of Ray Dataset."""
        from truthound.execution.distributed.protocols import ComputeBackend
        from truthound.execution.distributed.registry import (
            get_engine_registry,
        )

        # Mock the module detection
        dataset = type("Dataset", (), {"__module__": "ray.data.dataset"})()

        registry = get_engine_registry()
        backend = registry._detect_backend(dataset)

        assert backend == ComputeBackend.RAY


@pytest.mark.contract
def test_cached_nullable_stats_preserve_none_wildcard_count_and_result_isolation(mock_ray_available, mock_ray_dataset):
    from truthound.execution._protocols import AggregationType
    from truthound.execution.distributed.protocols import AggregationSpec
    from truthound.execution.distributed.ray_engine import RayExecutionEngine

    mock_ray_dataset.count.return_value = 3
    mock_ray_dataset.schema.return_value.names = ["value", "*"]
    with RayExecutionEngine(mock_ray_dataset) as engine:
        stats = {"count": 0, "null_count": 3, "mean": None, "std": None, "min": None, "max": None}
        engine._set_cached(engine._cache_key("get_stats", "value"), stats)
        engine._set_cached(engine._cache_key("get_stats", "*"), stats)
        mock_ray_dataset.mean.side_effect = AssertionError("unexpected duplicate native mean")
        mock_ray_dataset.std.side_effect = AssertionError("unexpected duplicate native std")
        mock_ray_dataset.min.side_effect = AssertionError("unexpected duplicate native min")
        mock_ray_dataset.max.side_effect = AssertionError("unexpected duplicate native max")
        for operation in (AggregationType.MEAN, AggregationType.STD, AggregationType.VAR, AggregationType.MIN, AggregationType.MAX):
            assert engine.aggregate_column("value", operation) is None
        assert engine.aggregate_column("value", AggregationType.COUNT) == 0
        assert engine.aggregate_column("value", AggregationType.NULL_COUNT) == 3
        assert engine.aggregate_column("*", AggregationType.COUNT) == 3
        spec = AggregationSpec().add("value", "minmax", alias="range")
        observed = engine._aggregate_distributed(spec)
        observed["range"]["min"] = 999
        assert engine._aggregate_distributed(spec) == {"range": {"min": None, "max": None}}
