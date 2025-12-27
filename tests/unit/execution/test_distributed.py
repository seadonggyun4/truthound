"""Tests for distributed execution framework.

This module provides comprehensive tests for the distributed execution
framework, covering:
- Protocol definitions
- Aggregators
- Spark execution engine
- Arrow bridge
- Validator adapters
- Registry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocols:
    """Tests for distributed execution protocols."""

    def test_partition_info(self):
        """Test PartitionInfo creation and properties."""
        from truthound.execution.distributed.protocols import PartitionInfo

        info = PartitionInfo(
            partition_id=0,
            total_partitions=4,
            row_start=0,
            row_end=1000,
            columns=("a", "b", "c"),
        )

        assert info.partition_id == 0
        assert info.total_partitions == 4
        assert info.row_count == 1000
        assert info.columns == ("a", "b", "c")

    def test_distributed_result(self):
        """Test DistributedResult creation and properties."""
        from truthound.execution.distributed.protocols import DistributedResult

        result = DistributedResult(
            partition_id=0,
            operation="count_nulls",
            value={"col1": 10, "col2": 5},
            row_count=1000,
            duration_ms=50.5,
        )

        assert result.success
        assert result.row_count == 1000
        assert result.value["col1"] == 10

        # Test with errors
        result_with_error = DistributedResult(
            partition_id=0,
            operation="count_nulls",
            value=None,
            errors=["Connection failed"],
        )

        assert not result_with_error.success

    def test_distributed_aggregation(self):
        """Test DistributedAggregation creation."""
        from truthound.execution.distributed.protocols import DistributedAggregation

        agg = DistributedAggregation(
            column="price",
            operation="mean",
        )

        assert agg.column == "price"
        assert agg.operation == "mean"
        assert agg.alias == "price_mean"  # Auto-generated

        agg_with_alias = DistributedAggregation(
            column="price",
            operation="mean",
            alias="avg_price",
        )

        assert agg_with_alias.alias == "avg_price"

    def test_aggregation_spec(self):
        """Test AggregationSpec builder."""
        from truthound.execution.distributed.protocols import AggregationSpec

        spec = AggregationSpec()
        spec.add("col1", "count", alias="total")
        spec.add("col2", "mean")

        assert len(spec.aggregations) == 2
        assert spec.aggregations[0].alias == "total"
        assert spec.aggregations[1].alias == "col2_mean"

    def test_partition_strategy_enum(self):
        """Test PartitionStrategy enum values."""
        from truthound.execution.distributed.protocols import PartitionStrategy

        assert PartitionStrategy.ROW_HASH == "row_hash"
        assert PartitionStrategy.COLUMN == "column"
        assert PartitionStrategy.ROUND_ROBIN == "round_robin"

    def test_compute_backend_enum(self):
        """Test ComputeBackend enum values."""
        from truthound.execution.distributed.protocols import ComputeBackend

        assert ComputeBackend.SPARK == "spark"
        assert ComputeBackend.DASK == "dask"
        assert ComputeBackend.RAY == "ray"
        assert ComputeBackend.LOCAL == "local"


# =============================================================================
# Aggregator Tests
# =============================================================================


class TestAggregators:
    """Tests for distributed aggregators."""

    def test_count_aggregator(self):
        """Test CountAggregator."""
        from truthound.execution.distributed.protocols import CountAggregator

        agg = CountAggregator()
        state = agg.initialize()

        # Accumulate values
        for _ in range(10):
            state = agg.accumulate(state, 1)

        assert agg.finalize(state) == 10

    def test_count_aggregator_merge(self):
        """Test CountAggregator merge."""
        from truthound.execution.distributed.protocols import CountAggregator

        agg = CountAggregator()

        state1 = agg.initialize()
        for _ in range(5):
            state1 = agg.accumulate(state1, 1)

        state2 = agg.initialize()
        for _ in range(3):
            state2 = agg.accumulate(state2, 1)

        merged = agg.merge(state1, state2)
        assert agg.finalize(merged) == 8

    def test_sum_aggregator(self):
        """Test SumAggregator."""
        from truthound.execution.distributed.protocols import SumAggregator

        agg = SumAggregator()
        state = agg.initialize()

        for i in range(1, 11):
            state = agg.accumulate(state, i)

        assert agg.finalize(state) == 55  # Sum of 1..10

    def test_sum_aggregator_with_nulls(self):
        """Test SumAggregator handles nulls."""
        from truthound.execution.distributed.protocols import SumAggregator

        agg = SumAggregator()
        state = agg.initialize()

        state = agg.accumulate(state, 10)
        state = agg.accumulate(state, None)
        state = agg.accumulate(state, 5)

        assert agg.finalize(state) == 15

    def test_mean_aggregator(self):
        """Test MeanAggregator (Welford's algorithm)."""
        from truthound.execution.distributed.protocols import MeanAggregator

        agg = MeanAggregator()
        state = agg.initialize()

        values = [2, 4, 6, 8, 10]
        for v in values:
            state = agg.accumulate(state, v)

        result = agg.finalize(state)
        assert abs(result - 6.0) < 1e-10  # Mean of [2,4,6,8,10] = 6

    def test_mean_aggregator_merge(self):
        """Test MeanAggregator parallel merge."""
        from truthound.execution.distributed.protocols import MeanAggregator

        agg = MeanAggregator()

        # Partition 1
        state1 = agg.initialize()
        for v in [1, 2, 3]:
            state1 = agg.accumulate(state1, v)

        # Partition 2
        state2 = agg.initialize()
        for v in [4, 5, 6]:
            state2 = agg.accumulate(state2, v)

        merged = agg.merge(state1, state2)
        result = agg.finalize(merged)

        assert abs(result - 3.5) < 1e-10  # Mean of [1..6] = 3.5

    def test_std_aggregator(self):
        """Test StdAggregator."""
        from truthound.execution.distributed.protocols import StdAggregator

        agg = StdAggregator(ddof=0)  # Population std
        state = agg.initialize()

        values = [2, 4, 4, 4, 5, 5, 7, 9]
        for v in values:
            state = agg.accumulate(state, v)

        result = agg.finalize(state)
        assert abs(result - 2.0) < 1e-10  # Known population std

    def test_minmax_aggregator(self):
        """Test MinMaxAggregator."""
        from truthound.execution.distributed.protocols import MinMaxAggregator

        agg = MinMaxAggregator()
        state = agg.initialize()

        for v in [5, 2, 8, 1, 9, 3]:
            state = agg.accumulate(state, v)

        result = agg.finalize(state)
        assert result["min"] == 1
        assert result["max"] == 9

    def test_minmax_aggregator_merge(self):
        """Test MinMaxAggregator merge."""
        from truthound.execution.distributed.protocols import MinMaxAggregator

        agg = MinMaxAggregator()

        state1 = agg.initialize()
        for v in [5, 2, 8]:
            state1 = agg.accumulate(state1, v)

        state2 = agg.initialize()
        for v in [1, 9, 3]:
            state2 = agg.accumulate(state2, v)

        merged = agg.merge(state1, state2)
        result = agg.finalize(merged)

        assert result["min"] == 1
        assert result["max"] == 9

    def test_null_count_aggregator(self):
        """Test NullCountAggregator."""
        from truthound.execution.distributed.protocols import NullCountAggregator

        agg = NullCountAggregator()
        state = agg.initialize()

        values = [1, None, 2, None, None, 3]
        for v in values:
            state = agg.accumulate(state, v)

        result = agg.finalize(state)
        assert result["null_count"] == 3
        assert result["total_count"] == 6

    def test_distinct_count_aggregator(self):
        """Test DistinctCountAggregator."""
        from truthound.execution.distributed.protocols import DistinctCountAggregator

        agg = DistinctCountAggregator()
        state = agg.initialize()

        values = ["a", "b", "a", "c", "b", "a"]
        for v in values:
            state = agg.accumulate(state, v)

        result = agg.finalize(state)
        assert result == 3  # 3 distinct values

    def test_get_aggregator(self):
        """Test aggregator registry lookup."""
        from truthound.execution.distributed.protocols import get_aggregator

        agg = get_aggregator("count")
        assert agg.name == "count"

        agg = get_aggregator("mean")
        assert agg.name == "mean"

        with pytest.raises(KeyError):
            get_aggregator("unknown_aggregator")

    def test_register_aggregator(self):
        """Test custom aggregator registration."""
        from truthound.execution.distributed.protocols import (
            BaseAggregator,
            register_aggregator,
            get_aggregator,
        )

        @dataclass
        class ProductState:
            product: float = 1.0

        class ProductAggregator(BaseAggregator[ProductState]):
            name = "product"

            def initialize(self):
                return ProductState()

            def accumulate(self, state, value):
                if value is not None:
                    state.product *= float(value)
                return state

            def merge(self, state1, state2):
                return ProductState(product=state1.product * state2.product)

            def finalize(self, state):
                return state.product

        register_aggregator("product", ProductAggregator)
        agg = get_aggregator("product")
        assert agg.name == "product"


# =============================================================================
# Aggregation Plan Tests
# =============================================================================


class TestAggregationPlan:
    """Tests for AggregationPlan builder."""

    def test_create_empty_plan(self):
        """Test creating empty plan."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = AggregationPlan()
        assert len(plan) == 0

    def test_add_aggregations(self):
        """Test adding aggregations."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = AggregationPlan()
        plan.add_count()
        plan.add_null_count("email")
        plan.add_mean("price")

        assert len(plan) == 3

    def test_method_chaining(self):
        """Test fluent interface."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = (
            AggregationPlan()
            .add_count()
            .add_null_count("email")
            .add_distinct_count("category")
            .add_stats("price")
        )

        # add_stats adds 5 aggregations
        assert len(plan) >= 8

    def test_add_stats(self):
        """Test add_stats helper."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = AggregationPlan()
        plan.add_stats("price")

        # Should add count, null_count, mean, std, minmax
        assert len(plan) == 5
        ops = [a[1] for a in plan.aggregations]
        assert "count" in ops
        assert "null_count" in ops
        assert "mean" in ops
        assert "std" in ops
        assert "minmax" in ops

    def test_group_by(self):
        """Test group by."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = AggregationPlan()
        plan.add_count()
        plan.add_group_by("category", "region")

        assert plan.group_by == ["category", "region"]

    def test_add_filter(self):
        """Test filter condition."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = AggregationPlan()
        plan.add_filter("price > 0")

        assert plan.filter_condition == "price > 0"

    def test_to_spec(self):
        """Test conversion to AggregationSpec."""
        from truthound.execution.distributed.aggregations import AggregationPlan

        plan = (
            AggregationPlan()
            .add_count("*", alias="total")
            .add_mean("price")
        )

        spec = plan.to_spec()
        assert len(spec.aggregations) == 2


# =============================================================================
# Distributed Aggregator Tests
# =============================================================================


class TestDistributedAggregator:
    """Tests for DistributedAggregator."""

    def test_infer_backend_local(self):
        """Test backend inference for local data."""
        import polars as pl
        from truthound.execution.distributed.aggregations import DistributedAggregator
        from truthound.execution.distributed.protocols import ComputeBackend

        df = pl.DataFrame({"a": [1, 2, 3]})
        agg = DistributedAggregator(df)

        assert agg._inferred_backend == ComputeBackend.LOCAL

    def test_execute_local(self):
        """Test local execution."""
        import polars as pl
        from truthound.execution.distributed.aggregations import (
            DistributedAggregator,
            AggregationPlan,
        )

        df = pl.DataFrame({
            "price": [10.0, 20.0, 30.0, None, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        })

        plan = (
            AggregationPlan()
            .add_count("price")
            .add_null_count("price", alias="price_nulls")
            .add_mean("price")
        )

        agg = DistributedAggregator(df)
        results = agg.execute(plan)

        assert results["price_count"] == 4  # Non-null count
        assert results["price_nulls"]["null_count"] == 1
        assert abs(results["price_mean"] - 27.5) < 0.1

    def test_execute_distinct_count_local(self):
        """Test distinct count execution."""
        import polars as pl
        from truthound.execution.distributed.aggregations import (
            DistributedAggregator,
            AggregationPlan,
        )

        df = pl.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A"],
        })

        plan = AggregationPlan().add_distinct_count("category")
        agg = DistributedAggregator(df)
        results = agg.execute(plan)

        assert results["category_distinct_count"] == 3

    def test_execute_minmax_local(self):
        """Test minmax execution."""
        import polars as pl
        from truthound.execution.distributed.aggregations import (
            DistributedAggregator,
            AggregationPlan,
        )

        df = pl.DataFrame({
            "value": [5, 2, 8, 1, 9],
        })

        plan = AggregationPlan().add_minmax("value")
        agg = DistributedAggregator(df)
        results = agg.execute(plan)

        assert results["value_minmax"]["min"] == 1
        assert results["value_minmax"]["max"] == 9


# =============================================================================
# Arrow Bridge Tests
# =============================================================================

# Check if pyarrow is available
try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class TestArrowBridge:
    """Tests for Arrow bridge."""

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_polars_to_arrow(self):
        """Test Polars to Arrow conversion."""
        import polars as pl
        import pyarrow as pa
        from truthound.execution.distributed.arrow_bridge import ArrowBridge

        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        bridge = ArrowBridge()
        table = bridge.to_arrow(df)

        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert table.column_names == ["a", "b"]

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_arrow_to_polars(self):
        """Test Arrow to Polars conversion."""
        import polars as pl
        import pyarrow as pa
        from truthound.execution.distributed.arrow_bridge import ArrowBridge

        table = pa.table({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        })

        bridge = ArrowBridge()
        df = bridge.to_polars(table)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == ["a", "b"]

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_to_polars_lazy(self):
        """Test conversion to LazyFrame."""
        import polars as pl
        import pyarrow as pa
        from truthound.execution.distributed.arrow_bridge import ArrowBridge

        table = pa.table({"a": [1, 2, 3]})
        bridge = ArrowBridge()

        lf = bridge.to_polars(table, collect=False)
        assert isinstance(lf, pl.LazyFrame)

        df = bridge.to_polars(table, collect=True)
        assert isinstance(df, pl.DataFrame)

    def test_config_options(self):
        """Test bridge configuration."""
        from truthound.execution.distributed.arrow_bridge import (
            ArrowBridge,
            ArrowBridgeConfig,
            ArrowConversionStrategy,
        )

        config = ArrowBridgeConfig(
            strategy=ArrowConversionStrategy.PANDAS,
            batch_size=10000,
            preserve_index=True,
        )

        bridge = ArrowBridge(config)
        assert bridge.config.batch_size == 10000
        assert bridge.config.preserve_index


# =============================================================================
# Registry Tests
# =============================================================================


class TestDistributedEngineRegistry:
    """Tests for engine registry."""

    def test_list_engines(self):
        """Test listing registered engines."""
        from truthound.execution.distributed.registry import list_distributed_engines

        engines = list_distributed_engines()
        assert "spark" in engines

    def test_register_custom_engine(self):
        """Test registering custom engine."""
        from truthound.execution.distributed.registry import (
            register_distributed_engine,
            list_distributed_engines,
        )
        from truthound.execution.distributed.base import BaseDistributedEngine
        from truthound.execution.distributed.protocols import ComputeBackend

        class CustomEngine(BaseDistributedEngine):
            backend_type = ComputeBackend.LOCAL

            def _get_partition_count(self):
                return 1

            def _get_partition_info(self):
                return []

            def _execute_on_partitions(self, op, func, cols=None):
                return []

            def _aggregate_distributed(self, spec):
                return {}

            def _to_arrow_batches(self, batch_size=None):
                return []

            def _repartition(self, num):
                return self

            def count_rows(self):
                return 0

            def get_columns(self):
                return []

            def count_nulls(self, col):
                return 0

            def count_distinct(self, col):
                return 0

            def get_stats(self, col):
                return {}

            def to_polars_lazyframe(self):
                import polars as pl
                return pl.DataFrame().lazy()

            def sample(self, n=1000, seed=None):
                return self

        register_distributed_engine(
            "custom",
            CustomEngine,
            ComputeBackend.LOCAL,
        )

        assert "custom" in list_distributed_engines()


# =============================================================================
# Validator Adapter Tests
# =============================================================================


class TestValidatorAdapter:
    """Tests for validator adapter."""

    def test_classify_null_validator(self):
        """Test validator classification for null check."""
        from truthound.execution.distributed.validator_adapter import (
            classify_validator,
            ValidatorCapability,
        )

        class MockNullValidator:
            name = "null_checker"
            category = "completeness"

        validator = MockNullValidator()
        caps = classify_validator(validator)

        assert ValidatorCapability.NULL_CHECK in caps

    def test_classify_duplicate_validator(self):
        """Test validator classification for duplicate check."""
        from truthound.execution.distributed.validator_adapter import (
            classify_validator,
            ValidatorCapability,
        )

        class MockDuplicateValidator:
            name = "duplicate_detector"
            category = "uniqueness"

        validator = MockDuplicateValidator()
        caps = classify_validator(validator)

        assert ValidatorCapability.DUPLICATE_CHECK in caps

    def test_classify_ml_validator(self):
        """Test validator classification for ML-based."""
        from truthound.execution.distributed.validator_adapter import (
            classify_validator,
            ValidatorCapability,
        )

        class MockMLValidator:
            name = "anomaly_detector"
            category = "ml"

        validator = MockMLValidator()
        caps = classify_validator(validator)

        assert ValidatorCapability.ML_BASED in caps

    def test_execution_strategy_enum(self):
        """Test ExecutionStrategy enum."""
        from truthound.execution.distributed.validator_adapter import ExecutionStrategy

        assert ExecutionStrategy.NATIVE == "native"
        assert ExecutionStrategy.SAMPLE_AND_LOCAL == "sample_and_local"
        assert ExecutionStrategy.FULL_CONVERSION == "full_conversion"

    def test_adapter_config(self):
        """Test AdapterConfig defaults."""
        from truthound.execution.distributed.validator_adapter import (
            AdapterConfig,
            ExecutionStrategy,
        )

        config = AdapterConfig()
        assert config.strategy == ExecutionStrategy.AUTO
        assert config.sample_size == 100_000
        assert config.native_threshold == 1_000_000


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_stats_plan(self):
        """Test create_stats_plan helper."""
        from truthound.execution.distributed.aggregations import create_stats_plan

        plan = create_stats_plan(["price", "quantity"])

        # Each column gets 5 aggregations
        assert len(plan) == 10

    def test_create_null_count_plan(self):
        """Test create_null_count_plan helper."""
        from truthound.execution.distributed.aggregations import create_null_count_plan

        plan = create_null_count_plan(["email", "phone", "name"])

        assert len(plan) == 3
        for col, op, _ in plan.aggregations:
            assert op == "null_count"

    def test_aggregate_distributed(self):
        """Test aggregate_distributed function."""
        import polars as pl
        from truthound.execution.distributed.aggregations import (
            aggregate_distributed,
            AggregationPlan,
        )

        df = pl.DataFrame({
            "value": [1, 2, 3, 4, 5],
        })

        plan = AggregationPlan().add_count("value").add_sum("value")
        results = aggregate_distributed(df, plan)

        assert results["value_count"] == 5
        assert results["value_sum"] == 15


# =============================================================================
# Integration Tests
# =============================================================================


class TestDistributedIntegration:
    """Integration tests for distributed execution framework."""

    def test_full_workflow_local(self):
        """Test complete workflow with local data."""
        import polars as pl
        from truthound.execution.distributed.aggregations import (
            DistributedAggregator,
            AggregationPlan,
        )

        # Create test data
        df = pl.DataFrame({
            "id": list(range(1000)),
            "price": [i * 1.5 if i % 10 != 0 else None for i in range(1000)],
            "category": ["A", "B", "C"] * 333 + ["A"],
        })

        # Create plan
        plan = (
            AggregationPlan(name="full_analysis")
            .add_count("id")
            .add_null_count("price", alias="price_nulls")
            .add_mean("price")
            .add_std("price")
            .add_minmax("price")
            .add_distinct_count("category")
        )

        # Execute
        agg = DistributedAggregator(df)
        results = agg.execute(plan)

        # Verify results
        assert results["id_count"] == 1000
        assert results["price_nulls"]["null_count"] == 100  # 10% nulls
        assert results["category_distinct_count"] == 3
        assert results["price_minmax"]["min"] is not None
        assert results["price_minmax"]["max"] is not None

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_arrow_roundtrip(self):
        """Test Polars -> Arrow -> Polars roundtrip."""
        import polars as pl
        import pyarrow as pa
        from truthound.execution.distributed.arrow_bridge import ArrowBridge

        # Original data
        original = pl.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
            "null_col": [1, None, 3, None, 5],
        })

        bridge = ArrowBridge()

        # Convert to Arrow
        arrow_table = bridge.to_arrow(original)

        # Convert back to Polars
        result = bridge.to_polars(arrow_table)

        # Verify data integrity
        assert result.shape == original.shape
        assert result.columns == original.columns
        assert result["int_col"].to_list() == original["int_col"].to_list()
        assert result["str_col"].to_list() == original["str_col"].to_list()
