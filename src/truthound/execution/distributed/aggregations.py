"""Distributed aggregation framework for validation operations.

This module provides a flexible aggregation framework that:
- Supports custom aggregators with proper map-reduce semantics
- Batches multiple aggregations for efficiency
- Provides streaming aggregation for very large datasets
- Includes common aggregators for validation tasks

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     AggregationExecutor                          │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │                  AggregationPlan                          │  │
    │   │   (Defines what aggregations to perform)                  │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │              DistributedAggregator                        │  │
    │   │   (Executes aggregations in distributed manner)           │  │
    │   │                                                           │  │
    │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
    │   │   │  Partition  │  │  Partition  │  │  Partition  │      │  │
    │   │   │   1 (Map)   │  │   2 (Map)   │  │   N (Map)   │      │  │
    │   │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │  │
    │   │          │                │                │              │  │
    │   │          └────────────────┼────────────────┘              │  │
    │   │                           ▼                               │  │
    │   │                    ┌─────────────┐                        │  │
    │   │                    │   Reduce    │                        │  │
    │   │                    │  (Merge)    │                        │  │
    │   │                    └─────────────┘                        │  │
    │   │                           │                               │  │
    │   │                           ▼                               │  │
    │   │                    ┌─────────────┐                        │  │
    │   │                    │  Finalize   │                        │  │
    │   │                    └─────────────┘                        │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Example:
    >>> from truthound.execution.distributed.aggregations import (
    ...     AggregationPlan,
    ...     DistributedAggregator,
    ... )
    >>>
    >>> # Create aggregation plan
    >>> plan = AggregationPlan()
    >>> plan.add_null_count("email")
    >>> plan.add_stats("price")
    >>> plan.add_distinct_count("category")
    >>>
    >>> # Execute on Spark DataFrame
    >>> aggregator = DistributedAggregator(spark_df)
    >>> results = aggregator.execute(plan)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, TypeVar

from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    BaseAggregator,
    ComputeBackend,
    DistributedAggregation,
    DistributedResult,
    get_aggregator,
    register_aggregator,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

logger = logging.getLogger(__name__)


# =============================================================================
# Aggregation Plan
# =============================================================================


@dataclass
class AggregationPlan:
    """Plan for executing multiple aggregations.

    AggregationPlan allows you to define what aggregations should be
    performed and then execute them efficiently in a single pass
    over the data when possible.

    Example:
        >>> plan = AggregationPlan()
        >>> plan.add_null_count("email")
        >>> plan.add_null_count("phone")
        >>> plan.add_stats("price")
        >>> plan.add_distinct_count("category")
        >>>
        >>> # Single-pass execution
        >>> results = executor.execute(plan)
    """

    name: str = "default"
    aggregations: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    filter_condition: str | None = None

    def add(
        self,
        column: str,
        operation: str,
        alias: str = "",
        **params: Any,
    ) -> "AggregationPlan":
        """Add an aggregation to the plan.

        Args:
            column: Column to aggregate.
            operation: Aggregation operation name.
            alias: Optional result alias.
            **params: Additional parameters.

        Returns:
            Self for method chaining.
        """
        self.aggregations.append((
            column,
            operation,
            {"alias": alias or f"{column}_{operation}", **params},
        ))
        return self

    def add_count(self, column: str = "*", alias: str = "") -> "AggregationPlan":
        """Add count aggregation."""
        return self.add(column, "count", alias)

    def add_null_count(self, column: str, alias: str = "") -> "AggregationPlan":
        """Add null count aggregation."""
        return self.add(column, "null_count", alias)

    def add_distinct_count(
        self,
        column: str,
        alias: str = "",
        max_sample: int = 100_000,
    ) -> "AggregationPlan":
        """Add distinct count aggregation."""
        return self.add(column, "distinct_count", alias, max_sample=max_sample)

    def add_sum(self, column: str, alias: str = "") -> "AggregationPlan":
        """Add sum aggregation."""
        return self.add(column, "sum", alias)

    def add_mean(self, column: str, alias: str = "") -> "AggregationPlan":
        """Add mean aggregation."""
        return self.add(column, "mean", alias)

    def add_std(
        self,
        column: str,
        alias: str = "",
        ddof: int = 1,
    ) -> "AggregationPlan":
        """Add standard deviation aggregation."""
        return self.add(column, "std", alias, ddof=ddof)

    def add_minmax(self, column: str, alias: str = "") -> "AggregationPlan":
        """Add min/max aggregation."""
        return self.add(column, "minmax", alias)

    def add_stats(self, column: str) -> "AggregationPlan":
        """Add comprehensive statistics (count, null, mean, std, min, max)."""
        return (
            self.add_count(column, f"{column}_count")
            .add_null_count(column, f"{column}_nulls")
            .add_mean(column, f"{column}_mean")
            .add_std(column, f"{column}_std")
            .add_minmax(column, f"{column}_minmax")
        )

    def add_group_by(self, *columns: str) -> "AggregationPlan":
        """Add group by columns."""
        self.group_by.extend(columns)
        return self

    def add_filter(self, condition: str) -> "AggregationPlan":
        """Add filter condition."""
        self.filter_condition = condition
        return self

    def to_spec(self) -> AggregationSpec:
        """Convert to AggregationSpec."""
        spec = AggregationSpec(group_by=self.group_by)
        for column, operation, params in self.aggregations:
            alias = params.pop("alias", f"{column}_{operation}")
            spec.add(column, operation, alias, **params)
        return spec

    def __len__(self) -> int:
        """Get number of aggregations."""
        return len(self.aggregations)


# =============================================================================
# Aggregation Executor
# =============================================================================


class AggregationExecutor(ABC):
    """Abstract base class for aggregation executors.

    Executors implement the actual execution of aggregation plans
    on different backends (Spark, Dask, Ray, etc.).
    """

    @abstractmethod
    def execute(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute an aggregation plan.

        Args:
            plan: Aggregation plan to execute.

        Returns:
            Dictionary of results keyed by alias.
        """
        pass

    @abstractmethod
    def execute_streaming(
        self,
        plan: AggregationPlan,
        batch_size: int = 10000,
    ) -> Iterator[dict[str, Any]]:
        """Execute aggregation plan in streaming mode.

        Args:
            plan: Aggregation plan.
            batch_size: Batch size for streaming.

        Yields:
            Partial results for each batch.
        """
        pass


# =============================================================================
# Distributed Aggregator
# =============================================================================


StateT = TypeVar("StateT")


@dataclass
class PartialAggregateState:
    """State for partial aggregates from a partition."""

    partition_id: int
    states: dict[str, Any]  # alias -> aggregator state
    row_count: int = 0
    duration_ms: float = 0.0


class DistributedAggregator:
    """Executes aggregations in a distributed manner.

    This class orchestrates the map-reduce style execution of
    aggregations across partitions, handling:
    - Efficient batching of multiple aggregations
    - Proper state management for custom aggregators
    - Parallel execution on distributed backends

    Example:
        >>> aggregator = DistributedAggregator(spark_df)
        >>> plan = AggregationPlan().add_stats("price").add_null_count("email")
        >>> results = aggregator.execute(plan)
    """

    def __init__(
        self,
        data: Any,
        backend: ComputeBackend = ComputeBackend.AUTO,
    ) -> None:
        """Initialize distributed aggregator.

        Args:
            data: Data to aggregate (DataFrame, etc.).
            backend: Compute backend to use.
        """
        self._data = data
        self._backend = backend
        self._inferred_backend = self._infer_backend()

    def _infer_backend(self) -> ComputeBackend:
        """Infer backend from data type."""
        if self._backend != ComputeBackend.AUTO:
            return self._backend

        # Try to infer from data type
        data_type = type(self._data).__name__

        if "DataFrame" in data_type:
            module = type(self._data).__module__
            if "pyspark" in module:
                return ComputeBackend.SPARK
            elif "dask" in module:
                return ComputeBackend.DASK

        if "Dataset" in data_type:
            module = type(self._data).__module__
            if "ray" in module:
                return ComputeBackend.RAY

        return ComputeBackend.LOCAL

    def execute(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute aggregation plan.

        Args:
            plan: Plan to execute.

        Returns:
            Aggregation results.
        """
        backend = self._inferred_backend

        if backend == ComputeBackend.SPARK:
            return self._execute_spark(plan)
        elif backend == ComputeBackend.DASK:
            return self._execute_dask(plan)
        elif backend == ComputeBackend.RAY:
            return self._execute_ray(plan)
        else:
            return self._execute_local(plan)

    def execute_streaming(
        self,
        plan: AggregationPlan,
        batch_size: int = 10000,
    ) -> Iterator[dict[str, Any]]:
        """Execute plan in streaming mode.

        Args:
            plan: Plan to execute.
            batch_size: Batch size.

        Yields:
            Partial results.
        """
        # For now, just execute normally and yield once
        # TODO: Implement true streaming aggregation
        yield self.execute(plan)

    def _execute_spark(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute on Spark using native aggregations where possible."""
        from pyspark.sql import functions as F

        df = self._data

        # Apply filter if specified
        if plan.filter_condition:
            df = df.filter(plan.filter_condition)

        # Separate native Spark aggs from custom aggs
        spark_native = {"count", "sum", "mean", "min", "max", "std", "var"}
        native_aggs = []
        custom_aggs = []

        for column, operation, params in plan.aggregations:
            if operation in spark_native:
                native_aggs.append((column, operation, params))
            else:
                custom_aggs.append((column, operation, params))

        results = {}

        # Execute native Spark aggregations
        if native_aggs:
            spark_funcs = {
                "count": lambda c: F.count(c if c != "*" else F.lit(1)),
                "sum": F.sum,
                "mean": F.avg,
                "min": F.min,
                "max": F.max,
                "std": F.stddev,
                "var": F.variance,
            }

            exprs = []
            for column, operation, params in native_aggs:
                alias = params.get("alias", f"{column}_{operation}")
                expr = spark_funcs[operation](column).alias(alias)
                exprs.append(expr)

            if plan.group_by:
                agg_df = df.groupBy(*plan.group_by).agg(*exprs)
                rows = agg_df.collect()
                # Return as list of dicts for grouped results
                for row in rows:
                    for column, operation, params in native_aggs:
                        alias = params.get("alias", f"{column}_{operation}")
                        results[alias] = row[alias]
            else:
                row = df.agg(*exprs).collect()[0]
                for column, operation, params in native_aggs:
                    alias = params.get("alias", f"{column}_{operation}")
                    results[alias] = row[alias]

        # Execute custom aggregations using map-reduce
        for column, operation, params in custom_aggs:
            alias = params.get("alias", f"{column}_{operation}")
            aggregator = get_aggregator(operation, **{k: v for k, v in params.items() if k != "alias"})
            result = self._map_reduce_spark(df, column, aggregator)
            results[alias] = result

        return results

    def _map_reduce_spark(
        self,
        df: "SparkDataFrame",
        column: str,
        aggregator: BaseAggregator,
    ) -> Any:
        """Execute aggregator using Spark map-reduce.

        Args:
            df: Spark DataFrame.
            column: Column to aggregate.
            aggregator: Aggregator to use.

        Returns:
            Aggregation result.
        """
        # Map phase: compute partial aggregate per partition
        def map_partition(iterator: Iterator) -> Iterator:
            state = aggregator.initialize()
            for row in iterator:
                value = row[column] if column in row.asDict() else None
                state = aggregator.accumulate(state, value)
            yield state

        partial_states = df.rdd.mapPartitions(map_partition).collect()

        # Reduce phase: merge all partial states
        if not partial_states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, partial_states)
        return aggregator.finalize(final_state)

    def _execute_dask(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute on Dask."""
        import dask.dataframe as dd
        import pandas as pd

        ddf = self._data

        # Apply filter
        if plan.filter_condition:
            ddf = ddf.query(plan.filter_condition)

        results = {}

        # Native Dask aggregations
        native_ops = {
            "count": lambda s: s.count(),
            "sum": lambda s: s.sum(),
            "mean": lambda s: s.mean(),
            "min": lambda s: s.min(),
            "max": lambda s: s.max(),
            "std": lambda s: s.std(),
            "var": lambda s: s.var(),
        }

        for column, operation, params in plan.aggregations:
            alias = params.get("alias", f"{column}_{operation}")

            if operation in native_ops:
                if column == "*":
                    value = len(ddf)
                else:
                    value = native_ops[operation](ddf[column]).compute()
                results[alias] = value
            elif operation == "null_count":
                null_count = ddf[column].isna().sum().compute()
                total_count = len(ddf)
                results[alias] = {"null_count": null_count, "total_count": total_count}
            elif operation == "distinct_count":
                distinct = ddf[column].nunique().compute()
                results[alias] = distinct
            elif operation == "minmax":
                min_val = ddf[column].min().compute()
                max_val = ddf[column].max().compute()
                results[alias] = {"min": min_val, "max": max_val}
            else:
                # Use custom aggregator
                aggregator = get_aggregator(
                    operation,
                    **{k: v for k, v in params.items() if k != "alias"},
                )
                result = self._map_reduce_dask(ddf, column, aggregator)
                results[alias] = result

        return results

    def _map_reduce_dask(
        self,
        ddf: Any,
        column: str,
        aggregator: BaseAggregator,
    ) -> Any:
        """Execute aggregator using Dask map-reduce."""
        import pandas as pd

        def map_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            state = aggregator.initialize()
            for value in pdf[column]:
                state = aggregator.accumulate(state, value)
            return pd.DataFrame([{"state": state}])

        partial_states = ddf.map_partitions(
            map_partition,
            meta={"state": object},
        ).compute()

        states = partial_states["state"].tolist()

        if not states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, states)
        return aggregator.finalize(final_state)

    def _execute_ray(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute on Ray."""
        import ray

        dataset = self._data
        results = {}

        for column, operation, params in plan.aggregations:
            alias = params.get("alias", f"{column}_{operation}")

            if operation == "count":
                if column == "*":
                    results[alias] = dataset.count()
                else:
                    results[alias] = dataset.count()
            elif operation == "sum":
                results[alias] = dataset.sum(column)
            elif operation == "mean":
                results[alias] = dataset.mean(column)
            elif operation == "min":
                results[alias] = dataset.min(column)
            elif operation == "max":
                results[alias] = dataset.max(column)
            elif operation == "std":
                results[alias] = dataset.std(column)
            else:
                # Use map-reduce with custom aggregator
                aggregator = get_aggregator(
                    operation,
                    **{k: v for k, v in params.items() if k != "alias"},
                )
                result = self._map_reduce_ray(dataset, column, aggregator)
                results[alias] = result

        return results

    def _map_reduce_ray(
        self,
        dataset: Any,
        column: str,
        aggregator: BaseAggregator,
    ) -> Any:
        """Execute aggregator using Ray map-reduce."""
        import ray

        @ray.remote
        def map_batch(batch: dict) -> Any:
            state = aggregator.initialize()
            for value in batch[column]:
                state = aggregator.accumulate(state, value)
            return state

        # Map phase
        batch_refs = [
            map_batch.remote(batch)
            for batch in dataset.iter_batches(batch_format="pydict")
        ]

        partial_states = ray.get(batch_refs)

        if not partial_states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, partial_states)
        return aggregator.finalize(final_state)

    def _execute_local(self, plan: AggregationPlan) -> dict[str, Any]:
        """Execute locally using Polars or Pandas."""
        import polars as pl

        # Try to convert to Polars
        if isinstance(self._data, pl.DataFrame):
            df = self._data
        elif isinstance(self._data, pl.LazyFrame):
            df = self._data.collect()
        else:
            # Try pandas
            try:
                import pandas as pd
                if isinstance(self._data, pd.DataFrame):
                    df = pl.from_pandas(self._data)
                else:
                    raise ValueError(f"Unsupported data type: {type(self._data)}")
            except ImportError:
                raise ValueError(f"Unsupported data type: {type(self._data)}")

        # Apply filter
        if plan.filter_condition:
            # Basic filter parsing
            df = df.filter(pl.sql_expr(plan.filter_condition))

        results = {}

        for column, operation, params in plan.aggregations:
            alias = params.get("alias", f"{column}_{operation}")

            if operation == "count":
                if column == "*":
                    results[alias] = len(df)
                else:
                    results[alias] = df.get_column(column).count()
            elif operation == "sum":
                results[alias] = df.get_column(column).sum()
            elif operation == "mean":
                results[alias] = df.get_column(column).mean()
            elif operation == "min":
                results[alias] = df.get_column(column).min()
            elif operation == "max":
                results[alias] = df.get_column(column).max()
            elif operation == "std":
                ddof = params.get("ddof", 1)
                results[alias] = df.get_column(column).std(ddof=ddof)
            elif operation == "var":
                ddof = params.get("ddof", 1)
                results[alias] = df.get_column(column).var(ddof=ddof)
            elif operation == "null_count":
                null_count = df.get_column(column).null_count()
                total_count = len(df)
                results[alias] = {"null_count": null_count, "total_count": total_count}
            elif operation == "distinct_count":
                results[alias] = df.get_column(column).n_unique()
            elif operation == "minmax":
                min_val = df.get_column(column).min()
                max_val = df.get_column(column).max()
                results[alias] = {"min": min_val, "max": max_val}
            else:
                # Use custom aggregator
                aggregator = get_aggregator(
                    operation,
                    **{k: v for k, v in params.items() if k != "alias"},
                )
                state = aggregator.initialize()
                for value in df.get_column(column):
                    state = aggregator.accumulate(state, value)
                results[alias] = aggregator.finalize(state)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def aggregate_distributed(
    data: Any,
    plan: AggregationPlan,
    backend: ComputeBackend = ComputeBackend.AUTO,
) -> dict[str, Any]:
    """Convenience function for distributed aggregation.

    Args:
        data: Data to aggregate.
        plan: Aggregation plan.
        backend: Compute backend.

    Returns:
        Aggregation results.
    """
    aggregator = DistributedAggregator(data, backend)
    return aggregator.execute(plan)


def create_stats_plan(columns: list[str]) -> AggregationPlan:
    """Create a plan for computing statistics on multiple columns.

    Args:
        columns: Columns to compute stats for.

    Returns:
        AggregationPlan with stats for all columns.
    """
    plan = AggregationPlan(name="stats")
    for column in columns:
        plan.add_stats(column)
    return plan


def create_null_count_plan(columns: list[str]) -> AggregationPlan:
    """Create a plan for counting nulls in multiple columns.

    Args:
        columns: Columns to check.

    Returns:
        AggregationPlan with null counts.
    """
    plan = AggregationPlan(name="null_counts")
    for column in columns:
        plan.add_null_count(column)
    return plan
