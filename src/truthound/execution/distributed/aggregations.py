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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    ComputeBackend,
    get_aggregator,
)

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
    scope: AggregationScope = AggregationScope.GLOBAL

    def add(
        self,
        column: str,
        operation: str,
        alias: str = "",
        **params: Any,
    ) -> AggregationPlan:
        """Add an aggregation to the plan.

        Args:
            column: Column to aggregate.
            operation: Aggregation operation name.
            alias: Optional result alias.
            **params: Additional parameters.

        Returns:
            Self for method chaining.
        """
        self.aggregations.append(
            (
                column,
                operation,
                {"alias": alias or f"{column}_{operation}", **params},
            )
        )
        return self

    def add_count(self, column: str = "*", alias: str = "") -> AggregationPlan:
        """Add count aggregation."""
        return self.add(column, "count", alias)

    def add_null_count(self, column: str, alias: str = "") -> AggregationPlan:
        """Add null count aggregation."""
        return self.add(column, "null_count", alias)

    def add_distinct_count(
        self,
        column: str,
        alias: str = "",
        max_sample: int = 100_000,
    ) -> AggregationPlan:
        """Add distinct count aggregation."""
        return self.add(column, "distinct_count", alias, max_sample=max_sample)

    def add_sum(self, column: str, alias: str = "") -> AggregationPlan:
        """Add sum aggregation."""
        return self.add(column, "sum", alias)

    def add_mean(self, column: str, alias: str = "") -> AggregationPlan:
        """Add mean aggregation."""
        return self.add(column, "mean", alias)

    def add_std(
        self,
        column: str,
        alias: str = "",
        ddof: int = 1,
    ) -> AggregationPlan:
        """Add standard deviation aggregation."""
        return self.add(column, "std", alias, ddof=ddof)

    def add_minmax(self, column: str, alias: str = "") -> AggregationPlan:
        """Add min/max aggregation."""
        return self.add(column, "minmax", alias)

    def add_stats(self, column: str) -> AggregationPlan:
        """Add comprehensive statistics (count, null, mean, std, min, max)."""
        return (
            self.add_count(column, f"{column}_count")
            .add_null_count(column, f"{column}_nulls")
            .add_mean(column, f"{column}_mean")
            .add_std(column, f"{column}_std")
            .add_minmax(column, f"{column}_minmax")
        )

    def add_group_by(self, *columns: str) -> AggregationPlan:
        """Add group by columns."""
        self.group_by.extend(columns)
        return self

    def add_filter(self, condition: str) -> AggregationPlan:
        """Add filter condition."""
        self.filter_condition = condition
        return self

    def to_spec(self) -> AggregationSpec:
        """Convert to AggregationSpec."""
        from copy import deepcopy

        spec = AggregationSpec(
            group_by=list(self.group_by),
            scope=AggregationScope.GROUPED
            if self.group_by and self.scope == AggregationScope.GLOBAL
            else AggregationScope(self.scope),
            filter_condition=self.filter_condition,
        )
        for column, operation, params in self.aggregations:
            alias = params.get("alias", f"{column}_{operation}")
            spec.add(
                column,
                operation,
                alias,
                **deepcopy({k: v for k, v in params.items() if k != "alias"}),
            )
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
        *,
        dask_scheduler: Any = None,
    ) -> None:
        """Initialize distributed aggregator.

        Args:
            data: Data to aggregate (DataFrame, etc.).
            backend: Compute backend to use.
        """
        self._data = data
        self._dask_scheduler = dask_scheduler
        self._backend = backend
        self._inferred_backend = self._infer_backend()
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        """Check owner authority before new work and after each observed batch."""
        self._execution_guard = guard

    def _check_guard(self) -> None:
        if self._execution_guard is not None:
            self._execution_guard()

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
        """Return the final merged result, retaining every group and typed key."""
        final = None
        for snapshot in self.execute_streaming(plan):
            final = snapshot
        assert final is not None
        return final

    def execute_streaming(
        self, plan: AggregationPlan, batch_size: int = 10000
    ) -> Iterator[dict[str, Any]]:
        """Yield cumulative independent snapshots after each native batch.

        Only aggregate states cross the worker boundary. Filters are SQL
        expressions evaluated by Core's Polars bridge inside each batch; no
        caller Python is evaluated. Grouped output is {"groups": [{"keys": ...,
        "aggregations": ...}]}; global output remains an alias-to-value mapping.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        captured, aggregators = self._prepare_plan(plan)
        if captured.scope == AggregationScope.PARTITION:
            return self._stream_partitions(captured, aggregators, batch_size)
        return self._stream(captured, aggregators, batch_size)

    def _prepare_plan(self, plan):
        from copy import deepcopy

        self._check_guard()
        captured = deepcopy(plan)
        captured.scope = AggregationScope(captured.scope)
        if captured.scope == AggregationScope.GROUPED and not captured.group_by:
            raise ValueError("Grouped aggregation requires group columns")
        aliases = [
            params.get("alias", f"{column}_{operation}")
            for column, operation, params in captured.aggregations
        ]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Aggregation aliases must be unique")
        if len(captured.group_by) != len(set(captured.group_by)):
            raise ValueError("Group columns must be unique")
        aggregators = [
            _plan_aggregator(column, operation, params)
            for column, operation, params in captured.aggregations
        ]
        return captured, aggregators

    def execute_partitioned(self, plan: AggregationPlan) -> dict[str, Any]:
        """Finalize each native partition separately, including empty partitions.

        IDs are zero-based ordinals in the selected native iteration. They are
        not durable IDs across repartitioning or separate backend executions.
        """

        captured, aggregators = self._prepare_plan(plan)
        final = {"partitions": []}
        for snapshot in self._stream_partitions(captured, aggregators, 10000):
            final = snapshot
        return final

    def _stream_partitions(self, captured, aggregators, batch_size):
        from contextlib import closing
        from copy import deepcopy

        import cloudpickle

        result = []
        with closing(self._partial_payloads(captured, aggregators, batch_size, True)) as partials:
            while True:
                self._check_guard()
                try:
                    payload = next(partials)
                except StopIteration:
                    break
                self._check_guard()
                result.append(
                    {
                        "partition_id": len(result),
                        "result": _finalize_plan(captured, aggregators, cloudpickle.loads(payload)),
                    }
                )
                yield {"partitions": deepcopy(result)}
        if not result:
            yield {"partitions": []}

    def _stream(self, plan, aggregators, batch_size):
        from contextlib import closing
        from copy import deepcopy

        import cloudpickle

        states = {} if plan.group_by else {(): ((), [agg.initialize() for agg in aggregators])}
        observed = False
        with closing(self._partial_payloads(plan, aggregators, batch_size)) as partials:
            while True:
                self._check_guard()
                try:
                    payload = next(partials)
                except StopIteration:
                    break
                # Only trusted in-operation workers create these state bytes.
                self._check_guard()
                partial = cloudpickle.loads(payload)
                for key, (values, incoming) in partial.items():
                    if key in states:
                        previous = states[key][1]
                        states[key] = (
                            values,
                            [
                                agg.merge(left, right)
                                for agg, left, right in zip(
                                    aggregators, previous, incoming, strict=True
                                )
                            ],
                        )
                    else:
                        states[key] = (values, incoming)
                observed = True
                yield deepcopy(_finalize_plan(plan, aggregators, states))
        if not observed:
            self._check_guard()
            yield deepcopy(_finalize_plan(plan, aggregators, states))

    def _partial_payloads(self, plan, aggregators, batch_size, whole_partition=False):
        self._check_guard()
        backend = self._inferred_backend
        if backend == ComputeBackend.DASK:
            import dask

            for partition in self._data.to_delayed():
                self._check_guard()
                task = dask.delayed(_pandas_partials)(
                    partition, plan, aggregators, batch_size, whole_partition
                )
                yield from task.compute(scheduler=self._dask_scheduler)
        elif backend == ComputeBackend.SPARK:
            columns = self._data.columns

            def batches(rows):
                import pyarrow as pa

                batch = []
                for row in rows:
                    batch.append(row.asDict(recursive=True))
                    if len(batch) == batch_size:
                        yield from _arrow_partials(
                            pa.Table.from_pylist(batch), plan, aggregators, batch_size
                        )
                        batch = []
                if batch:
                    yield from _arrow_partials(
                        pa.Table.from_pylist(batch), plan, aggregators, batch_size
                    )

            def partial_partition(rows):
                payloads = batches(rows)
                if whole_partition:
                    yield _merge_payloads(payloads, plan, aggregators)
                else:
                    yield from payloads

            # Validate names even when Spark has no rows/partitions to inspect.
            _validate_plan_columns(plan, columns)
            yield from self._data.rdd.mapPartitions(partial_partition).toLocalIterator()
        elif backend == ComputeBackend.RAY:
            import pyarrow as pa

            if whole_partition:
                import ray

                aggregate_block = ray.remote(_aggregate_arrow_partition)
                for block in self._data.to_arrow_refs():
                    self._check_guard()
                    yield ray.get(aggregate_block.remote(block, plan, aggregators, batch_size))
                return

            def partial_batch(table):
                payloads = _arrow_partials(table, plan, aggregators, batch_size)
                if whole_partition:
                    payloads = [_merge_payloads(payloads, plan, aggregators)]
                return pa.table(
                    {
                        "state": pa.array(
                            list(payloads),
                            type=pa.binary(),
                        )
                    }
                )

            result = self._data.map_batches(
                partial_batch,
                batch_format="pyarrow",
                batch_size=None if whole_partition else batch_size,
            )
            for batch in result.iter_batches(batch_format="pyarrow", batch_size=1):
                yield from batch.column("state").to_pylist()
        else:
            import polars as pl
            import pyarrow as pa

            if isinstance(self._data, pl.LazyFrame):
                data = self._data.collect().to_arrow()
            elif isinstance(self._data, pl.DataFrame):
                data = self._data.to_arrow()
            elif isinstance(self._data, pa.Table):
                data = self._data
            else:
                import pandas as pd

                if not isinstance(self._data, pd.DataFrame):
                    raise TypeError("Unsupported aggregation input")
                data = pa.Table.from_pandas(self._data, preserve_index=False)
            payloads = _arrow_partials(data, plan, aggregators, batch_size)
            if whole_partition:
                yield _merge_payloads(payloads, plan, aggregators)
            else:
                yield from payloads


def execute_spec(data, spec, *, dask_scheduler=None):
    """Execute a captured native grouped/filtered spec through Core plan states."""
    from copy import deepcopy

    scope = AggregationScope(spec.scope)
    if scope == AggregationScope.GROUPED and not spec.group_by:
        raise ValueError("Grouped aggregation requires group columns")
    plan = AggregationPlan(
        group_by=list(spec.group_by), filter_condition=spec.filter_condition, scope=scope
    )
    for item in spec.aggregations:
        plan.add(item.column, item.operation, item.alias, **deepcopy(item.params))
    executor = DistributedAggregator(data, dask_scheduler=dask_scheduler)
    if scope == AggregationScope.PARTITION:
        return executor.execute_partitioned(plan)
    return executor.execute(plan)


def _plan_aggregator(column, operation, params):
    if column == "*" and operation != "count":
        raise ValueError("Only count supports the wildcard column")
    return get_aggregator(
        operation, **{key: value for key, value in params.items() if key != "alias"}
    )


def _validate_plan_columns(plan, columns):
    required = set(plan.group_by) | {column for column, _, _ in plan.aggregations if column != "*"}
    if not required.issubset(columns):
        raise ValueError("Unknown aggregation column")


def _pandas_partials(frame, plan, aggregators, batch_size, whole_partition=False):
    import pyarrow as pa

    payloads = _arrow_partials(
        pa.Table.from_pandas(frame, preserve_index=False), plan, aggregators, batch_size
    )
    if whole_partition:
        return [_merge_payloads(payloads, plan, aggregators)]
    return list(payloads)


def _merge_payloads(payloads, plan, aggregators):
    """Merge trusted worker batches without retaining raw partition rows."""
    import cloudpickle

    states = {} if plan.group_by else {(): ((), [agg.initialize() for agg in aggregators])}
    for payload in payloads:
        for key, (values, incoming) in cloudpickle.loads(payload).items():
            if key in states:
                incoming = [
                    agg.merge(left, right)
                    for agg, left, right in zip(aggregators, states[key][1], incoming, strict=True)
                ]
            states[key] = (values, incoming)
    return cloudpickle.dumps(states)


def _aggregate_arrow_partition(table, plan, aggregators, batch_size):
    # map_batches skips zero-row blocks in Ray; explicit block tasks retain
    # their empty aggregate without moving block values into the driver.
    return _merge_payloads(_arrow_partials(table, plan, aggregators, batch_size), plan, aggregators)


def _arrow_partials(table, plan, aggregators, batch_size):
    import cloudpickle
    import polars as pl

    from truthound.execution.distributed.protocols import _distinct_identity

    _validate_plan_columns(plan, table.column_names)
    for batch in table.to_batches(max_chunksize=batch_size):
        frame = pl.from_arrow(batch)
        if plan.filter_condition:
            frame = frame.filter(pl.sql_expr(plan.filter_condition))
        states = {}
        for row in frame.iter_rows(named=True):
            values = tuple(row[column] for column in plan.group_by)
            key = tuple(_distinct_identity(value) for value in values)
            if key not in states:
                states[key] = (values, [agg.initialize() for agg in aggregators])
            current = states[key][1]
            for index, ((column, operation, _), agg) in enumerate(
                zip(plan.aggregations, aggregators, strict=True)
            ):
                value = row[column] if column != "*" else 1
                if operation == "count" and column != "*" and value is None:
                    continue
                current[index] = agg.accumulate(current[index], value)
        yield cloudpickle.dumps(states)


def _finalize_plan(plan, aggregators, states):
    def finalize(current):
        return {
            params.get("alias", f"{column}_{operation}"): agg.finalize(state)
            for (column, operation, params), agg, state in zip(
                plan.aggregations, aggregators, current, strict=True
            )
        }

    if plan.group_by:
        return {
            "groups": [
                {
                    "keys": dict(zip(plan.group_by, values, strict=True)),
                    "aggregations": finalize(current),
                }
                for values, current in states.values()
            ]
        }
    return finalize(states[()][1])


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
