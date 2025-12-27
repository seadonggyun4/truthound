"""Protocol definitions for distributed execution engines.

This module defines the structural typing protocols that all distributed
execution engine implementations should follow. These protocols enable:
- Type-safe distributed operations
- Backend-agnostic interfaces
- Extensibility for custom backends

Design Principles:
1. Protocol-first: Define interfaces before implementations
2. Composable: Small, focused protocols that can be combined
3. Backend-agnostic: Same interface for Spark, Dask, Ray, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa


# =============================================================================
# Enums
# =============================================================================


class ExecutionMode(str, Enum):
    """Execution modes for distributed operations."""

    EAGER = "eager"  # Execute immediately
    LAZY = "lazy"  # Build execution plan, execute on collect
    STREAMING = "streaming"  # Process data in streaming fashion


class PartitionStrategy(str, Enum):
    """Strategies for data partitioning."""

    ROW_HASH = "row_hash"  # Hash-based row partitioning
    ROW_RANGE = "row_range"  # Range-based row partitioning
    COLUMN = "column"  # Partition by columns
    ROUND_ROBIN = "round_robin"  # Round-robin distribution
    CUSTOM = "custom"  # Custom partitioning function


class AggregationScope(str, Enum):
    """Scope of aggregation operations."""

    GLOBAL = "global"  # Aggregate across all partitions
    PARTITION = "partition"  # Aggregate within partition
    COLUMN = "column"  # Aggregate per column
    GROUPED = "grouped"  # Aggregate by group key


class ComputeBackend(str, Enum):
    """Supported distributed compute backends."""

    SPARK = "spark"
    DASK = "dask"
    RAY = "ray"
    LOCAL = "local"
    AUTO = "auto"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class PartitionInfo:
    """Information about a data partition.

    Attributes:
        partition_id: Unique identifier for this partition.
        total_partitions: Total number of partitions.
        row_start: Starting row index (inclusive).
        row_end: Ending row index (exclusive).
        columns: Columns in this partition.
        size_bytes: Estimated size in bytes.
        host: Host where this partition resides.
        metadata: Additional partition metadata.
    """

    partition_id: int
    total_partitions: int
    row_start: int = 0
    row_end: int = 0
    columns: tuple[str, ...] = field(default_factory=tuple)
    size_bytes: int = 0
    host: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        """Get number of rows in this partition."""
        return self.row_end - self.row_start


@dataclass
class DistributedResult:
    """Result from a distributed operation.

    Attributes:
        partition_id: Source partition ID.
        operation: Operation that produced this result.
        value: The computed value.
        row_count: Number of rows processed.
        duration_ms: Processing duration in milliseconds.
        errors: List of errors encountered.
        warnings: List of warnings.
        metadata: Additional result metadata.
    """

    partition_id: int
    operation: str
    value: Any
    row_count: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if operation succeeded without errors."""
        return len(self.errors) == 0


@dataclass
class DistributedAggregation:
    """Specification for a distributed aggregation.

    Attributes:
        column: Column to aggregate.
        operation: Aggregation operation name.
        params: Additional parameters for the operation.
        alias: Result column alias.
    """

    column: str
    operation: str  # "count", "sum", "mean", "min", "max", "std", "var", etc.
    params: dict[str, Any] = field(default_factory=dict)
    alias: str = ""

    def __post_init__(self) -> None:
        if not self.alias:
            self.alias = f"{self.column}_{self.operation}"


@dataclass
class AggregationSpec:
    """Specification for multiple aggregations."""

    aggregations: list[DistributedAggregation] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    scope: AggregationScope = AggregationScope.GLOBAL

    def add(
        self,
        column: str,
        operation: str,
        alias: str = "",
        **params: Any,
    ) -> "AggregationSpec":
        """Add an aggregation to the spec."""
        self.aggregations.append(
            DistributedAggregation(
                column=column,
                operation=operation,
                params=params,
                alias=alias,
            )
        )
        return self


# =============================================================================
# Protocols
# =============================================================================


T = TypeVar("T")
ResultT = TypeVar("ResultT")


@runtime_checkable
class DistributedDataProtocol(Protocol):
    """Protocol for distributed data representations.

    This protocol abstracts over Spark DataFrames, Dask DataFrames,
    Ray Datasets, etc.
    """

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        ...

    @property
    def num_partitions(self) -> int:
        """Get number of partitions."""
        ...

    def repartition(self, num_partitions: int) -> "DistributedDataProtocol":
        """Repartition the data."""
        ...

    def get_partition_info(self) -> list[PartitionInfo]:
        """Get information about all partitions."""
        ...


@runtime_checkable
class MapReduceProtocol(Protocol[T, ResultT]):
    """Protocol for map-reduce style operations.

    Type Parameters:
        T: Input type for map function.
        ResultT: Output type from reduce function.
    """

    def map_partitions(
        self,
        func: Callable[[Iterator[T]], Iterator[ResultT]],
    ) -> "MapReduceProtocol[ResultT, ResultT]":
        """Apply function to each partition."""
        ...

    def reduce(
        self,
        func: Callable[[ResultT, ResultT], ResultT],
    ) -> ResultT:
        """Reduce all partitions to a single value."""
        ...

    def collect(self) -> list[ResultT]:
        """Collect all results to driver."""
        ...


@runtime_checkable
class DistributedAggregatorProtocol(Protocol):
    """Protocol for distributed aggregation operations.

    Aggregators must support:
    - Partial aggregation (per-partition)
    - Final aggregation (cross-partition merge)
    - Incremental updates
    """

    def initialize(self) -> Any:
        """Initialize accumulator state."""
        ...

    def accumulate(self, state: Any, value: Any) -> Any:
        """Add a value to the accumulator."""
        ...

    def merge(self, state1: Any, state2: Any) -> Any:
        """Merge two accumulator states."""
        ...

    def finalize(self, state: Any) -> Any:
        """Finalize and return the result."""
        ...


@runtime_checkable
class DistributedBackendProtocol(Protocol):
    """Protocol for distributed computing backends.

    All distributed backends (Spark, Dask, Ray) must implement
    this protocol to be usable with Truthound's distributed
    execution framework.
    """

    @property
    def backend_type(self) -> ComputeBackend:
        """Get the backend type."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the backend is available."""
        ...

    def initialize(self) -> None:
        """Initialize the backend (connect, start cluster, etc.)."""
        ...

    def shutdown(self) -> None:
        """Shutdown the backend (disconnect, stop cluster, etc.)."""
        ...

    def distribute_data(
        self,
        data: Any,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_HASH,
    ) -> DistributedDataProtocol:
        """Distribute data across the cluster."""
        ...

    def map_partitions(
        self,
        data: DistributedDataProtocol,
        func: Callable[[Any], DistributedResult],
    ) -> list[DistributedResult]:
        """Execute function on each partition."""
        ...

    def aggregate(
        self,
        data: DistributedDataProtocol,
        spec: AggregationSpec,
    ) -> dict[str, Any]:
        """Perform distributed aggregation."""
        ...

    def collect(
        self,
        data: DistributedDataProtocol,
        limit: int | None = None,
    ) -> Any:
        """Collect distributed data to local."""
        ...


@runtime_checkable
class ArrowConvertibleProtocol(Protocol):
    """Protocol for types that can convert to/from Arrow.

    Arrow is used as the zero-copy interchange format between
    different compute backends.
    """

    def to_arrow(self) -> "pa.Table":
        """Convert to PyArrow Table."""
        ...

    @classmethod
    def from_arrow(cls, table: "pa.Table") -> Any:
        """Create from PyArrow Table."""
        ...


@runtime_checkable
class DistributedExecutionProtocol(Protocol):
    """Protocol for distributed execution engines.

    Execution engines provide the high-level interface for
    running validations in a distributed manner.
    """

    @property
    def backend(self) -> DistributedBackendProtocol:
        """Get the underlying backend."""
        ...

    def count_rows(self) -> int:
        """Count total rows (distributed)."""
        ...

    def count_nulls(self, column: str) -> int:
        """Count nulls in a column (distributed)."""
        ...

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns (distributed)."""
        ...

    def count_distinct(self, column: str) -> int:
        """Count distinct values (distributed)."""
        ...

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics (distributed)."""
        ...

    def aggregate(self, spec: AggregationSpec) -> dict[str, Any]:
        """Perform distributed aggregation."""
        ...

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame (via Arrow)."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseAggregator(ABC, Generic[T]):
    """Abstract base class for distributed aggregators.

    Aggregators implement the map-reduce pattern for computing
    aggregate statistics across partitions.

    Type Parameters:
        T: Type of the accumulated state.
    """

    name: str = "base"

    @abstractmethod
    def initialize(self) -> T:
        """Create initial accumulator state."""
        pass

    @abstractmethod
    def accumulate(self, state: T, value: Any) -> T:
        """Add a value to the accumulator."""
        pass

    @abstractmethod
    def merge(self, state1: T, state2: T) -> T:
        """Merge two accumulator states."""
        pass

    @abstractmethod
    def finalize(self, state: T) -> Any:
        """Convert accumulator state to final result."""
        pass


# =============================================================================
# Built-in Aggregators
# =============================================================================


@dataclass
class CountState:
    """State for count aggregator."""

    count: int = 0


class CountAggregator(BaseAggregator[CountState]):
    """Distributed count aggregator."""

    name = "count"

    def initialize(self) -> CountState:
        return CountState()

    def accumulate(self, state: CountState, value: Any) -> CountState:
        state.count += 1
        return state

    def merge(self, state1: CountState, state2: CountState) -> CountState:
        return CountState(count=state1.count + state2.count)

    def finalize(self, state: CountState) -> int:
        return state.count


@dataclass
class SumState:
    """State for sum aggregator."""

    total: float = 0.0


class SumAggregator(BaseAggregator[SumState]):
    """Distributed sum aggregator."""

    name = "sum"

    def initialize(self) -> SumState:
        return SumState()

    def accumulate(self, state: SumState, value: Any) -> SumState:
        if value is not None:
            state.total += float(value)
        return state

    def merge(self, state1: SumState, state2: SumState) -> SumState:
        return SumState(total=state1.total + state2.total)

    def finalize(self, state: SumState) -> float:
        return state.total


@dataclass
class MeanState:
    """State for mean aggregator (uses Welford's online algorithm)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean


class MeanAggregator(BaseAggregator[MeanState]):
    """Distributed mean aggregator using parallel Welford's algorithm."""

    name = "mean"

    def initialize(self) -> MeanState:
        return MeanState()

    def accumulate(self, state: MeanState, value: Any) -> MeanState:
        if value is None:
            return state
        x = float(value)
        state.count += 1
        delta = x - state.mean
        state.mean += delta / state.count
        delta2 = x - state.mean
        state.m2 += delta * delta2
        return state

    def merge(self, state1: MeanState, state2: MeanState) -> MeanState:
        """Merge using parallel Welford's algorithm."""
        if state1.count == 0:
            return state2
        if state2.count == 0:
            return state1

        count = state1.count + state2.count
        delta = state2.mean - state1.mean
        mean = state1.mean + delta * state2.count / count
        m2 = (
            state1.m2
            + state2.m2
            + delta**2 * state1.count * state2.count / count
        )
        return MeanState(count=count, mean=mean, m2=m2)

    def finalize(self, state: MeanState) -> float:
        return state.mean if state.count > 0 else 0.0


@dataclass
class StdState(MeanState):
    """State for standard deviation aggregator (extends MeanState)."""

    pass


class StdAggregator(BaseAggregator[StdState]):
    """Distributed standard deviation aggregator."""

    name = "std"

    def __init__(self, ddof: int = 1) -> None:
        self.ddof = ddof  # Delta degrees of freedom

    def initialize(self) -> StdState:
        return StdState()

    def accumulate(self, state: StdState, value: Any) -> StdState:
        if value is None:
            return state
        x = float(value)
        state.count += 1
        delta = x - state.mean
        state.mean += delta / state.count
        delta2 = x - state.mean
        state.m2 += delta * delta2
        return state

    def merge(self, state1: StdState, state2: StdState) -> StdState:
        if state1.count == 0:
            return state2
        if state2.count == 0:
            return state1

        count = state1.count + state2.count
        delta = state2.mean - state1.mean
        mean = state1.mean + delta * state2.count / count
        m2 = (
            state1.m2
            + state2.m2
            + delta**2 * state1.count * state2.count / count
        )
        return StdState(count=count, mean=mean, m2=m2)

    def finalize(self, state: StdState) -> float:
        if state.count <= self.ddof:
            return 0.0
        variance = state.m2 / (state.count - self.ddof)
        return variance**0.5


@dataclass
class MinMaxState:
    """State for min/max aggregator."""

    min_value: float | None = None
    max_value: float | None = None


class MinMaxAggregator(BaseAggregator[MinMaxState]):
    """Distributed min/max aggregator."""

    name = "minmax"

    def initialize(self) -> MinMaxState:
        return MinMaxState()

    def accumulate(self, state: MinMaxState, value: Any) -> MinMaxState:
        if value is None:
            return state
        x = float(value)
        if state.min_value is None or x < state.min_value:
            state.min_value = x
        if state.max_value is None or x > state.max_value:
            state.max_value = x
        return state

    def merge(self, state1: MinMaxState, state2: MinMaxState) -> MinMaxState:
        min_val = None
        max_val = None

        if state1.min_value is not None and state2.min_value is not None:
            min_val = min(state1.min_value, state2.min_value)
        else:
            min_val = state1.min_value or state2.min_value

        if state1.max_value is not None and state2.max_value is not None:
            max_val = max(state1.max_value, state2.max_value)
        else:
            max_val = state1.max_value or state2.max_value

        return MinMaxState(min_value=min_val, max_value=max_val)

    def finalize(self, state: MinMaxState) -> dict[str, float | None]:
        return {"min": state.min_value, "max": state.max_value}


@dataclass
class NullCountState:
    """State for null count aggregator."""

    null_count: int = 0
    total_count: int = 0


class NullCountAggregator(BaseAggregator[NullCountState]):
    """Distributed null count aggregator."""

    name = "null_count"

    def initialize(self) -> NullCountState:
        return NullCountState()

    def accumulate(self, state: NullCountState, value: Any) -> NullCountState:
        state.total_count += 1
        if value is None:
            state.null_count += 1
        return state

    def merge(self, state1: NullCountState, state2: NullCountState) -> NullCountState:
        return NullCountState(
            null_count=state1.null_count + state2.null_count,
            total_count=state1.total_count + state2.total_count,
        )

    def finalize(self, state: NullCountState) -> dict[str, int]:
        return {"null_count": state.null_count, "total_count": state.total_count}


@dataclass
class DistinctState:
    """State for approximate distinct count (HyperLogLog)."""

    # Simplified version - for production use HyperLogLog
    seen: set = field(default_factory=set)
    count: int = 0


class DistinctCountAggregator(BaseAggregator[DistinctState]):
    """Distributed distinct count aggregator.

    Note: For very large cardinalities, consider using HyperLogLog.
    """

    name = "distinct_count"

    def __init__(self, max_sample: int = 100_000) -> None:
        self.max_sample = max_sample

    def initialize(self) -> DistinctState:
        return DistinctState()

    def accumulate(self, state: DistinctState, value: Any) -> DistinctState:
        if value is not None and len(state.seen) < self.max_sample:
            # Use hash for memory efficiency
            try:
                state.seen.add(hash(value))
            except TypeError:
                state.seen.add(hash(str(value)))
        state.count += 1
        return state

    def merge(self, state1: DistinctState, state2: DistinctState) -> DistinctState:
        merged = DistinctState()
        merged.seen = state1.seen | state2.seen
        merged.count = state1.count + state2.count
        return merged

    def finalize(self, state: DistinctState) -> int:
        return len(state.seen)


# =============================================================================
# Aggregator Registry
# =============================================================================


AGGREGATOR_REGISTRY: dict[str, type[BaseAggregator]] = {
    "count": CountAggregator,
    "sum": SumAggregator,
    "mean": MeanAggregator,
    "std": StdAggregator,
    "minmax": MinMaxAggregator,
    "null_count": NullCountAggregator,
    "distinct_count": DistinctCountAggregator,
}


def get_aggregator(name: str, **kwargs: Any) -> BaseAggregator:
    """Get an aggregator by name.

    Args:
        name: Aggregator name.
        **kwargs: Additional arguments for the aggregator.

    Returns:
        Aggregator instance.

    Raises:
        KeyError: If aggregator not found.
    """
    if name not in AGGREGATOR_REGISTRY:
        raise KeyError(
            f"Unknown aggregator: {name}. "
            f"Available: {list(AGGREGATOR_REGISTRY.keys())}"
        )
    return AGGREGATOR_REGISTRY[name](**kwargs)


def register_aggregator(name: str, aggregator_class: type[BaseAggregator]) -> None:
    """Register a custom aggregator.

    Args:
        name: Name to register under.
        aggregator_class: Aggregator class to register.
    """
    AGGREGATOR_REGISTRY[name] = aggregator_class
