# Validator Optimization

Truthound's optimization module provides tools to maximize validator execution performance on large-scale datasets.

## Overview

The optimization module covers the following areas:

| Area | Module | Key Features |
|------|--------|--------------|
| **DAG Execution** | `orchestrator` | Dependency-based execution order, parallel execution |
| **Graph Algorithms** | `graph` | Cycle detection, topological sort, DFS |
| **Covariance Computation** | `covariance` | Incremental covariance, Woodbury updates |
| **Geographic Operations** | `geo` | Vectorized Haversine, spatial indexing |
| **Aggregation Optimization** | `aggregation` | Lazy aggregation, streaming joins |
| **Profiling** | `profiling` | Execution time, memory, throughput measurement |

## DAG-Based Execution Orchestration

`ValidatorDAG` manages dependencies between validators and determines optimal execution order.

### Basic Usage

```python
from truthound.validators.optimization import (
    ValidatorDAG,
    ExecutionPlan,
    ParallelExecutionStrategy,
)

# Create DAG
dag = ValidatorDAG()

# Add validators
dag.add_validators([
    NullValidator(),
    DuplicateValidator(),
    RangeValidator(),
])

# Build execution plan
plan = dag.build_execution_plan()

# Execute with parallel strategy
strategy = ParallelExecutionStrategy(max_workers=4)
results = plan.execute(lf, strategy)
```

### Execution Phases (ValidatorPhase)

Validators are automatically assigned to execution phases:

| Phase | Description | Priority |
|-------|-------------|----------|
| `SCHEMA` | Schema validation (column existence, types) | 1 |
| `COMPLETENESS` | Null checks, missing values | 2 |
| `UNIQUENESS` | Duplicate detection, key validation | 3 |
| `FORMAT` | Pattern matching, format validation | 4 |
| `RANGE` | Value range, distribution checks | 5 |
| `STATISTICAL` | Statistics, outliers | 6 |
| `CROSS_TABLE` | Multi-table validation | 7 |
| `CUSTOM` | User-defined | 8 |

### ValidatorNode

Metadata can be specified when adding validators to the DAG:

```python
from truthound.validators.optimization import ValidatorNode, ValidatorPhase

node = dag.add_validator(
    validator=MyValidator(),
    dependencies={"null_validator"},  # Explicit dependencies
    provides={"completeness"},         # Provided capabilities
    phase=ValidatorPhase.STATISTICAL,  # Execution phase
    priority=50,                       # Priority within phase (lower = earlier)
    estimated_cost=2.5,                # Estimated cost (for adaptive scheduling)
)
```

### Execution Strategies

#### SequentialExecutionStrategy

Executes validators sequentially:

```python
from truthound.validators.optimization import SequentialExecutionStrategy

strategy = SequentialExecutionStrategy()
result = plan.execute(lf, strategy)
```

#### ParallelExecutionStrategy

Parallel execution using `ThreadPoolExecutor`:

```python
from truthound.validators.optimization import ParallelExecutionStrategy

strategy = ParallelExecutionStrategy(max_workers=8)
result = plan.execute(lf, strategy)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_workers` | `int \| None` | `None` | Maximum workers. If None, uses `min(32, cpu_count + 4)` |

#### AdaptiveExecutionStrategy

Dynamically selects sequential/parallel strategy based on the number of validators per level:

```python
from truthound.validators.optimization import AdaptiveExecutionStrategy

strategy = AdaptiveExecutionStrategy(
    parallel_threshold=3,  # Use parallel when 3 or more validators
    max_workers=4,
)
result = plan.execute(lf, strategy)
```

### ExecutionResult

Execution results provide various metrics:

```python
result = plan.execute(lf, strategy)

# Basic metrics
print(f"Total duration: {result.total_duration_ms}ms")
print(f"Total validators: {result.total_validators}")
print(f"Success: {result.success_count}")
print(f"Failure: {result.failure_count}")
print(f"Total issues: {len(result.all_issues)}")

# Detailed metrics
metrics = result.get_metrics()
# {
#     "total_duration_ms": 150.5,
#     "total_validators": 10,
#     "total_issues": 25,
#     "success_count": 8,
#     "failure_count": 2,
#     "levels": 4,
#     "strategy": "parallel",
#     "parallelism_factor": 2.5,  # Sequential time / actual time
# }
```

### Convenience Functions

```python
from truthound.validators.optimization import (
    create_execution_plan,
    execute_validators,
)

# Create execution plan
plan = create_execution_plan(
    validators=[v1, v2, v3],
    dependencies={"v2": {"v1"}},  # v2 depends on v1
)

# Execute at once
result = execute_validators(
    validators=[v1, v2, v3],
    lf=df.lazy(),
    strategy=ParallelExecutionStrategy(max_workers=4),
)
```

## Graph Traversal Algorithms

Optimized graph algorithms for hierarchy and relationship data validation.

### IterativeDFS

Iterative depth-first search to avoid recursion limits:

```python
from truthound.validators.optimization import IterativeDFS

adjacency = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": [],
}

dfs = IterativeDFS(adjacency)

# Preorder traversal
for node in dfs.traverse(start="A", order="preorder"):
    print(node)  # A, B, D, C

# Find path
path = dfs.find_path("A", "D")  # ["A", "B", "D"]

# Compute depths
depths = dfs.compute_depths(roots=["A"])  # {"A": 0, "B": 1, "C": 1, "D": 2}
```

### TarjanSCC

Tarjan's Strongly Connected Components algorithm for O(V+E) cycle detection:

```python
from truthound.validators.optimization import TarjanSCC, CycleInfo

adjacency = {
    "A": ["B"],
    "B": ["C"],
    "C": ["A", "D"],  # A-B-C cycle
    "D": ["D"],       # Self-loop
}

tarjan = TarjanSCC(adjacency)

# Find all SCCs
sccs = tarjan.find_sccs()
# [["A", "C", "B"], ["D"]]  # SCCs with size > 1 are cycles

# Get cycle info
cycles = tarjan.find_cycles()
for cycle in cycles:
    print(f"Cycle: {cycle.nodes}")
    print(f"Length: {cycle.length}")
    print(f"Self-loop: {cycle.is_self_loop}")
```

### TopologicalSort

Topological sort using Kahn's algorithm:

```python
from truthound.validators.optimization import TopologicalSort

dependencies = {
    "task_a": ["task_b", "task_c"],  # a before b, c
    "task_b": ["task_d"],
    "task_c": ["task_d"],
    "task_d": [],
}

sorter = TopologicalSort(dependencies)

# Perform sort
try:
    order = sorter.sort()  # ["task_a", "task_b", "task_c", "task_d"]
except ValueError:
    print("Cycle exists")

# Check for cycles
if sorter.has_cycles():
    print("Cycle detected")
```

### GraphTraversalMixin

Mixin for graph operations in validators:

```python
from truthound.validators.optimization import GraphTraversalMixin

class HierarchyValidator(BaseValidator, GraphTraversalMixin):
    def validate(self, df):
        # Build adjacency list
        adj = self.build_adjacency_list(
            df,
            id_column="id",
            parent_column="parent_id",
            cache_key="my_hierarchy",  # Caching key
        )

        # Find all cycles
        cycles = self.find_all_cycles(adj)

        # Find hierarchy cycles (parent-child relationships)
        child_to_parent = self.build_child_to_parent(df, "id", "parent_id")
        hierarchy_cycles = self.find_hierarchy_cycles(
            child_to_parent,
            max_depth=1000,
        )

        # Compute node depths
        depths = self.compute_node_depths(adj, roots=["root"])

        # Topological sort
        sorted_nodes = self.topological_sort(adj)

        # Clear cache
        self.clear_adjacency_cache()
```

## Covariance Computation Optimization

Efficient covariance computation for multivariate methods like Mahalanobis distance on large datasets.

### IncrementalCovariance

Incremental covariance using Welford's online algorithm:

```python
from truthound.validators.optimization import IncrementalCovariance

# Incremental covariance for 10 features
cov = IncrementalCovariance(n_features=10)

# Update with streaming data
for batch in data_stream:
    cov.update_batch(batch)  # Batch update

# Single sample update
cov.update(sample)

# Get results
result = cov.get_result(regularization=1e-6)
print(result.mean)       # Mean vector
print(result.covariance) # Covariance matrix
print(result.n_samples)  # Sample count
```

### WoodburyCovariance

Woodbury matrix identity for efficient rank-k updates:

```python
from truthound.validators.optimization import WoodburyCovariance

# Create from data
cov = WoodburyCovariance.from_data(
    training_data,
    regularization=1e-6,
)

# Efficient sample add/remove
cov.add_sample(new_sample)
cov.remove_sample(old_sample)

# Compute Mahalanobis distance
distance = cov.mahalanobis(query_point)
distances = cov.mahalanobis_batch(query_points)
```

### RobustCovarianceEstimator

MCD (Minimum Covariance Determinant) estimation robust to outliers:

```python
from truthound.validators.optimization import RobustCovarianceEstimator

estimator = RobustCovarianceEstimator(
    contamination=0.1,    # Expected outlier ratio
    n_subsamples=10,      # Number of subsamples
    subsample_size=500,   # Each subsample size
    random_state=42,
)

result = estimator.fit(large_data)
# result.is_robust = True
```

### BatchCovarianceMixin

Mixin for batch covariance computation in validators:

```python
from truthound.validators.optimization import BatchCovarianceMixin

class MahalanobisValidator(BaseValidator, BatchCovarianceMixin):
    def validate(self, data):
        # Automatically select optimal method
        result = self.compute_covariance_auto(
            data,
            use_robust=True,
        )

        # Compute Mahalanobis distances
        distances = self.compute_mahalanobis_distances(data, result)

        # Create Woodbury covariance (for efficient updates)
        woodbury = self.create_woodbury_covariance(data)
```

**Configuration Attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `_batch_size` | 10000 | Batch size for incremental processing |
| `_robust_threshold` | 5000 | Use subsampled robust estimation above this |
| `_regularization` | 1e-6 | Regularization parameter |

## Geographic Operations Optimization

Vectorized geospatial computations for improved distance and polygon validation performance.

### VectorizedGeoMixin

```python
from truthound.validators.optimization import (
    VectorizedGeoMixin,
    DistanceUnit,
    BoundingBox,
)

class GeoValidator(BaseValidator, VectorizedGeoMixin):
    def validate(self, coords):
        lats = coords[:, 0]
        lons = coords[:, 1]

        # Vectorized Haversine distance
        distances = self.haversine_vectorized(
            lat1=lats,
            lon1=lons,
            lat2=target_lat,
            lon2=target_lon,
            unit=DistanceUnit.KILOMETERS,
        )

        # Pairwise distance matrix
        dist_matrix = self.pairwise_distances(
            lats1, lons1, lats2, lons2,
            unit=DistanceUnit.METERS,
        )

        # Memory-efficient chunked computation
        for start, end, chunk in self.pairwise_distances_chunked(
            lats1, lons1, lats2, lons2,
            chunk_size=50000,
        ):
            process_chunk(chunk)

        # Find nearest point
        idx, dist = self.nearest_point(
            query_lat, query_lon, lats, lons,
        )

        # k-nearest neighbors
        indices, distances = self.k_nearest_points(
            query_lat, query_lon, lats, lons, k=5,
        )

        # Find points within radius
        indices, distances = self.points_within_radius(
            center_lat, center_lon, lats, lons,
            radius=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # Bearing calculation
        bearings = self.bearing_vectorized(lat1, lon1, lat2, lon2)

        # Destination point calculation
        dest_lat, dest_lon = self.destination_point(
            lat, lon,
            bearing=45,
            distance=100,
            unit=DistanceUnit.KILOMETERS,
        )

        # Create bounding box
        bbox = self.create_bounding_box(
            center_lat, center_lon,
            radius=50,
            unit=DistanceUnit.KILOMETERS,
        )

        # Filter by bounding box
        mask, filtered_lats, filtered_lons = self.filter_by_bounding_box(
            lats, lons, bbox,
        )

        # Coordinate validation
        valid_mask = self.validate_coordinates(lats, lons)
```

### DistanceUnit

Supported distance units:

| Unit | Earth Radius |
|------|--------------|
| `METERS` | 6,371,000 m |
| `KILOMETERS` | 6,371 km |
| `MILES` | 3,958.8 mi |
| `NAUTICAL_MILES` | 3,440.1 nm |

### SpatialIndexMixin

Efficient spatial queries using BallTree:

```python
from truthound.validators.optimization import SpatialIndexMixin

class IndexedGeoValidator(BaseValidator, SpatialIndexMixin):
    def setup(self, reference_coords):
        # Build spatial index
        self.build_spatial_index(
            lats=reference_coords[:, 0],
            lons=reference_coords[:, 1],
            leaf_size=40,
        )

    def validate(self, query_coords):
        # Query within radius
        results = self.query_radius(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            radius_km=10,
        )

        # k-nearest neighbor query
        distances_km, indices = self.query_nearest(
            query_lats=query_coords[:, 0],
            query_lons=query_coords[:, 1],
            k=5,
        )

        # Clear index
        self.clear_spatial_index()
```

## Aggregation Optimization

Leverage Polars' lazy evaluation for memory-efficient aggregation operations.

### LazyAggregationMixin

```python
from truthound.validators.optimization import LazyAggregationMixin
import polars as pl

class CrossTableValidator(BaseValidator, LazyAggregationMixin):
    def validate(self, orders, order_items):
        # Lazy aggregation
        result = self.aggregate_lazy(
            lf=order_items.lazy(),
            group_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("total_qty"),
                pl.col("price").sum().alias("total_price"),
            ],
            cache_key="order_totals",  # Caching
        )

        # Join and aggregate in one operation
        result = self.aggregate_with_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            left_on="order_id",
            right_on="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            how="left",
        )

        # Streaming join for large tables
        result = self.streaming_aggregate_join(
            left=orders.lazy(),
            right=order_items.lazy(),
            join_key="order_id",
            agg_exprs=[pl.col("quantity").sum()],
            slice_size=100000,
        )

        # Compare aggregate values
        mismatches = self.compare_aggregates(
            source=orders,
            aggregated=result,
            key_column="order_id",
            source_column="expected_total",
            agg_column="total_qty",
            tolerance=0.01,
        )

        # Incremental aggregation
        updated = self.incremental_aggregate(
            existing=previous_result,
            new_data=new_items.lazy(),
            group_by="order_id",
            sum_columns=["quantity", "price"],
            count_column="item_count",
        )

        # Window aggregation
        with_windows = self.window_aggregate(
            lf=order_items.lazy(),
            partition_by=["order_id"],
            agg_exprs=[
                pl.col("quantity").sum().alias("order_total"),
                pl.col("price").mean().alias("order_avg_price"),
            ],
        )

        # Semi-join filter
        filtered = self.semi_join_filter(
            main=orders.lazy(),
            filter_by=active_orders.lazy(),
            on="order_id",
            anti=False,  # True for anti-join
        )

        # Multi-table aggregation
        result = self.multi_table_aggregate(
            tables={
                "orders": orders.lazy(),
                "items": order_items.lazy(),
                "products": products.lazy(),
            },
            joins=[
                ("orders", "items", ["order_id"]),
                ("items", "products", ["product_id"]),
            ],
            final_agg=[
                pl.col("quantity").sum(),
                pl.col("unit_price").mean(),
            ],
            final_group_by="category",
        )

        # Clear cache
        self.clear_aggregation_cache()
```

### AggregationExpressionBuilder

Fluent interface for building aggregation expressions:

```python
from truthound.validators.optimization import AggregationExpressionBuilder

builder = AggregationExpressionBuilder()

exprs = (
    builder
    .sum("quantity", alias="total_qty")
    .mean("price", alias="avg_price")
    .min("created_at", alias="first_order")
    .max("created_at", alias="last_order")
    .std("price", alias="price_std")
    .count(alias="order_count")
    .n_unique("customer_id", alias="unique_customers")
    .first("status")
    .last("updated_at")
    .custom(pl.col("discount").filter(pl.col("discount") > 0).mean())
    .build()
)

# Usage
result = lf.group_by("category").agg(exprs)
```

## Validator Profiling

Measure and analyze validator execution performance.

### ProfilerConfig

Profiler configuration:

```python
from truthound.validators.optimization import (
    ProfilerConfig,
    ProfilerMode,
)

# Basic configuration (timing only)
config = ProfilerConfig.basic()

# Standard configuration (timing + memory)
config = ProfilerConfig()  # Default

# Detailed configuration (with snapshots)
config = ProfilerConfig.detailed()

# Diagnostic configuration (maximum detail)
config = ProfilerConfig.diagnostic()

# Custom configuration
config = ProfilerConfig(
    mode=ProfilerMode.STANDARD,
    track_memory=True,
    track_gc=True,
    track_throughput=True,
    record_snapshots=False,
    max_snapshots=1000,
    memory_warning_mb=100,
    memory_critical_mb=500,
)
```

### ValidatorProfiler

```python
from truthound.validators.optimization import (
    ValidatorProfiler,
    ProfilerConfig,
)

# Create profiler
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())

# Start session
profiler.start_session("validation_run_1", attributes={"env": "prod"})

# Profile validators
for validator in validators:
    with profiler.profile(validator, rows_processed=100000) as ctx:
        issues = validator.validate(lf)
        ctx.set_issue_count(len(issues))
        ctx.add_attribute("columns", ["a", "b", "c"])

# End session
session = profiler.end_session()

# Analyze results
print(session.summary())
print(session.to_json())

# Slowest validators
slowest = profiler.get_slowest_validators(n=10)
# [("SlowValidator", 150.5), ("AnotherValidator", 100.2), ...]

# Memory-intensive validators
memory_heavy = profiler.get_memory_intensive_validators(n=10)

# Overall summary
summary = profiler.summary()
# {
#     "total_validators": 15,
#     "total_executions": 150,
#     "total_issues": 2500,
#     "total_time_ms": 5000.5,
#     "completed_sessions": 3,
#     "current_session_active": False,
#     "memory_tracking_available": True,
# }
```

### Convenience Functions

```python
from truthound.validators.optimization import (
    profile_validator,
    profiled,
    get_default_profiler,
)

# Context manager
with profile_validator(my_validator, rows_processed=10000) as ctx:
    issues = my_validator.validate(lf)
    ctx.set_issue_count(len(issues))

print(ctx.metrics.timing.mean_ms)

# Decorator
class MyValidator(Validator):
    @profiled(track_issues=True)
    def validate(self, lf):
        return [issue1, issue2]

# Global profiler
profiler = get_default_profiler()
```

### ValidatorMetrics

Collected metrics:

```python
metrics = profiler.get_metrics("MyValidator")

# Timing metrics
print(metrics.timing.mean_ms)    # Mean execution time
print(metrics.timing.median_ms)  # Median
print(metrics.timing.p95_ms)     # 95th percentile
print(metrics.timing.p99_ms)     # 99th percentile
print(metrics.timing.std_ms)     # Standard deviation

# Memory metrics
print(metrics.memory.mean_peak_mb)       # Mean peak memory
print(metrics.memory.max_peak_mb)        # Maximum peak memory
print(metrics.memory.total_gc_collections)  # GC collection count

# Throughput metrics
print(metrics.throughput.mean_rows_per_sec)  # Rows per second
print(metrics.throughput.total_rows)         # Total rows processed

# Issue metrics
print(metrics.total_issues)  # Total issues found
print(metrics.mean_issues)   # Mean issues
print(metrics.error_counts)  # Error counts
```

### ProfilingReport

Report generation:

```python
from truthound.validators.optimization import ProfilingReport

report = ProfilingReport(profiler)

# Text summary
print(report.text_summary())
# ============================================================
# VALIDATOR PROFILING REPORT
# ============================================================
# Total Validators: 15
# Total Executions: 150
# Total Issues Found: 2500
# Total Time: 5000.50ms
#
# ------------------------------------------------------------
# TOP 10 SLOWEST VALIDATORS (by mean execution time)
# ------------------------------------------------------------
#  1. SlowValidator: 150.50ms
#  2. AnotherValidator: 100.20ms
# ...

# HTML report
html = report.html_report()
with open("profile_report.html", "w") as f:
    f.write(html)

# Prometheus format export
prometheus_metrics = profiler.to_prometheus()
# # HELP validator_execution_duration_ms Validator execution duration
# # TYPE validator_execution_duration_ms gauge
# validator_execution_duration_ms_mean{validator="MyValidator",category="completeness"} 150.500
# ...
```

## Integration Example

Example using all optimization features together:

```python
import polars as pl
from truthound.validators.optimization import (
    # DAG orchestration
    ValidatorDAG,
    ParallelExecutionStrategy,
    # Profiling
    ValidatorProfiler,
    ProfilerConfig,
    # Mixins
    GraphTraversalMixin,
    BatchCovarianceMixin,
    VectorizedGeoMixin,
    LazyAggregationMixin,
)
from truthound.validators import NullValidator, RangeValidator


# Define custom validator
class OptimizedHierarchyValidator(
    BaseValidator,
    GraphTraversalMixin,
    BatchCovarianceMixin,
):
    def validate(self, df):
        # Graph cycle detection
        adj = self.build_adjacency_list(df, "id", "parent_id")
        cycles = self.find_all_cycles(adj)

        # Covariance for outlier detection
        numeric_data = df.select(pl.col(pl.Float64)).to_numpy()
        cov_result = self.compute_covariance_auto(numeric_data, use_robust=True)
        distances = self.compute_mahalanobis_distances(numeric_data, cov_result)

        return []


# Configure profiler
profiler = ValidatorProfiler(config=ProfilerConfig.detailed())
profiler.start_session("optimized_validation")

# Build DAG
dag = ValidatorDAG()
dag.add_validators([
    NullValidator(),
    RangeValidator(min_value=0),
    OptimizedHierarchyValidator(),
])

# Create and execute plan
plan = dag.build_execution_plan()
print(plan.get_summary())

# Parallel execution
strategy = ParallelExecutionStrategy(max_workers=4)

df = pl.DataFrame({"id": [1, 2, 3], "parent_id": [None, 1, 2], "value": [10, 20, 30]})

with profiler.profile(plan, rows_processed=len(df)) as ctx:
    result = plan.execute(df.lazy(), strategy)
    ctx.set_issue_count(len(result.all_issues))

# End session and report
session = profiler.end_session()
print(session.to_json())
```

## API Reference

### orchestrator Module

| Class/Function | Description |
|----------------|-------------|
| `ValidatorDAG` | Validator dependency DAG |
| `ValidatorNode` | Validator node wrapper |
| `ValidatorPhase` | Execution phase enum |
| `ExecutionPlan` | Execution plan |
| `ExecutionLevel` | Group of validators that can run in parallel |
| `ExecutionResult` | Execution result |
| `ExecutionStrategy` | Execution strategy abstract class |
| `SequentialExecutionStrategy` | Sequential execution strategy |
| `ParallelExecutionStrategy` | Parallel execution strategy |
| `AdaptiveExecutionStrategy` | Adaptive execution strategy |
| `create_execution_plan()` | Convenience function for creating execution plan |
| `execute_validators()` | Convenience function for executing validators |

### graph Module

| Class/Function | Description |
|----------------|-------------|
| `IterativeDFS` | Iterative depth-first search |
| `TarjanSCC` | Tarjan SCC algorithm |
| `TopologicalSort` | Topological sort |
| `GraphTraversalMixin` | Graph traversal mixin |
| `CycleInfo` | Cycle info dataclass |
| `NodeState` | Node state enum |

### covariance Module

| Class/Function | Description |
|----------------|-------------|
| `IncrementalCovariance` | Welford's incremental covariance |
| `WoodburyCovariance` | Woodbury update covariance |
| `RobustCovarianceEstimator` | MCD-based robust estimation |
| `BatchCovarianceMixin` | Batch covariance mixin |
| `CovarianceResult` | Covariance result dataclass |

### geo Module

| Class/Function | Description |
|----------------|-------------|
| `VectorizedGeoMixin` | Vectorized geographic operations mixin |
| `SpatialIndexMixin` | Spatial indexing mixin |
| `DistanceUnit` | Distance unit enum |
| `BoundingBox` | Bounding box dataclass |

### aggregation Module

| Class/Function | Description |
|----------------|-------------|
| `LazyAggregationMixin` | Lazy aggregation mixin |
| `AggregationExpressionBuilder` | Aggregation expression builder |
| `AggregationResult` | Aggregation result dataclass |
| `JoinStrategy` | Join strategy configuration |

### profiling Module

| Class/Function | Description |
|----------------|-------------|
| `ValidatorProfiler` | Main profiler class |
| `ProfilerConfig` | Profiler configuration |
| `ProfilerMode` | Profiler mode enum |
| `ValidatorMetrics` | Validator metrics |
| `TimingMetrics` | Timing metrics |
| `MemoryMetrics` | Memory metrics |
| `ThroughputMetrics` | Throughput metrics |
| `ProfilingSession` | Profiling session |
| `ExecutionSnapshot` | Execution snapshot |
| `ProfilingReport` | Report generator |
| `profile_validator()` | Profiling context manager |
| `profiled()` | Profiling decorator |
