# Distributed Processing

This document describes the distributed profiling system for large-scale datasets.

## Overview

The distributed processing system implemented in `src/truthound/profiler/distributed.py` supports Spark, Dask, Ray, and Local backends with a unified interface.

## DistributedBackend

```python
class DistributedBackend(str, Enum):
    """Distributed processing backends"""

    LOCAL = "local"    # Local multithread/multiprocess
    SPARK = "spark"    # Apache Spark
    DASK = "dask"      # Dask Distributed
    RAY = "ray"        # Ray
    AUTO = "auto"      # Automatic detection
```

## PartitionStrategy

```python
class PartitionStrategy(str, Enum):
    """Partition strategies"""

    ROW_BASED = "row_based"       # Row-based partitioning
    COLUMN_BASED = "column_based" # Column-based partitioning
    HYBRID = "hybrid"             # Hybrid (row + column)
    HASH = "hash"                 # Hash-based partitioning
```

## Backend-Specific Configuration

### SparkConfig

```python
@dataclass
class SparkConfig:
    """Spark configuration"""

    app_name: str = "truthound-profiler"
    master: str = "local[*]"
    executor_memory: str = "4g"
    executor_cores: int = 2
    num_executors: int = 4
    spark_config: dict[str, str] = field(default_factory=dict)
```

### DaskConfig

```python
@dataclass
class DaskConfig:
    """Dask configuration"""

    scheduler: str = "threads"  # threads, processes, synchronous
    num_workers: int | None = None  # None = number of CPU cores
    memory_limit: str = "4GB"
    dashboard_address: str = ":8787"
```

### RayConfig

```python
@dataclass
class RayConfig:
    """Ray configuration"""

    address: str | None = None  # None = local cluster
    num_cpus: int | None = None
    num_gpus: int | None = None
    object_store_memory: int | None = None
```

## DistributedProfiler

A unified distributed profiler interface.

```python
from truthound.profiler.distributed import DistributedProfiler, DistributedBackend

# Automatic backend selection
profiler = DistributedProfiler.create(backend=DistributedBackend.AUTO)

# Spark backend
profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=SparkConfig(master="spark://master:7077"),
)

# Dask backend
profiler = DistributedProfiler.create(
    backend=DistributedBackend.DASK,
    config=DaskConfig(scheduler="distributed"),
)

# Execute profiling
profile = profiler.profile("hdfs://data/large.parquet")
```

## LocalBackend

Local multithread/multiprocess backend.

```python
from truthound.profiler.distributed import LocalBackend

backend = LocalBackend(
    num_workers=4,
    use_processes=True,  # True = processes, False = threads
)

# Parallel column profiling
results = backend.profile_columns_parallel(lf, columns)
```

## SparkBackend

Apache Spark backend.

```python
from truthound.profiler.distributed import SparkBackend, SparkConfig

config = SparkConfig(
    master="spark://master:7077",
    executor_memory="8g",
    num_executors=10,
)

backend = SparkBackend(config)

# Profile Spark DataFrame
profile = backend.profile(spark_df)

# Profile HDFS file
profile = backend.profile_file("hdfs://data/large.parquet")
```

## DaskBackend

Dask Distributed backend.

```python
from truthound.profiler.distributed import DaskBackend, DaskConfig

config = DaskConfig(
    scheduler="distributed",
    num_workers=8,
    memory_limit="8GB",
)

backend = DaskBackend(config)

# Profile Dask DataFrame
profile = backend.profile(dask_df)

# Profile partitioned files
profile = backend.profile_file("s3://bucket/data/*.parquet")
```

## RayBackend

Ray backend.

```python
from truthound.profiler.distributed import RayBackend, RayConfig

config = RayConfig(
    address="ray://cluster:10001",
    num_cpus=16,
)

backend = RayBackend(config)

# Profile Ray Dataset
profile = backend.profile(ray_dataset)
```

## BackendRegistry

A pluggable backend registry.

```python
from truthound.profiler.distributed import BackendRegistry

# List registered backends
backends = BackendRegistry.list()  # ["local", "spark", "dask", "ray"]

# Create backend instance
backend = BackendRegistry.create("spark", config=spark_config)

# Register custom backend
@BackendRegistry.register("my_backend")
class MyCustomBackend:
    def profile(self, data):
        # Custom profiling logic
        pass
```

## Partition Strategies

### ROW_BASED - Row-Based

```python
from truthound.profiler.distributed import RowBasedPartitioner

partitioner = RowBasedPartitioner(num_partitions=10)
partitions = partitioner.partition(lf)

# Each partition contains 1/10 of total rows
```

### COLUMN_BASED - Column-Based

```python
from truthound.profiler.distributed import ColumnBasedPartitioner

partitioner = ColumnBasedPartitioner()
column_groups = partitioner.partition(lf, num_workers=4)

# Each worker processes 1/4 of columns
```

### HYBRID - Hybrid

```python
from truthound.profiler.distributed import HybridPartitioner

partitioner = HybridPartitioner(
    row_partitions=4,
    column_partitions=2,
)
# 4 x 2 = 8 partitions
```

## Automatic Backend Selection

```python
from truthound.profiler.distributed import auto_select_backend

# Automatic selection based on environment
backend = auto_select_backend()

# Selection logic:
# 1. Spark session exists → Spark
# 2. Dask client exists → Dask
# 3. Ray initialized → Ray
# 4. Default → Local
```

## Result Merging

Merge distributed processing results.

```python
from truthound.profiler.distributed import ProfileMerger

merger = ProfileMerger()

# Merge partition profiles
partial_profiles = [profile1, profile2, profile3, profile4]
merged = merger.merge(partial_profiles)

# Statistics merge strategy:
# - count: sum
# - mean: weighted average
# - min/max: global min/max
# - quantiles: approximate merge
```

## Monitoring

```python
from truthound.profiler.distributed import DistributedMonitor

monitor = DistributedMonitor()

# Progress monitoring
profiler = DistributedProfiler.create(
    backend="spark",
    monitor=monitor,
)

# Register callbacks
monitor.on_partition_complete(lambda p: print(f"Partition {p} complete"))
monitor.on_progress(lambda pct: print(f"Progress: {pct}%"))

profile = profiler.profile(data)
```

## CLI Usage

```bash
# Local parallel processing
th profile data.csv --distributed --backend local --workers 4

# Spark processing
th profile hdfs://data/large.parquet --distributed --backend spark \
  --master spark://master:7077

# Dask processing
th profile s3://bucket/data/*.parquet --distributed --backend dask \
  --scheduler distributed

# Automatic backend selection
th profile data.parquet --distributed --backend auto
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRUTHOUND_DISTRIBUTED_BACKEND` | Default backend | `local` |
| `TRUTHOUND_NUM_WORKERS` | Number of workers | CPU core count |
| `SPARK_HOME` | Spark home | - |
| `DASK_SCHEDULER_ADDRESS` | Dask scheduler | - |
| `RAY_ADDRESS` | Ray cluster | - |

## Integration Example

```python
from truthound.profiler.distributed import (
    DistributedProfiler,
    DistributedBackend,
    SparkConfig,
    PartitionStrategy,
)

# Profile large data on Spark cluster
config = SparkConfig(
    master="spark://master:7077",
    executor_memory="16g",
    num_executors=20,
    spark_config={
        "spark.sql.shuffle.partitions": "200",
    },
)

profiler = DistributedProfiler.create(
    backend=DistributedBackend.SPARK,
    config=config,
    partition_strategy=PartitionStrategy.HYBRID,
)

# Profile 100GB data
profile = profiler.profile("hdfs://data/100gb_dataset.parquet")

print(f"Rows: {profile.row_count:,}")
print(f"Columns: {profile.column_count}")
print(f"Duration: {profile.profile_duration_ms / 1000:.2f}s")
```

## Performance Tips

| Data Size | Recommended Backend | Recommended Settings |
|-----------|--------------------|--------------------|
| < 1GB | Local | 4-8 threads |
| 1-10GB | Local/Dask | Process mode |
| 10-100GB | Dask/Spark | Cluster |
| > 100GB | Spark | Large cluster |

## Next Steps

- [Sampling](sampling.md) - Sampling in distributed environments
- [Caching](caching.md) - Distributed cache sharing
