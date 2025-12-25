"""Distributed processing backends for large-scale data profiling.

This module provides pluggable distributed computing backends:
- Spark: Apache Spark for cluster computing
- Dask: Parallel computing with task scheduling
- Ray: Distributed computing framework

Key features:
- Unified API across all backends
- Automatic backend detection
- Fallback to local processing
- Resource-aware partitioning

Example:
    from truthound.profiler.distributed import (
        DistributedProfiler,
        SparkBackend,
        DaskBackend,
        RayBackend,
    )

    # Auto-detect backend
    profiler = DistributedProfiler.create(backend="auto")

    # Or specify backend
    profiler = DistributedProfiler.create(
        backend="spark",
        spark_config={"spark.executor.memory": "4g"}
    )

    # Profile large dataset
    profile = profiler.profile("hdfs://data/large_dataset.parquet")
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    TableProfile,
    ValueFrequency,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class BackendType(str, Enum):
    """Supported distributed computing backends."""

    LOCAL = "local"
    SPARK = "spark"
    DASK = "dask"
    RAY = "ray"
    AUTO = "auto"


class PartitionStrategy(str, Enum):
    """Data partitioning strategies."""

    ROW_BASED = "row_based"       # Split by row ranges
    COLUMN_BASED = "column_based"  # Profile columns in parallel
    HYBRID = "hybrid"             # Combine both strategies
    HASH = "hash"                 # Hash-based partitioning


@dataclass
class PartitionInfo:
    """Information about a data partition."""

    partition_id: int
    total_partitions: int
    start_row: int = 0
    end_row: int = 0
    columns: list[str] = field(default_factory=list)
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    """Result from a worker processing a partition."""

    partition_id: int
    column_stats: dict[str, dict[str, Any]]
    row_count: int
    processing_time_ms: float
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Backend Configuration
# =============================================================================


@dataclass
class BackendConfig:
    """Base configuration for distributed backends."""

    backend_type: BackendType = BackendType.LOCAL
    num_workers: int = 0  # 0 = auto-detect
    memory_per_worker: str = "2g"
    parallelism: int = 0  # 0 = auto
    timeout_seconds: int = 3600
    retry_count: int = 3
    checkpoint_enabled: bool = False
    checkpoint_dir: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_type": self.backend_type.value,
            "num_workers": self.num_workers,
            "memory_per_worker": self.memory_per_worker,
            "parallelism": self.parallelism,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_dir": self.checkpoint_dir,
            "metadata": self.metadata,
        }


@dataclass
class SparkConfig(BackendConfig):
    """Spark-specific configuration."""

    backend_type: BackendType = BackendType.SPARK
    master: str = "local[*]"
    app_name: str = "truthound-profiler"
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    executor_cores: int = 2
    num_executors: int = 0
    spark_config: dict[str, str] = field(default_factory=dict)
    hadoop_config: dict[str, str] = field(default_factory=dict)


@dataclass
class DaskConfig(BackendConfig):
    """Dask-specific configuration."""

    backend_type: BackendType = BackendType.DASK
    scheduler: str = "threads"  # threads, processes, distributed
    address: str = ""  # For distributed scheduler
    n_workers: int = 0
    threads_per_worker: int = 2
    memory_limit: str = "auto"
    dashboard_address: str = ":8787"


@dataclass
class RayConfig(BackendConfig):
    """Ray-specific configuration."""

    backend_type: BackendType = BackendType.RAY
    address: str = ""  # Empty = local, "auto" = cluster
    num_cpus: int = 0
    num_gpus: int = 0
    object_store_memory: int = 0
    runtime_env: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Distributed Backend Protocol
# =============================================================================


class DistributedBackend(ABC):
    """Abstract base class for distributed computing backends.

    Implement this to create custom distributed backends.
    All backends must provide a consistent interface for:
    - Initialization and cleanup
    - Data distribution
    - Parallel execution
    - Result aggregation
    """

    name: str = "base"
    available: bool = False

    def __init__(self, config: BackendConfig):
        self.config = config
        self._initialized = False
        self._lock = threading.Lock()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the distributed backend.

        Sets up connections, creates cluster, etc.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the distributed backend.

        Cleans up resources, closes connections.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available.

        Returns:
            True if backend dependencies are installed
        """
        pass

    @abstractmethod
    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        """Distribute data across workers.

        Args:
            data: DataFrame, LazyFrame, or path to data
            num_partitions: Number of partitions (None = auto)
            strategy: Partitioning strategy

        Returns:
            List of partition information
        """
        pass

    @abstractmethod
    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        """Execute function on each partition.

        Args:
            func: Function to execute on each partition
            partitions: List of partitions to process
            data: Reference to distributed data

        Returns:
            List of results from each partition
        """
        pass

    @abstractmethod
    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate results from all partitions.

        Args:
            results: Results from map_partitions

        Returns:
            Aggregated statistics per column
        """
        pass

    def __enter__(self) -> "DistributedBackend":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()


# =============================================================================
# Local Backend (Fallback)
# =============================================================================


class LocalBackend(DistributedBackend):
    """Local multi-threaded backend as fallback.

    Uses ThreadPoolExecutor for parallel column processing.
    """

    name = "local"
    available = True

    def __init__(self, config: BackendConfig | None = None):
        super().__init__(config or BackendConfig())
        self._executor: ThreadPoolExecutor | None = None

    def initialize(self) -> None:
        num_workers = self.config.num_workers
        if num_workers <= 0:
            num_workers = min(os.cpu_count() or 4, 8)

        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._initialized = True
        logger.info(f"LocalBackend initialized with {num_workers} workers")

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
        self._initialized = False

    def is_available(self) -> bool:
        return True

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.COLUMN_BASED,
    ) -> list[PartitionInfo]:
        # Load data if path
        if isinstance(data, str):
            df = pl.scan_parquet(data).collect()
        elif isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        columns = df.columns
        row_count = len(df)

        if strategy == PartitionStrategy.COLUMN_BASED:
            # One partition per column
            num_parts = len(columns) if num_partitions is None else num_partitions
            partitions = []

            cols_per_part = max(1, len(columns) // num_parts)
            for i in range(num_parts):
                start = i * cols_per_part
                end = start + cols_per_part if i < num_parts - 1 else len(columns)
                partitions.append(PartitionInfo(
                    partition_id=i,
                    total_partitions=num_parts,
                    start_row=0,
                    end_row=row_count,
                    columns=columns[start:end],
                ))

            return partitions

        else:  # ROW_BASED or HYBRID
            num_parts = num_partitions or (os.cpu_count() or 4)
            rows_per_part = max(1, row_count // num_parts)

            partitions = []
            for i in range(num_parts):
                start = i * rows_per_part
                end = start + rows_per_part if i < num_parts - 1 else row_count
                partitions.append(PartitionInfo(
                    partition_id=i,
                    total_partitions=num_parts,
                    start_row=start,
                    end_row=end,
                    columns=columns,
                ))

            return partitions

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not self._executor:
            raise RuntimeError("Backend not initialized")

        futures = {
            self._executor.submit(func, partition, data): partition
            for partition in partitions
        }

        results = []
        for future in as_completed(futures):
            partition = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Partition {partition.partition_id} failed: {e}")
                results.append(WorkerResult(
                    partition_id=partition.partition_id,
                    column_stats={},
                    row_count=0,
                    processing_time_ms=0,
                    errors=[str(e)],
                ))

        return sorted(results, key=lambda r: r.partition_id)

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}

        for result in results:
            for col_name, stats in result.column_stats.items():
                if col_name not in aggregated:
                    aggregated[col_name] = {
                        "row_count": 0,
                        "null_count": 0,
                        "distinct_values": set(),
                        "min_value": None,
                        "max_value": None,
                        "sum_value": 0,
                        "sum_squared": 0,
                    }

                agg = aggregated[col_name]
                agg["row_count"] += stats.get("row_count", 0)
                agg["null_count"] += stats.get("null_count", 0)

                if "distinct_values" in stats:
                    agg["distinct_values"].update(stats["distinct_values"])

                # Min/Max
                if stats.get("min_value") is not None:
                    if agg["min_value"] is None or stats["min_value"] < agg["min_value"]:
                        agg["min_value"] = stats["min_value"]
                if stats.get("max_value") is not None:
                    if agg["max_value"] is None or stats["max_value"] > agg["max_value"]:
                        agg["max_value"] = stats["max_value"]

                # For computing variance
                agg["sum_value"] += stats.get("sum_value", 0)
                agg["sum_squared"] += stats.get("sum_squared", 0)

        # Finalize aggregations
        for col_name, agg in aggregated.items():
            agg["distinct_count"] = len(agg.pop("distinct_values", set()))
            n = agg["row_count"]
            if n > 0:
                mean = agg["sum_value"] / n
                agg["mean"] = mean
                variance = (agg["sum_squared"] / n) - (mean ** 2)
                agg["std"] = variance ** 0.5 if variance > 0 else 0

        return aggregated


# =============================================================================
# Spark Backend
# =============================================================================


class SparkBackend(DistributedBackend):
    """Apache Spark backend for cluster computing.

    Supports:
    - Local mode
    - Standalone cluster
    - YARN
    - Kubernetes
    - Databricks
    """

    name = "spark"

    def __init__(self, config: SparkConfig | None = None):
        super().__init__(config or SparkConfig())
        self._spark = None
        self._sc = None

    @property
    def spark_config(self) -> SparkConfig:
        return self.config  # type: ignore

    def is_available(self) -> bool:
        try:
            import pyspark
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        if not self.is_available():
            raise ImportError(
                "PySpark is required for Spark backend. "
                "Install with: pip install pyspark"
            )

        from pyspark.sql import SparkSession

        builder = SparkSession.builder.appName(self.spark_config.app_name)

        if self.spark_config.master:
            builder = builder.master(self.spark_config.master)

        # Set memory configurations
        builder = builder.config(
            "spark.executor.memory", self.spark_config.executor_memory
        ).config(
            "spark.driver.memory", self.spark_config.driver_memory
        ).config(
            "spark.executor.cores", str(self.spark_config.executor_cores)
        )

        # Custom Spark configs
        for key, value in self.spark_config.spark_config.items():
            builder = builder.config(key, value)

        # Hadoop configs
        for key, value in self.spark_config.hadoop_config.items():
            builder = builder.config(f"spark.hadoop.{key}", value)

        self._spark = builder.getOrCreate()
        self._sc = self._spark.sparkContext
        self._initialized = True

        logger.info(f"SparkBackend initialized: {self._spark.version}")

    def shutdown(self) -> None:
        if self._spark:
            self._spark.stop()
        self._spark = None
        self._sc = None
        self._initialized = False

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        if not self._spark:
            raise RuntimeError("Spark not initialized")

        # Load data into Spark DataFrame
        if isinstance(data, str):
            # Path to file
            spark_df = self._spark.read.parquet(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            # Convert Polars to Spark
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            pandas_df = data.to_pandas()
            spark_df = self._spark.createDataFrame(pandas_df)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Repartition
        num_parts = num_partitions or spark_df.rdd.getNumPartitions()
        spark_df = spark_df.repartition(num_parts)

        # Store in context for later use
        self._current_df = spark_df

        # Create partition info
        columns = spark_df.columns
        return [
            PartitionInfo(
                partition_id=i,
                total_partitions=num_parts,
                columns=columns,
            )
            for i in range(num_parts)
        ]

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not self._spark or not hasattr(self, "_current_df"):
            raise RuntimeError("No data distributed")

        spark_df = self._current_df
        columns = spark_df.columns

        # Define Spark UDF for profiling
        def profile_partition(iterator: Iterator) -> Iterator:
            import time
            import pandas as pd

            start = time.time()
            rows = list(iterator)

            if not rows:
                yield {
                    "partition_id": 0,
                    "column_stats": {},
                    "row_count": 0,
                    "processing_time_ms": 0,
                }
                return

            pdf = pd.DataFrame(rows, columns=columns)
            stats = {}

            for col in columns:
                col_data = pdf[col]
                stats[col] = {
                    "row_count": len(col_data),
                    "null_count": int(col_data.isna().sum()),
                    "distinct_count": int(col_data.nunique()),
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    stats[col].update({
                        "min_value": float(col_data.min()) if not col_data.isna().all() else None,
                        "max_value": float(col_data.max()) if not col_data.isna().all() else None,
                        "sum_value": float(col_data.sum()),
                        "sum_squared": float((col_data ** 2).sum()),
                    })

            elapsed = (time.time() - start) * 1000

            yield {
                "partition_id": 0,
                "column_stats": stats,
                "row_count": len(pdf),
                "processing_time_ms": elapsed,
            }

        # Execute on partitions
        results_rdd = spark_df.rdd.mapPartitions(profile_partition)
        results = results_rdd.collect()

        return [
            WorkerResult(
                partition_id=i,
                column_stats=r["column_stats"],
                row_count=r["row_count"],
                processing_time_ms=r["processing_time_ms"],
            )
            for i, r in enumerate(results)
        ]

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        # Use same logic as LocalBackend
        return LocalBackend(self.config).aggregate_results(results)


# =============================================================================
# Dask Backend
# =============================================================================


class DaskBackend(DistributedBackend):
    """Dask backend for parallel computing.

    Supports:
    - Threaded scheduler (single machine)
    - Process scheduler (single machine, multiprocessing)
    - Distributed scheduler (cluster)
    """

    name = "dask"

    def __init__(self, config: DaskConfig | None = None):
        super().__init__(config or DaskConfig())
        self._client = None
        self._cluster = None

    @property
    def dask_config(self) -> DaskConfig:
        return self.config  # type: ignore

    def is_available(self) -> bool:
        try:
            import dask
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        if not self.is_available():
            raise ImportError(
                "Dask is required for Dask backend. "
                "Install with: pip install dask[complete]"
            )

        import dask

        scheduler = self.dask_config.scheduler

        if scheduler == "distributed":
            from dask.distributed import Client, LocalCluster

            if self.dask_config.address:
                # Connect to existing cluster
                self._client = Client(self.dask_config.address)
            else:
                # Create local cluster
                n_workers = self.dask_config.n_workers or (os.cpu_count() or 4)
                self._cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=self.dask_config.threads_per_worker,
                    memory_limit=self.dask_config.memory_limit,
                    dashboard_address=self.dask_config.dashboard_address,
                )
                self._client = Client(self._cluster)

            logger.info(f"DaskBackend (distributed) initialized: {self._client}")
        else:
            # Use simple scheduler
            dask.config.set(scheduler=scheduler)
            logger.info(f"DaskBackend ({scheduler}) initialized")

        self._initialized = True

    def shutdown(self) -> None:
        if self._client:
            self._client.close()
        if self._cluster:
            self._cluster.close()
        self._client = None
        self._cluster = None
        self._initialized = False

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        import dask.dataframe as dd

        # Load data as Dask DataFrame
        if isinstance(data, str):
            ddf = dd.read_parquet(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            pdf = data.to_pandas()
            ddf = dd.from_pandas(pdf, npartitions=num_partitions or (os.cpu_count() or 4))
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Repartition if needed
        if num_partitions:
            ddf = ddf.repartition(npartitions=num_partitions)

        self._current_ddf = ddf
        num_parts = ddf.npartitions
        columns = list(ddf.columns)

        return [
            PartitionInfo(
                partition_id=i,
                total_partitions=num_parts,
                columns=columns,
            )
            for i in range(num_parts)
        ]

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not hasattr(self, "_current_ddf"):
            raise RuntimeError("No data distributed")

        import dask
        import pandas as pd

        ddf = self._current_ddf
        columns = list(ddf.columns)

        def profile_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            """Profile a single partition."""
            stats = {}

            for col in pdf.columns:
                col_data = pdf[col]
                stats[col] = {
                    "row_count": len(col_data),
                    "null_count": int(col_data.isna().sum()),
                    "distinct_count": int(col_data.nunique()),
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    non_null = col_data.dropna()
                    if len(non_null) > 0:
                        stats[col].update({
                            "min_value": float(non_null.min()),
                            "max_value": float(non_null.max()),
                            "sum_value": float(non_null.sum()),
                            "sum_squared": float((non_null ** 2).sum()),
                        })

            # Return as single-row DataFrame
            import json
            return pd.DataFrame([{
                "stats": json.dumps(stats),
                "row_count": len(pdf),
            }])

        # Apply to all partitions
        result_ddf = ddf.map_partitions(
            profile_partition,
            meta={"stats": str, "row_count": int},
        )

        # Compute results
        results_pdf = result_ddf.compute()

        import json
        return [
            WorkerResult(
                partition_id=i,
                column_stats=json.loads(row["stats"]),
                row_count=row["row_count"],
                processing_time_ms=0,
            )
            for i, (_, row) in enumerate(results_pdf.iterrows())
        ]

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        return LocalBackend(self.config).aggregate_results(results)


# =============================================================================
# Ray Backend
# =============================================================================


class RayBackend(DistributedBackend):
    """Ray backend for distributed computing.

    Features:
    - Automatic cluster management
    - Object store for shared data
    - Actor-based processing
    """

    name = "ray"

    def __init__(self, config: RayConfig | None = None):
        super().__init__(config or RayConfig())
        self._ray = None

    @property
    def ray_config(self) -> RayConfig:
        return self.config  # type: ignore

    def is_available(self) -> bool:
        try:
            import ray
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        if not self.is_available():
            raise ImportError(
                "Ray is required for Ray backend. "
                "Install with: pip install ray"
            )

        import ray

        init_kwargs: dict[str, Any] = {}

        if self.ray_config.address:
            init_kwargs["address"] = self.ray_config.address
        if self.ray_config.num_cpus:
            init_kwargs["num_cpus"] = self.ray_config.num_cpus
        if self.ray_config.num_gpus:
            init_kwargs["num_gpus"] = self.ray_config.num_gpus
        if self.ray_config.object_store_memory:
            init_kwargs["object_store_memory"] = self.ray_config.object_store_memory
        if self.ray_config.runtime_env:
            init_kwargs["runtime_env"] = self.ray_config.runtime_env

        if not ray.is_initialized():
            ray.init(**init_kwargs)

        self._ray = ray
        self._initialized = True
        logger.info("RayBackend initialized")

    def shutdown(self) -> None:
        if self._ray and self._ray.is_initialized():
            self._ray.shutdown()
        self._ray = None
        self._initialized = False

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        if not self._ray:
            raise RuntimeError("Ray not initialized")

        # Load data
        if isinstance(data, str):
            df = pl.read_parquet(data)
        elif isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        num_parts = num_partitions or (os.cpu_count() or 4)
        rows_per_part = max(1, len(df) // num_parts)
        columns = df.columns

        # Create partitions and store in Ray object store
        self._partitioned_data = []
        partitions = []

        for i in range(num_parts):
            start = i * rows_per_part
            end = start + rows_per_part if i < num_parts - 1 else len(df)

            partition_df = df.slice(start, end - start)
            ref = self._ray.put(partition_df)
            self._partitioned_data.append(ref)

            partitions.append(PartitionInfo(
                partition_id=i,
                total_partitions=num_parts,
                start_row=start,
                end_row=end,
                columns=columns,
            ))

        return partitions

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not self._ray or not hasattr(self, "_partitioned_data"):
            raise RuntimeError("No data distributed")

        ray = self._ray

        @ray.remote
        def profile_partition_remote(
            df: pl.DataFrame,
            partition_id: int,
        ) -> dict[str, Any]:
            """Remote function to profile a partition."""
            import time
            start = time.time()

            stats = {}
            for col in df.columns:
                col_data = df.get_column(col)
                row_count = len(col_data)
                null_count = col_data.null_count()

                stats[col] = {
                    "row_count": row_count,
                    "null_count": null_count,
                    "distinct_count": col_data.n_unique(),
                }

                # Numeric stats
                if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    non_null = col_data.drop_nulls()
                    if len(non_null) > 0:
                        stats[col].update({
                            "min_value": float(non_null.min()),
                            "max_value": float(non_null.max()),
                            "sum_value": float(non_null.sum()),
                            "sum_squared": float((non_null ** 2).sum()),
                        })

            elapsed = (time.time() - start) * 1000

            return {
                "partition_id": partition_id,
                "column_stats": stats,
                "row_count": len(df),
                "processing_time_ms": elapsed,
            }

        # Launch tasks
        futures = [
            profile_partition_remote.remote(
                self._partitioned_data[p.partition_id],
                p.partition_id,
            )
            for p in partitions
        ]

        # Collect results
        results_raw = ray.get(futures)

        return [
            WorkerResult(
                partition_id=r["partition_id"],
                column_stats=r["column_stats"],
                row_count=r["row_count"],
                processing_time_ms=r["processing_time_ms"],
            )
            for r in results_raw
        ]

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        return LocalBackend(self.config).aggregate_results(results)


# =============================================================================
# Backend Registry
# =============================================================================


class BackendRegistry:
    """Registry for distributed backends.

    Allows dynamic registration of custom backends.
    """

    def __init__(self) -> None:
        self._backends: dict[str, type[DistributedBackend]] = {}

    def register(
        self,
        name: str,
        backend_class: type[DistributedBackend],
    ) -> None:
        """Register a backend class."""
        self._backends[name] = backend_class

    def get(self, name: str) -> type[DistributedBackend]:
        """Get a registered backend class."""
        if name not in self._backends:
            raise KeyError(
                f"Unknown backend: {name}. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[name]

    def create(
        self,
        name: str,
        config: BackendConfig | None = None,
    ) -> DistributedBackend:
        """Create a backend instance."""
        backend_class = self.get(name)
        return backend_class(config)

    def list_backends(self) -> list[str]:
        """List available backends."""
        return list(self._backends.keys())

    def get_available_backends(self) -> list[str]:
        """List backends with available dependencies."""
        available = []
        for name, backend_class in self._backends.items():
            try:
                instance = backend_class()
                if instance.is_available():
                    available.append(name)
            except Exception:
                pass
        return available


# Global registry
backend_registry = BackendRegistry()
backend_registry.register("local", LocalBackend)
backend_registry.register("spark", SparkBackend)
backend_registry.register("dask", DaskBackend)
backend_registry.register("ray", RayBackend)


# =============================================================================
# Distributed Profiler
# =============================================================================


@dataclass
class DistributedProfileConfig:
    """Configuration for distributed profiling."""

    backend: str = "auto"
    backend_config: BackendConfig | None = None
    partition_strategy: PartitionStrategy = PartitionStrategy.ROW_BASED
    num_partitions: int | None = None
    include_patterns: bool = True
    sample_size: int | None = None
    timeout_seconds: int = 3600


class DistributedProfiler:
    """High-level distributed data profiler.

    Provides a unified interface for profiling large datasets
    using any of the supported distributed backends.

    Example:
        profiler = DistributedProfiler.create(backend="dask")

        with profiler:
            profile = profiler.profile("hdfs://data/large.parquet")
    """

    def __init__(
        self,
        backend: DistributedBackend,
        config: DistributedProfileConfig | None = None,
    ):
        self._backend = backend
        self._config = config or DistributedProfileConfig()

    @classmethod
    def create(
        cls,
        backend: str = "auto",
        backend_config: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "DistributedProfiler":
        """Create a distributed profiler with the specified backend.

        Args:
            backend: Backend name or "auto" for auto-detection
            backend_config: Backend-specific configuration
            **kwargs: Additional profiler configuration

        Returns:
            Configured DistributedProfiler
        """
        # Auto-detect backend
        if backend == "auto":
            available = backend_registry.get_available_backends()
            # Prefer in order: ray, dask, spark, local
            for preferred in ["ray", "dask", "spark", "local"]:
                if preferred in available:
                    backend = preferred
                    break
            else:
                backend = "local"

            logger.info(f"Auto-selected backend: {backend}")

        backend_instance = backend_registry.create(backend, backend_config)
        config = DistributedProfileConfig(
            backend=backend,
            backend_config=backend_config,
            **kwargs,
        )

        return cls(backend_instance, config)

    def __enter__(self) -> "DistributedProfiler":
        self._backend.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._backend.shutdown()

    @property
    def backend(self) -> DistributedBackend:
        """Access the underlying backend."""
        return self._backend

    def profile(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        name: str = "",
    ) -> TableProfile:
        """Profile a dataset using distributed computing.

        Args:
            data: DataFrame, LazyFrame, or path to data file
            name: Name for the profile

        Returns:
            Complete TableProfile
        """
        start_time = time.time()

        # Distribute data
        partitions = self._backend.distribute_data(
            data,
            num_partitions=self._config.num_partitions,
            strategy=self._config.partition_strategy,
        )

        logger.info(f"Data distributed into {len(partitions)} partitions")

        # Profile partitions
        results = self._backend.map_partitions(
            self._profile_partition,
            partitions,
            data,
        )

        logger.info(f"Collected results from {len(results)} partitions")

        # Aggregate results
        aggregated = self._backend.aggregate_results(results)

        # Build TableProfile
        total_rows = sum(r.row_count for r in results)
        columns = list(aggregated.keys())

        column_profiles = []
        for col_name, stats in aggregated.items():
            profile = self._build_column_profile(col_name, stats)
            column_profiles.append(profile)

        elapsed_ms = (time.time() - start_time) * 1000

        return TableProfile(
            name=name or "distributed_profile",
            row_count=total_rows,
            column_count=len(columns),
            columns=tuple(column_profiles),
            source=str(data) if isinstance(data, str) else "dataframe",
            profile_duration_ms=elapsed_ms,
        )

    def _profile_partition(
        self,
        partition: PartitionInfo,
        data: Any,
    ) -> WorkerResult:
        """Profile a single partition (called by workers)."""
        start = time.time()

        # This is a placeholder - actual implementation in backend
        stats: dict[str, dict[str, Any]] = {}
        row_count = 0

        elapsed = (time.time() - start) * 1000

        return WorkerResult(
            partition_id=partition.partition_id,
            column_stats=stats,
            row_count=row_count,
            processing_time_ms=elapsed,
        )

    def _build_column_profile(
        self,
        name: str,
        stats: dict[str, Any],
    ) -> ColumnProfile:
        """Build ColumnProfile from aggregated stats."""
        row_count = stats.get("row_count", 0)
        null_count = stats.get("null_count", 0)
        distinct_count = stats.get("distinct_count", 0)

        distribution = None
        if "mean" in stats:
            distribution = DistributionStats(
                mean=stats.get("mean"),
                std=stats.get("std"),
                min=stats.get("min_value"),
                max=stats.get("max_value"),
            )

        return ColumnProfile(
            name=name,
            physical_type="unknown",  # Would need type info
            row_count=row_count,
            null_count=null_count,
            null_ratio=null_count / row_count if row_count > 0 else 0,
            distinct_count=distinct_count,
            unique_ratio=distinct_count / row_count if row_count > 0 else 0,
            distribution=distribution,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_distributed_profiler(
    backend: str = "auto",
    **kwargs: Any,
) -> DistributedProfiler:
    """Create a distributed profiler.

    Args:
        backend: Backend name ("local", "spark", "dask", "ray", "auto")
        **kwargs: Backend configuration options

    Returns:
        Configured DistributedProfiler
    """
    return DistributedProfiler.create(backend=backend, **kwargs)


def profile_distributed(
    data: pl.DataFrame | pl.LazyFrame | str,
    backend: str = "auto",
    name: str = "",
    **kwargs: Any,
) -> TableProfile:
    """Profile data using distributed computing.

    Args:
        data: Data to profile
        backend: Backend to use
        name: Profile name
        **kwargs: Additional options

    Returns:
        TableProfile
    """
    profiler = DistributedProfiler.create(backend=backend, **kwargs)
    with profiler:
        return profiler.profile(data, name=name)


def get_available_backends() -> list[str]:
    """Get list of available distributed backends.

    Returns:
        List of backend names with installed dependencies
    """
    return backend_registry.get_available_backends()
