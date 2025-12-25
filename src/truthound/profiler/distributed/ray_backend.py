"""Ray Backend for Distributed Processing.

This module provides the Ray integration for distributed data profiling.
Ray offers easy-to-use distributed computing with good performance.

Example:
    from truthound.profiler.distributed import RayBackend, RayConfig

    config = RayConfig(
        address="ray://cluster:10001",
        num_cpus=8,
    )

    with RayBackend(config) as backend:
        profile = backend.profile(data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import polars as pl

from truthound.profiler.distributed.base import (
    BackendConfig,
    BackendType,
    DistributedBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
    backend_registry,
)


logger = logging.getLogger(__name__)


@dataclass
class RayConfig(BackendConfig):
    """Configuration for Ray backend.

    Attributes:
        address: Ray cluster address (None for local, "auto" for auto-detect)
        num_cpus: Number of CPUs to use
        num_gpus: Number of GPUs to use
        memory: Total memory limit (bytes)
        object_store_memory: Object store memory limit
        runtime_env: Runtime environment configuration
    """

    backend_type: BackendType = field(default=BackendType.RAY)
    address: Optional[str] = None  # None = start local, "auto" = detect
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    memory: Optional[int] = None
    object_store_memory: Optional[int] = None
    runtime_env: dict[str, Any] = field(default_factory=dict)
    include_dashboard: bool = False
    dashboard_port: int = 8265

    def to_ray_init_kwargs(self) -> dict[str, Any]:
        """Convert to ray.init() keyword arguments."""
        kwargs: dict[str, Any] = {}

        if self.address is not None:
            kwargs["address"] = self.address
        if self.num_cpus is not None:
            kwargs["num_cpus"] = self.num_cpus
        if self.num_gpus is not None:
            kwargs["num_gpus"] = self.num_gpus
        if self.memory is not None:
            kwargs["_memory"] = self.memory
        if self.object_store_memory is not None:
            kwargs["object_store_memory"] = self.object_store_memory
        if self.runtime_env:
            kwargs["runtime_env"] = self.runtime_env
        if self.include_dashboard:
            kwargs["include_dashboard"] = True
            kwargs["dashboard_port"] = self.dashboard_port

        return kwargs


class RayBackend(DistributedBackend):
    """Ray distributed backend.

    Provides distributed data profiling using Ray, which offers
    simple and flexible distributed computing.

    Features:
    - Easy-to-use API with minimal setup
    - Automatic resource management
    - Shared object store for zero-copy data sharing
    - Support for heterogeneous clusters

    Example:
        config = RayConfig(num_cpus=8)
        backend = RayBackend(config)

        with backend:
            partitions = backend.distribute_data(df)
            results = backend.map_partitions(profile_func, partitions, df)
            stats = backend.aggregate_results(results)
    """

    name = "ray"
    available = False

    def __init__(self, config: RayConfig | None = None):
        """Initialize Ray backend.

        Args:
            config: Ray configuration
        """
        super().__init__(config or RayConfig())
        self._ray_initialized = False
        self._data_ref = None

    @property
    def ray_config(self) -> RayConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def is_available(self) -> bool:
        """Check if Ray is available."""
        try:
            import ray
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize Ray runtime.

        Starts a local Ray instance or connects to existing cluster.
        """
        if self._initialized:
            return

        if not self.is_available():
            raise ImportError(
                "Ray is required for RayBackend. "
                "Install with: pip install ray"
            )

        with self._lock:
            if self._initialized:
                return

            import ray

            # Check if already connected
            if not ray.is_initialized():
                ray.init(**self.ray_config.to_ray_init_kwargs())
                self._ray_initialized = True

            # Get cluster info
            resources = ray.cluster_resources()
            logger.info(
                f"Ray initialized: {resources.get('CPU', 0)} CPUs, "
                f"{resources.get('GPU', 0)} GPUs, "
                f"{resources.get('memory', 0) / 1e9:.1f} GB memory"
            )

            self._initialized = True

    def shutdown(self) -> None:
        """Shutdown Ray if we started it."""
        if self._ray_initialized:
            import ray
            ray.shutdown()
            self._ray_initialized = False
            self._initialized = False
            logger.info("Ray shutdown")

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        """Distribute data using Ray object store.

        Args:
            data: Polars DataFrame, LazyFrame, or path to data
            num_partitions: Number of partitions
            strategy: Partitioning strategy

        Returns:
            List of partition information
        """
        if not self._initialized:
            self.initialize()

        import ray

        # Convert to DataFrame
        if isinstance(data, str):
            df = pl.read_parquet(data)
        elif isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        row_count = len(df)
        columns = df.columns

        # Determine partitions
        if num_partitions is None:
            # Based on available CPUs
            num_cpus = int(ray.cluster_resources().get("CPU", 4))
            num_partitions = max(1, min(num_cpus * 2, row_count // 10000))

        # Calculate partition sizes
        rows_per_partition = row_count // num_partitions
        partitions = []

        # Put data in object store
        self._data_ref = ray.put(df)

        for i in range(num_partitions):
            start_row = i * rows_per_partition
            end_row = (i + 1) * rows_per_partition if i < num_partitions - 1 else row_count

            partitions.append(PartitionInfo(
                partition_id=i,
                total_partitions=num_partitions,
                start_row=start_row,
                end_row=end_row,
                columns=columns,
                metadata={"data_ref": str(self._data_ref)},
            ))

        logger.debug(f"Created {num_partitions} partitions for {row_count} rows")
        return partitions

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        """Execute profiling function on partitions using Ray.

        Args:
            func: Profiling function
            partitions: Partition info list
            data: Reference to data (ignored, uses object store)

        Returns:
            List of worker results
        """
        if not self._initialized:
            self.initialize()

        import ray
        import time

        # Define remote function
        @ray.remote
        def process_partition(partition: PartitionInfo, data_ref) -> WorkerResult:
            """Process a single partition."""
            start_time = time.time()

            # Get data from object store
            df = ray.get(data_ref)

            # Slice to partition
            partition_df = df.slice(partition.start_row, partition.end_row - partition.start_row)

            # Calculate statistics
            stats = {}
            for col in partition_df.columns:
                series = partition_df.get_column(col)
                stats[col] = {
                    "count": len(series),
                    "null_count": series.null_count(),
                    "distinct_count": series.n_unique(),
                }

                if series.dtype.is_numeric():
                    non_null = series.drop_nulls()
                    if len(non_null) > 0:
                        stats[col].update({
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                            "sum": float(non_null.sum()),
                        })

            processing_time = (time.time() - start_time) * 1000

            return WorkerResult(
                partition_id=partition.partition_id,
                column_stats=stats,
                row_count=len(partition_df),
                processing_time_ms=processing_time,
            )

        # Submit all tasks
        futures = [
            process_partition.remote(partition, self._data_ref)
            for partition in partitions
        ]

        # Collect results
        results = ray.get(futures)

        logger.debug(f"Processed {len(results)} partitions")
        return results

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate statistics from all partitions.

        Args:
            results: Worker results from each partition

        Returns:
            Aggregated column statistics
        """
        if not results:
            return {}

        # Gather all columns
        all_columns = set()
        for result in results:
            all_columns.update(result.column_stats.keys())

        aggregated = {}

        for col in all_columns:
            col_stats = [r.column_stats.get(col, {}) for r in results if col in r.column_stats]

            if not col_stats:
                continue

            # Aggregate counts
            total_count = sum(s.get("count", 0) for s in col_stats)
            total_null = sum(s.get("null_count", 0) for s in col_stats)

            # Distinct count needs exact computation or HyperLogLog
            max_distinct = max((s.get("distinct_count", 0) for s in col_stats), default=0)

            aggregated[col] = {
                "count": total_count,
                "null_count": total_null,
                "null_ratio": total_null / total_count if total_count > 0 else 0,
                "distinct_count_estimate": max_distinct,
            }

            # Aggregate numeric stats
            mins = [s.get("min") for s in col_stats if s.get("min") is not None]
            maxs = [s.get("max") for s in col_stats if s.get("max") is not None]
            sums = [s.get("sum") for s in col_stats if s.get("sum") is not None]
            counts = [s.get("count", 0) for s in col_stats]

            if mins:
                aggregated[col]["min"] = min(mins)
            if maxs:
                aggregated[col]["max"] = max(maxs)
            if sums and sum(counts) > 0:
                aggregated[col]["sum"] = sum(sums)
                aggregated[col]["mean"] = sum(sums) / sum(counts)

        return aggregated


# Register backend
backend_registry.register("ray", RayBackend)
