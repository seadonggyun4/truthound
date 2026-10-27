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

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from collections.abc import Callable


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
    address: str | None = None  # None = start local, "auto" = detect
    num_cpus: int | None = None
    num_gpus: int | None = None
    memory: int | None = None
    object_store_memory: int | None = None
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
            importlib.import_module("ray")
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
        self._data_ref = None
        self._initialized = False
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
        sample_size: int | None = None,
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

        if sample_size is not None:
            df = df.head(sample_size)
        row_count = len(df)
        columns = df.columns

        # Determine partitions
        if num_partitions is None:
            # Based on available CPUs
            num_cpus = int(ray.cluster_resources().get("CPU", 4))
            num_partitions = self.config.parallelism or max(1, min(num_cpus * 2, row_count // 10000))

        from truthound.profiler.distributed._partitions import polars_plan
        self._checkpoint_polars(df)
        self._data_ref = ray.put(df)
        return polars_plan(row_count, columns, num_partitions, strategy)

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
        policy = self.config
        @ray.remote(max_retries=0)
        def process_partition(partition, frame):
            from truthound.profiler.distributed._partitions import select_polars
            from truthound.profiler.distributed.base import execute_partition
            selected = select_polars(frame, partition)
            return execute_partition(func, partition, selected, policy)

        futures = [
            process_partition.remote(partition, self._data_ref)
            for partition in partitions
        ]

        # Collect results
        results = ray.get(futures)

        logger.debug(f"Processed {len(results)} partitions")
        return results

    def sample_column(self, name, limit):
        import ray
        @ray.remote
        def select(frame, column, size):
            return frame.select(column).drop_nulls().head(size)
        return ray.get(select.remote(self._data_ref, name, limit))

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
        import ray
        if any(result.errors for result in results):
            raise RuntimeError("Native profiling partition failed")
        if self._data_ref is None:
            raise RuntimeError("No active native profiling input")

        @ray.remote
        def global_statistics(frame):
            from truthound.profiler.distributed.local_backend import LocalBackend
            native = LocalBackend()
            return {column.name: native._profile_column(column) for column in frame}

        return ray.get(global_statistics.remote(self._data_ref))


# Register backend
backend_registry.register("ray", RayBackend)
