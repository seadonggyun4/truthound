"""Dask distributed backend for parallel computing.

This module provides a complete Dask integration for distributed profiling,
supporting threaded, process, and distributed schedulers.

Features:
- Local cluster with automatic worker management
- Remote cluster connection
- Adaptive scaling
- Dashboard for monitoring
- Fault tolerance and retry logic

Example:
    from truthound.profiler.distributed import DaskBackend, DaskConfig

    # Local cluster with process scheduler
    config = DaskConfig(scheduler="distributed", n_workers=4)
    backend = DaskBackend(config)

    with backend:
        profile = backend.profile_dataframe(df)

    # Connect to existing cluster
    config = DaskConfig(scheduler="distributed", address="tcp://scheduler:8786")
    backend = DaskBackend(config)
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
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
class DaskConfig(BackendConfig):
    """Dask-specific configuration."""

    backend_type: BackendType = BackendType.DASK
    scheduler: str = "threads"  # threads, processes, distributed, synchronous
    address: str = ""  # For distributed scheduler
    n_workers: int = 0  # 0 = auto
    threads_per_worker: int = 2
    memory_limit: str = "auto"
    dashboard_address: str = ":8787"
    # Distributed cluster options
    local_directory: str = ""
    silence_logs: int = logging.WARNING
    # Performance tuning
    work_stealing: bool = True
    scheduler_port: int = 0  # 0 = auto
    # Adaptive scaling
    adaptive: bool = False
    minimum_workers: int = 1
    maximum_workers: int = 10


class DaskBackend(DistributedBackend):
    """Dask backend for parallel computing.

    Supports:
    - Threaded scheduler (single machine, GIL-bound)
    - Process scheduler (single machine, multiprocessing)
    - Distributed scheduler (cluster or local)
    - Synchronous scheduler (for debugging)

    The distributed scheduler is recommended for production use as it
    provides:
    - Dashboard for monitoring
    - Better error handling
    - Adaptive scaling
    - Work stealing for load balancing
    """

    name = "dask"

    def __init__(self, config: DaskConfig | None = None):
        super().__init__(config or DaskConfig())
        self._client = None
        self._cluster = None
        self._dask = None
        self._current_ddf = None

    @property
    def dask_config(self) -> DaskConfig:
        return self.config  # type: ignore

    def is_available(self) -> bool:
        try:
            importlib.import_module("dask.dataframe")
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        if not self.is_available():
            raise ImportError(
                "Dask is required for Dask backend. "
                "Install with: pip install dask[complete] distributed"
            )

        import dask

        self._dask = dask
        scheduler = self.dask_config.scheduler

        if scheduler == "distributed":
            self._init_distributed()
        else:
            # Use simple scheduler (threads, processes, synchronous)
            logger.info(f"DaskBackend ({scheduler} scheduler) initialized")

        self._initialized = True

    def _init_distributed(self) -> None:
        """Initialize distributed scheduler."""
        from dask.distributed import Client, LocalCluster

        if self.dask_config.address:
            # Connect to existing cluster
            logger.info(f"Connecting to Dask cluster at {self.dask_config.address}")
            self._client = Client(
                self.dask_config.address,
                timeout=self.dask_config.timeout_seconds,
            )
        else:
            # Create local cluster
            n_workers = self.dask_config.n_workers
            if n_workers <= 0:
                n_workers = os.cpu_count() or 4

            cluster_kwargs: dict[str, Any] = {
                "n_workers": n_workers,
                "threads_per_worker": self.dask_config.threads_per_worker,
                "memory_limit": self.dask_config.memory_limit,
                "dashboard_address": self.dask_config.dashboard_address,
                "silence_logs": self.dask_config.silence_logs,
            }

            if self.dask_config.local_directory:
                cluster_kwargs["local_directory"] = self.dask_config.local_directory

            if self.dask_config.scheduler_port:
                cluster_kwargs["scheduler_port"] = self.dask_config.scheduler_port

            self._cluster = LocalCluster(**cluster_kwargs)
            self._client = Client(self._cluster)

            # Enable adaptive scaling if configured
            if self.dask_config.adaptive:
                self._cluster.adapt(
                    minimum=self.dask_config.minimum_workers,
                    maximum=self.dask_config.maximum_workers,
                )
                logger.info(
                    f"Adaptive scaling enabled: {self.dask_config.minimum_workers}-"
                    f"{self.dask_config.maximum_workers} workers"
                )

        dashboard_link = getattr(self._client, "dashboard_link", "N/A")
        logger.info(f"DaskBackend (distributed) initialized. Dashboard: {dashboard_link}")

    def shutdown(self) -> None:
        failures = []
        for name in ("_client", "_cluster"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    failures.append(error)
                else:
                    setattr(self, name, None)
        self._current_ddf = None
        self._initialized = False
        if failures:
            raise ExceptionGroup("Native Dask profiler cleanup failed", failures)

    @property
    def dashboard_link(self) -> str | None:
        """Get the dashboard link if using distributed scheduler."""
        if self._client:
            return getattr(self._client, "dashboard_link", None)
        return None

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
        sample_size: int | None = None,
    ) -> list[PartitionInfo]:
        import dask.dataframe as dd

        # Determine number of partitions
        if num_partitions is None and self.config.parallelism:
            num_partitions = self.config.parallelism
        if num_partitions is None:
            if self._client:
                # Use number of workers
                num_partitions = len(self._client.scheduler_info()["workers"])
            else:
                num_partitions = self.config.parallelism or os.cpu_count() or 4

        # Load data as Dask DataFrame
        if isinstance(data, str):
            if data.endswith(".parquet"):
                ddf = dd.read_parquet(data)
            elif data.endswith(".csv"):
                ddf = dd.read_csv(data)
            else:
                # Try parquet by default
                ddf = dd.read_parquet(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            pdf = data.to_pandas()
            ddf = dd.from_pandas(pdf, npartitions=num_partitions)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if sample_size is not None:
            snapshot = ddf.head(sample_size, npartitions=-1, compute=False).compute(scheduler=self._client or self.dask_config.scheduler)
            num_partitions = min(num_partitions, max(1, len(snapshot)))
            ddf = dd.from_pandas(snapshot, npartitions=num_partitions)
        observed_rows = int(ddf.shape[0].compute(scheduler=self._client or self.dask_config.scheduler))
        num_partitions = min(num_partitions, max(1, observed_rows))
        from truthound.profiler.distributed._partitions import column_groups, expand
        if not isinstance(strategy, PartitionStrategy):
            raise ValueError("Unknown partition strategy")
        requested = num_partitions
        groups = column_groups(list(ddf.columns), requested, strategy)
        physical = 1 if strategy is PartitionStrategy.COLUMN_BASED else max(1, (requested + len(groups) - 1) // len(groups))
        if strategy is PartitionStrategy.HASH and len(ddf.columns):
            ddf = ddf.shuffle(on=list(ddf.columns), npartitions=physical)
        elif ddf.npartitions != physical:
            ddf = ddf.repartition(npartitions=physical)
        def write(directory):
            graph = ddf.to_parquet(str(directory), write_index=False, compute=False)
            graph.compute(scheduler=self._client or self.dask_config.scheduler)
        self._publish_checkpoint(write)
        self._current_ddf = ddf
        return expand(ddf.npartitions, list(ddf.columns), requested, strategy)

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not hasattr(self, "_current_ddf") or self._current_ddf is None:
            raise RuntimeError("No data distributed")

        import dask

        from truthound.profiler.distributed.base import execute_partition
        policy = self.config
        delayed = self._current_ddf.to_delayed()
        def apply(partition, frame):
            return execute_partition(func, partition, pl.from_pandas(frame).select(partition.columns), policy)
        tasks = [dask.delayed(apply)(partition, delayed[partition.metadata["source_partition"]])
                 for partition in partitions]
        return list(dask.compute(*tasks, scheduler=self._client or self.dask_config.scheduler))

    def sample_column(self, name, limit):
        sample = self._current_ddf[[name]].dropna().head(limit, npartitions=-1, compute=False)
        return pl.from_pandas(sample.compute(scheduler=self._client or self.dask_config.scheduler))

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate results from all partitions."""
        import dask
        import pandas as pd

        if any(result.errors for result in results):
            raise RuntimeError("Native profiling partition failed")
        if self._current_ddf is None:
            raise RuntimeError("No active native profiling input")
        tasks = {}
        for name, dtype in self._current_ddf.dtypes.items():
            column = self._current_ddf[name]
            values = {"row_count": column.size, "null_count": column.isna().sum(),
                      "distinct_count": column.nunique(dropna=False)}
            if pd.api.types.is_numeric_dtype(dtype):
                values.update(mean=column.mean(), std=column.std(),
                              min_value=column.min(), max_value=column.max())
            elif pd.api.types.is_string_dtype(dtype):
                lengths = column.str.len()
                values.update(min_length=lengths.min(), max_length=lengths.max(), avg_length=lengths.mean())
            tasks[name] = values
        computed = dask.compute(tasks, scheduler=self._client or self.dask_config.scheduler)[0]
        for name, values in computed.items():
            values["dtype"] = str(self._current_ddf.dtypes[name])
            for key, value in list(values.items()):
                if key == "dtype":
                    continue
                if pd.isna(value):
                    values[key] = None
                elif key in {"row_count", "null_count", "distinct_count", "min_length", "max_length"}:
                    values[key] = int(value)
                else:
                    values[key] = float(value)
        return computed

    def profile_dataframe(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Convenience method to profile a DataFrame.

        Args:
            data: DataFrame or path to data file
            num_partitions: Number of partitions

        Returns:
            Column statistics dictionary
        """
        partitions = self.distribute_data(data, num_partitions)
        from truthound.profiler.distributed.profiler import DistributedProfiler
        results = self.map_partitions(DistributedProfiler._profile_partition, partitions, data)
        return self.aggregate_results(results)

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about the Dask cluster.

        Returns:
            Dictionary with cluster information
        """
        if not self._client:
            return {"status": "not_initialized", "scheduler": self.dask_config.scheduler}

        info = self._client.scheduler_info()
        workers = info.get("workers", {})

        return {
            "status": "running",
            "scheduler": self.dask_config.scheduler,
            "dashboard_link": self.dashboard_link,
            "n_workers": len(workers),
            "total_threads": sum(w.get("nthreads", 0) for w in workers.values()),
            "total_memory": sum(w.get("memory_limit", 0) for w in workers.values()),
            "workers": [
                {
                    "address": addr,
                    "nthreads": w.get("nthreads", 0),
                    "memory_limit": w.get("memory_limit", 0),
                    "status": w.get("status", "unknown"),
                }
                for addr, w in workers.items()
            ],
        }

    def scatter_data(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to workers.

        Args:
            data: Data to scatter
            broadcast: If True, replicate to all workers

        Returns:
            Future reference to scattered data
        """
        if not self._client:
            raise RuntimeError("Distributed client not initialized")

        return self._client.scatter(data, broadcast=broadcast)

    def submit(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Submit a function to be executed on the cluster.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        """
        if not self._client:
            raise RuntimeError("Distributed client not initialized")

        return self._client.submit(func, *args, **kwargs)

    def gather(self, futures: list) -> list:
        """Gather results from futures.

        Args:
            futures: List of future objects

        Returns:
            List of results
        """
        if not self._client:
            raise RuntimeError("Distributed client not initialized")

        return self._client.gather(futures)


# Register backend

backend_registry.register("dask", DaskBackend)
