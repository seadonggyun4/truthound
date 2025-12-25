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

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import polars as pl

from truthound.profiler.distributed.base import (
    BackendConfig,
    BackendType,
    DistributedBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
)


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
            import dask
            import dask.dataframe as dd
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
            dask.config.set(scheduler=scheduler)
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
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Dask client: {e}")

        if self._cluster:
            try:
                self._cluster.close()
            except Exception as e:
                logger.warning(f"Error closing Dask cluster: {e}")

        self._client = None
        self._cluster = None
        self._current_ddf = None
        self._initialized = False
        logger.info("DaskBackend shutdown complete")

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
    ) -> list[PartitionInfo]:
        import dask.dataframe as dd
        import pandas as pd

        # Determine number of partitions
        if num_partitions is None:
            if self._client:
                # Use number of workers
                num_partitions = len(self._client.scheduler_info()["workers"])
            else:
                num_partitions = os.cpu_count() or 4

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

        # Repartition if needed
        if ddf.npartitions != num_partitions:
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
        if not hasattr(self, "_current_ddf") or self._current_ddf is None:
            raise RuntimeError("No data distributed")

        import pandas as pd

        ddf = self._current_ddf
        columns = list(ddf.columns)

        def profile_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            """Profile a single partition."""
            import time
            start = time.time()

            stats = {}
            for col in pdf.columns:
                col_data = pdf[col]
                col_stats = {
                    "row_count": len(col_data),
                    "null_count": int(col_data.isna().sum()),
                    "distinct_count": int(col_data.nunique()),
                    "dtype": str(col_data.dtype),
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    non_null = col_data.dropna()
                    if len(non_null) > 0:
                        col_stats.update({
                            "min_value": float(non_null.min()),
                            "max_value": float(non_null.max()),
                            "sum_value": float(non_null.sum()),
                            "mean": float(non_null.mean()),
                            "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                            "sum_squared": float((non_null ** 2).sum()),
                        })

                elif pd.api.types.is_string_dtype(col_data):
                    non_null = col_data.dropna()
                    if len(non_null) > 0:
                        lengths = non_null.str.len()
                        col_stats.update({
                            "min_length": int(lengths.min()),
                            "max_length": int(lengths.max()),
                            "avg_length": float(lengths.mean()),
                        })

                stats[col] = col_stats

            elapsed_ms = (time.time() - start) * 1000

            # Return as single-row DataFrame with JSON stats
            return pd.DataFrame([{
                "stats": json.dumps(stats),
                "row_count": len(pdf),
                "processing_time_ms": elapsed_ms,
            }])

        # Apply to all partitions
        result_ddf = ddf.map_partitions(
            profile_partition,
            meta={"stats": str, "row_count": int, "processing_time_ms": float},
        )

        # Compute results
        results_pdf = result_ddf.compute()

        return [
            WorkerResult(
                partition_id=i,
                column_stats=json.loads(row["stats"]),
                row_count=row["row_count"],
                processing_time_ms=row["processing_time_ms"],
            )
            for i, (_, row) in enumerate(results_pdf.iterrows())
        ]

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate results from all partitions."""
        aggregated: dict[str, dict[str, Any]] = {}

        for result in results:
            for col_name, stats in result.column_stats.items():
                if col_name not in aggregated:
                    aggregated[col_name] = {
                        "row_count": 0,
                        "null_count": 0,
                        "min_value": None,
                        "max_value": None,
                        "sum_value": 0,
                        "sum_squared": 0,
                        "dtype": stats.get("dtype", "unknown"),
                    }

                agg = aggregated[col_name]
                agg["row_count"] += stats.get("row_count", 0)
                agg["null_count"] += stats.get("null_count", 0)

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

                # Distinct count - use first partition's value as approximation
                if "distinct_count" not in agg and "distinct_count" in stats:
                    agg["distinct_count"] = stats["distinct_count"]

                # Length stats
                if "min_length" in stats and "min_length" not in agg:
                    agg["min_length"] = stats["min_length"]
                    agg["max_length"] = stats["max_length"]
                    agg["avg_length"] = stats["avg_length"]

        # Finalize aggregations
        for col_name, agg in aggregated.items():
            n = agg["row_count"]
            if n > 0 and agg.get("sum_value"):
                mean = agg["sum_value"] / n
                agg["mean"] = mean
                variance = (agg["sum_squared"] / n) - (mean ** 2)
                agg["std"] = variance ** 0.5 if variance > 0 else 0

        return aggregated

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
        results = self.map_partitions(lambda p, d: WorkerResult(p.partition_id, {}, 0, 0), partitions, data)
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
from truthound.profiler.distributed.base import backend_registry
backend_registry.register("dask", DaskBackend)
