"""Distributed profiler interface.

High-level interface for distributed data profiling that wraps
the various backends into a unified API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    TableProfile,
)
from truthound.profiler.distributed.base import (
    BackendConfig,
    DistributedBackend,
    PartitionStrategy,
    backend_registry,
)


logger = logging.getLogger(__name__)


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

        # Or with explicit configuration
        from truthound.profiler.distributed import DaskConfig

        config = DaskConfig(scheduler="distributed", n_workers=8)
        profiler = DistributedProfiler.create(
            backend="dask",
            backend_config=config,
        )

        with profiler:
            profile = profiler.profile(df, name="my_table")
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
            # Prefer in order: dask, ray, spark, local
            for preferred in ["dask", "ray", "spark", "local"]:
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

    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        return self._backend.name

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
        total_errors = sum(len(r.errors) for r in results)
        columns = list(aggregated.keys())

        if total_errors > 0:
            logger.warning(f"Profiling completed with {total_errors} errors")

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
        partition: Any,
        data: Any,
    ) -> Any:
        """Profile a single partition (called by workers)."""
        # This is a placeholder - actual implementation in backend
        pass

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
        if "mean" in stats or "min_value" in stats:
            distribution = DistributionStats(
                mean=stats.get("mean"),
                std=stats.get("std"),
                min=stats.get("min_value"),
                max=stats.get("max_value"),
            )

        # Infer type from dtype string
        dtype_str = stats.get("dtype", "").lower()
        if "int" in dtype_str:
            inferred_type = DataType.INTEGER
        elif "float" in dtype_str:
            inferred_type = DataType.FLOAT
        elif "bool" in dtype_str:
            inferred_type = DataType.BOOLEAN
        elif "date" in dtype_str:
            inferred_type = DataType.DATE
        elif "str" in dtype_str or "object" in dtype_str:
            inferred_type = DataType.STRING
        else:
            inferred_type = DataType.STRING

        return ColumnProfile(
            name=name,
            physical_type=stats.get("dtype", "unknown"),
            inferred_type=inferred_type,
            row_count=row_count,
            null_count=null_count,
            null_ratio=null_count / row_count if row_count > 0 else 0,
            distinct_count=distinct_count,
            unique_ratio=distinct_count / row_count if row_count > 0 else 0,
            distribution=distribution,
            min_length=stats.get("min_length"),
            max_length=stats.get("max_length"),
            avg_length=stats.get("avg_length"),
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
