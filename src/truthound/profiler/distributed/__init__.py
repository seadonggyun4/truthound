"""Distributed processing backends for large-scale data profiling.

This package provides pluggable distributed computing backends with real integrations.

Backends:
- LocalBackend: Multi-threaded local processing (default fallback)
- DaskBackend: Dask distributed computing (primary recommended)
- SparkBackend: Apache Spark for cluster computing
- RayBackend: Ray for distributed computing

Example:
    from truthound.profiler.distributed import (
        create_backend,
        profile_distributed,
        DaskBackend,
    )

    # Auto-detect best backend
    with create_backend("auto") as backend:
        profile = backend.profile(data)

    # Or use specific backend
    from truthound.profiler.distributed.dask_backend import DaskBackend
    backend = DaskBackend(scheduler="distributed", n_workers=4)
"""

from truthound.profiler.distributed.base import (
    BackendType,
    PartitionStrategy,
    PartitionInfo,
    WorkerResult,
    BackendConfig,
    DistributedBackend,
    BackendRegistry,
    backend_registry,
)
from truthound.profiler.distributed.local_backend import LocalBackend
from truthound.profiler.distributed.dask_backend import (
    DaskBackend,
    DaskConfig,
)
from truthound.profiler.distributed.spark_backend import (
    SparkBackend,
    SparkConfig,
)
from truthound.profiler.distributed.ray_backend import (
    RayBackend,
    RayConfig,
)
from truthound.profiler.distributed.profiler import (
    DistributedProfiler,
    DistributedProfileConfig,
    create_distributed_profiler,
    profile_distributed,
    get_available_backends,
)

__all__ = [
    # Types
    "BackendType",
    "PartitionStrategy",
    "PartitionInfo",
    "WorkerResult",
    "BackendConfig",
    # Base classes
    "DistributedBackend",
    "BackendRegistry",
    "backend_registry",
    # Backends
    "LocalBackend",
    "DaskBackend",
    "DaskConfig",
    "SparkBackend",
    "SparkConfig",
    "RayBackend",
    "RayConfig",
    # Profiler
    "DistributedProfiler",
    "DistributedProfileConfig",
    "create_distributed_profiler",
    "profile_distributed",
    "get_available_backends",
]
