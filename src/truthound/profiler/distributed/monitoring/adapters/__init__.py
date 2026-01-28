"""Backend-specific monitoring adapters.

This module provides monitoring adapters for different distributed backends,
enabling seamless integration with Dask, Spark, Ray, and local backends.
"""

from __future__ import annotations

from truthound.profiler.distributed.monitoring.adapters.base import (
    BackendMonitorAdapter,
    BackendMonitorProtocol,
)
from truthound.profiler.distributed.monitoring.adapters.local import LocalMonitorAdapter
from truthound.profiler.distributed.monitoring.adapters.dask import DaskMonitorAdapter
from truthound.profiler.distributed.monitoring.adapters.spark import SparkMonitorAdapter
from truthound.profiler.distributed.monitoring.adapters.ray import RayMonitorAdapter

__all__ = [
    "BackendMonitorAdapter",
    "BackendMonitorProtocol",
    "LocalMonitorAdapter",
    "DaskMonitorAdapter",
    "SparkMonitorAdapter",
    "RayMonitorAdapter",
]
