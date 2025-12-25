"""Base classes and types for distributed backends.

This module provides the abstract interfaces and common types used by all
distributed processing backends.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

import polars as pl


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

    ROW_BASED = "row_based"        # Split by row ranges
    COLUMN_BASED = "column_based"  # Profile columns in parallel
    HYBRID = "hybrid"              # Combine both strategies
    HASH = "hash"                  # Hash-based partitioning


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

    def __init__(self, config: BackendConfig | None = None):
        self.config = config or BackendConfig()
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
# Backend Registry
# =============================================================================


class BackendRegistry:
    """Registry for distributed backends.

    Allows dynamic registration of custom backends.
    """

    _instance: "BackendRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "BackendRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._backends: dict[str, type[DistributedBackend]] = {}
        return cls._instance

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
