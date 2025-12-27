"""Registry for distributed execution engines.

This module provides a registry pattern for discovering and instantiating
distributed execution engines. It enables:
- Dynamic engine registration
- Auto-detection of available backends
- Lazy loading of engine implementations
- Plugin support for custom engines

Example:
    >>> from truthound.execution.distributed import (
    ...     get_distributed_engine,
    ...     register_distributed_engine,
    ... )
    >>>
    >>> # Get auto-detected best engine
    >>> engine = get_distributed_engine(spark_df)
    >>>
    >>> # Register custom engine
    >>> register_distributed_engine("custom", CustomDistributedEngine)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Type

from truthound.execution.distributed.protocols import ComputeBackend

if TYPE_CHECKING:
    from truthound.execution.distributed.base import (
        BaseDistributedEngine,
        DistributedEngineConfig,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Engine Entry
# =============================================================================


@dataclass
class EngineEntry:
    """Entry in the engine registry.

    Attributes:
        name: Engine name.
        engine_class: Engine class (or lazy loader).
        backend: Compute backend type.
        priority: Priority for auto-detection (higher = preferred).
        is_lazy: Whether the engine class is lazily loaded.
    """

    name: str
    engine_class: Type["BaseDistributedEngine"] | Callable[[], Type["BaseDistributedEngine"]]
    backend: ComputeBackend
    priority: int = 0
    is_lazy: bool = False

    def get_class(self) -> Type["BaseDistributedEngine"]:
        """Get the engine class, loading lazily if needed."""
        if self.is_lazy and callable(self.engine_class):
            return self.engine_class()
        return self.engine_class  # type: ignore


# =============================================================================
# Engine Registry
# =============================================================================


class DistributedEngineRegistry:
    """Registry for distributed execution engines.

    The registry provides:
    - Registration of engine implementations
    - Auto-detection of available engines
    - Priority-based engine selection
    - Lazy loading support

    Example:
        >>> registry = DistributedEngineRegistry()
        >>> registry.register("spark", SparkExecutionEngine, ComputeBackend.SPARK)
        >>> engine = registry.get("spark", spark_df)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._entries: dict[str, EngineEntry] = {}
        self._backend_map: dict[ComputeBackend, str] = {}

    def register(
        self,
        name: str,
        engine_class: Type["BaseDistributedEngine"] | Callable[[], Type["BaseDistributedEngine"]],
        backend: ComputeBackend,
        priority: int = 0,
        is_lazy: bool = False,
    ) -> None:
        """Register an engine.

        Args:
            name: Engine name.
            engine_class: Engine class or lazy loader function.
            backend: Compute backend type.
            priority: Priority for auto-detection.
            is_lazy: Whether to load lazily.
        """
        entry = EngineEntry(
            name=name,
            engine_class=engine_class,
            backend=backend,
            priority=priority,
            is_lazy=is_lazy,
        )
        self._entries[name] = entry
        self._backend_map[backend] = name
        logger.debug(f"Registered engine: {name} ({backend.value})")

    def unregister(self, name: str) -> None:
        """Unregister an engine.

        Args:
            name: Engine name.
        """
        if name in self._entries:
            entry = self._entries.pop(name)
            if self._backend_map.get(entry.backend) == name:
                del self._backend_map[entry.backend]

    def get_class(self, name: str) -> Type["BaseDistributedEngine"]:
        """Get an engine class by name.

        Args:
            name: Engine name.

        Returns:
            Engine class.

        Raises:
            KeyError: If engine not found.
        """
        if name not in self._entries:
            raise KeyError(
                f"Unknown engine: {name}. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[name].get_class()

    def get(
        self,
        name_or_data: str | Any,
        config: "DistributedEngineConfig | None" = None,
        **kwargs: Any,
    ) -> "BaseDistributedEngine":
        """Get an engine instance.

        Args:
            name_or_data: Engine name or data to auto-detect engine for.
            config: Optional configuration.
            **kwargs: Additional arguments for engine constructor.

        Returns:
            Engine instance.
        """
        if isinstance(name_or_data, str):
            # Explicit engine name
            engine_class = self.get_class(name_or_data)
            return engine_class(config=config, **kwargs)
        else:
            # Auto-detect from data
            return self.create_for_data(name_or_data, config, **kwargs)

    def create_for_data(
        self,
        data: Any,
        config: "DistributedEngineConfig | None" = None,
        **kwargs: Any,
    ) -> "BaseDistributedEngine":
        """Create an engine for the given data.

        Args:
            data: Data to create engine for.
            config: Optional configuration.
            **kwargs: Additional arguments.

        Returns:
            Engine instance appropriate for the data.

        Raises:
            ValueError: If no suitable engine found.
        """
        backend = self._detect_backend(data)
        engine_name = self._backend_map.get(backend)

        if not engine_name:
            raise ValueError(
                f"No engine registered for backend: {backend.value}. "
                f"Available backends: {list(self._backend_map.keys())}"
            )

        engine_class = self.get_class(engine_name)

        # Handle different engine constructors based on backend
        if backend == ComputeBackend.SPARK:
            return engine_class(spark_df=data, config=config, **kwargs)
        elif backend == ComputeBackend.DASK:
            return engine_class(dask_df=data, config=config, **kwargs)
        elif backend == ComputeBackend.RAY:
            return engine_class(dataset=data, config=config, **kwargs)
        else:
            return engine_class(data=data, config=config, **kwargs)

    def _detect_backend(self, data: Any) -> ComputeBackend:
        """Detect the compute backend from data type.

        Args:
            data: Data to inspect.

        Returns:
            Detected compute backend.
        """
        data_type = type(data).__name__
        module = type(data).__module__

        # Check for Spark
        if "pyspark" in module or data_type == "DataFrame" and hasattr(data, "sparkSession"):
            return ComputeBackend.SPARK

        # Check for Dask
        if "dask" in module:
            return ComputeBackend.DASK

        # Check for Ray
        if "ray" in module:
            return ComputeBackend.RAY

        # Default to local
        return ComputeBackend.LOCAL

    def list_engines(self) -> list[str]:
        """List all registered engine names.

        Returns:
            List of engine names.
        """
        return list(self._entries.keys())

    def list_available(self) -> list[str]:
        """List engines with available dependencies.

        Returns:
            List of available engine names.
        """
        available = []
        for name, entry in self._entries.items():
            try:
                engine_class = entry.get_class()
                # Check if dependencies are available
                if entry.backend == ComputeBackend.SPARK:
                    import pyspark  # noqa: F401
                    available.append(name)
                elif entry.backend == ComputeBackend.DASK:
                    import dask  # noqa: F401
                    available.append(name)
                elif entry.backend == ComputeBackend.RAY:
                    import ray  # noqa: F401
                    available.append(name)
                else:
                    available.append(name)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Engine {name} not available: {e}")

        return available

    def get_best_available(self) -> str | None:
        """Get the best available engine by priority.

        Returns:
            Engine name or None if no engines available.
        """
        available = self.list_available()
        if not available:
            return None

        # Sort by priority
        available.sort(
            key=lambda n: self._entries[n].priority,
            reverse=True,
        )
        return available[0]


# =============================================================================
# Global Registry
# =============================================================================


_global_registry = DistributedEngineRegistry()


def _lazy_spark_engine() -> Type["BaseDistributedEngine"]:
    """Lazy loader for SparkExecutionEngine."""
    from truthound.execution.distributed.spark_engine import SparkExecutionEngine
    return SparkExecutionEngine


# =============================================================================
# Lazy Loaders for Engines
# =============================================================================


def _lazy_dask_engine() -> Type["BaseDistributedEngine"]:
    """Lazy loader for DaskExecutionEngine."""
    from truthound.execution.distributed.dask_engine import DaskExecutionEngine
    return DaskExecutionEngine


def _lazy_ray_engine() -> Type["BaseDistributedEngine"]:
    """Lazy loader for RayExecutionEngine."""
    from truthound.execution.distributed.ray_engine import RayExecutionEngine
    return RayExecutionEngine


# =============================================================================
# Register Built-in Engines
# =============================================================================

# Register Spark engine (highest priority for big data workloads)
_global_registry.register(
    "spark",
    _lazy_spark_engine,
    ComputeBackend.SPARK,
    priority=100,
    is_lazy=True,
)

# Register Dask engine (good for pandas-like workflows)
_global_registry.register(
    "dask",
    _lazy_dask_engine,
    ComputeBackend.DASK,
    priority=80,
    is_lazy=True,
)

# Register Ray engine (good for ML workloads)
_global_registry.register(
    "ray",
    _lazy_ray_engine,
    ComputeBackend.RAY,
    priority=90,
    is_lazy=True,
)


# =============================================================================
# Public API
# =============================================================================


def get_distributed_engine(
    name_or_data: str | Any,
    config: "DistributedEngineConfig | None" = None,
    **kwargs: Any,
) -> "BaseDistributedEngine":
    """Get a distributed execution engine.

    This is the main entry point for creating distributed engines.

    Args:
        name_or_data: Engine name (e.g., "spark") or data to auto-detect.
        config: Optional configuration.
        **kwargs: Additional arguments for engine constructor.

    Returns:
        Distributed execution engine.

    Example:
        >>> # Auto-detect from data
        >>> engine = get_distributed_engine(spark_df)
        >>>
        >>> # Explicit engine name
        >>> engine = get_distributed_engine("spark", spark_df=my_df)
    """
    return _global_registry.get(name_or_data, config, **kwargs)


def register_distributed_engine(
    name: str,
    engine_class: Type["BaseDistributedEngine"],
    backend: ComputeBackend,
    priority: int = 0,
) -> None:
    """Register a custom distributed execution engine.

    Args:
        name: Engine name.
        engine_class: Engine class.
        backend: Compute backend type.
        priority: Priority for auto-detection.

    Example:
        >>> class CustomSparkEngine(BaseDistributedEngine):
        ...     pass
        >>>
        >>> register_distributed_engine(
        ...     "custom_spark",
        ...     CustomSparkEngine,
        ...     ComputeBackend.SPARK,
        ...     priority=200,  # Higher than default
        ... )
    """
    _global_registry.register(name, engine_class, backend, priority)


def list_distributed_engines() -> list[str]:
    """List all registered distributed engines.

    Returns:
        List of engine names.
    """
    return _global_registry.list_engines()


def list_available_engines() -> list[str]:
    """List distributed engines with available dependencies.

    Returns:
        List of available engine names.
    """
    return _global_registry.list_available()


def get_engine_registry() -> DistributedEngineRegistry:
    """Get the global engine registry.

    Returns:
        The global registry instance.
    """
    return _global_registry
