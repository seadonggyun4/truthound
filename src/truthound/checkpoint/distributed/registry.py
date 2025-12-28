"""Backend Registry and Factory for Distributed Checkpoints.

This module provides a registry for backend implementations and
factory functions for creating orchestrators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Type

from truthound.checkpoint.distributed.protocols import (
    BackendNotAvailableError,
    DistributedConfig,
)

if TYPE_CHECKING:
    from truthound.checkpoint.distributed.base import (
        BaseDistributedBackend,
        BaseDistributedOrchestrator,
    )
    from truthound.checkpoint.distributed.orchestrator import (
        DistributedCheckpointOrchestrator,
    )


logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a registered backend.

    Attributes:
        name: Backend name.
        factory: Factory function or class.
        priority: Priority for auto-selection (higher = preferred).
        description: Human-readable description.
        install_hint: Installation instructions.
        check_available: Function to check if backend is available.
    """

    name: str
    factory: Callable[..., "BaseDistributedBackend"]
    priority: int = 0
    description: str = ""
    install_hint: str = ""
    check_available: Callable[[], bool] | None = None

    def is_available(self) -> bool:
        """Check if backend is available."""
        if self.check_available:
            try:
                return self.check_available()
            except Exception:
                return False
        return True


class BackendRegistry:
    """Registry for distributed backend implementations.

    The registry maintains a mapping of backend names to their
    implementations, supporting dynamic registration and auto-selection.

    Example:
        >>> from truthound.checkpoint.distributed import BackendRegistry
        >>>
        >>> # Get registered backends
        >>> backends = BackendRegistry.list_backends()
        >>>
        >>> # Get best available backend
        >>> backend = BackendRegistry.get_best_backend()
        >>>
        >>> # Register custom backend
        >>> BackendRegistry.register(
        ...     name="custom",
        ...     factory=MyCustomBackend,
        ...     priority=50,
        ... )
    """

    _backends: dict[str, BackendInfo] = {}
    _initialized = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure default backends are registered."""
        if cls._initialized:
            return

        cls._initialized = True

        # Register built-in backends
        cls._register_builtin_backends()

    @classmethod
    def _register_builtin_backends(cls) -> None:
        """Register built-in backends."""
        # Local backend (always available)
        def create_local(**kwargs: Any) -> "BaseDistributedBackend":
            from truthound.checkpoint.distributed.backends.local_backend import LocalBackend
            return LocalBackend(**kwargs)

        cls.register(
            name="local",
            factory=create_local,
            priority=10,
            description="Local multi-process/thread execution",
            check_available=lambda: True,
        )

        # Celery backend
        def check_celery() -> bool:
            try:
                import celery
                return True
            except ImportError:
                return False

        def create_celery(**kwargs: Any) -> "BaseDistributedBackend":
            from truthound.checkpoint.distributed.backends.celery_backend import CeleryBackend
            return CeleryBackend(**kwargs)

        cls.register(
            name="celery",
            factory=create_celery,
            priority=80,
            description="Celery distributed task queue",
            install_hint="pip install celery[redis]",
            check_available=check_celery,
        )

        # Ray backend
        def check_ray() -> bool:
            try:
                import ray
                return True
            except ImportError:
                return False

        def create_ray(**kwargs: Any) -> "BaseDistributedBackend":
            from truthound.checkpoint.distributed.backends.ray_backend import RayBackend
            return RayBackend(**kwargs)

        cls.register(
            name="ray",
            factory=create_ray,
            priority=90,
            description="Ray distributed computing framework",
            install_hint="pip install ray",
            check_available=check_ray,
        )

        # Ray Actor backend
        def create_ray_actor(**kwargs: Any) -> "BaseDistributedBackend":
            from truthound.checkpoint.distributed.backends.ray_backend import RayActorBackend
            return RayActorBackend(**kwargs)

        cls.register(
            name="ray-actor",
            factory=create_ray_actor,
            priority=85,
            description="Ray with persistent actors",
            install_hint="pip install ray",
            check_available=check_ray,
        )

        # Kubernetes backend
        def check_kubernetes() -> bool:
            try:
                import kubernetes
                return True
            except ImportError:
                return False

        def create_kubernetes(**kwargs: Any) -> "BaseDistributedBackend":
            from truthound.checkpoint.distributed.backends.kubernetes_backend import KubernetesBackend
            return KubernetesBackend(**kwargs)

        cls.register(
            name="kubernetes",
            factory=create_kubernetes,
            priority=70,
            description="Kubernetes Job-based execution",
            install_hint="pip install kubernetes",
            check_available=check_kubernetes,
        )

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., "BaseDistributedBackend"],
        priority: int = 0,
        description: str = "",
        install_hint: str = "",
        check_available: Callable[[], bool] | None = None,
    ) -> None:
        """Register a backend implementation.

        Args:
            name: Backend name (used for lookup).
            factory: Factory function or class to create backend.
            priority: Priority for auto-selection (higher = preferred).
            description: Human-readable description.
            install_hint: Installation instructions.
            check_available: Function to check if backend is available.
        """
        cls._backends[name] = BackendInfo(
            name=name,
            factory=factory,
            priority=priority,
            description=description,
            install_hint=install_hint,
            check_available=check_available,
        )
        logger.debug(f"Registered backend: {name} (priority={priority})")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a backend.

        Args:
            name: Backend name.

        Returns:
            True if removed, False if not found.
        """
        if name in cls._backends:
            del cls._backends[name]
            return True
        return False

    @classmethod
    def get(cls, name: str) -> BackendInfo | None:
        """Get backend info by name.

        Args:
            name: Backend name.

        Returns:
            BackendInfo if found, None otherwise.
        """
        cls._ensure_initialized()
        return cls._backends.get(name)

    @classmethod
    def list_backends(cls, available_only: bool = False) -> list[BackendInfo]:
        """List all registered backends.

        Args:
            available_only: If True, only return available backends.

        Returns:
            List of BackendInfo, sorted by priority.
        """
        cls._ensure_initialized()

        backends = list(cls._backends.values())

        if available_only:
            backends = [b for b in backends if b.is_available()]

        return sorted(backends, key=lambda b: -b.priority)

    @classmethod
    def get_best_backend(cls) -> BackendInfo:
        """Get the best available backend.

        Returns:
            BackendInfo for the highest priority available backend.

        Raises:
            BackendNotAvailableError: If no backends are available.
        """
        available = cls.list_backends(available_only=True)

        if not available:
            raise BackendNotAvailableError(
                "any",
                reason="No distributed backends are available",
                install_hint="pip install ray  # or: pip install celery[redis]",
            )

        return available[0]

    @classmethod
    def create_backend(
        cls,
        name: str | None = None,
        **kwargs: Any,
    ) -> "BaseDistributedBackend":
        """Create a backend instance.

        Args:
            name: Backend name (None = auto-select best available).
            **kwargs: Backend-specific configuration.

        Returns:
            Backend instance.

        Raises:
            BackendNotAvailableError: If backend not found or not available.
        """
        cls._ensure_initialized()

        if name is None:
            info = cls.get_best_backend()
        else:
            info = cls.get(name)
            if info is None:
                available = ", ".join(b.name for b in cls.list_backends())
                raise BackendNotAvailableError(
                    name,
                    reason=f"Backend not registered. Available: {available}",
                )
            if not info.is_available():
                raise BackendNotAvailableError(
                    name,
                    reason="Backend dependencies not installed",
                    install_hint=info.install_hint,
                )

        return info.factory(**kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def register_backend(
    name: str,
    factory: Callable[..., "BaseDistributedBackend"],
    priority: int = 0,
    description: str = "",
    install_hint: str = "",
    check_available: Callable[[], bool] | None = None,
) -> None:
    """Register a backend implementation.

    See BackendRegistry.register() for details.
    """
    BackendRegistry.register(
        name=name,
        factory=factory,
        priority=priority,
        description=description,
        install_hint=install_hint,
        check_available=check_available,
    )


def get_backend(
    name: str | None = None,
    **kwargs: Any,
) -> "BaseDistributedBackend":
    """Get a backend instance.

    Args:
        name: Backend name (None = auto-select best available).
        **kwargs: Backend-specific configuration.

    Returns:
        Backend instance.

    Example:
        >>> # Auto-select best backend
        >>> backend = get_backend()
        >>>
        >>> # Specific backend
        >>> backend = get_backend("celery", broker_url="redis://localhost:6379")
        >>>
        >>> # With configuration
        >>> backend = get_backend("ray", num_cpus=4)
    """
    return BackendRegistry.create_backend(name, **kwargs)


def get_orchestrator(
    backend: str | "BaseDistributedBackend" | None = None,
    config: DistributedConfig | None = None,
    **kwargs: Any,
) -> "DistributedCheckpointOrchestrator":
    """Get a distributed checkpoint orchestrator.

    Args:
        backend: Backend name, instance, or None for auto-select.
        config: Distributed configuration.
        **kwargs: Backend-specific configuration.

    Returns:
        DistributedCheckpointOrchestrator instance.

    Example:
        >>> # Auto-select backend
        >>> orchestrator = get_orchestrator()
        >>>
        >>> # Specific backend
        >>> orchestrator = get_orchestrator("celery", broker_url="redis://localhost")
        >>>
        >>> # With existing backend instance
        >>> backend = get_backend("ray")
        >>> orchestrator = get_orchestrator(backend)
        >>>
        >>> # Use orchestrator
        >>> with orchestrator:
        ...     task = orchestrator.submit(checkpoint)
        ...     result = task.result(timeout=300)
    """
    from truthound.checkpoint.distributed.orchestrator import DistributedCheckpointOrchestrator
    from truthound.checkpoint.distributed.base import BaseDistributedBackend

    if isinstance(backend, BaseDistributedBackend):
        backend_instance = backend
    else:
        backend_instance = get_backend(backend, **kwargs)

    return DistributedCheckpointOrchestrator(
        backend=backend_instance,
        config=config,
    )


def list_backends(available_only: bool = False) -> list[str]:
    """List registered backend names.

    Args:
        available_only: If True, only return available backends.

    Returns:
        List of backend names.
    """
    return [b.name for b in BackendRegistry.list_backends(available_only)]


def is_backend_available(name: str) -> bool:
    """Check if a backend is available.

    Args:
        name: Backend name.

    Returns:
        True if available, False otherwise.
    """
    info = BackendRegistry.get(name)
    return info is not None and info.is_available()
