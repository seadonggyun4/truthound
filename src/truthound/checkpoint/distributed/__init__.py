"""Distributed Checkpoint Orchestration Framework.

This module provides multi-node distributed checkpoint execution with support
for various backends including Celery, Ray, and Kubernetes.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  DistributedCheckpointOrchestrator                   │
    │                     (Unified Abstraction Layer)                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │   CeleryBackend   │    RayBackend    │   KubernetesBackend          │
    │  (Task Queue)     │  (Actor-based)   │   (Pod-based)                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                   DistributedBackendProtocol                         │
    │          (Interface for custom backend implementations)              │
    └─────────────────────────────────────────────────────────────────────┘

Example:
    >>> from truthound.checkpoint.distributed import (
    ...     DistributedCheckpointOrchestrator,
    ...     get_orchestrator,
    ... )
    >>>
    >>> # Auto-detect best available backend
    >>> orchestrator = get_orchestrator()
    >>>
    >>> # Or specify backend explicitly
    >>> orchestrator = get_orchestrator("celery", broker_url="redis://localhost:6379")
    >>>
    >>> # Submit checkpoint for distributed execution
    >>> task = orchestrator.submit(checkpoint, priority=1)
    >>>
    >>> # Wait for result
    >>> result = task.result(timeout=300)
    >>>
    >>> # Or run multiple checkpoints in parallel
    >>> tasks = orchestrator.submit_batch([cp1, cp2, cp3])
    >>> results = orchestrator.gather(tasks, timeout=600)
"""

from truthound.checkpoint.distributed.protocols import (
    # Core Protocols
    DistributedBackendProtocol,
    DistributedTaskProtocol,
    # Data Classes
    DistributedTask,
    TaskState,
    TaskPriority,
    DistributedTaskResult,
    WorkerInfo,
    WorkerState,
    ClusterState,
    # Configuration
    DistributedConfig,
    BackendCapability,
    # Exceptions
    DistributedError,
    TaskSubmissionError,
    TaskTimeoutError,
    TaskCancelledError,
    WorkerNotAvailableError,
    BackendNotAvailableError,
)

from truthound.checkpoint.distributed.base import (
    BaseDistributedBackend,
    BaseDistributedOrchestrator,
)

from truthound.checkpoint.distributed.orchestrator import (
    DistributedCheckpointOrchestrator,
)

from truthound.checkpoint.distributed.registry import (
    BackendRegistry,
    register_backend,
    get_backend,
    get_orchestrator,
    list_backends,
    is_backend_available,
)

# Backend implementations (lazy loaded)
def __getattr__(name: str):
    if name == "CeleryBackend":
        from truthound.checkpoint.distributed.backends.celery_backend import CeleryBackend
        return CeleryBackend
    elif name == "RayBackend":
        from truthound.checkpoint.distributed.backends.ray_backend import RayBackend
        return RayBackend
    elif name == "KubernetesBackend":
        from truthound.checkpoint.distributed.backends.kubernetes_backend import KubernetesBackend
        return KubernetesBackend
    elif name == "LocalBackend":
        from truthound.checkpoint.distributed.backends.local_backend import LocalBackend
        return LocalBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core Protocols
    "DistributedBackendProtocol",
    "DistributedTaskProtocol",
    # Data Classes
    "DistributedTask",
    "TaskState",
    "TaskPriority",
    "DistributedTaskResult",
    "WorkerInfo",
    "WorkerState",
    "ClusterState",
    # Configuration
    "DistributedConfig",
    "BackendCapability",
    # Exceptions
    "DistributedError",
    "TaskSubmissionError",
    "TaskTimeoutError",
    "TaskCancelledError",
    "WorkerNotAvailableError",
    "BackendNotAvailableError",
    # Base Classes
    "BaseDistributedBackend",
    "BaseDistributedOrchestrator",
    # Orchestrator
    "DistributedCheckpointOrchestrator",
    # Registry
    "BackendRegistry",
    "register_backend",
    "get_backend",
    "get_orchestrator",
    "list_backends",
    "is_backend_available",
    # Backends (lazy loaded)
    "CeleryBackend",
    "RayBackend",
    "KubernetesBackend",
    "LocalBackend",
]
