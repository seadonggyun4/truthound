"""Distributed Checkpoint Backend Implementations.

This module provides backend implementations for various distributed
execution frameworks.

Available Backends:
    - LocalBackend: Local multi-process execution (default)
    - CeleryBackend: Celery task queue
    - RayBackend: Ray distributed computing
    - KubernetesBackend: Kubernetes Job/Pod execution
"""

from truthound.checkpoint.distributed.backends.local_backend import LocalBackend


def __getattr__(name: str):
    """Lazy load backends to avoid import errors for optional dependencies."""
    if name == "CeleryBackend":
        from truthound.checkpoint.distributed.backends.celery_backend import CeleryBackend
        return CeleryBackend
    elif name == "RayBackend":
        from truthound.checkpoint.distributed.backends.ray_backend import RayBackend
        return RayBackend
    elif name == "KubernetesBackend":
        from truthound.checkpoint.distributed.backends.kubernetes_backend import KubernetesBackend
        return KubernetesBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LocalBackend",
    "CeleryBackend",
    "RayBackend",
    "KubernetesBackend",
]
