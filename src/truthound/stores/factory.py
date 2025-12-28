"""Factory functions for creating stores.

This module provides a registry-based factory pattern for creating store
instances. New store backends can be registered at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from truthound.stores.base import BaseStore, StoreConfig, StoreError

# Type for store constructor functions
StoreConstructor = Callable[..., BaseStore[Any, Any]]

# Registry of store constructors
_store_registry: dict[str, StoreConstructor] = {}


def register_store(name: str) -> Callable[[StoreConstructor], StoreConstructor]:
    """Decorator to register a store backend.

    Args:
        name: Name to register the store under.

    Returns:
        Decorator function.

    Example:
        >>> @register_store("my_store")
        ... class MyStore(BaseStore):
        ...     pass
    """

    def decorator(cls: StoreConstructor) -> StoreConstructor:
        _store_registry[name] = cls
        return cls

    return decorator


def get_store(backend: str, **kwargs: Any) -> BaseStore[Any, Any]:
    """Create a store instance for the specified backend.

    This is the primary entry point for creating stores. It handles
    lazy loading of backend modules and provides a uniform interface.

    Args:
        backend: Name of the store backend to use. Options:
            - "filesystem": Local filesystem storage (default)
            - "s3": AWS S3 storage (requires boto3)
            - "gcs": Google Cloud Storage (requires google-cloud-storage)
            - "database": SQL database storage (requires sqlalchemy)
            - "memory": In-memory storage (for testing)
        **kwargs: Backend-specific configuration options.

    Returns:
        Configured store instance.

    Raises:
        StoreError: If the backend is not available or configuration fails.

    Example:
        >>> # Filesystem store
        >>> store = get_store("filesystem", base_path=".truthound/results")
        >>>
        >>> # S3 store
        >>> store = get_store("s3", bucket="my-bucket", prefix="validations/")
        >>>
        >>> # Memory store (for testing)
        >>> store = get_store("memory")
    """
    # Normalize backend name
    backend = backend.lower().strip()

    # Check if already registered
    if backend in _store_registry:
        return _store_registry[backend](**kwargs)

    # Lazy load built-in backends
    if backend == "filesystem":
        from truthound.stores.backends.filesystem import FileSystemStore

        return FileSystemStore(**kwargs)

    elif backend == "memory":
        from truthound.stores.backends.memory import MemoryStore

        return MemoryStore(**kwargs)

    elif backend == "s3":
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise StoreError(
                "S3 backend requires boto3. Install with: pip install truthound[s3]"
            )
        from truthound.stores.backends.s3 import S3Store

        return S3Store(**kwargs)

    elif backend == "gcs":
        try:
            import google.cloud.storage  # noqa: F401
        except ImportError:
            raise StoreError(
                "GCS backend requires google-cloud-storage. "
                "Install with: pip install truthound[gcs]"
            )
        from truthound.stores.backends.gcs import GCSStore

        return GCSStore(**kwargs)

    elif backend in ("database", "db", "sql"):
        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            raise StoreError(
                "Database backend requires sqlalchemy. "
                "Install with: pip install truthound[database]"
            )
        from truthound.stores.backends.database import DatabaseStore

        return DatabaseStore(**kwargs)

    elif backend in ("azure", "azure_blob", "azureblob"):
        try:
            import azure.storage.blob  # noqa: F401
        except ImportError:
            raise StoreError(
                "Azure Blob backend requires azure-storage-blob. "
                "Install with: pip install truthound[azure]"
            )
        from truthound.stores.backends.azure_blob import AzureBlobStore

        return AzureBlobStore(**kwargs)

    else:
        available = list(_store_registry.keys()) + [
            "filesystem",
            "memory",
            "s3",
            "gcs",
            "database",
            "azure",
        ]
        raise StoreError(
            f"Unknown store backend: {backend}. "
            f"Available backends: {', '.join(sorted(set(available)))}"
        )


def list_available_backends() -> list[str]:
    """List all available store backends.

    Returns:
        List of backend names that can be used with get_store().
    """
    # Built-in backends always available
    backends = ["filesystem", "memory"]

    # Check optional backends
    try:
        import boto3

        backends.append("s3")
    except ImportError:
        pass

    try:
        import google.cloud.storage

        backends.append("gcs")
    except ImportError:
        pass

    try:
        import sqlalchemy

        backends.append("database")
    except ImportError:
        pass

    try:
        import azure.storage.blob

        backends.append("azure")
    except ImportError:
        pass

    # Add registered backends
    backends.extend(_store_registry.keys())

    return sorted(set(backends))


def is_backend_available(backend: str) -> bool:
    """Check if a backend is available.

    Args:
        backend: Name of the backend to check.

    Returns:
        True if the backend is available, False otherwise.
    """
    backend = backend.lower().strip()

    if backend in ("filesystem", "memory"):
        return True

    if backend in _store_registry:
        return True

    if backend == "s3":
        try:
            import boto3

            return True
        except ImportError:
            return False

    if backend == "gcs":
        try:
            import google.cloud.storage

            return True
        except ImportError:
            return False

    if backend in ("database", "db", "sql"):
        try:
            import sqlalchemy

            return True
        except ImportError:
            return False

    if backend in ("azure", "azure_blob", "azureblob"):
        try:
            import azure.storage.blob

            return True
        except ImportError:
            return False

    return False
