"""Stores module for persisting validation results and expectations.

This module provides a unified interface for storing and retrieving validation
results across different backends (filesystem, S3, GCS, databases).

Example:
    >>> from truthound.stores import get_store, ValidationResult
    >>>
    >>> # Use filesystem store (default)
    >>> store = get_store("filesystem", base_path=".truthound/results")
    >>>
    >>> # Save validation result
    >>> result = ValidationResult.from_report(report, data_asset="customers.csv")
    >>> run_id = store.save(result)
    >>>
    >>> # Retrieve result
    >>> result = store.get(run_id)
    >>>
    >>> # List all results for a data asset
    >>> run_ids = store.list_runs("customers.csv")

Concurrent Access:
    For multi-threaded or multi-process access, use ConcurrentFileSystemStore:

    >>> from truthound.stores.backends.concurrent_filesystem import (
    ...     ConcurrentFileSystemStore,
    ...     ConcurrencyConfig,
    ... )
    >>>
    >>> store = ConcurrentFileSystemStore(
    ...     base_path=".truthound/results",
    ...     concurrency=ConcurrencyConfig(lock_strategy="auto"),
    ... )

Enterprise Database Pooling:
    For enterprise-grade database connection pooling with circuit breaker,
    retry logic, and health monitoring:

    >>> from truthound.stores.backends.database import DatabaseStore, PoolingConfig
    >>>
    >>> store = DatabaseStore(
    ...     connection_url="postgresql://user:pass@localhost/db",
    ...     pooling=PoolingConfig(
    ...         pool_size=10,
    ...         max_overflow=20,
    ...         enable_circuit_breaker=True,
    ...         enable_health_checks=True,
    ...     ),
    ... )
    >>>
    >>> # Access pool metrics
    >>> print(store.pool_metrics.to_dict())
    >>> print(store.get_pool_status())

Streaming Storage:
    For large-scale validation results that cannot fit in memory, use
    streaming stores:

    >>> from truthound.stores.streaming import StreamingFileSystemStore
    >>>
    >>> store = StreamingFileSystemStore(base_path=".truthound/streaming")
    >>>
    >>> # Create a streaming session
    >>> session = store.create_session("run_001", "large_dataset.csv")
    >>>
    >>> # Write results incrementally
    >>> with store.create_writer(session) as writer:
    ...     for result in validation_results:
    ...         writer.write_result(result)
    >>>
    >>> # Read results back efficiently
    >>> for result in store.iter_results("run_001"):
    ...     process(result)

Encryption:
    For encrypting sensitive validation results:

    >>> from truthound.stores.encryption import (
    ...     get_encryptor,
    ...     generate_key,
    ...     EncryptionAlgorithm,
    ...     create_secure_pipeline,
    ... )
    >>>
    >>> # Generate a key and encrypt data
    >>> key = generate_key(EncryptionAlgorithm.AES_256_GCM)
    >>> encryptor = get_encryptor("aes-256-gcm")
    >>> encrypted = encryptor.encrypt(sensitive_data, key)
    >>>
    >>> # Or use a pipeline for compress-then-encrypt
    >>> pipeline = create_secure_pipeline(key, compression="gzip")
    >>> result = pipeline.process(data)
    >>> original = pipeline.reverse(result.data, result.header)
"""

from truthound.stores.base import (
    BaseStore,
    StoreConfig,
    StoreQuery,
    StoreError,
    StoreNotFoundError,
    StoreConnectionError,
)
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)
from truthound.stores.expectations import (
    Expectation,
    ExpectationSuite,
)
from truthound.stores.factory import get_store, register_store

# Lazy import for submodules
_SUBMODULES = {
    "streaming",
    "encryption",
    "compression",
    "versioning",
    "caching",
    "tiering",
    "backpressure",
    "batching",
    "retention",
    "replication",
    "migration",
    "observability",
    "concurrency",
}


def __getattr__(name: str):
    """Lazy import submodules."""
    if name in _SUBMODULES:
        import importlib
        module = importlib.import_module(f"truthound.stores.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "BaseStore",
    "StoreConfig",
    "StoreQuery",
    "StoreError",
    "StoreNotFoundError",
    "StoreConnectionError",
    # Result types
    "ValidationResult",
    "ValidatorResult",
    "ResultStatistics",
    "ResultStatus",
    # Expectation types
    "Expectation",
    "ExpectationSuite",
    # Factory functions
    "get_store",
    "register_store",
    # Submodules (lazy loaded)
    "streaming",
    "encryption",
    "compression",
    "versioning",
    "caching",
    "tiering",
    "backpressure",
    "batching",
    "retention",
    "replication",
    "migration",
    "observability",
    "concurrency",
]
