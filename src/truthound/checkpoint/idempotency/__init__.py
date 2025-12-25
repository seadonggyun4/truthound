"""Enhanced Idempotency Framework for Truthound.

This module provides a comprehensive idempotency system with support for:
- Multiple storage backends (Memory, File, Redis, SQL)
- Distributed locking for concurrent access
- Request fingerprinting for deduplication
- Middleware integration for actions
- Async support

Architecture:
    The framework uses a layered approach:
    1. Fingerprint Layer - Generates unique identifiers for requests
    2. Storage Layer - Persists idempotency state
    3. Lock Layer - Prevents concurrent execution
    4. Middleware Layer - Integrates with action execution

Example:
    >>> from truthound.checkpoint.idempotency import (
    ...     IdempotencyService,
    ...     RedisIdempotencyStore,
    ...     RequestFingerprint,
    ... )
    >>>
    >>> # Create service with Redis backend
    >>> service = IdempotencyService(
    ...     store=RedisIdempotencyStore(redis_url="redis://localhost:6379"),
    ...     ttl_seconds=3600,
    ... )
    >>>
    >>> # Generate fingerprint
    >>> fingerprint = RequestFingerprint.from_dict({
    ...     "action": "validate_data",
    ...     "dataset": "users",
    ...     "version": "1.0",
    ... })
    >>>
    >>> # Execute with idempotency
    >>> result = service.execute(
    ...     key=fingerprint.key,
    ...     execute_fn=lambda: run_validation(),
    ... )
"""

from truthound.checkpoint.idempotency.core import (
    # Core types
    IdempotencyRecord,
    IdempotencyStatus,
    IdempotencyConfig,
    # Errors
    IdempotencyError,
    IdempotencyConflictError,
    IdempotencyExpiredError,
    IdempotencyHashMismatchError,
)

from truthound.checkpoint.idempotency.fingerprint import (
    RequestFingerprint,
    FingerprintStrategy,
    ContentHashStrategy,
    StructuralHashStrategy,
    CompositeFingerprint,
)

from truthound.checkpoint.idempotency.stores import (
    # Protocol
    IdempotencyStore,
    # Implementations
    InMemoryIdempotencyStore,
    FileIdempotencyStore,
    SQLIdempotencyStore,
)

from truthound.checkpoint.idempotency.locking import (
    # Protocol
    DistributedLock,
    LockAcquisitionError,
    # Implementations
    InMemoryLock,
    FileLock,
)

from truthound.checkpoint.idempotency.service import (
    IdempotencyService,
    IdempotencyMiddleware,
    idempotent,
    idempotent_action,
)

__all__ = [
    # Core types
    "IdempotencyRecord",
    "IdempotencyStatus",
    "IdempotencyConfig",
    # Errors
    "IdempotencyError",
    "IdempotencyConflictError",
    "IdempotencyExpiredError",
    "IdempotencyHashMismatchError",
    # Fingerprint
    "RequestFingerprint",
    "FingerprintStrategy",
    "ContentHashStrategy",
    "StructuralHashStrategy",
    "CompositeFingerprint",
    # Stores
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "FileIdempotencyStore",
    "SQLIdempotencyStore",
    # Locking
    "DistributedLock",
    "LockAcquisitionError",
    "InMemoryLock",
    "FileLock",
    # Service
    "IdempotencyService",
    "IdempotencyMiddleware",
    "idempotent",
    "idempotent_action",
]
