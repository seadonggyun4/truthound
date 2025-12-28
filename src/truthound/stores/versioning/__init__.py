"""Result versioning system for validation stores.

This module provides a versioning layer that can be applied to any store backend,
enabling version tracking, history, rollback, and diff capabilities.

Example:
    >>> from truthound.stores import get_store
    >>> from truthound.stores.versioning import VersionedStore, VersioningConfig
    >>>
    >>> # Wrap any store with versioning
    >>> base_store = get_store("filesystem", base_path=".truthound/results")
    >>> versioned_store = VersionedStore(base_store, VersioningConfig(max_versions=10))
    >>>
    >>> # Save with automatic versioning
    >>> version_id = versioned_store.save(result)
    >>>
    >>> # Get version history
    >>> history = versioned_store.get_version_history(result.run_id)
    >>>
    >>> # Rollback to previous version
    >>> versioned_store.rollback(result.run_id, version=2)
    >>>
    >>> # Compare versions
    >>> diff = versioned_store.diff(result.run_id, version_a=1, version_b=3)
"""

from truthound.stores.versioning.base import (
    VersionInfo,
    VersioningConfig,
    VersioningStrategy,
    VersionDiff,
    VersionConflictError,
)
from truthound.stores.versioning.store import VersionedStore
from truthound.stores.versioning.strategies import (
    IncrementalStrategy,
    SemanticStrategy,
    TimestampStrategy,
    GitLikeStrategy,
)

__all__ = [
    # Core types
    "VersionInfo",
    "VersioningConfig",
    "VersioningStrategy",
    "VersionDiff",
    "VersionConflictError",
    # Main store
    "VersionedStore",
    # Strategies
    "IncrementalStrategy",
    "SemanticStrategy",
    "TimestampStrategy",
    "GitLikeStrategy",
]
