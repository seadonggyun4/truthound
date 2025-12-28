"""Schema migration system for validation stores.

This module provides a schema migration layer that handles versioning
of data formats and automatic migration between schema versions.

Example:
    >>> from truthound.stores import get_store
    >>> from truthound.stores.migration import (
    ...     MigratableStore,
    ...     MigrationConfig,
    ...     SchemaMigration,
    ...     MigrationRegistry,
    ... )
    >>>
    >>> # Define migrations
    >>> registry = MigrationRegistry()
    >>>
    >>> @registry.register("1.0.0", "2.0.0")
    ... def migrate_v1_to_v2(data: dict) -> dict:
    ...     # Add new field
    ...     data["new_field"] = data.get("old_field", "")
    ...     return data
    >>>
    >>> @registry.register("2.0.0", "3.0.0")
    ... def migrate_v2_to_v3(data: dict) -> dict:
    ...     # Rename field
    ...     data["renamed"] = data.pop("old_name", None)
    ...     return data
    >>>
    >>> # Create migratable store
    >>> base_store = get_store("filesystem", base_path=".truthound/results")
    >>> store = MigratableStore(
    ...     base_store,
    ...     MigrationConfig(
    ...         current_version="3.0.0",
    ...         registry=registry,
    ...         auto_migrate=True,
    ...     ),
    ... )
    >>>
    >>> # Old data is automatically migrated on read
    >>> result = store.get(run_id)
"""

from truthound.stores.migration.base import (
    MigrationConfig,
    MigrationInfo,
    MigrationResult,
    SchemaMigration,
    SchemaVersion,
    MigrationError,
    IncompatibleVersionError,
    MigrationPathNotFoundError,
)
from truthound.stores.migration.registry import MigrationRegistry
from truthound.stores.migration.store import MigratableStore
from truthound.stores.migration.manager import MigrationManager

__all__ = [
    # Base types
    "MigrationConfig",
    "MigrationInfo",
    "MigrationResult",
    "SchemaMigration",
    "SchemaVersion",
    "MigrationError",
    "IncompatibleVersionError",
    "MigrationPathNotFoundError",
    # Registry
    "MigrationRegistry",
    # Store wrapper
    "MigratableStore",
    # Manager
    "MigrationManager",
]
