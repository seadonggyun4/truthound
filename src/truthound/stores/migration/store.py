"""Migratable store wrapper implementation.

This module provides the main MigratableStore class that wraps any
BaseStore with schema migration capabilities.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Generic, TypeVar

from truthound.stores.base import (
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    ValidationStore,
)
from truthound.stores.migration.base import (
    IncompatibleVersionError,
    MigrationConfig,
    MigrationFailedError,
    MigrationResult,
    MigrationStrategy,
    SchemaVersion,
)
from truthound.stores.migration.registry import MigrationRegistry, get_default_registry
from truthound.stores.results import ValidationResult

ConfigT = TypeVar("ConfigT", bound=StoreConfig)

logger = logging.getLogger(__name__)


# Key used to store schema version in data
SCHEMA_VERSION_KEY = "_schema_version"


class MigratableStore(ValidationStore[ConfigT], Generic[ConfigT]):
    """A store wrapper that adds schema migration to any base store.

    This class wraps an existing store and automatically handles
    schema versioning and migration of data.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.migration import (
        ...     MigratableStore,
        ...     MigrationConfig,
        ...     MigrationRegistry,
        ... )
        >>>
        >>> # Create registry with migrations
        >>> registry = MigrationRegistry()
        >>>
        >>> @registry.register("1.0.0", "2.0.0")
        ... def add_metadata(data: dict) -> dict:
        ...     data["metadata"] = data.get("metadata", {})
        ...     return data
        >>>
        >>> # Create migratable store
        >>> base = get_store("filesystem", base_path=".truthound/results")
        >>> config = MigrationConfig(
        ...     current_version="2.0.0",
        ...     auto_migrate=True,
        ... )
        >>> store = MigratableStore(base, config, registry)
        >>>
        >>> # Old data is automatically migrated
        >>> result = store.get(run_id)  # Migrated if needed
    """

    def __init__(
        self,
        base_store: ValidationStore[Any],
        config: MigrationConfig | None = None,
        registry: MigrationRegistry | None = None,
    ) -> None:
        """Initialize the migratable store.

        Args:
            base_store: The underlying store to wrap.
            config: Migration configuration.
            registry: Migration registry (uses default if None).
        """
        self._io_guard = None
        self._base_store = base_store
        self._config = config or MigrationConfig()
        self._registry = registry or get_default_registry()
        self._current_version = SchemaVersion.parse(self._config.current_version)
        self._min_version = SchemaVersion.parse(self._config.min_supported_version)
        self._initialized = False
        self._migration_stats = {
            "items_migrated": 0,
            "migrations_applied": 0,
            "errors": 0,
        }

    def set_io_guard(self, guard) -> None:
        """Set an optional trusted caller authorization check before native effects."""
        self._io_guard = guard

    def _guard(self) -> None:
        if self._io_guard is not None:
            self._io_guard()

    def _backup(self, item: ValidationResult) -> None:
        if self._config.backup_before_migrate:
            data = copy.deepcopy(item.to_dict())
            data["run_id"] = f"_backup_{item.id}"
            data["metadata"] = {**data.get("metadata", {}), "_schema_migration_backup_of": item.id}
            self._guard()
            if self._base_store.exists(data["run_id"]):
                existing = self._base_store.get(data["run_id"])
                if existing.metadata.get("_schema_migration_backup_of") != item.id:
                    raise ValueError("Migration backup identity conflict")
            else:
                self._base_store.save(ValidationResult.from_dict(data))

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return StoreConfig()  # type: ignore

    def _do_initialize(self) -> None:
        """Initialize the store."""
        self._guard()
        self._base_store.initialize()

    def _get_data_version(self, data: dict[str, Any]) -> SchemaVersion:
        """Extract schema version from data.

        Args:
            data: The data dictionary.

        Returns:
            SchemaVersion (defaults to min version if not present).
        """
        version_str = data.get("metadata", {}).get(
            SCHEMA_VERSION_KEY,
            data.get("meta", {}).get(SCHEMA_VERSION_KEY, data.get(SCHEMA_VERSION_KEY, str(self._min_version))),
        )
        return SchemaVersion.parse(version_str)

    def _set_data_version(self, data: dict[str, Any], version: SchemaVersion) -> None:
        """Set schema version in data.

        Args:
            data: The data dictionary.
            version: Version to set.
        """
        data[SCHEMA_VERSION_KEY] = str(version)
        data["metadata"] = {**data.get("metadata", {}), SCHEMA_VERSION_KEY: str(version)}

    def _needs_migration(self, data: dict[str, Any]) -> bool:
        """Check if data needs migration.

        Args:
            data: The data dictionary.

        Returns:
            True if migration is needed.
        """
        data_version = self._get_data_version(data)
        return data_version != self._current_version

    def _migrate_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate data to current version.

        Args:
            data: Data to migrate.

        Returns:
            Migrated data.

        Raises:
            IncompatibleVersionError: If data version is too old.
            MigrationFailedError: If migration fails.
        """
        data_version = self._get_data_version(data)

        # Check if version is supported
        if data_version < self._min_version:
            raise IncompatibleVersionError(
                str(data_version), str(self._current_version)
            )

        if data_version == self._current_version:
            return data

        # Find migration path
        try:
            path = self._registry.find_path(
                str(data_version), str(self._current_version)
            )
        except Exception as e:
            raise MigrationFailedError(
                str(data_version), str(self._current_version), str(e)
            ) from e

        # Apply migrations
        migrated_data = copy.deepcopy(data)

        for migration in path:
            try:
                self._guard()
                # Validate input
                if self._config.validate_after_migrate and not migration.validate_input(migrated_data):
                    raise ValueError("Migration input validation failed")

                # Apply migration
                migrated_data = migration.migrate(migrated_data)
                self._migration_stats["migrations_applied"] += 1

                # Validate output
                if self._config.validate_after_migrate and not migration.validate_output(migrated_data):
                    raise ValueError("Migration output validation failed")

                logger.debug(
                    f"Applied migration: {migration.from_version} -> "
                    f"{migration.to_version}"
                )

            except Exception as e:
                self._migration_stats["errors"] += 1
                raise MigrationFailedError(
                    str(migration.from_version),
                    str(migration.to_version),
                    str(e),
                ) from e

        # Set new version
        self._set_data_version(migrated_data, self._current_version)
        self._migration_stats["items_migrated"] += 1

        return migrated_data

    # -------------------------------------------------------------------------
    # CRUD Operations with Migration
    # -------------------------------------------------------------------------

    def save(self, item: ValidationResult) -> str:
        """Save a validation result with current schema version.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.
        """
        self.initialize()

        # Add current version to data
        self._guard()
        data = copy.deepcopy(item.to_dict())
        self._set_data_version(data, self._current_version)

        # Create new result with versioned data
        # Store the version in meta for persistence
        result = ValidationResult.from_dict(data)
        return self._base_store.save(result)

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result, migrating if necessary.

        Args:
            item_id: The run ID of the result.

        Returns:
            The validation result (migrated to current version).

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            IncompatibleVersionError: If data version is too old.
            MigrationFailedError: If migration fails.
        """
        self.initialize()

        self._guard()
        result = self._base_store.get(item_id)
        data = result.to_dict()
        # Required originals must remain readable without upgrading the backup.
        if result.metadata.get("_schema_migration_backup_of"):
            return result

        # Check for version in meta
        if "meta" in data and SCHEMA_VERSION_KEY in data["meta"]:
            data[SCHEMA_VERSION_KEY] = data["meta"][SCHEMA_VERSION_KEY]

        # Check if migration is needed
        if self._needs_migration(data):
            if not self._config.auto_migrate:
                data_version = self._get_data_version(data)
                raise IncompatibleVersionError(
                    str(data_version), str(self._current_version)
                )

            # Preserve the required original before executing user migrations.
            if self._config.strategy == MigrationStrategy.LAZY:
                self._backup(result)
            # Migrate data
            migrated_data = self._migrate_data(data)

            # Save migrated data if using lazy strategy
            if self._config.strategy == MigrationStrategy.LAZY:
                migrated_result = ValidationResult.from_dict(migrated_data)
                self._guard()
                self._base_store.save(migrated_result)

            return ValidationResult.from_dict(migrated_data)

        return result

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists."""
        self.initialize()
        return self._base_store.exists(item_id)

    def delete(self, item_id: str) -> bool:
        """Delete a validation result."""
        self.initialize()
        return self._base_store.delete(item_id)

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query."""
        self.initialize()
        return self._base_store.list_ids(query)

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results, migrating each if needed."""
        self.initialize()

        ids = self.list_ids(query)
        results = []

        for item_id in ids:
            # A partial result is not a successful complete query.
            results.append(self.get(item_id))

        return results

    # -------------------------------------------------------------------------
    # Migration Operations
    # -------------------------------------------------------------------------

    def migrate_all(
        self,
        dry_run: bool = False,
        batch_size: int | None = None,
    ) -> MigrationResult:
        """Migrate all items to current version.

        Args:
            dry_run: If True, only report what would be migrated.
            batch_size: Items to process per batch.

        Returns:
            Result of the migration.
        """
        self.initialize()

        start_time = datetime.now()
        batch_size = self._config.batch_size if batch_size is None else batch_size
        if batch_size < 1 or self._config.max_retries < 1:
            raise ValueError("batch_size and max_retries must be positive")

        result = MigrationResult(
            start_time=start_time,
            from_version=str(self._min_version),
            to_version=str(self._current_version),
            dry_run=dry_run,
        )

        # Get all item IDs
        all_ids = self._base_store.list_ids()

        # Process in batches
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]

            for item_id in batch_ids:
                self._guard()
                try:
                    # Get item
                    item = self._base_store.get(item_id)
                    data = item.to_dict()
                    if data.get("metadata", {}).get("_schema_migration_backup_of"):
                        result.items_skipped += 1
                        continue

                    # Check for version in meta
                    if "meta" in data and SCHEMA_VERSION_KEY in data["meta"]:
                        data[SCHEMA_VERSION_KEY] = data["meta"][SCHEMA_VERSION_KEY]

                    if not self._needs_migration(data):
                        result.items_skipped += 1
                        continue

                    data_version = self._get_data_version(data)

                    if dry_run:
                        if data_version < self._min_version:
                            raise IncompatibleVersionError(str(data_version), str(self._current_version))
                        self._registry.find_path(str(data_version), str(self._current_version))
                        result.items_migrated += 1
                        result.migrations_applied.append(
                            f"{item_id}: {data_version} -> {self._current_version}"
                        )
                    else:
                        self._backup(item)

                        # Migrate
                        migrated_data = self._migrate_data(data)
                        migrated_result = ValidationResult.from_dict(migrated_data)

                        # Save
                        retries = 0
                        while retries < self._config.max_retries:
                            try:
                                self._guard()
                                self._base_store.save(migrated_result)
                                result.items_migrated += 1
                                result.migrations_applied.append(
                                    f"{item_id}: {data_version} -> "
                                    f"{self._current_version}"
                                )
                                break
                            except Exception:
                                retries += 1
                                if retries >= self._config.max_retries:
                                    raise

                except Exception as e:
                    result.items_failed += 1
                    result.errors.append(f"{item_id}: {str(e)}")
                    logger.error("Native store migration failed (%s)", type(e).__name__)

        result.end_time = datetime.now()
        return result

    def get_migration_status(self) -> dict[str, Any]:
        """Get migration status for the store.

        Returns:
            Dictionary with migration status.
        """
        self.initialize()

        # Sample items to check versions
        all_ids = self._base_store.list_ids()
        sample_size = min(100, len(all_ids))
        sample_ids = all_ids[:sample_size]

        version_counts: dict[str, int] = {}
        needs_migration = 0
        failed_items = 0

        for item_id in sample_ids:
            self._guard()
            try:
                item = self._base_store.get(item_id)
                data = item.to_dict()
                if data.get("metadata", {}).get("_schema_migration_backup_of"):
                    continue

                if "meta" in data and SCHEMA_VERSION_KEY in data["meta"]:
                    data[SCHEMA_VERSION_KEY] = data["meta"][SCHEMA_VERSION_KEY]

                version = str(self._get_data_version(data))
                version_counts[version] = version_counts.get(version, 0) + 1

                if self._needs_migration(data):
                    needs_migration += 1

            except Exception:
                failed_items += 1

        return {
            "current_version": str(self._current_version),
            "min_supported_version": str(self._min_version),
            "total_items": len(all_ids),
            "sampled_items": sample_size,
            "failed_items": failed_items,
            "needs_migration_estimated": (
                int(needs_migration / sample_size * len(all_ids))
                if sample_size > 0
                else 0
            ),
            "version_distribution": version_counts,
            "migration_stats": self._migration_stats.copy(),
        }

    def validate_item(self, item_id: str) -> dict[str, Any]:
        """Validate migration for a specific item.

        Args:
            item_id: The item to validate.

        Returns:
            Dictionary with validation result.
        """
        self.initialize()

        self._guard()
        try:
            item = self._base_store.get(item_id)
            data = item.to_dict()

            if "meta" in data and SCHEMA_VERSION_KEY in data["meta"]:
                data[SCHEMA_VERSION_KEY] = data["meta"][SCHEMA_VERSION_KEY]

            data_version = self._get_data_version(data)
            needs_migration = self._needs_migration(data)

            # Try migration if needed
            if needs_migration:
                try:
                    if data_version < self._min_version:
                        raise IncompatibleVersionError(str(data_version), str(self._current_version))
                    path = self._registry.find_path(
                        str(data_version), str(self._current_version)
                    )
                    migration_path = [
                        f"{m.from_version} -> {m.to_version}" for m in path
                    ]
                except Exception as e:
                    return {
                        "item_id": item_id,
                        "current_version": str(data_version),
                        "target_version": str(self._current_version),
                        "needs_migration": True,
                        "can_migrate": False,
                        "error": str(e),
                    }

                return {
                    "item_id": item_id,
                    "current_version": str(data_version),
                    "target_version": str(self._current_version),
                    "needs_migration": True,
                    "can_migrate": True,
                    "migration_path": migration_path,
                }

            return {
                "item_id": item_id,
                "current_version": str(data_version),
                "target_version": str(self._current_version),
                "needs_migration": False,
                "can_migrate": True,
            }

        except StoreNotFoundError:
            return {
                "item_id": item_id,
                "error": "Item not found",
            }
        except Exception as e:
            return {
                "item_id": item_id,
                "error": str(e),
            }

    def close(self) -> None:
        """Close the store."""
        self._base_store.close()

    @property
    def current_version(self) -> SchemaVersion:
        """Get the current schema version."""
        return self._current_version

    @property
    def registry(self) -> MigrationRegistry:
        """Get the migration registry."""
        return self._registry

    @property
    def config(self) -> MigrationConfig:
        """Get the migration configuration."""
        return self._config
