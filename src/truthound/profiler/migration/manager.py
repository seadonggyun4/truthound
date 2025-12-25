"""Migration Manager.

Provides high-level migration management and execution.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from truthound.profiler.migration.base import (
    CURRENT_VERSION,
    Migration,
    MigrationDirection,
    MigrationError,
    MigrationRegistry,
    MigrationResult,
    SchemaVersion,
    migration_registry,
)


logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for migration operations."""

    # Behavior
    auto_migrate: bool = True
    fail_on_error: bool = True
    validate_after: bool = True
    create_backup: bool = True
    backup_dir: Optional[str] = None
    # History
    track_history: bool = True
    history_file: Optional[str] = None
    # Limits
    max_migrations: int = 100
    timeout_seconds: int = 300


class MigrationManager:
    """Manage schema migrations.

    Provides:
    - Automatic version detection
    - Migration path finding
    - Execution with rollback on failure
    - Migration history tracking

    Example:
        manager = MigrationManager()

        # Migrate to latest
        result = manager.migrate(profile_data)

        # Migrate to specific version
        result = manager.migrate(profile_data, target_version="1.1")

        # Check if migration needed
        if manager.needs_migration(profile_data):
            result = manager.migrate(profile_data)
    """

    def __init__(
        self,
        config: MigrationConfig | None = None,
        registry: MigrationRegistry | None = None,
    ):
        """Initialize manager.

        Args:
            config: Migration configuration
            registry: Custom migration registry
        """
        self.config = config or MigrationConfig()
        self._registry = registry or migration_registry
        self._history: List[MigrationResult] = []

    def get_version(self, data: Dict[str, Any]) -> SchemaVersion:
        """Get schema version from data.

        Args:
            data: Profile data

        Returns:
            Schema version
        """
        version_str = data.get("schema_version", data.get("_version", "1.0"))

        if isinstance(version_str, dict):
            version_str = f"{version_str.get('major', 1)}.{version_str.get('minor', 0)}"

        return SchemaVersion.parse(str(version_str))

    def needs_migration(
        self,
        data: Dict[str, Any],
        target_version: Optional[str | SchemaVersion] = None,
    ) -> bool:
        """Check if data needs migration.

        Args:
            data: Profile data
            target_version: Target version (default: current)

        Returns:
            True if migration needed
        """
        current = self.get_version(data)
        target = self._resolve_version(target_version)

        return current != target

    def migrate(
        self,
        data: Dict[str, Any],
        target_version: Optional[str | SchemaVersion] = None,
        in_place: bool = False,
    ) -> Dict[str, Any]:
        """Migrate data to target version.

        Args:
            data: Profile data to migrate
            target_version: Target version (default: current)
            in_place: If True, modify data in place

        Returns:
            Migrated data

        Raises:
            MigrationError: If migration fails
        """
        if not in_place:
            data = copy.deepcopy(data)

        current_version = self.get_version(data)
        target = self._resolve_version(target_version)

        if current_version == target:
            logger.info(f"Data already at version {target}")
            return data

        # Get migration path
        path = self._registry.get_path(current_version, target)

        if not path:
            raise MigrationError(
                f"No migration path from {current_version} to {target}",
                from_version=current_version,
                to_version=target,
            )

        logger.info(f"Migrating from {current_version} to {target} ({len(path)} steps)")

        # Create backup
        backup = copy.deepcopy(data) if self.config.create_backup else None

        # Determine direction
        direction = MigrationDirection.FORWARD if target > current_version else MigrationDirection.BACKWARD

        # Execute migrations
        total_changes: List[str] = []
        total_warnings: List[str] = []
        start_time = time.time()

        try:
            for migration in path:
                result = self._execute_migration(data, migration, direction)

                if not result.success:
                    raise MigrationError(
                        f"Migration {migration.from_version} -> {migration.to_version} failed: "
                        f"{'; '.join(result.errors)}",
                        from_version=migration.from_version,
                        to_version=migration.to_version,
                    )

                # Apply changes
                total_changes.extend(result.changes)
                total_warnings.extend(result.warnings)

            # Update version in data
            data["schema_version"] = str(target)
            data["_migration_timestamp"] = datetime.now().isoformat()

            # Validate if configured
            if self.config.validate_after:
                errors = self._validate_data(data, target)
                if errors:
                    raise MigrationError(
                        f"Validation failed after migration: {'; '.join(errors)}",
                        from_version=current_version,
                        to_version=target,
                    )

            # Record result
            duration_ms = (time.time() - start_time) * 1000
            result = MigrationResult(
                success=True,
                from_version=current_version,
                to_version=target,
                direction=direction,
                changes=total_changes,
                warnings=total_warnings,
                duration_ms=duration_ms,
            )

            self._record_history(result)

            logger.info(f"Migration complete: {len(total_changes)} changes, {duration_ms:.2f}ms")

            return data

        except MigrationError:
            # Rollback if backup available
            if backup and self.config.fail_on_error:
                logger.warning("Migration failed, rolling back")
                data.clear()
                data.update(backup)
            raise

        except Exception as e:
            raise MigrationError(
                f"Unexpected error during migration: {e}",
                from_version=current_version,
                to_version=target,
            ) from e

    def _execute_migration(
        self,
        data: Dict[str, Any],
        migration: Migration,
        direction: MigrationDirection,
    ) -> MigrationResult:
        """Execute a single migration.

        Args:
            data: Data to migrate (modified in place)
            migration: Migration to execute
            direction: Migration direction

        Returns:
            Migration result
        """
        start_time = time.time()
        changes: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        try:
            if direction == MigrationDirection.FORWARD:
                logger.debug(f"Upgrading: {migration.from_version} -> {migration.to_version}")
                result_data = migration.upgrade(data)
                validation_errors = migration.validate_upgrade(result_data)
            else:
                logger.debug(f"Downgrading: {migration.to_version} -> {migration.from_version}")
                result_data = migration.downgrade(data)
                validation_errors = migration.validate_downgrade(result_data)

            if validation_errors:
                errors.extend(validation_errors)

            # Update data in place
            data.clear()
            data.update(result_data)

            duration_ms = (time.time() - start_time) * 1000

            return MigrationResult(
                success=len(errors) == 0,
                from_version=migration.from_version if direction == MigrationDirection.FORWARD else migration.to_version,
                to_version=migration.to_version if direction == MigrationDirection.FORWARD else migration.from_version,
                direction=direction,
                changes=changes,
                warnings=warnings,
                errors=errors,
                duration_ms=duration_ms,
            )

        except Exception as e:
            return MigrationResult(
                success=False,
                from_version=migration.from_version,
                to_version=migration.to_version,
                direction=direction,
                errors=[str(e)],
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _resolve_version(
        self,
        version: Optional[str | SchemaVersion],
    ) -> SchemaVersion:
        """Resolve version specification.

        Args:
            version: Version string, SchemaVersion, or None (= current)

        Returns:
            SchemaVersion
        """
        if version is None:
            return CURRENT_VERSION
        if isinstance(version, SchemaVersion):
            return version
        return SchemaVersion.parse(version)

    def _validate_data(
        self,
        data: Dict[str, Any],
        version: SchemaVersion,
    ) -> List[str]:
        """Validate data structure for version.

        Args:
            data: Data to validate
            version: Expected version

        Returns:
            List of validation errors
        """
        errors = []

        # Basic structure validation
        required_fields = ["schema_version"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Version-specific validation would go here
        # This could be extended based on schema definitions

        return errors

    def _record_history(self, result: MigrationResult) -> None:
        """Record migration in history.

        Args:
            result: Migration result to record
        """
        self._history.append(result)

        if self.config.track_history and self.config.history_file:
            try:
                path = Path(self.config.history_file)
                history_data = []

                if path.exists():
                    with open(path) as f:
                        history_data = json.load(f)

                history_data.append(result.to_dict())

                with open(path, "w") as f:
                    json.dump(history_data, f, indent=2)

            except Exception as e:
                logger.warning(f"Failed to save migration history: {e}")

    def get_history(self) -> List[MigrationResult]:
        """Get migration history.

        Returns:
            List of migration results
        """
        return self._history.copy()

    def list_available_migrations(self) -> List[Dict[str, Any]]:
        """List all available migrations.

        Returns:
            List of migration info
        """
        return self._registry.list_migrations()

    def get_migration_path(
        self,
        from_version: str | SchemaVersion,
        to_version: Optional[str | SchemaVersion] = None,
    ) -> List[str]:
        """Get the migration path between versions.

        Args:
            from_version: Source version
            to_version: Target version (default: current)

        Returns:
            List of migration descriptions
        """
        from_v = self._resolve_version(from_version) if isinstance(from_version, str) else from_version
        to_v = self._resolve_version(to_version)

        path = self._registry.get_path(from_v, to_v)

        return [
            f"{m.from_version} -> {m.to_version}: {m.description or 'No description'}"
            for m in path
        ]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_schema_version(data: Dict[str, Any]) -> str:
    """Get schema version from profile data.

    Args:
        data: Profile data

    Returns:
        Version string
    """
    manager = MigrationManager()
    return str(manager.get_version(data))


def migrate_profile(
    data: Dict[str, Any],
    target_version: Optional[str] = None,
    in_place: bool = False,
) -> Dict[str, Any]:
    """Migrate profile data to target version.

    Args:
        data: Profile data
        target_version: Target version (default: current)
        in_place: Modify in place

    Returns:
        Migrated data
    """
    manager = MigrationManager()
    return manager.migrate(data, target_version, in_place)
