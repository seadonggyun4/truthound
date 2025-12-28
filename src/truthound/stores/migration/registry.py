"""Migration registry for managing schema migrations.

This module provides a registry for registering and discovering
schema migrations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from truthound.stores.migration.base import (
    FunctionalMigration,
    MigrationFunc,
    MigrationInfo,
    MigrationPathNotFoundError,
    SchemaMigration,
    SchemaVersion,
)

logger = logging.getLogger(__name__)


class MigrationRegistry:
    """Registry for schema migrations.

    The registry maintains a collection of migrations and provides
    methods to find migration paths between versions.

    Example:
        >>> registry = MigrationRegistry()
        >>>
        >>> # Register with decorator
        >>> @registry.register("1.0.0", "2.0.0")
        ... def migrate_v1_v2(data: dict) -> dict:
        ...     data["new_field"] = "default"
        ...     return data
        >>>
        >>> # Register migration class
        >>> registry.add(MyMigration())
        >>>
        >>> # Find migration path
        >>> path = registry.find_path("1.0.0", "3.0.0")
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._migrations: dict[tuple[str, str], SchemaMigration] = {}
        self._graph: dict[str, list[str]] = {}

    def add(self, migration: SchemaMigration) -> None:
        """Add a migration to the registry.

        Args:
            migration: The migration to add.
        """
        from_key = str(migration.from_version)
        to_key = str(migration.to_version)

        self._migrations[(from_key, to_key)] = migration

        # Update graph for path finding
        if from_key not in self._graph:
            self._graph[from_key] = []
        self._graph[from_key].append(to_key)

        logger.debug(f"Registered migration: {from_key} -> {to_key}")

    def register(
        self,
        from_version: str,
        to_version: str,
        description: str = "",
    ) -> Callable[[MigrationFunc], MigrationFunc]:
        """Decorator to register a migration function.

        Args:
            from_version: Source version string.
            to_version: Target version string.
            description: Human-readable description.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register("1.0.0", "2.0.0")
            ... def upgrade(data: dict) -> dict:
            ...     return data
        """

        def decorator(func: MigrationFunc) -> MigrationFunc:
            migration = FunctionalMigration(
                from_version=from_version,
                to_version=to_version,
                migrate_func=func,
                description=description or func.__doc__ or "",
            )
            self.add(migration)
            return func

        return decorator

    def register_bidirectional(
        self,
        from_version: str,
        to_version: str,
        description: str = "",
    ) -> Callable[
        [tuple[MigrationFunc, MigrationFunc]], tuple[MigrationFunc, MigrationFunc]
    ]:
        """Decorator to register both upgrade and downgrade migrations.

        Args:
            from_version: Source version string.
            to_version: Target version string.
            description: Human-readable description.

        Returns:
            Decorator function.

        Example:
            >>> @registry.register_bidirectional("1.0.0", "2.0.0")
            ... def migrations():
            ...     def upgrade(data):
            ...         data["new"] = data.pop("old", None)
            ...         return data
            ...     def downgrade(data):
            ...         data["old"] = data.pop("new", None)
            ...         return data
            ...     return upgrade, downgrade
        """

        def decorator(
            func: Callable[[], tuple[MigrationFunc, MigrationFunc]],
        ) -> tuple[MigrationFunc, MigrationFunc]:
            upgrade, downgrade = func()

            # Forward migration
            self.add(
                FunctionalMigration(
                    from_version=from_version,
                    to_version=to_version,
                    migrate_func=upgrade,
                    rollback_func=downgrade,
                    description=description,
                )
            )

            # Reverse migration
            self.add(
                FunctionalMigration(
                    from_version=to_version,
                    to_version=from_version,
                    migrate_func=downgrade,
                    rollback_func=upgrade,
                    description=f"Rollback: {description}",
                )
            )

            return upgrade, downgrade

        return decorator

    def get(
        self,
        from_version: str | SchemaVersion,
        to_version: str | SchemaVersion,
    ) -> SchemaMigration | None:
        """Get a specific migration.

        Args:
            from_version: Source version.
            to_version: Target version.

        Returns:
            The migration, or None if not found.
        """
        from_key = str(from_version)
        to_key = str(to_version)
        return self._migrations.get((from_key, to_key))

    def find_path(
        self,
        from_version: str | SchemaVersion,
        to_version: str | SchemaVersion,
    ) -> list[SchemaMigration]:
        """Find a migration path between versions.

        Uses BFS to find the shortest path.

        Args:
            from_version: Starting version.
            to_version: Target version.

        Returns:
            List of migrations to apply in order.

        Raises:
            MigrationPathNotFoundError: If no path exists.
        """
        from_key = str(from_version)
        to_key = str(to_version)

        if from_key == to_key:
            return []

        # BFS to find shortest path
        visited = {from_key}
        queue: list[tuple[str, list[str]]] = [(from_key, [])]

        while queue:
            current, path = queue.pop(0)

            if current not in self._graph:
                continue

            for next_version in self._graph[current]:
                if next_version in visited:
                    continue

                new_path = path + [(current, next_version)]

                if next_version == to_key:
                    # Found path - convert to migrations
                    return [
                        self._migrations[(f, t)]
                        for f, t in new_path
                    ]

                visited.add(next_version)
                queue.append((next_version, new_path))

        raise MigrationPathNotFoundError(from_key, to_key)

    def list_versions(self) -> list[str]:
        """List all known versions.

        Returns:
            List of version strings.
        """
        versions = set()
        for from_v, to_v in self._migrations.keys():
            versions.add(from_v)
            versions.add(to_v)
        return sorted(versions, key=lambda v: SchemaVersion.parse(v))

    def list_migrations(self) -> list[MigrationInfo]:
        """List all registered migrations.

        Returns:
            List of migration info objects.
        """
        return [m.info for m in self._migrations.values()]

    def get_latest_version(self) -> str:
        """Get the latest/highest version.

        Returns:
            Version string.
        """
        versions = self.list_versions()
        return versions[-1] if versions else "1.0.0"

    def get_migrations_from(self, version: str) -> list[SchemaMigration]:
        """Get all migrations available from a version.

        Args:
            version: Starting version.

        Returns:
            List of migrations.
        """
        return [
            m for (from_v, _), m in self._migrations.items() if from_v == version
        ]

    def get_migrations_to(self, version: str) -> list[SchemaMigration]:
        """Get all migrations available to a version.

        Args:
            version: Target version.

        Returns:
            List of migrations.
        """
        return [
            m for (_, to_v), m in self._migrations.items() if to_v == version
        ]

    def validate_path(
        self,
        from_version: str,
        to_version: str,
        sample_data: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate a migration path with sample data.

        Args:
            from_version: Starting version.
            to_version: Target version.
            sample_data: Sample data to test.

        Returns:
            Tuple of (success, list of error messages).
        """
        errors: list[str] = []

        try:
            path = self.find_path(from_version, to_version)
        except MigrationPathNotFoundError as e:
            return False, [str(e)]

        data = sample_data.copy()

        for migration in path:
            try:
                # Validate input
                if not migration.validate_input(data):
                    errors.append(
                        f"Input validation failed for {migration.from_version} "
                        f"-> {migration.to_version}"
                    )
                    continue

                # Apply migration
                data = migration.migrate(data)

                # Validate output
                if not migration.validate_output(data):
                    errors.append(
                        f"Output validation failed for {migration.from_version} "
                        f"-> {migration.to_version}"
                    )

            except Exception as e:
                errors.append(
                    f"Migration {migration.from_version} -> {migration.to_version} "
                    f"failed: {str(e)}"
                )

        return len(errors) == 0, errors

    def clear(self) -> None:
        """Clear all registered migrations."""
        self._migrations.clear()
        self._graph.clear()

    def __len__(self) -> int:
        """Get number of registered migrations."""
        return len(self._migrations)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if a migration exists."""
        return key in self._migrations


# Global default registry
_default_registry = MigrationRegistry()


def get_default_registry() -> MigrationRegistry:
    """Get the default global migration registry."""
    return _default_registry


def register(
    from_version: str,
    to_version: str,
    description: str = "",
) -> Callable[[MigrationFunc], MigrationFunc]:
    """Register a migration in the default registry.

    This is a convenience function for simple use cases.

    Example:
        >>> from truthound.stores.migration import register
        >>>
        >>> @register("1.0.0", "2.0.0")
        ... def upgrade(data: dict) -> dict:
        ...     return data
    """
    return _default_registry.register(from_version, to_version, description)
