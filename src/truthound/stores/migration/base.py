"""Base classes and interfaces for schema migration.

This module defines the abstract base classes and data structures that all
schema migration implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


# =============================================================================
# Exceptions
# =============================================================================


class MigrationError(Exception):
    """Base exception for migration-related errors."""

    pass


class IncompatibleVersionError(MigrationError):
    """Raised when data version is incompatible with current version."""

    def __init__(self, data_version: str, current_version: str) -> None:
        self.data_version = data_version
        self.current_version = current_version
        super().__init__(
            f"Data version {data_version} is incompatible with "
            f"current version {current_version}"
        )


class MigrationPathNotFoundError(MigrationError):
    """Raised when no migration path exists between versions."""

    def __init__(self, from_version: str, to_version: str) -> None:
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"No migration path found from version {from_version} "
            f"to version {to_version}"
        )


class MigrationFailedError(MigrationError):
    """Raised when a migration operation fails."""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        message: str,
    ) -> None:
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"Migration from {from_version} to {to_version} failed: {message}"
        )


# =============================================================================
# Enums
# =============================================================================


class MigrationDirection(Enum):
    """Direction of migration."""

    UPGRADE = "upgrade"  # From older to newer version
    DOWNGRADE = "downgrade"  # From newer to older version (if supported)


class MigrationStrategy(Enum):
    """Strategy for handling migrations."""

    EAGER = "eager"  # Migrate all data immediately
    LAZY = "lazy"  # Migrate data on access
    BATCH = "batch"  # Migrate in batches


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """Represents a schema version.

    Supports semantic versioning (major.minor.patch) and comparison.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
    """

    major: int
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        """Format as version string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """Parse a version string.

        Args:
            version_str: Version string (e.g., "1.2.3", "2.0", "3").

        Returns:
            SchemaVersion instance.

        Raises:
            ValueError: If version string is invalid.
        """
        parts = version_str.strip().split(".")

        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return cls(major, minor, patch)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid version string: {version_str}") from e

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is compatible with another.

        Major version must match for compatibility.
        """
        return self.major == other.major


@dataclass
class MigrationInfo:
    """Information about a single migration step.

    Attributes:
        from_version: Source version.
        to_version: Target version.
        description: Human-readable description.
        reversible: Whether this migration can be reversed.
        breaking: Whether this is a breaking change.
        deprecated_fields: Fields deprecated in this migration.
        new_fields: Fields added in this migration.
    """

    from_version: SchemaVersion
    to_version: SchemaVersion
    description: str = ""
    reversible: bool = False
    breaking: bool = False
    deprecated_fields: list[str] = field(default_factory=list)
    new_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_version": str(self.from_version),
            "to_version": str(self.to_version),
            "description": self.description,
            "reversible": self.reversible,
            "breaking": self.breaking,
            "deprecated_fields": self.deprecated_fields,
            "new_fields": self.new_fields,
        }


@dataclass
class MigrationResult:
    """Result of a migration operation.

    Attributes:
        start_time: When migration started.
        end_time: When migration finished.
        from_version: Original version.
        to_version: Target version.
        items_migrated: Number of items migrated.
        items_failed: Number of items that failed.
        items_skipped: Number of items skipped.
        migrations_applied: List of migrations applied.
        errors: List of errors encountered.
        dry_run: Whether this was a dry run.
    """

    start_time: datetime
    end_time: datetime | None = None
    from_version: str = ""
    to_version: str = ""
    items_migrated: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    migrations_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get migration duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success(self) -> bool:
        """Check if migration was successful."""
        return self.items_failed == 0 and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "items_migrated": self.items_migrated,
            "items_failed": self.items_failed,
            "items_skipped": self.items_skipped,
            "migrations_applied": self.migrations_applied,
            "errors": self.errors,
            "dry_run": self.dry_run,
            "success": self.success,
        }


@dataclass
class MigrationConfig:
    """Configuration for schema migration.

    Attributes:
        current_version: The current/target schema version.
        min_supported_version: Minimum version that can be migrated from.
        auto_migrate: Whether to automatically migrate data on read.
        strategy: Migration strategy to use.
        backup_before_migrate: Whether to backup data before migration.
        validate_after_migrate: Whether to validate data after migration.
        batch_size: Items to process per batch.
        max_retries: Maximum retries for failed migrations.
    """

    current_version: str = "1.0.0"
    min_supported_version: str = "1.0.0"
    auto_migrate: bool = True
    strategy: MigrationStrategy = MigrationStrategy.LAZY
    backup_before_migrate: bool = True
    validate_after_migrate: bool = True
    batch_size: int = 100
    max_retries: int = 3


# =============================================================================
# Protocols
# =============================================================================


# Type alias for migration functions
MigrationFunc = Callable[[dict[str, Any]], dict[str, Any]]


@runtime_checkable
class Versioned(Protocol):
    """Protocol for objects that have a schema version."""

    @property
    def schema_version(self) -> str:
        """Get the schema version of this object."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class SchemaMigration(ABC):
    """Abstract base class for schema migrations.

    A schema migration defines how to transform data from one version
    to another.

    Example:
        >>> class MigrateV1ToV2(SchemaMigration):
        ...     @property
        ...     def from_version(self) -> SchemaVersion:
        ...         return SchemaVersion(1, 0, 0)
        ...
        ...     @property
        ...     def to_version(self) -> SchemaVersion:
        ...         return SchemaVersion(2, 0, 0)
        ...
        ...     def migrate(self, data: dict) -> dict:
        ...         # Add new required field with default
        ...         data["new_field"] = data.get("old_field", "default")
        ...         return data
    """

    @property
    @abstractmethod
    def from_version(self) -> SchemaVersion:
        """The source version this migration handles."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> SchemaVersion:
        """The target version this migration produces."""
        pass

    @abstractmethod
    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate data from source to target version.

        Args:
            data: Data in source version format.

        Returns:
            Data in target version format.

        Raises:
            MigrationFailedError: If migration fails.
        """
        pass

    def rollback(self, data: dict[str, Any]) -> dict[str, Any]:
        """Rollback migration (reverse direction).

        Override this method to support downgrade migrations.

        Args:
            data: Data in target version format.

        Returns:
            Data in source version format.

        Raises:
            NotImplementedError: If rollback not supported.
        """
        raise NotImplementedError(
            f"Rollback not supported for migration "
            f"{self.from_version} -> {self.to_version}"
        )

    @property
    def info(self) -> MigrationInfo:
        """Get migration information."""
        return MigrationInfo(
            from_version=self.from_version,
            to_version=self.to_version,
            description=self.description,
            reversible=self.reversible,
        )

    @property
    def description(self) -> str:
        """Human-readable description of this migration."""
        return f"Migrate from {self.from_version} to {self.to_version}"

    @property
    def reversible(self) -> bool:
        """Whether this migration supports rollback."""
        try:
            # Check if rollback is implemented
            self.rollback({})
            return True
        except NotImplementedError:
            return False
        except Exception:
            return True  # Implemented but may have validation

    def validate_input(self, data: dict[str, Any]) -> bool:
        """Validate input data before migration.

        Args:
            data: Data to validate.

        Returns:
            True if data is valid for migration.
        """
        return True

    def validate_output(self, data: dict[str, Any]) -> bool:
        """Validate output data after migration.

        Args:
            data: Migrated data to validate.

        Returns:
            True if migrated data is valid.
        """
        return True


class FunctionalMigration(SchemaMigration):
    """A migration defined by a function.

    This allows defining simple migrations without creating a class.

    Example:
        >>> def upgrade_v1_v2(data: dict) -> dict:
        ...     data["version"] = "2.0.0"
        ...     return data
        >>>
        >>> migration = FunctionalMigration(
        ...     SchemaVersion(1, 0, 0),
        ...     SchemaVersion(2, 0, 0),
        ...     upgrade_v1_v2,
        ...     description="Add version field",
        ... )
    """

    def __init__(
        self,
        from_version: SchemaVersion | str,
        to_version: SchemaVersion | str,
        migrate_func: MigrationFunc,
        rollback_func: MigrationFunc | None = None,
        description: str = "",
    ) -> None:
        """Initialize the migration.

        Args:
            from_version: Source version.
            to_version: Target version.
            migrate_func: Function to perform migration.
            rollback_func: Optional function to reverse migration.
            description: Human-readable description.
        """
        self._from_version = (
            SchemaVersion.parse(from_version)
            if isinstance(from_version, str)
            else from_version
        )
        self._to_version = (
            SchemaVersion.parse(to_version)
            if isinstance(to_version, str)
            else to_version
        )
        self._migrate_func = migrate_func
        self._rollback_func = rollback_func
        self._description = description

    @property
    def from_version(self) -> SchemaVersion:
        """The source version."""
        return self._from_version

    @property
    def to_version(self) -> SchemaVersion:
        """The target version."""
        return self._to_version

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate data using the provided function."""
        return self._migrate_func(data)

    def rollback(self, data: dict[str, Any]) -> dict[str, Any]:
        """Rollback migration using the provided function."""
        if self._rollback_func is None:
            raise NotImplementedError("Rollback function not provided")
        return self._rollback_func(data)

    @property
    def description(self) -> str:
        """Migration description."""
        return self._description or super().description

    @property
    def reversible(self) -> bool:
        """Whether rollback is supported."""
        return self._rollback_func is not None
