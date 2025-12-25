"""Base types and classes for schema migration.

Provides the foundation for implementing version migrations.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class MigrationDirection(str, Enum):
    """Direction of migration."""

    FORWARD = "forward"   # Upgrade to newer version
    BACKWARD = "backward" # Downgrade to older version


@dataclass
class SchemaVersion:
    """Schema version representation.

    Uses semantic versioning: major.minor.patch
    """

    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = SchemaVersion.parse(other)
        if not isinstance(other, SchemaVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "SchemaVersion") -> bool:
        if isinstance(other, str):
            other = SchemaVersion.parse(other)
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SchemaVersion") -> bool:
        if isinstance(other, str):
            other = SchemaVersion.parse(other)
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SchemaVersion") -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.0", "1.0.0", "v1.0")

        Returns:
            SchemaVersion instance
        """
        version_str = version_str.strip().lstrip("v")
        parts = version_str.split(".")

        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch)


# Current schema version
CURRENT_VERSION = SchemaVersion(1, 1, 0)


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    from_version: SchemaVersion
    to_version: SchemaVersion
    direction: MigrationDirection
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "from_version": str(self.from_version),
            "to_version": str(self.to_version),
            "direction": self.direction.value,
            "changes": self.changes,
            "warnings": self.warnings,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class MigrationError(Exception):
    """Error during migration."""

    def __init__(
        self,
        message: str,
        from_version: Optional[SchemaVersion] = None,
        to_version: Optional[SchemaVersion] = None,
    ):
        super().__init__(message)
        self.from_version = from_version
        self.to_version = to_version


# =============================================================================
# Migration Base Class
# =============================================================================


class Migration(ABC):
    """Abstract base class for schema migrations.

    Implement this to create version-specific migrations.

    Example:
        class V1_0_to_V1_1_Migration(Migration):
            from_version = SchemaVersion(1, 0)
            to_version = SchemaVersion(1, 1)

            def upgrade(self, data):
                # Add new fields
                data["new_field"] = "default_value"
                return data

            def downgrade(self, data):
                # Remove new fields
                del data["new_field"]
                return data
    """

    from_version: SchemaVersion
    to_version: SchemaVersion
    description: str = ""
    reversible: bool = True

    @abstractmethod
    def upgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade data from from_version to to_version.

        Args:
            data: Data in from_version format

        Returns:
            Data in to_version format
        """
        pass

    @abstractmethod
    def downgrade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Downgrade data from to_version to from_version.

        Args:
            data: Data in to_version format

        Returns:
            Data in from_version format
        """
        pass

    def validate_upgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate data after upgrade.

        Args:
            data: Upgraded data

        Returns:
            List of validation errors (empty if valid)
        """
        return []

    def validate_downgrade(self, data: Dict[str, Any]) -> List[str]:
        """Validate data after downgrade.

        Args:
            data: Downgraded data

        Returns:
            List of validation errors (empty if valid)
        """
        return []


# =============================================================================
# Migration Registry
# =============================================================================


class MigrationRegistry:
    """Registry for schema migrations.

    Manages available migrations and determines migration paths.
    """

    _instance: Optional["MigrationRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MigrationRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._migrations: Dict[Tuple[SchemaVersion, SchemaVersion], Migration] = {}
        return cls._instance

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration to register
        """
        key = (migration.from_version, migration.to_version)
        self._migrations[key] = migration
        logger.debug(f"Registered migration: {migration.from_version} -> {migration.to_version}")

    def get(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Optional[Migration]:
        """Get a specific migration.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Migration or None
        """
        key = (from_version, to_version)
        return self._migrations.get(key)

    def get_path(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> List[Migration]:
        """Find migration path between versions.

        Uses BFS to find shortest path.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of migrations to apply in order
        """
        if from_version == to_version:
            return []

        direction = MigrationDirection.FORWARD if to_version > from_version else MigrationDirection.BACKWARD

        # Build graph
        graph: Dict[SchemaVersion, List[Tuple[SchemaVersion, Migration]]] = {}

        for (v1, v2), migration in self._migrations.items():
            if direction == MigrationDirection.FORWARD:
                graph.setdefault(v1, []).append((v2, migration))
            else:
                if migration.reversible:
                    graph.setdefault(v2, []).append((v1, migration))

        # BFS
        from collections import deque

        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current, path = queue.popleft()

            if current == to_version:
                return path

            for next_version, migration in graph.get(current, []):
                if next_version not in visited:
                    visited.add(next_version)
                    queue.append((next_version, path + [migration]))

        return []  # No path found

    def list_versions(self) -> List[SchemaVersion]:
        """List all known versions.

        Returns:
            Sorted list of schema versions
        """
        versions = set()
        for (v1, v2) in self._migrations.keys():
            versions.add(v1)
            versions.add(v2)
        return sorted(versions)

    def list_migrations(self) -> List[Dict[str, Any]]:
        """List all registered migrations.

        Returns:
            List of migration info dictionaries
        """
        return [
            {
                "from_version": str(m.from_version),
                "to_version": str(m.to_version),
                "description": m.description,
                "reversible": m.reversible,
            }
            for m in self._migrations.values()
        ]


# Global registry
migration_registry = MigrationRegistry()
