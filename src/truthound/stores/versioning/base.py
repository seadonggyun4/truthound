"""Base classes and interfaces for result versioning.

This module defines the abstract base classes and protocols that all versioning
implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Exceptions
# =============================================================================


class VersioningError(Exception):
    """Base exception for versioning-related errors."""

    pass


class VersionConflictError(VersioningError):
    """Raised when there's a version conflict during save."""

    def __init__(
        self,
        item_id: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.item_id = item_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Version conflict for {item_id}: "
            f"expected version {expected_version}, but current version is {actual_version}"
        )


class VersionNotFoundError(VersioningError):
    """Raised when a requested version doesn't exist."""

    def __init__(self, item_id: str, version: int) -> None:
        self.item_id = item_id
        self.version = version
        super().__init__(f"Version {version} not found for {item_id}")


# =============================================================================
# Enums
# =============================================================================


class VersioningMode(Enum):
    """Versioning mode for different use cases."""

    INCREMENTAL = "incremental"  # Simple integer versioning: 1, 2, 3...
    SEMANTIC = "semantic"  # Semantic versioning: 1.0.0, 1.1.0...
    TIMESTAMP = "timestamp"  # ISO timestamp-based: 2025-01-01T12:00:00
    GIT_LIKE = "git_like"  # Short hash-based: abc1234


class DiffType(Enum):
    """Type of difference between versions."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VersionInfo:
    """Information about a specific version.

    Attributes:
        version: Version number or identifier.
        item_id: ID of the versioned item.
        created_at: When this version was created.
        created_by: Who created this version (optional).
        message: Optional commit-like message.
        parent_version: Previous version (for history tracking).
        metadata: Additional version metadata.
        checksum: Content checksum for integrity.
        size_bytes: Size of the version data.
    """

    version: int
    item_id: str
    created_at: datetime
    created_by: str | None = None
    message: str | None = None
    parent_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "item_id": self.item_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "message": self.message,
            "parent_version": self.parent_version,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """Create from dictionary."""
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            version=data["version"],
            item_id=data["item_id"],
            created_at=created_at,
            created_by=data.get("created_by"),
            message=data.get("message"),
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class VersionDiff:
    """Difference between two versions.

    Attributes:
        item_id: ID of the versioned item.
        version_a: First version for comparison.
        version_b: Second version for comparison.
        changes: List of changes between versions.
        summary: Human-readable summary.
    """

    item_id: str
    version_a: int
    version_b: int
    changes: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "changes": self.changes,
            "summary": self.summary,
        }


@dataclass
class VersioningConfig:
    """Configuration for versioning behavior.

    Attributes:
        mode: Versioning mode to use.
        max_versions: Maximum versions to keep per item (0 for unlimited).
        auto_cleanup: Whether to automatically clean up old versions.
        track_changes: Whether to store change details.
        require_message: Whether save requires a commit message.
        enable_branching: Whether to support version branching.
        checksum_algorithm: Algorithm for content checksums.
    """

    mode: VersioningMode = VersioningMode.INCREMENTAL
    max_versions: int = 0  # 0 = unlimited
    auto_cleanup: bool = True
    track_changes: bool = True
    require_message: bool = False
    enable_branching: bool = False
    checksum_algorithm: str = "sha256"


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class Versionable(Protocol):
    """Protocol for objects that can be versioned."""

    @property
    def id(self) -> str:
        """Unique identifier for the item."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Versionable":
        """Deserialize from dictionary."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class VersioningStrategy(ABC):
    """Abstract base class for versioning strategies.

    Different strategies determine how version numbers are generated
    and managed. Subclasses implement the specific logic for each mode.
    """

    @abstractmethod
    def get_next_version(
        self,
        item_id: str,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            item_id: ID of the item being versioned.
            current_version: Current version number, or None if first version.
            metadata: Optional metadata for version calculation.

        Returns:
            The next version number.
        """
        pass

    @abstractmethod
    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: The version number.

        Returns:
            Formatted version string.
        """
        pass

    @abstractmethod
    def parse_version(self, version_str: str) -> int:
        """Parse a formatted version string.

        Args:
            version_str: The formatted version string.

        Returns:
            The version number.
        """
        pass

    def compare_versions(self, version_a: int, version_b: int) -> int:
        """Compare two versions.

        Args:
            version_a: First version.
            version_b: Second version.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b.
        """
        if version_a < version_b:
            return -1
        elif version_a > version_b:
            return 1
        return 0


class VersionStore(ABC):
    """Abstract base class for version metadata storage.

    This handles storing and retrieving version information,
    separate from the actual versioned data.
    """

    @abstractmethod
    def save_version_info(self, info: VersionInfo) -> None:
        """Save version information.

        Args:
            info: Version information to save.
        """
        pass

    @abstractmethod
    def get_version_info(self, item_id: str, version: int) -> VersionInfo:
        """Get version information.

        Args:
            item_id: ID of the item.
            version: Version number.

        Returns:
            Version information.

        Raises:
            VersionNotFoundError: If version doesn't exist.
        """
        pass

    @abstractmethod
    def list_versions(
        self,
        item_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """List all versions for an item.

        Args:
            item_id: ID of the item.
            limit: Maximum versions to return.
            offset: Versions to skip.

        Returns:
            List of version information, newest first.
        """
        pass

    @abstractmethod
    def get_latest_version(self, item_id: str) -> VersionInfo | None:
        """Get the latest version for an item.

        Args:
            item_id: ID of the item.

        Returns:
            Latest version info, or None if no versions exist.
        """
        pass

    @abstractmethod
    def delete_version(self, item_id: str, version: int) -> bool:
        """Delete a specific version.

        Args:
            item_id: ID of the item.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def count_versions(self, item_id: str) -> int:
        """Count versions for an item.

        Args:
            item_id: ID of the item.

        Returns:
            Number of versions.
        """
        pass
