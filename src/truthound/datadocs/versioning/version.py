"""Version information and strategies for report versioning.

This module defines the core data structures and strategies
for managing report versions.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class VersionInfo:
    """Information about a specific report version.

    Attributes:
        version: Version number or identifier.
        report_id: ID of the report being versioned.
        created_at: When this version was created.
        created_by: Who created this version.
        message: Optional version message (like commit message).
        parent_version: Previous version number.
        checksum: Content checksum for integrity.
        size_bytes: Size of the report content.
        metadata: Additional version metadata.
    """
    version: int
    report_id: str
    created_at: datetime
    created_by: str | None = None
    message: str | None = None
    parent_version: int | None = None
    checksum: str | None = None
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "version": self.version,
            "report_id": self.report_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "message": self.message,
            "parent_version": self.parent_version,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionInfo":
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            VersionInfo instance.
        """
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            version=data["version"],
            report_id=data["report_id"],
            created_at=created_at,
            created_by=data.get("created_by"),
            message=data.get("message"),
            parent_version=data.get("parent_version"),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReportVersion:
    """A complete versioned report.

    Attributes:
        info: Version information.
        content: Report content (bytes or string).
        format: Output format (html, pdf, etc.).
    """
    info: VersionInfo
    content: bytes | str
    format: str = "html"

    @property
    def version(self) -> int:
        """Get the version number."""
        return self.info.version

    @property
    def report_id(self) -> str:
        """Get the report ID."""
        return self.info.report_id

    @property
    def checksum(self) -> str:
        """Get or compute the content checksum."""
        if self.info.checksum:
            return self.info.checksum

        content_bytes = (
            self.content
            if isinstance(self.content, bytes)
            else self.content.encode("utf-8")
        )
        return hashlib.sha256(content_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        content = self.content
        if isinstance(content, bytes):
            # Store as base64 for binary content
            import base64
            content = base64.b64encode(content).decode("ascii")

        return {
            "info": self.info.to_dict(),
            "content": content,
            "format": self.format,
            "is_binary": isinstance(self.content, bytes),
        }


class VersioningStrategy(ABC):
    """Abstract base class for versioning strategies.

    Different strategies determine how version numbers are generated.
    """

    @abstractmethod
    def next_version(
        self,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            current_version: Current version, or None for first.
            metadata: Optional metadata for version calculation.

        Returns:
            Next version number.
        """
        pass

    @abstractmethod
    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: Version number.

        Returns:
            Formatted version string.
        """
        pass

    @abstractmethod
    def parse_version(self, version_str: str) -> int:
        """Parse a version string to number.

        Args:
            version_str: Version string.

        Returns:
            Version number.
        """
        pass


class IncrementalStrategy(VersioningStrategy):
    """Simple incremental versioning: 1, 2, 3, ...

    The most straightforward versioning strategy.
    """

    def __init__(self, start: int = 1) -> None:
        """Initialize with starting version.

        Args:
            start: First version number.
        """
        self._start = start

    def next_version(
        self,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if current_version is None:
            return self._start
        return current_version + 1

    def format_version(self, version: int) -> str:
        return str(version)

    def parse_version(self, version_str: str) -> int:
        return int(version_str)


class SemanticStrategy(VersioningStrategy):
    """Semantic versioning: major.minor.patch.

    Version changes based on change type:
    - major: Breaking changes
    - minor: New features
    - patch: Bug fixes
    """

    def __init__(self, initial: str = "1.0.0") -> None:
        """Initialize with initial version.

        Args:
            initial: Initial version string.
        """
        self._initial = initial

    def next_version(
        self,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate next version.

        Uses metadata to determine version bump type:
        - {"bump": "major"} -> increment major
        - {"bump": "minor"} -> increment minor
        - {"bump": "patch"} or default -> increment patch
        """
        if current_version is None:
            return self._version_to_int(self._initial)

        metadata = metadata or {}
        bump = metadata.get("bump", "patch")

        major, minor, patch = self._int_to_parts(current_version)

        if bump == "major":
            return self._parts_to_int(major + 1, 0, 0)
        elif bump == "minor":
            return self._parts_to_int(major, minor + 1, 0)
        else:
            return self._parts_to_int(major, minor, patch + 1)

    def format_version(self, version: int) -> str:
        major, minor, patch = self._int_to_parts(version)
        return f"{major}.{minor}.{patch}"

    def parse_version(self, version_str: str) -> int:
        return self._version_to_int(version_str)

    def _version_to_int(self, version_str: str) -> int:
        """Convert version string to integer.

        Encodes version as: major * 1000000 + minor * 1000 + patch
        """
        parts = version_str.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return self._parts_to_int(major, minor, patch)

    def _int_to_parts(self, version: int) -> tuple[int, int, int]:
        """Convert integer to version parts."""
        major = version // 1000000
        minor = (version % 1000000) // 1000
        patch = version % 1000
        return major, minor, patch

    def _parts_to_int(self, major: int, minor: int, patch: int) -> int:
        """Convert version parts to integer."""
        return major * 1000000 + minor * 1000 + patch


class TimestampStrategy(VersioningStrategy):
    """Timestamp-based versioning.

    Uses Unix timestamp as version number, formatted as ISO date.
    """

    def next_version(
        self,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate version from current timestamp."""
        return int(datetime.now().timestamp())

    def format_version(self, version: int) -> str:
        """Format as ISO datetime."""
        dt = datetime.fromtimestamp(version)
        return dt.isoformat()

    def parse_version(self, version_str: str) -> int:
        """Parse ISO datetime to timestamp."""
        dt = datetime.fromisoformat(version_str)
        return int(dt.timestamp())


class GitLikeStrategy(VersioningStrategy):
    """Git-like versioning with short hashes.

    Uses content hash as version identifier, stored as integer
    representation of the first 8 hex characters.
    """

    def __init__(self) -> None:
        """Initialize the strategy."""
        self._counter = 0  # Fallback counter

    def next_version(
        self,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate version from content hash.

        If metadata contains 'content', uses hash of content.
        Otherwise, uses timestamp-based version.
        """
        metadata = metadata or {}

        if "content" in metadata:
            content = metadata["content"]
            if isinstance(content, str):
                content = content.encode("utf-8")
            hash_hex = hashlib.sha256(content).hexdigest()[:8]
            return int(hash_hex, 16)
        else:
            # Fallback to timestamp
            return int(datetime.now().timestamp() * 1000)

    def format_version(self, version: int) -> str:
        """Format as short hash."""
        return f"{version:08x}"

    def parse_version(self, version_str: str) -> int:
        """Parse hex string to integer."""
        return int(version_str, 16)
