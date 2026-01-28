"""Schema history storage and management.

This module provides persistent storage for schema versions with
diff computation, rollback support, and versioning strategies.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from truthound.profiler.evolution.changes import (
    ChangeType,
    ChangeSeverity,
    CompatibilityLevel,
    SchemaChange,
    SchemaChangeSummary,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class SchemaHistoryStorage(Protocol):
    """Protocol for schema history storage backends."""

    @abstractmethod
    def save_version(
        self,
        schema: dict[str, Any],
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SchemaVersion":
        """Save a new schema version.

        Args:
            schema: Schema to save as dict mapping column names to types.
            version: Optional version string. Auto-generated if not provided.
            metadata: Additional metadata (source, author, description, etc.)

        Returns:
            The created SchemaVersion.
        """
        ...

    @abstractmethod
    def get_version(self, version_id: str) -> "SchemaVersion | None":
        """Get a specific schema version by ID.

        Args:
            version_id: Unique version identifier.

        Returns:
            SchemaVersion if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_latest(self) -> "SchemaVersion | None":
        """Get the most recent schema version.

        Returns:
            Latest SchemaVersion, or None if history is empty.
        """
        ...

    @abstractmethod
    def get_baseline(self) -> "SchemaVersion | None":
        """Get the baseline (oldest) schema version.

        Returns:
            Oldest SchemaVersion, or None if history is empty.
        """
        ...

    @abstractmethod
    def list_versions(
        self,
        limit: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list["SchemaVersion"]:
        """List schema versions.

        Args:
            limit: Maximum number of versions to return.
            since: Only include versions after this time.
            until: Only include versions before this time.

        Returns:
            List of SchemaVersion objects, most recent first.
        """
        ...

    @abstractmethod
    def delete_version(self, version_id: str) -> bool:
        """Delete a schema version.

        Args:
            version_id: Version identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...


@runtime_checkable
class VersionStrategy(Protocol):
    """Protocol for version string generation strategies."""

    @abstractmethod
    def generate(
        self,
        previous_version: str | None,
        changes: list[SchemaChange],
    ) -> str:
        """Generate the next version string.

        Args:
            previous_version: Previous version string, or None if first version.
            changes: List of changes from previous version.

        Returns:
            New version string.
        """
        ...


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SchemaVersion:
    """Represents a single schema version.

    Attributes:
        version_id: Unique identifier for this version.
        version: Human-readable version string (e.g., "1.0.0", "v2").
        schema: Column name to type mapping.
        created_at: When this version was created.
        metadata: Additional metadata (source, author, tags, etc.)
        parent_id: ID of the previous version (None for first version).
        changes_from_parent: Changes from the parent version.
    """

    version_id: str
    version: str
    schema: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    changes_from_parent: list[SchemaChange] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "version": self.version,
            "schema": {k: str(v) for k, v in self.schema.items()},
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "changes_from_parent": [c.to_dict() for c in self.changes_from_parent],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            version=data["version"],
            schema=data["schema"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            changes_from_parent=[
                SchemaChange.from_dict(c)
                for c in data.get("changes_from_parent", [])
            ],
        )

    def column_count(self) -> int:
        """Get the number of columns in this schema."""
        return len(self.schema)

    def has_breaking_changes(self) -> bool:
        """Check if this version has breaking changes from parent."""
        return any(c.breaking for c in self.changes_from_parent)


@dataclass
class SchemaDiff:
    """Diff between two schema versions.

    Attributes:
        source_version: The older/source version.
        target_version: The newer/target version.
        changes: List of changes between versions.
        summary: Summary of changes.
        computed_at: When the diff was computed.
    """

    source_version: SchemaVersion
    target_version: SchemaVersion
    changes: list[SchemaChange]
    summary: SchemaChangeSummary
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_version": self.source_version.to_dict(),
            "target_version": self.target_version.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary.to_dict(),
            "computed_at": self.computed_at.isoformat(),
        }

    def is_breaking(self) -> bool:
        """Check if this diff contains breaking changes."""
        return self.summary.is_breaking()

    def format_text(self, include_hints: bool = True) -> str:
        """Format the diff as human-readable text.

        Args:
            include_hints: Whether to include migration hints.

        Returns:
            Formatted diff string.
        """
        lines = [
            f"Schema Diff: {self.source_version.version} → {self.target_version.version}",
            f"  Total changes: {self.summary.total_changes}",
            f"  Breaking changes: {self.summary.breaking_changes}",
            f"  Compatibility: {self.summary.compatibility_level.value}",
            "",
        ]

        if self.changes:
            lines.append("Changes:")
            for change in self.changes:
                prefix = "  [BREAKING] " if change.breaking else "  "
                lines.append(f"{prefix}{change.description}")
                if include_hints and change.migration_hint:
                    lines.append(f"    Hint: {change.migration_hint}")

        return "\n".join(lines)


# =============================================================================
# Version Strategies
# =============================================================================


class IncrementalVersionStrategy:
    """Simple incrementing version numbers (1, 2, 3, ...)."""

    def generate(
        self,
        previous_version: str | None,
        changes: list[SchemaChange],
    ) -> str:
        """Generate next version by incrementing."""
        if previous_version is None:
            return "1"
        try:
            num = int(previous_version)
            return str(num + 1)
        except ValueError:
            # Fallback if previous isn't a number
            return "1"


class SemanticVersionStrategy:
    """Semantic versioning (major.minor.patch).

    - Major: Breaking changes
    - Minor: Non-breaking additions
    - Patch: Non-breaking changes (type widening, etc.)
    """

    def generate(
        self,
        previous_version: str | None,
        changes: list[SchemaChange],
    ) -> str:
        """Generate next semantic version based on change types."""
        if previous_version is None:
            return "1.0.0"

        parts = previous_version.split(".")
        if len(parts) != 3:
            return "1.0.0"

        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return "1.0.0"

        # Analyze changes to determine version bump
        has_breaking = any(c.breaking for c in changes)
        has_additions = any(
            c.change_type == ChangeType.COLUMN_ADDED for c in changes
        )

        if has_breaking:
            return f"{major + 1}.0.0"
        elif has_additions:
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"


class TimestampVersionStrategy:
    """Timestamp-based versions (YYYYMMDD.HHMMSS)."""

    def generate(
        self,
        previous_version: str | None,
        changes: list[SchemaChange],
    ) -> str:
        """Generate timestamp-based version."""
        now = datetime.now()
        return now.strftime("%Y%m%d.%H%M%S")


class GitLikeVersionStrategy:
    """Git-like short hash versions."""

    def generate(
        self,
        previous_version: str | None,
        changes: list[SchemaChange],
    ) -> str:
        """Generate git-like short hash version."""
        return uuid.uuid4().hex[:8]


# =============================================================================
# Storage Implementations
# =============================================================================


class InMemorySchemaHistory(SchemaHistoryStorage):
    """In-memory schema history storage.

    Suitable for testing and short-lived processes.

    Example:
        history = InMemorySchemaHistory()
        v1 = history.save_version({"id": "Int64", "name": "Utf8"})
        v2 = history.save_version({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        diff = history.diff(v1.version_id, v2.version_id)
    """

    def __init__(
        self,
        max_versions: int = 100,
        version_strategy: VersionStrategy | None = None,
    ):
        """Initialize in-memory storage.

        Args:
            max_versions: Maximum versions to keep.
            version_strategy: Strategy for version string generation.
        """
        self._max_versions = max_versions
        self._version_strategy = version_strategy or SemanticVersionStrategy()
        self._versions: dict[str, SchemaVersion] = {}
        self._order: list[str] = []  # Ordered list of version_ids

    def save_version(
        self,
        schema: dict[str, Any],
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchemaVersion:
        """Save a new schema version."""
        from truthound.profiler.evolution.detector import SchemaEvolutionDetector

        version_id = str(uuid.uuid4())
        parent = self.get_latest()

        # Detect changes from parent
        changes: list[SchemaChange] = []
        if parent:
            detector = SchemaEvolutionDetector()
            changes = detector.detect_changes(schema, parent.schema)

        # Generate version string
        if version is None:
            parent_version = parent.version if parent else None
            version = self._version_strategy.generate(parent_version, changes)

        schema_version = SchemaVersion(
            version_id=version_id,
            version=version,
            schema=schema,
            created_at=datetime.now(),
            metadata=metadata or {},
            parent_id=parent.version_id if parent else None,
            changes_from_parent=changes,
        )

        self._versions[version_id] = schema_version
        self._order.append(version_id)

        # Cleanup old versions
        while len(self._order) > self._max_versions:
            old_id = self._order.pop(0)
            del self._versions[old_id]

        return schema_version

    def get_version(self, version_id: str) -> SchemaVersion | None:
        """Get a specific version by ID."""
        return self._versions.get(version_id)

    def get_latest(self) -> SchemaVersion | None:
        """Get the most recent version."""
        if not self._order:
            return None
        return self._versions[self._order[-1]]

    def get_baseline(self) -> SchemaVersion | None:
        """Get the oldest version."""
        if not self._order:
            return None
        return self._versions[self._order[0]]

    def list_versions(
        self,
        limit: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SchemaVersion]:
        """List versions, most recent first."""
        versions = [self._versions[vid] for vid in reversed(self._order)]

        if since:
            versions = [v for v in versions if v.created_at > since]
        if until:
            versions = [v for v in versions if v.created_at < until]
        if limit:
            versions = versions[:limit]

        return versions

    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        if version_id not in self._versions:
            return False

        del self._versions[version_id]
        self._order.remove(version_id)
        return True

    def diff(
        self,
        source_id: str,
        target_id: str,
    ) -> SchemaDiff | None:
        """Compute diff between two versions.

        Args:
            source_id: Source (older) version ID.
            target_id: Target (newer) version ID.

        Returns:
            SchemaDiff or None if versions not found.
        """
        source = self.get_version(source_id)
        target = self.get_version(target_id)

        if not source or not target:
            return None

        from truthound.profiler.evolution.detector import SchemaEvolutionDetector

        detector = SchemaEvolutionDetector()
        changes = detector.detect_changes(target.schema, source.schema)
        summary = detector.get_change_summary(changes)

        return SchemaDiff(
            source_version=source,
            target_version=target,
            changes=changes,
            summary=summary,
        )

    def rollback_to(self, version_id: str) -> SchemaVersion | None:
        """Rollback schema to a specific version.

        Creates a new version that copies the schema from the target version.

        Args:
            version_id: Version to rollback to.

        Returns:
            New SchemaVersion representing the rollback.
        """
        target = self.get_version(version_id)
        if not target:
            return None

        return self.save_version(
            schema=target.schema.copy(),
            metadata={
                "rollback_from": self.get_latest().version_id if self.get_latest() else None,
                "rollback_to": version_id,
                "rollback_reason": "manual_rollback",
            },
        )

    def get_version_by_string(self, version: str) -> SchemaVersion | None:
        """Find version by version string.

        Args:
            version: Version string (e.g., "1.0.0").

        Returns:
            SchemaVersion if found, None otherwise.
        """
        for v in self._versions.values():
            if v.version == version:
                return v
        return None


class FileSchemaHistory(SchemaHistoryStorage):
    """File-based schema history storage with persistence.

    Stores schema versions as JSON files with an index for fast lookup.

    Example:
        history = FileSchemaHistory("./schema_history")
        v1 = history.save_version({"id": "Int64", "name": "Utf8"})

        # History persists across restarts
        history2 = FileSchemaHistory("./schema_history")
        assert history2.get_latest().version_id == v1.version_id
    """

    def __init__(
        self,
        base_path: str | Path,
        max_versions: int = 1000,
        version_strategy: VersionStrategy | None = None,
        compress: bool = False,
    ):
        """Initialize file-based storage.

        Args:
            base_path: Base directory for storage.
            max_versions: Maximum versions to keep.
            version_strategy: Strategy for version string generation.
            compress: Whether to compress stored files.
        """
        self._base_path = Path(base_path)
        self._max_versions = max_versions
        self._version_strategy = version_strategy or SemanticVersionStrategy()
        self._compress = compress

        # Create directories
        self._versions_dir = self._base_path / "versions"
        self._versions_dir.mkdir(parents=True, exist_ok=True)

        self._index_file = self._base_path / "index.json"
        self._index: list[str] = self._load_index()

    def _load_index(self) -> list[str]:
        """Load the version index."""
        if not self._index_file.exists():
            return []

        try:
            with open(self._index_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load schema index: {e}")
            return []

    def _save_index(self) -> None:
        """Save the version index."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save schema index: {e}")

    def _get_version_path(self, version_id: str) -> Path:
        """Get path for a version file."""
        suffix = ".json.gz" if self._compress else ".json"
        return self._versions_dir / f"{version_id}{suffix}"

    def save_version(
        self,
        schema: dict[str, Any],
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchemaVersion:
        """Save a new schema version."""
        from truthound.profiler.evolution.detector import SchemaEvolutionDetector

        version_id = str(uuid.uuid4())
        parent = self.get_latest()

        # Detect changes from parent
        changes: list[SchemaChange] = []
        if parent:
            detector = SchemaEvolutionDetector()
            changes = detector.detect_changes(schema, parent.schema)

        # Generate version string
        if version is None:
            parent_version = parent.version if parent else None
            version = self._version_strategy.generate(parent_version, changes)

        schema_version = SchemaVersion(
            version_id=version_id,
            version=version,
            schema=schema,
            created_at=datetime.now(),
            metadata=metadata or {},
            parent_id=parent.version_id if parent else None,
            changes_from_parent=changes,
        )

        # Save to file
        version_path = self._get_version_path(version_id)
        try:
            data = schema_version.to_dict()
            if self._compress:
                import gzip
                with gzip.open(version_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
            else:
                with open(version_path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save schema version: {e}")
            raise

        # Update index
        self._index.append(version_id)
        self._save_index()

        # Cleanup old versions
        self._cleanup_old_versions()

        return schema_version

    def get_version(self, version_id: str) -> SchemaVersion | None:
        """Get a specific version by ID."""
        version_path = self._get_version_path(version_id)

        if not version_path.exists():
            return None

        try:
            if self._compress:
                import gzip
                with gzip.open(version_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(version_path, "r") as f:
                    data = json.load(f)

            return SchemaVersion.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load schema version {version_id}: {e}")
            return None

    def get_latest(self) -> SchemaVersion | None:
        """Get the most recent version."""
        if not self._index:
            return None
        return self.get_version(self._index[-1])

    def get_baseline(self) -> SchemaVersion | None:
        """Get the oldest version."""
        if not self._index:
            return None
        return self.get_version(self._index[0])

    def list_versions(
        self,
        limit: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SchemaVersion]:
        """List versions, most recent first."""
        versions = []
        for vid in reversed(self._index):
            v = self.get_version(vid)
            if v:
                if since and v.created_at <= since:
                    continue
                if until and v.created_at >= until:
                    continue
                versions.append(v)
                if limit and len(versions) >= limit:
                    break
        return versions

    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        if version_id not in self._index:
            return False

        version_path = self._get_version_path(version_id)
        try:
            if version_path.exists():
                version_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete version file: {e}")
            return False

        self._index.remove(version_id)
        self._save_index()
        return True

    def _cleanup_old_versions(self) -> None:
        """Remove old versions beyond the limit."""
        while len(self._index) > self._max_versions:
            old_id = self._index[0]
            self.delete_version(old_id)

    def diff(
        self,
        source_id: str,
        target_id: str,
    ) -> SchemaDiff | None:
        """Compute diff between two versions."""
        source = self.get_version(source_id)
        target = self.get_version(target_id)

        if not source or not target:
            return None

        from truthound.profiler.evolution.detector import SchemaEvolutionDetector

        detector = SchemaEvolutionDetector()
        changes = detector.detect_changes(target.schema, source.schema)
        summary = detector.get_change_summary(changes)

        return SchemaDiff(
            source_version=source,
            target_version=target,
            changes=changes,
            summary=summary,
        )

    def rollback_to(self, version_id: str) -> SchemaVersion | None:
        """Rollback schema to a specific version."""
        target = self.get_version(version_id)
        if not target:
            return None

        return self.save_version(
            schema=target.schema.copy(),
            metadata={
                "rollback_from": self.get_latest().version_id if self.get_latest() else None,
                "rollback_to": version_id,
                "rollback_reason": "manual_rollback",
            },
        )

    def get_version_by_string(self, version: str) -> SchemaVersion | None:
        """Find version by version string."""
        for vid in self._index:
            v = self.get_version(vid)
            if v and v.version == version:
                return v
        return None


# =============================================================================
# Schema History Manager (Unified Interface)
# =============================================================================


class SchemaHistory:
    """High-level manager for schema history operations.

    Provides a unified interface for schema versioning, diff computation,
    and rollback operations.

    Example:
        # Initialize with file storage
        history = SchemaHistory.create(
            storage_type="file",
            path="./schema_history",
            version_strategy="semantic",
        )

        # Save schema versions
        v1 = history.save({"id": "Int64", "name": "Utf8"})
        v2 = history.save({"id": "Int64", "name": "Utf8", "email": "Utf8"})

        # Get diff
        diff = history.diff(v1, v2)
        print(diff.format_text())

        # Rollback to v1
        v3 = history.rollback(v1)
    """

    def __init__(
        self,
        storage: SchemaHistoryStorage,
    ):
        """Initialize with a storage backend.

        Args:
            storage: Storage backend implementing SchemaHistoryStorage.
        """
        self._storage = storage

    @classmethod
    def create(
        cls,
        storage_type: str = "memory",
        path: str | Path | None = None,
        max_versions: int = 100,
        version_strategy: str = "semantic",
        compress: bool = False,
    ) -> "SchemaHistory":
        """Factory method to create SchemaHistory.

        Args:
            storage_type: Type of storage ("memory" or "file").
            path: Path for file storage (required if storage_type="file").
            max_versions: Maximum versions to keep.
            version_strategy: Version strategy ("incremental", "semantic", "timestamp", "git").
            compress: Whether to compress stored files.

        Returns:
            Configured SchemaHistory instance.
        """
        # Select version strategy
        strategies: dict[str, VersionStrategy] = {
            "incremental": IncrementalVersionStrategy(),
            "semantic": SemanticVersionStrategy(),
            "timestamp": TimestampVersionStrategy(),
            "git": GitLikeVersionStrategy(),
        }
        strategy = strategies.get(version_strategy, SemanticVersionStrategy())

        # Create storage
        if storage_type == "file":
            if not path:
                raise ValueError("path is required for file storage")
            storage: SchemaHistoryStorage = FileSchemaHistory(
                base_path=path,
                max_versions=max_versions,
                version_strategy=strategy,
                compress=compress,
            )
        else:
            storage = InMemorySchemaHistory(
                max_versions=max_versions,
                version_strategy=strategy,
            )

        return cls(storage)

    def save(
        self,
        schema: dict[str, Any],
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SchemaVersion:
        """Save a new schema version.

        Args:
            schema: Schema to save.
            version: Optional version string.
            metadata: Additional metadata.

        Returns:
            Created SchemaVersion.
        """
        return self._storage.save_version(schema, version, metadata)

    def get(self, version_id: str) -> SchemaVersion | None:
        """Get a specific version by ID."""
        return self._storage.get_version(version_id)

    def get_by_version(self, version: str) -> SchemaVersion | None:
        """Get a version by version string (e.g., "1.0.0")."""
        if hasattr(self._storage, "get_version_by_string"):
            return self._storage.get_version_by_string(version)
        # Fallback: search through all versions
        for v in self._storage.list_versions():
            if v.version == version:
                return v
        return None

    @property
    def latest(self) -> SchemaVersion | None:
        """Get the most recent version."""
        return self._storage.get_latest()

    @property
    def baseline(self) -> SchemaVersion | None:
        """Get the baseline (oldest) version."""
        return self._storage.get_baseline()

    def list(
        self,
        limit: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SchemaVersion]:
        """List schema versions."""
        return self._storage.list_versions(limit, since, until)

    def diff(
        self,
        source: str | SchemaVersion,
        target: str | SchemaVersion | None = None,
    ) -> SchemaDiff | None:
        """Compute diff between two versions.

        Args:
            source: Source version ID, version string, or SchemaVersion.
            target: Target version (defaults to latest if None).

        Returns:
            SchemaDiff or None if versions not found.
        """
        # Resolve source
        if isinstance(source, SchemaVersion):
            source_version = source
        else:
            source_version = self._storage.get_version(source)
            if not source_version:
                source_version = self.get_by_version(source)

        if not source_version:
            return None

        # Resolve target
        if target is None:
            target_version = self._storage.get_latest()
        elif isinstance(target, SchemaVersion):
            target_version = target
        else:
            target_version = self._storage.get_version(target)
            if not target_version:
                target_version = self.get_by_version(target)

        if not target_version:
            return None

        # Use storage's diff if available
        if hasattr(self._storage, "diff"):
            return self._storage.diff(source_version.version_id, target_version.version_id)

        # Manual diff
        from truthound.profiler.evolution.detector import SchemaEvolutionDetector

        detector = SchemaEvolutionDetector()
        changes = detector.detect_changes(target_version.schema, source_version.schema)
        summary = detector.get_change_summary(changes)

        return SchemaDiff(
            source_version=source_version,
            target_version=target_version,
            changes=changes,
            summary=summary,
        )

    def rollback(
        self,
        target: str | SchemaVersion,
        reason: str = "manual_rollback",
    ) -> SchemaVersion | None:
        """Rollback to a specific version.

        Args:
            target: Target version ID, version string, or SchemaVersion.
            reason: Reason for rollback.

        Returns:
            New SchemaVersion representing the rollback.
        """
        # Resolve target
        if isinstance(target, SchemaVersion):
            target_version = target
        else:
            target_version = self._storage.get_version(target)
            if not target_version:
                target_version = self.get_by_version(target)

        if not target_version:
            return None

        # Use storage's rollback if available
        if hasattr(self._storage, "rollback_to"):
            # Update metadata with reason
            current = self._storage.get_latest()
            return self._storage.save_version(
                schema=target_version.schema.copy(),
                metadata={
                    "rollback_from": current.version_id if current else None,
                    "rollback_to": target_version.version_id,
                    "rollback_reason": reason,
                },
            )

        return None

    def delete(self, version_id: str) -> bool:
        """Delete a version."""
        return self._storage.delete_version(version_id)

    @property
    def version_count(self) -> int:
        """Get total number of versions."""
        return len(self._storage.list_versions())

    def has_breaking_changes_since(self, version: str | SchemaVersion) -> bool:
        """Check if there are breaking changes since a version.

        Args:
            version: Version to check from.

        Returns:
            True if any breaking changes exist.
        """
        diff = self.diff(version)
        return diff.is_breaking() if diff else False
