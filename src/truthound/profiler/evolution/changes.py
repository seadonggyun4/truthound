"""Schema change data structures.

This module defines the types and structures for representing
schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChangeType(str, Enum):
    """Types of schema changes."""

    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    NULLABLE_CHANGED = "nullable_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    DEFAULT_CHANGED = "default_changed"
    ORDER_CHANGED = "order_changed"


class ChangeSeverity(str, Enum):
    """Severity levels for schema changes."""

    INFO = "info"  # Non-breaking, informational
    WARNING = "warning"  # Potentially breaking, needs review
    CRITICAL = "critical"  # Breaking change


class CompatibilityLevel(str, Enum):
    """Schema compatibility levels."""

    FULL = "full"  # Fully compatible (both forward and backward)
    FORWARD = "forward"  # Forward compatible only (new can read old)
    BACKWARD = "backward"  # Backward compatible only (old can read new)
    NONE = "none"  # Not compatible


@dataclass
class SchemaChange:
    """Represents a single schema change.

    Attributes:
        change_type: Type of the change.
        column: Affected column name (if applicable).
        old_value: Previous value.
        new_value: New value.
        breaking: Whether this is a breaking change.
        severity: Severity level of the change.
        description: Human-readable description.
        migration_hint: Hint for how to handle the change.
        metadata: Additional metadata.
    """

    change_type: ChangeType
    column: str | None = None
    old_value: Any = None
    new_value: Any = None
    breaking: bool = False
    severity: ChangeSeverity = ChangeSeverity.INFO
    description: str = ""
    migration_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate description if not provided."""
        if not self.description:
            self.description = self._generate_description()

        # Set severity based on breaking flag
        if self.breaking and self.severity == ChangeSeverity.INFO:
            self.severity = ChangeSeverity.CRITICAL

    def _generate_description(self) -> str:
        """Generate a human-readable description."""
        descriptions = {
            ChangeType.COLUMN_ADDED: f"Column '{self.column}' added with type {self.new_value}",
            ChangeType.COLUMN_REMOVED: f"Column '{self.column}' removed (was type {self.old_value})",
            ChangeType.COLUMN_RENAMED: f"Column renamed from '{self.old_value}' to '{self.new_value}'",
            ChangeType.TYPE_CHANGED: f"Column '{self.column}' type changed from {self.old_value} to {self.new_value}",
            ChangeType.NULLABLE_CHANGED: f"Column '{self.column}' nullability changed from {self.old_value} to {self.new_value}",
            ChangeType.CONSTRAINT_ADDED: f"Constraint added on column '{self.column}': {self.new_value}",
            ChangeType.CONSTRAINT_REMOVED: f"Constraint removed from column '{self.column}': {self.old_value}",
            ChangeType.DEFAULT_CHANGED: f"Column '{self.column}' default changed from {self.old_value} to {self.new_value}",
            ChangeType.ORDER_CHANGED: f"Column order changed",
        }
        return descriptions.get(self.change_type, f"Unknown change: {self.change_type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type.value,
            "column": self.column,
            "old_value": str(self.old_value) if self.old_value else None,
            "new_value": str(self.new_value) if self.new_value else None,
            "breaking": self.breaking,
            "severity": self.severity.value,
            "description": self.description,
            "migration_hint": self.migration_hint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaChange":
        """Create from dictionary."""
        return cls(
            change_type=ChangeType(data["change_type"]),
            column=data.get("column"),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            breaking=data.get("breaking", False),
            severity=ChangeSeverity(data.get("severity", "info")),
            description=data.get("description", ""),
            migration_hint=data.get("migration_hint"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SchemaChangeSummary:
    """Summary of schema changes.

    Attributes:
        total_changes: Total number of changes.
        breaking_changes: Number of breaking changes.
        columns_added: Number of columns added.
        columns_removed: Number of columns removed.
        type_changes: Number of type changes.
        changes_by_severity: Changes grouped by severity.
        compatibility_level: Overall compatibility level.
        detected_at: When the changes were detected.
        baseline_version: Version of the baseline schema.
        current_version: Version of the current schema.
    """

    total_changes: int = 0
    breaking_changes: int = 0
    columns_added: int = 0
    columns_removed: int = 0
    type_changes: int = 0
    changes_by_severity: dict[str, int] = field(default_factory=dict)
    compatibility_level: CompatibilityLevel = CompatibilityLevel.FULL
    detected_at: datetime = field(default_factory=datetime.now)
    baseline_version: str | None = None
    current_version: str | None = None

    @classmethod
    def from_changes(cls, changes: list[SchemaChange]) -> "SchemaChangeSummary":
        """Create summary from list of changes."""
        summary = cls(
            total_changes=len(changes),
            detected_at=datetime.now(),
        )

        for change in changes:
            if change.breaking:
                summary.breaking_changes += 1

            if change.change_type == ChangeType.COLUMN_ADDED:
                summary.columns_added += 1
            elif change.change_type == ChangeType.COLUMN_REMOVED:
                summary.columns_removed += 1
            elif change.change_type == ChangeType.TYPE_CHANGED:
                summary.type_changes += 1

            severity = change.severity.value
            summary.changes_by_severity[severity] = (
                summary.changes_by_severity.get(severity, 0) + 1
            )

        # Determine compatibility level
        if summary.breaking_changes > 0:
            if summary.columns_removed > 0 or summary.type_changes > 0:
                summary.compatibility_level = CompatibilityLevel.NONE
            else:
                summary.compatibility_level = CompatibilityLevel.FORWARD
        elif summary.columns_added > 0:
            summary.compatibility_level = CompatibilityLevel.BACKWARD

        return summary

    def is_breaking(self) -> bool:
        """Check if there are any breaking changes."""
        return self.breaking_changes > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_changes": self.total_changes,
            "breaking_changes": self.breaking_changes,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "type_changes": self.type_changes,
            "changes_by_severity": self.changes_by_severity,
            "compatibility_level": self.compatibility_level.value,
            "detected_at": self.detected_at.isoformat(),
            "baseline_version": self.baseline_version,
            "current_version": self.current_version,
        }
