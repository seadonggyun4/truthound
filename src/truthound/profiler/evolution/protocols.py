"""Protocol definitions for schema evolution detection.

This module defines the interfaces for schema change detection,
compatibility checking, and change notification.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import polars as pl
    from truthound.profiler.evolution.changes import SchemaChange, SchemaChangeSummary


@runtime_checkable
class SchemaChangeDetector(Protocol):
    """Protocol for schema change detection."""

    @abstractmethod
    def detect_changes(
        self,
        current_schema: Any,
        baseline_schema: Any | None = None,
    ) -> list["SchemaChange"]:
        """Detect schema changes between current and baseline.

        Args:
            current_schema: Current schema to analyze.
            baseline_schema: Baseline schema to compare against.

        Returns:
            List of detected schema changes.
        """
        ...

    @abstractmethod
    def get_change_summary(
        self,
        changes: list["SchemaChange"],
    ) -> "SchemaChangeSummary":
        """Get a summary of schema changes.

        Args:
            changes: List of changes to summarize.

        Returns:
            Summary of the changes.
        """
        ...


@runtime_checkable
class CompatibilityChecker(Protocol):
    """Protocol for schema compatibility checking."""

    @abstractmethod
    def is_compatible(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> bool:
        """Check if schemas are compatible.

        Args:
            old_schema: Previous schema.
            new_schema: New schema.

        Returns:
            True if schemas are compatible.
        """
        ...

    @abstractmethod
    def get_compatibility_level(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> str:
        """Get the compatibility level between schemas.

        Args:
            old_schema: Previous schema.
            new_schema: New schema.

        Returns:
            Compatibility level ('full', 'forward', 'backward', 'none').
        """
        ...


@runtime_checkable
class ChangeNotifier(Protocol):
    """Protocol for change notification handlers."""

    @abstractmethod
    def notify(
        self,
        changes: list["SchemaChange"],
        severity: str,
    ) -> None:
        """Send notification about schema changes.

        Args:
            changes: List of changes to notify about.
            severity: Severity level ('info', 'warning', 'critical').
        """
        ...

    @abstractmethod
    def supports_severity(self, severity: str) -> bool:
        """Check if this notifier handles the given severity.

        Args:
            severity: Severity level to check.

        Returns:
            True if this notifier handles the severity.
        """
        ...
