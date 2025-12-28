"""Schema evolution detector implementation.

This module provides the main schema change detection logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import polars as pl

from truthound.profiler.evolution.protocols import SchemaChangeDetector
from truthound.profiler.evolution.changes import (
    ChangeType,
    ChangeSeverity,
    SchemaChange,
    SchemaChangeSummary,
)

if TYPE_CHECKING:
    from truthound.profiler.scheduling.protocols import ProfileStorage

logger = logging.getLogger(__name__)


# Type compatibility mappings
TYPE_COMPATIBLE_UPGRADES: set[tuple[type, type]] = {
    (pl.Int8, pl.Int16),
    (pl.Int8, pl.Int32),
    (pl.Int8, pl.Int64),
    (pl.Int16, pl.Int32),
    (pl.Int16, pl.Int64),
    (pl.Int32, pl.Int64),
    (pl.UInt8, pl.UInt16),
    (pl.UInt8, pl.UInt32),
    (pl.UInt8, pl.UInt64),
    (pl.UInt16, pl.UInt32),
    (pl.UInt16, pl.UInt64),
    (pl.UInt32, pl.UInt64),
    (pl.Float32, pl.Float64),
}


@dataclass
class ColumnAddedChange(SchemaChange):
    """Change representing a new column added."""

    def __init__(self, column: str, dtype: Any):
        super().__init__(
            change_type=ChangeType.COLUMN_ADDED,
            column=column,
            new_value=dtype,
            breaking=False,
            severity=ChangeSeverity.INFO,
            migration_hint="New column - ensure consumers handle missing data for older records",
        )


@dataclass
class ColumnRemovedChange(SchemaChange):
    """Change representing a column removed."""

    def __init__(self, column: str, dtype: Any):
        super().__init__(
            change_type=ChangeType.COLUMN_REMOVED,
            column=column,
            old_value=dtype,
            breaking=True,
            severity=ChangeSeverity.CRITICAL,
            migration_hint="Column removed - update consumers to not depend on this column",
        )


@dataclass
class ColumnRenamedChange(SchemaChange):
    """Change representing a column renamed."""

    def __init__(self, old_name: str, new_name: str):
        super().__init__(
            change_type=ChangeType.COLUMN_RENAMED,
            old_value=old_name,
            new_value=new_name,
            breaking=True,
            severity=ChangeSeverity.CRITICAL,
            migration_hint=f"Rename references from '{old_name}' to '{new_name}'",
        )


@dataclass
class TypeChangedChange(SchemaChange):
    """Change representing a column type change."""

    def __init__(
        self,
        column: str,
        old_type: Any,
        new_type: Any,
        is_compatible: bool = False,
    ):
        breaking = not is_compatible
        super().__init__(
            change_type=ChangeType.TYPE_CHANGED,
            column=column,
            old_value=old_type,
            new_value=new_type,
            breaking=breaking,
            severity=ChangeSeverity.CRITICAL if breaking else ChangeSeverity.WARNING,
            migration_hint=(
                "Compatible type widening - no action needed"
                if is_compatible
                else f"Type changed - update data pipeline for {old_type} -> {new_type}"
            ),
        )


@dataclass
class NullabilityChangedChange(SchemaChange):
    """Change representing nullability change."""

    def __init__(self, column: str, was_nullable: bool, is_nullable: bool):
        # Non-nullable to nullable is safe; nullable to non-nullable is breaking
        breaking = was_nullable and not is_nullable
        super().__init__(
            change_type=ChangeType.NULLABLE_CHANGED,
            column=column,
            old_value=f"nullable={was_nullable}",
            new_value=f"nullable={is_nullable}",
            breaking=breaking,
            severity=ChangeSeverity.CRITICAL if breaking else ChangeSeverity.INFO,
            migration_hint=(
                "Column is now required - ensure all records have values"
                if breaking
                else "Column is now optional - no action needed"
            ),
        )


class SchemaEvolutionDetector(SchemaChangeDetector):
    """Detector for schema evolution.

    Analyzes schema changes and classifies them as breaking or non-breaking.

    Example:
        detector = SchemaEvolutionDetector(storage=profile_storage)

        # Detect changes from stored baseline
        changes = detector.detect_changes(current_schema)

        # Or compare specific schemas
        changes = detector.detect_changes(current_schema, baseline_schema)

        for change in changes:
            print(f"{change.severity}: {change.description}")
            if change.migration_hint:
                print(f"  Hint: {change.migration_hint}")
    """

    def __init__(
        self,
        storage: "ProfileStorage | None" = None,
        detect_renames: bool = True,
        rename_similarity_threshold: float = 0.8,
    ):
        """Initialize the detector.

        Args:
            storage: Profile storage for baseline retrieval.
            detect_renames: Whether to detect column renames.
            rename_similarity_threshold: Threshold for rename detection.
        """
        self._storage = storage
        self._detect_renames = detect_renames
        self._rename_threshold = rename_similarity_threshold

    def detect_changes(
        self,
        current_schema: Any,
        baseline_schema: Any | None = None,
    ) -> list[SchemaChange]:
        """Detect schema changes.

        Args:
            current_schema: Current schema (pl.Schema or dict).
            baseline_schema: Baseline to compare against. If None, uses storage.

        Returns:
            List of detected changes.
        """
        # Get baseline if not provided
        if baseline_schema is None and self._storage:
            baseline_schema = self._storage.get_baseline_schema()

        if baseline_schema is None:
            logger.debug("No baseline schema - treating current as baseline")
            return []

        # Normalize schemas
        current = self._normalize_schema(current_schema)
        baseline = self._normalize_schema(baseline_schema)

        changes: list[SchemaChange] = []

        # Get column sets
        current_cols = set(current.keys())
        baseline_cols = set(baseline.keys())

        # Detect added columns
        added_cols = current_cols - baseline_cols

        # Detect removed columns
        removed_cols = baseline_cols - current_cols

        # Try to detect renames
        if self._detect_renames and added_cols and removed_cols:
            renames, added_cols, removed_cols = self._detect_column_renames(
                added_cols,
                removed_cols,
                current,
                baseline,
            )
            changes.extend(renames)

        # Add remaining as added/removed
        for col in added_cols:
            changes.append(ColumnAddedChange(col, current[col]))

        for col in removed_cols:
            changes.append(ColumnRemovedChange(col, baseline[col]))

        # Detect type changes for common columns
        common_cols = current_cols & baseline_cols
        for col in common_cols:
            current_type = current[col]
            baseline_type = baseline[col]

            if current_type != baseline_type:
                is_compatible = self._is_compatible_type_change(
                    baseline_type, current_type
                )
                changes.append(TypeChangedChange(
                    col, baseline_type, current_type, is_compatible
                ))

        return changes

    def get_change_summary(
        self,
        changes: list[SchemaChange],
    ) -> SchemaChangeSummary:
        """Get a summary of schema changes."""
        return SchemaChangeSummary.from_changes(changes)

    def _normalize_schema(self, schema: Any) -> dict[str, Any]:
        """Normalize schema to a standard format."""
        if isinstance(schema, dict):
            return schema

        # Handle polars Schema
        if hasattr(schema, "names") and hasattr(schema, "__getitem__"):
            return {name: schema[name] for name in schema.names()}

        # Handle polars DataFrame
        if isinstance(schema, pl.DataFrame):
            return dict(schema.schema)

        # Handle polars LazyFrame
        if isinstance(schema, pl.LazyFrame):
            return dict(schema.collect_schema())

        raise TypeError(f"Cannot normalize schema of type {type(schema)}")

    def _is_compatible_type_change(
        self,
        old_type: Any,
        new_type: Any,
    ) -> bool:
        """Check if a type change is backward compatible.

        Compatible changes include:
        - Integer widening (Int32 -> Int64)
        - Float widening (Float32 -> Float64)
        - Same type (e.g., Utf8 -> Utf8)
        """
        if old_type == new_type:
            return True

        # Check known compatible upgrades
        old_base = type(old_type) if not isinstance(old_type, type) else old_type
        new_base = type(new_type) if not isinstance(new_type, type) else new_type

        return (old_base, new_base) in TYPE_COMPATIBLE_UPGRADES

    def _detect_column_renames(
        self,
        added: set[str],
        removed: set[str],
        current: dict[str, Any],
        baseline: dict[str, Any],
    ) -> tuple[list[SchemaChange], set[str], set[str]]:
        """Detect potential column renames.

        Uses column name similarity and type matching to detect renames.

        Returns:
            Tuple of (rename changes, remaining added, remaining removed).
        """
        renames: list[SchemaChange] = []
        matched_added: set[str] = set()
        matched_removed: set[str] = set()

        for removed_col in removed:
            removed_type = baseline[removed_col]

            for added_col in added:
                if added_col in matched_added:
                    continue

                added_type = current[added_col]

                # Check type compatibility
                if added_type != removed_type:
                    continue

                # Check name similarity
                similarity = self._name_similarity(removed_col, added_col)
                if similarity >= self._rename_threshold:
                    renames.append(ColumnRenamedChange(removed_col, added_col))
                    matched_added.add(added_col)
                    matched_removed.add(removed_col)
                    break

        return (
            renames,
            added - matched_added,
            removed - matched_removed,
        )

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between column names.

        Uses Levenshtein distance normalized by length.
        """
        # Simple Levenshtein distance implementation
        if name1 == name2:
            return 1.0

        len1, len2 = len(name1), len(name2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if name1[i-1].lower() == name2[j-1].lower() else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost, # substitution
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)
