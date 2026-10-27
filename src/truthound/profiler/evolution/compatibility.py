"""Schema compatibility analysis.

This module provides compatibility checking between schemas,
useful for determining if consumers can safely use new data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl

from truthound.profiler.evolution.changes import (
    ChangeType,
    CompatibilityLevel,
    SchemaChange,
)
from truthound.profiler.evolution.protocols import CompatibilityChecker

logger = logging.getLogger(__name__)


class CompatibilityAnalysisError(RuntimeError):
    """A requested native rule failed; no compatibility result was produced."""

    def __init__(self, rule_index: int, error_type: str):
        self.rule_index = rule_index
        self.error_type = error_type
        super().__init__("Native schema compatibility rule failed")


class CompatibilityRule(ABC):
    """Abstract base class for compatibility rules.

    Each rule checks a specific aspect of schema compatibility.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        ...

    @abstractmethod
    def check(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> tuple[bool, list[SchemaChange]]:
        """Check compatibility.

        Args:
            old_schema: Previous schema.
            new_schema: New schema.

        Returns:
            Tuple of (is_compatible, list of issues).
        """
        ...


class TypeCompatibilityChecker(CompatibilityRule):
    """Rule for checking type compatibility.

    Checks if type changes are backward compatible.
    """

    COMPATIBLE_CHANGES: set[tuple[type, type]] = {
        # Integer widening
        (pl.Int8, pl.Int16),
        (pl.Int8, pl.Int32),
        (pl.Int8, pl.Int64),
        (pl.Int16, pl.Int32),
        (pl.Int16, pl.Int64),
        (pl.Int32, pl.Int64),
        # Unsigned integer widening
        (pl.UInt8, pl.UInt16),
        (pl.UInt8, pl.UInt32),
        (pl.UInt8, pl.UInt64),
        (pl.UInt16, pl.UInt32),
        (pl.UInt16, pl.UInt64),
        (pl.UInt32, pl.UInt64),
        # Float widening
        (pl.Float32, pl.Float64),
        # Date to datetime is usually safe
        (pl.Date, pl.Datetime),
    }

    @property
    def name(self) -> str:
        return "type_compatibility"

    def check(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> tuple[bool, list[SchemaChange]]:
        """Check type compatibility."""
        issues: list[SchemaChange] = []
        is_compatible = True

        common_cols = set(old_schema.keys()) & set(new_schema.keys())

        for col in sorted(common_cols):
            old_type = old_schema[col]
            new_type = new_schema[col]

            if old_type == new_type:
                continue

            # Check if this is a compatible type change
            old_base = type(old_type) if not isinstance(old_type, type) else old_type
            new_base = type(new_type) if not isinstance(new_type, type) else new_type

            if (old_base, new_base) not in self.COMPATIBLE_CHANGES:
                is_compatible = False
                issues.append(SchemaChange(
                    change_type=ChangeType.TYPE_CHANGED,
                    column=col,
                    old_value=old_type,
                    new_value=new_type,
                    breaking=True,
                    description=f"Incompatible type change: {old_type} -> {new_type}",
                ))

        return is_compatible, issues


class NullabilityCompatibilityChecker(CompatibilityRule):
    """Rule for checking nullability changes.

    Non-nullable to nullable is safe.
    Nullable to non-nullable is breaking.
    """

    def __init__(self, old_nullable: dict[str, bool] | None = None,
                 new_nullable: dict[str, bool] | None = None):
        if (old_nullable is None) != (new_nullable is None):
            raise ValueError("Both declared nullability maps are required")
        for mapping in (old_nullable, new_nullable):
            if mapping is not None and (
                not isinstance(mapping, dict)
                or any(not isinstance(key, str) or type(value) is not bool for key, value in mapping.items())
            ):
                raise ValueError("Declared nullability requires column names and boolean values")
        self._old_nullable = dict(old_nullable) if old_nullable is not None else None
        self._new_nullable = dict(new_nullable) if new_nullable is not None else None

    @property
    def name(self) -> str:
        return "nullability_compatibility"

    def check(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> tuple[bool, list[SchemaChange]]:
        """Check nullability compatibility."""
        # Polars dtypes alone do not declare nullability. Preserve type-only
        # callers without inferring constraints from a sample's observed nulls.
        if self._old_nullable is None:
            return True, []
        old, new = self._old_nullable, self._new_nullable
        if old.keys() - old_schema.keys() or new.keys() - new_schema.keys():
            raise ValueError("Declared nullability contains unknown schema columns")
        issues = []
        for column in sorted(old_schema.keys() & new_schema.keys()):
            if (column in old) != (column in new):
                raise ValueError("A common column requires both nullability declarations")
            if column in old and old[column] and not new[column]:
                issues.append(SchemaChange(
                    change_type=ChangeType.NULLABLE_CHANGED, column=column,
                    old_value=True, new_value=False, breaking=True,
                    description="Declared nullable column became non-nullable",
                ))
        return not issues, issues


class ColumnPresenceChecker(CompatibilityRule):
    """Rule for checking column presence.

    Removed columns break backward compatibility.
    """

    @property
    def name(self) -> str:
        return "column_presence"

    def check(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
    ) -> tuple[bool, list[SchemaChange]]:
        """Check for removed columns."""
        issues: list[SchemaChange] = []
        is_compatible = True

        removed = set(old_schema.keys()) - set(new_schema.keys())

        for col in sorted(removed):
            is_compatible = False
            issues.append(SchemaChange(
                change_type=ChangeType.COLUMN_REMOVED,
                column=col,
                old_value=old_schema[col],
                breaking=True,
                description=f"Column '{col}' was removed",
            ))

        return is_compatible, issues


@dataclass
class CompatibilityResult:
    """Result of compatibility analysis.

    Attributes:
        is_compatible: Overall compatibility status.
        level: Compatibility level.
        issues: List of compatibility issues.
        rules_checked: Rules that were evaluated.
    """

    is_compatible: bool
    level: CompatibilityLevel
    issues: list[SchemaChange]
    rules_checked: list[str]

    def __bool__(self) -> bool:
        """Boolean evaluation returns compatibility status."""
        return self.is_compatible


class SchemaCompatibilityAnalyzer(CompatibilityChecker):
    """Analyzer for schema compatibility.

    Evaluates schema changes against a set of compatibility rules
    to determine the compatibility level.

    Example:
        analyzer = SchemaCompatibilityAnalyzer()

        result = analyzer.analyze(old_schema, new_schema)
        if result.is_compatible:
            print("Schemas are compatible")
        else:
            for issue in result.issues:
                print(f"Issue: {issue.description}")
    """

    DEFAULT_RULES: list[type[CompatibilityRule]] = [
        TypeCompatibilityChecker,
        NullabilityCompatibilityChecker,
        ColumnPresenceChecker,
    ]

    def __init__(
        self,
        rules: list[CompatibilityRule] | None = None,
        strict: bool = False,
    ):
        """Initialize the analyzer.

        Args:
            rules: Custom compatibility rules.
            strict: If True, any issue fails compatibility.
        """
        if type(strict) is not bool:
            raise ValueError("strict must be boolean")
        self._rules = list(rules) if rules else [cls() for cls in self.DEFAULT_RULES]
        self._strict = strict
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        if guard is not None and not callable(guard):
            raise TypeError("Execution guard must be callable")
        self._execution_guard = guard

    def _guard_execution(self) -> None:
        if self._execution_guard is not None:
            self._execution_guard()

    def is_compatible(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> bool:
        """Check if schemas are compatible."""
        result = self.analyze(old_schema, new_schema)
        return result.is_compatible

    def get_compatibility_level(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> str:
        """Get the compatibility level."""
        result = self.analyze(old_schema, new_schema)
        return result.level.value

    def analyze(
        self,
        old_schema: Any,
        new_schema: Any,
    ) -> CompatibilityResult:
        """Analyze schema compatibility.

        Args:
            old_schema: Previous schema.
            new_schema: New schema.

        Returns:
            Compatibility result with detailed analysis.
        """
        self._guard_execution()
        old = self._normalize_schema(old_schema)
        new = self._normalize_schema(new_schema)

        all_issues: list[SchemaChange] = []
        all_compatible = True
        rules_checked: list[str] = []

        for index, rule in enumerate(self._rules):
            self._guard_execution()
            try:
                name = rule.name
                if not isinstance(name, str) or not name:
                    raise TypeError("A compatibility rule requires a name")
                is_compatible, issues = rule.check(old, new)
                if type(is_compatible) is not bool or not isinstance(issues, list) or any(
                    not isinstance(issue, SchemaChange) for issue in issues
                ):
                    raise TypeError("A compatibility rule returned an invalid verdict")
                rules_checked.append(name)
                all_issues.extend(issues)

                if not is_compatible:
                    all_compatible = False

            except Exception as e:
                raise CompatibilityAnalysisError(index, type(e).__name__) from None
            self._guard_execution()

        # Determine compatibility level
        if self._strict and all_issues:
            all_compatible = False
            level = CompatibilityLevel.NONE
        else:
            level = self._determine_level(old, new, all_issues, all_compatible)

        return CompatibilityResult(
            is_compatible=all_compatible,
            level=level,
            issues=all_issues,
            rules_checked=rules_checked,
        )

    def _normalize_schema(self, schema: Any) -> dict[str, Any]:
        """Normalize schema to standard format."""
        if isinstance(schema, dict):
            return schema

        if hasattr(schema, "names") and hasattr(schema, "__getitem__"):
            return {name: schema[name] for name in schema.names()}

        if isinstance(schema, pl.DataFrame):
            return dict(schema.schema)

        if isinstance(schema, pl.LazyFrame):
            return dict(schema.collect_schema())

        raise TypeError(f"Cannot normalize schema of type {type(schema)}")

    def _determine_level(
        self,
        old_schema: dict[str, Any],
        new_schema: dict[str, Any],
        issues: list[SchemaChange],
        compatible: bool,
    ) -> CompatibilityLevel:
        """Determine the compatibility level."""
        if compatible and not issues:
            # Check for additions (forward compatible)
            added = set(new_schema.keys()) - set(old_schema.keys())
            if added:
                return CompatibilityLevel.BACKWARD
            return CompatibilityLevel.FULL

        if not compatible:
            # Check if it's at least forward compatible
            removed = set(old_schema.keys()) - set(new_schema.keys())
            type_breaks = any(
                i.change_type == ChangeType.TYPE_CHANGED and i.breaking
                for i in issues
            )

            if not removed and not type_breaks:
                return CompatibilityLevel.FORWARD

            return CompatibilityLevel.NONE

        return CompatibilityLevel.FULL

    def add_rule(self, rule: CompatibilityRule) -> None:
        """Add a compatibility rule.

        Args:
            rule: Rule to add.
        """
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name.

        Args:
            rule_name: Name of the rule to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                return True
        return False
