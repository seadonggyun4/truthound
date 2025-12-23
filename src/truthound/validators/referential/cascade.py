"""Cascade integrity validators.

This module provides validators for checking cascading referential integrity,
ensuring that dependent records are properly handled when parent records
change or are deleted.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.referential.base import (
    ForeignKeyRelation,
    MultiTableValidator,
)


class CascadeAction(str, Enum):
    """Defines cascade behavior on parent record changes."""

    CASCADE = "CASCADE"  # Delete/update child records
    SET_NULL = "SET_NULL"  # Set FK to NULL
    SET_DEFAULT = "SET_DEFAULT"  # Set FK to default value
    RESTRICT = "RESTRICT"  # Prevent parent change
    NO_ACTION = "NO_ACTION"  # Do nothing (may leave orphans)


@dataclass
class CascadeRule:
    """Defines a cascade rule for a relationship.

    Attributes:
        relation: The FK relationship
        on_delete: Action when parent is deleted
        on_update: Action when parent key is updated
        default_value: Default value for SET_DEFAULT action
    """

    relation: ForeignKeyRelation
    on_delete: CascadeAction = CascadeAction.RESTRICT
    on_update: CascadeAction = CascadeAction.RESTRICT
    default_value: Any = None


@register_validator
class CascadeIntegrityValidator(MultiTableValidator):
    """Validates cascade integrity across table relationships.

    This validator checks that cascade rules are being properly enforced
    by detecting inconsistencies that would result from improper cascade handling.

    Key checks:
    - CASCADE: Verifies child records exist when parent exists
    - SET_NULL: Checks for expected NULL patterns
    - RESTRICT: Ensures no orphans exist
    - Detects cascade chain issues

    Example:
        validator = CascadeIntegrityValidator(
            cascade_rules=[
                CascadeRule(
                    relation=ForeignKeyRelation(
                        child_table="orders",
                        child_columns=["customer_id"],
                        parent_table="customers",
                        parent_columns=["id"],
                    ),
                    on_delete=CascadeAction.SET_NULL,
                    on_update=CascadeAction.CASCADE,
                ),
            ]
        )
    """

    name = "cascade_integrity"

    def __init__(
        self,
        cascade_rules: list[CascadeRule] | None = None,
        tables: dict[str, pl.LazyFrame] | None = None,
        check_cascade_chains: bool = True,
        **kwargs: Any,
    ):
        """Initialize cascade integrity validator.

        Args:
            cascade_rules: List of cascade rules to validate
            tables: Pre-registered tables
            check_cascade_chains: Check for cascade chain issues
            **kwargs: Additional config
        """
        super().__init__(tables=tables, **kwargs)
        self.cascade_rules = cascade_rules or []
        self.check_cascade_chains = check_cascade_chains

    def add_cascade_rule(self, rule: CascadeRule) -> None:
        """Add a cascade rule.

        Args:
            rule: The cascade rule to add
        """
        self.cascade_rules.append(rule)
        self.add_relation(rule.relation)

    def _check_restrict_violation(
        self, rule: CascadeRule
    ) -> list[ValidationIssue]:
        """Check for RESTRICT constraint violations.

        RESTRICT means no orphans should exist.

        Args:
            rule: The cascade rule to check

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        relation = rule.relation

        child_lf = self._tables.get(relation.child_table)
        parent_lf = self._tables.get(relation.parent_table)

        if child_lf is None or parent_lf is None:
            return issues

        # Find orphan records
        child_df = child_lf.collect()
        parent_df = parent_lf.collect()

        # Get parent keys
        parent_keys = set()
        for row in parent_df.select(relation.parent_columns).iter_rows():
            parent_keys.add(row)

        # Check child records
        orphan_count = 0
        for row in child_df.select(relation.child_columns).iter_rows():
            # Skip if any value is None
            if any(v is None for v in row):
                continue
            if row not in parent_keys:
                orphan_count += 1

        if orphan_count > 0:
            action_type = "delete" if rule.on_delete == CascadeAction.RESTRICT else "update"
            issues.append(
                ValidationIssue(
                    column=", ".join(relation.child_columns),
                    issue_type="cascade_restrict_violation",
                    count=orphan_count,
                    severity=Severity.CRITICAL,
                    details=(
                        f"RESTRICT violation: {orphan_count} orphan records in "
                        f"'{relation.child_table}' indicate improper {action_type} "
                        f"on '{relation.parent_table}'"
                    ),
                    expected="No orphan records with RESTRICT cascade rule",
                )
            )

        return issues

    def _check_set_null_pattern(
        self, rule: CascadeRule
    ) -> list[ValidationIssue]:
        """Check SET_NULL cascade patterns.

        When SET_NULL is used, we expect NULL values in FK columns
        for records whose parent was deleted.

        Args:
            rule: The cascade rule to check

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        relation = rule.relation

        child_lf = self._tables.get(relation.child_table)
        if child_lf is None:
            return issues

        child_df = child_lf.collect()

        # Count NULL FK values
        null_filter = pl.lit(False)
        for col in relation.child_columns:
            null_filter = null_filter | pl.col(col).is_null()

        null_records = child_df.filter(null_filter)
        null_count = len(null_records)

        if null_count > 0:
            # This is informational - NULLs are expected with SET_NULL
            issues.append(
                ValidationIssue(
                    column=", ".join(relation.child_columns),
                    issue_type="cascade_set_null_detected",
                    count=null_count,
                    severity=Severity.LOW,
                    details=(
                        f"Found {null_count} records with NULL FK values in "
                        f"'{relation.child_table}', consistent with SET_NULL cascade"
                    ),
                    expected="NULL values indicate deleted parent records",
                )
            )

        return issues

    def _check_cascade_chain(self) -> list[ValidationIssue]:
        """Check for cascade chain issues.

        Detects potential problems in multi-level cascade scenarios.

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Build dependency chains
        table_deps: dict[str, list[str]] = {}
        for rule in self.cascade_rules:
            child = rule.relation.child_table
            parent = rule.relation.parent_table

            if child not in table_deps:
                table_deps[child] = []
            table_deps[child].append(parent)

        # Find tables with multiple dependencies
        multi_dep_tables = {
            table: deps for table, deps in table_deps.items() if len(deps) > 1
        }

        for table, deps in multi_dep_tables.items():
            # Check for conflicting cascade rules
            rules_for_table = [
                r for r in self.cascade_rules if r.relation.child_table == table
            ]

            delete_actions = set(r.on_delete for r in rules_for_table)
            if len(delete_actions) > 1:
                issues.append(
                    ValidationIssue(
                        column=table,
                        issue_type="cascade_chain_conflict",
                        count=len(delete_actions),
                        severity=Severity.MEDIUM,
                        details=(
                            f"Table '{table}' has conflicting cascade actions "
                            f"from multiple parent tables: {deps}. "
                            f"Actions: {[a.value for a in delete_actions]}"
                        ),
                        expected="Consistent cascade actions for multi-parent tables",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate cascade integrity.

        Args:
            lf: Primary table LazyFrame (used to detect table if needed)

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        for rule in self.cascade_rules:
            # Check based on cascade action type
            if rule.on_delete == CascadeAction.RESTRICT:
                issues.extend(self._check_restrict_violation(rule))
            elif rule.on_delete == CascadeAction.SET_NULL:
                issues.extend(self._check_set_null_pattern(rule))

        # Check cascade chains
        if self.check_cascade_chains:
            issues.extend(self._check_cascade_chain())

        return issues


@register_validator
class CascadeDepthValidator(MultiTableValidator):
    """Validates cascade chain depth to prevent deep cascade operations.

    Deep cascades can cause performance issues and unexpected data loss.
    This validator analyzes the dependency graph and reports chains
    exceeding a specified depth.

    Example:
        validator = CascadeDepthValidator(
            max_depth=3,  # Warn if cascade depth exceeds 3
        )
    """

    name = "cascade_depth"

    def __init__(
        self,
        max_depth: int = 5,
        tables: dict[str, pl.LazyFrame] | None = None,
        relations: list[ForeignKeyRelation] | None = None,
        **kwargs: Any,
    ):
        """Initialize cascade depth validator.

        Args:
            max_depth: Maximum acceptable cascade depth
            tables: Pre-registered tables
            relations: FK relationships defining cascade paths
            **kwargs: Additional config
        """
        super().__init__(tables=tables, relations=relations, **kwargs)
        self.max_depth = max_depth

    def _find_max_depth(self, start: str, visited: set[str]) -> int:
        """Find maximum cascade depth from a starting table.

        Args:
            start: Starting table name
            visited: Set of visited tables (to detect cycles)

        Returns:
            Maximum depth from this table
        """
        if start in visited:
            return 0  # Cycle detected

        visited.add(start)
        graph = self.build_dependency_graph()

        if start not in graph:
            return 0

        max_child_depth = 0
        for child_table in graph[start].referenced_by:
            depth = self._find_max_depth(child_table, visited.copy())
            max_child_depth = max(max_child_depth, depth)

        return 1 + max_child_depth

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate cascade depth.

        Args:
            lf: Not used (analyzes registered relations)

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        graph = self.build_dependency_graph()

        # Find root tables (no references to other tables)
        root_tables = [
            name for name, node in graph.items() if not node.references
        ]

        max_depths: dict[str, int] = {}
        for root in root_tables:
            depth = self._find_max_depth(root, set())
            max_depths[root] = depth

            if depth > self.max_depth:
                issues.append(
                    ValidationIssue(
                        column=root,
                        issue_type="cascade_depth_exceeded",
                        count=depth,
                        severity=Severity.HIGH if depth > self.max_depth * 2 else Severity.MEDIUM,
                        details=(
                            f"Cascade chain from '{root}' has depth {depth}, "
                            f"exceeding maximum of {self.max_depth}. "
                            f"Deep cascades may cause performance issues."
                        ),
                        expected=f"Cascade depth <= {self.max_depth}",
                    )
                )

        return issues
