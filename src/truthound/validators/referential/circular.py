"""Circular reference detection validators.

This module provides validators for detecting circular references in data,
which can occur in self-referential tables (hierarchies) or across
multiple related tables.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.referential.base import (
    ForeignKeyRelation,
    MultiTableValidator,
    ReferentialValidator,
)
from truthound.validators.optimization import GraphTraversalMixin


@register_validator
class CircularReferenceValidator(MultiTableValidator):
    """Detects circular references in table relationships.

    Circular references occur when:
    - A table references itself in a cycle (A -> A)
    - Multiple tables form a reference cycle (A -> B -> C -> A)

    This validator analyzes the schema's dependency graph to find all cycles.

    Example:
        validator = CircularReferenceValidator(
            relations=[
                ForeignKeyRelation("orders", ["customer_id"], "customers", ["id"]),
                ForeignKeyRelation("customers", ["primary_order_id"], "orders", ["id"]),
            ]
        )
    """

    name = "circular_reference"

    def __init__(
        self,
        tables: dict[str, pl.LazyFrame] | None = None,
        relations: list[ForeignKeyRelation] | None = None,
        allow_self_reference: bool = True,
        max_cycle_length: int | None = None,
        **kwargs: Any,
    ):
        """Initialize circular reference validator.

        Args:
            tables: Pre-registered tables
            relations: List of FK relationships
            allow_self_reference: Whether to allow A->A references
            max_cycle_length: Maximum acceptable cycle length (None = no cycles)
            **kwargs: Additional config
        """
        super().__init__(tables=tables, relations=relations, **kwargs)
        self.allow_self_reference = allow_self_reference
        self.max_cycle_length = max_cycle_length

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Detect circular references in schema.

        Args:
            lf: Not used (analyzes registered relations)

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Find all cycles
        cycles = self.find_cycles()

        for cycle in cycles:
            cycle_length = len(cycle) - 1  # Subtract 1 because start=end

            # Check if it's a self-reference
            is_self_ref = cycle_length == 1

            if is_self_ref and self.allow_self_reference:
                continue

            # Check against max allowed length
            if self.max_cycle_length is not None and cycle_length <= self.max_cycle_length:
                severity = Severity.LOW
            else:
                severity = Severity.HIGH if cycle_length > 2 else Severity.MEDIUM

            cycle_str = " -> ".join(cycle)
            issues.append(
                ValidationIssue(
                    column="(schema)",
                    issue_type="circular_reference_detected",
                    count=cycle_length,
                    severity=severity,
                    details=(
                        f"Circular reference detected: {cycle_str}. "
                        f"Cycle length: {cycle_length}. "
                        "This may cause issues with cascade operations, "
                        "queries, and data integrity."
                    ),
                    expected="Acyclic table relationships" if self.max_cycle_length is None
                    else f"Cycle length <= {self.max_cycle_length}",
                )
            )

        return issues


@register_validator
class HierarchyCircularValidator(ReferentialValidator):
    """Detects circular references in hierarchical/self-referential data.

    Used for tree structures like:
    - Organization hierarchies (employee -> manager)
    - Category trees (category -> parent_category)
    - Bill of materials (part -> component_of)

    Unlike schema-level CircularReferenceValidator, this checks actual data
    for cycles within a single table's self-reference.

    Example:
        validator = HierarchyCircularValidator(
            table="employees",
            id_column="employee_id",
            parent_column="manager_id",
        )
    """

    name = "hierarchy_circular"

    def __init__(
        self,
        table: str,
        id_column: str,
        parent_column: str,
        tables: dict[str, pl.LazyFrame] | None = None,
        max_depth: int = 100,
        **kwargs: Any,
    ):
        """Initialize hierarchy circular validator.

        Args:
            table: Table name
            id_column: Primary key column
            parent_column: Self-referencing FK column
            tables: Pre-registered tables
            max_depth: Maximum traversal depth (cycle detection limit)
            **kwargs: Additional config
        """
        super().__init__(tables=tables, **kwargs)
        self.table_name = table
        self.id_column = id_column
        self.parent_column = parent_column
        self.max_depth = max_depth
        self.relation = ForeignKeyRelation(
            child_table=table,
            child_columns=[parent_column],
            parent_table=table,
            parent_columns=[id_column],
        )

    def _find_cycles_in_data(
        self, df: pl.DataFrame
    ) -> list[tuple[list[Any], int]]:
        """Find cycles in hierarchical data.

        Args:
            df: DataFrame with hierarchy data

        Returns:
            List of (cycle_path, starting_id) tuples
        """
        cycles: list[tuple[list[Any], int]] = []

        # Build adjacency map: id -> parent_id
        id_to_parent: dict[Any, Any] = {}
        for row in df.select([self.id_column, self.parent_column]).iter_rows():
            id_val, parent_val = row
            if parent_val is not None:
                id_to_parent[id_val] = parent_val

        # Check each node for cycles
        checked: set[Any] = set()

        for start_id in id_to_parent:
            if start_id in checked:
                continue

            visited: set[Any] = set()
            path: list[Any] = []
            current = start_id

            while current is not None and current not in visited:
                if len(path) > self.max_depth:
                    break

                visited.add(current)
                path.append(current)
                current = id_to_parent.get(current)

                if current == start_id:
                    # Found a cycle back to start
                    path.append(current)
                    cycles.append((path.copy(), start_id))
                    break

            checked.update(visited)

        return cycles

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Find circular references in hierarchy data.

        Args:
            relation: The self-referential relation

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        table_lf = self.get_table(self.table_name)
        if table_lf is None:
            return issues

        df = table_lf.collect()

        if len(df) == 0:
            return issues

        cycles = self._find_cycles_in_data(df)

        if cycles:
            # Group by cycle length for reporting
            by_length: dict[int, list[list[Any]]] = {}
            for path, _ in cycles:
                length = len(path) - 1
                if length not in by_length:
                    by_length[length] = []
                by_length[length].append(path)

            for length, paths in by_length.items():
                sample_paths = paths[:3]
                sample_str = "; ".join([" -> ".join(str(x) for x in p) for p in sample_paths])

                issues.append(
                    ValidationIssue(
                        column=self.parent_column,
                        issue_type="hierarchy_cycle_detected",
                        count=len(paths),
                        severity=Severity.CRITICAL,
                        details=(
                            f"Found {len(paths)} circular reference(s) of length {length} "
                            f"in '{self.table_name}'. Sample cycles: {sample_str}"
                        ),
                        expected="Acyclic hierarchy (tree structure)",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate hierarchy for circular references.

        Args:
            lf: Table LazyFrame

        Returns:
            List of validation issues
        """
        self.register_table(self.table_name, lf)
        return self.validate_reference(self.relation)


@register_validator
class HierarchyDepthValidator(ReferentialValidator):
    """Validates hierarchy depth constraints.

    Ensures hierarchical data doesn't exceed a maximum depth,
    which can indicate data issues or design problems.

    Example:
        validator = HierarchyDepthValidator(
            table="categories",
            id_column="id",
            parent_column="parent_id",
            max_depth=5,
        )
    """

    name = "hierarchy_depth"

    def __init__(
        self,
        table: str,
        id_column: str,
        parent_column: str,
        max_depth: int = 10,
        tables: dict[str, pl.LazyFrame] | None = None,
        **kwargs: Any,
    ):
        """Initialize hierarchy depth validator.

        Args:
            table: Table name
            id_column: Primary key column
            parent_column: Self-referencing FK column
            max_depth: Maximum allowed hierarchy depth
            tables: Pre-registered tables
            **kwargs: Additional config
        """
        super().__init__(tables=tables, **kwargs)
        self.table_name = table
        self.id_column = id_column
        self.parent_column = parent_column
        self.max_depth = max_depth
        self.relation = ForeignKeyRelation(
            child_table=table,
            child_columns=[parent_column],
            parent_table=table,
            parent_columns=[id_column],
        )

    def _calculate_depths(self, df: pl.DataFrame) -> dict[Any, int]:
        """Calculate depth of each node in hierarchy.

        Args:
            df: DataFrame with hierarchy data

        Returns:
            Dictionary mapping node ID to its depth
        """
        # Build adjacency map
        id_to_parent: dict[Any, Any] = {}
        for row in df.select([self.id_column, self.parent_column]).iter_rows():
            id_val, parent_val = row
            id_to_parent[id_val] = parent_val

        # Calculate depths using memoization
        depths: dict[Any, int] = {}

        def get_depth(node_id: Any, visited: set[Any]) -> int:
            if node_id in depths:
                return depths[node_id]

            if node_id in visited:
                return -1  # Cycle detected

            parent = id_to_parent.get(node_id)
            if parent is None:
                depths[node_id] = 0
                return 0

            visited.add(node_id)
            parent_depth = get_depth(parent, visited)

            if parent_depth == -1:
                depths[node_id] = -1  # Part of cycle
                return -1

            depths[node_id] = parent_depth + 1
            return depths[node_id]

        for node_id in id_to_parent:
            if node_id not in depths:
                get_depth(node_id, set())

        return depths

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Validate hierarchy depth constraints.

        Args:
            relation: The self-referential relation

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        table_lf = self.get_table(self.table_name)
        if table_lf is None:
            return issues

        df = table_lf.collect()

        if len(df) == 0:
            return issues

        depths = self._calculate_depths(df)

        # Find nodes exceeding max depth
        exceeding = {
            node_id: depth
            for node_id, depth in depths.items()
            if depth > self.max_depth
        }

        if exceeding:
            max_actual_depth = max(exceeding.values())
            sample_nodes = list(exceeding.items())[:5]

            issues.append(
                ValidationIssue(
                    column=self.parent_column,
                    issue_type="hierarchy_depth_exceeded",
                    count=len(exceeding),
                    severity=Severity.HIGH if max_actual_depth > self.max_depth * 2 else Severity.MEDIUM,
                    details=(
                        f"Found {len(exceeding)} nodes exceeding max depth of "
                        f"{self.max_depth}. Maximum depth found: {max_actual_depth}. "
                        f"Sample nodes: {sample_nodes}"
                    ),
                    expected=f"Hierarchy depth <= {self.max_depth}",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate hierarchy depth.

        Args:
            lf: Table LazyFrame

        Returns:
            List of validation issues
        """
        self.register_table(self.table_name, lf)
        return self.validate_reference(self.relation)


@register_validator
class OptimizedHierarchyCircularValidator(ReferentialValidator, GraphTraversalMixin):
    """Optimized hierarchy circular validator using Tarjan's SCC algorithm.

    Uses GraphTraversalMixin for stack-safe, efficient cycle detection:
    - Iterative DFS (no Python recursion limit)
    - Tarjan's SCC algorithm O(V+E)
    - Handles graphs with millions of nodes

    Example:
        validator = OptimizedHierarchyCircularValidator(
            table="employees",
            id_column="employee_id",
            parent_column="manager_id",
        )
    """

    name = "optimized_hierarchy_circular"

    def __init__(
        self,
        table: str,
        id_column: str,
        parent_column: str,
        tables: dict[str, pl.LazyFrame] | None = None,
        max_depth: int = 1000,
        **kwargs: Any,
    ):
        """Initialize optimized hierarchy circular validator.

        Args:
            table: Table name
            id_column: Primary key column
            parent_column: Self-referencing FK column
            tables: Pre-registered tables
            max_depth: Maximum traversal depth (cycle detection limit)
            **kwargs: Additional config
        """
        super().__init__(tables=tables, **kwargs)
        self.table_name = table
        self.id_column = id_column
        self.parent_column = parent_column
        self.max_depth = max_depth
        self.relation = ForeignKeyRelation(
            child_table=table,
            child_columns=[parent_column],
            parent_table=table,
            parent_columns=[id_column],
        )

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Find circular references using optimized Tarjan's algorithm.

        Args:
            relation: The self-referential relation

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        table_lf = self.get_table(self.table_name)
        if table_lf is None:
            return issues

        df = table_lf.collect()

        if len(df) == 0:
            return issues

        # Build child->parent mapping using mixin
        child_to_parent = self.build_child_to_parent(
            df, self.id_column, self.parent_column
        )

        # Use optimized hierarchy cycle detection
        cycles = self.find_hierarchy_cycles(child_to_parent, max_depth=self.max_depth)

        if cycles:
            # Group by cycle length for reporting
            by_length: dict[int, list[list[Any]]] = {}
            for cycle_info in cycles:
                length = cycle_info.length
                if length not in by_length:
                    by_length[length] = []
                by_length[length].append(cycle_info.nodes)

            for length, paths in by_length.items():
                sample_paths = paths[:3]
                sample_str = "; ".join([str(cycle_info) for cycle_info in sample_paths[:3]])

                issues.append(
                    ValidationIssue(
                        column=self.parent_column,
                        issue_type="optimized_hierarchy_cycle_detected",
                        count=len(paths),
                        severity=Severity.CRITICAL,
                        details=(
                            f"Found {len(paths)} circular reference(s) of length {length} "
                            f"in '{self.table_name}' using Tarjan's SCC. "
                            f"Sample cycles: {sample_str}"
                        ),
                        expected="Acyclic hierarchy (tree structure)",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate hierarchy for circular references.

        Args:
            lf: Table LazyFrame

        Returns:
            List of validation issues
        """
        self.register_table(self.table_name, lf)
        return self.validate_reference(self.relation)
