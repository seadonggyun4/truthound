"""Base classes for referential integrity validators.

This module provides extensible base classes for implementing various
referential integrity checks between tables.

Performance Optimization:
    For large tables, use sample-based validation to reduce memory and time:

    validator = ForeignKeyValidator(
        child_table="orders",
        child_columns=["customer_id"],
        parent_table="customers",
        parent_columns=["id"],
        sample_size=100000,  # Validate on sample
        sample_seed=42,      # Reproducible sampling
    )
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


# Default sampling configuration
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_SAMPLE_SEED = 42


@dataclass
class ForeignKeyRelation:
    """Defines a foreign key relationship between tables.

    Attributes:
        child_table: The table containing the foreign key
        child_columns: Column(s) in child table referencing parent
        parent_table: The referenced table
        parent_columns: Column(s) in parent table being referenced
        relation_name: Optional name for this relationship
        on_delete: Expected behavior on delete (CASCADE, SET NULL, RESTRICT, etc.)
        on_update: Expected behavior on update
    """

    child_table: str
    child_columns: list[str]
    parent_table: str
    parent_columns: list[str]
    relation_name: str | None = None
    on_delete: str | None = None
    on_update: str | None = None

    def __post_init__(self) -> None:
        """Validate the relationship definition."""
        if len(self.child_columns) != len(self.parent_columns):
            raise ValueError(
                f"Column count mismatch: child has {len(self.child_columns)} columns, "
                f"parent has {len(self.parent_columns)} columns"
            )

    @property
    def name(self) -> str:
        """Get a descriptive name for this relation."""
        if self.relation_name:
            return self.relation_name
        return f"{self.child_table}({','.join(self.child_columns)}) -> {self.parent_table}({','.join(self.parent_columns)})"


@dataclass
class TableNode:
    """Represents a table in a dependency graph.

    Used for detecting circular references and analyzing table relationships.
    """

    name: str
    references: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)


class ReferentialValidator(Validator):
    """Base class for referential integrity validators.

    Referential validators check relationships between multiple tables,
    ensuring data consistency across table boundaries.

    Performance:
        For large tables, use sample_size parameter to validate on a sample:
        - Reduces memory usage and validation time
        - Provides statistical estimate of violation rate
        - Reproducible with sample_seed parameter

    Subclasses should implement:
        - validate_reference(): Check the specific referential constraint
    """

    category = "referential"

    def __init__(
        self,
        tables: dict[str, pl.LazyFrame] | None = None,
        sample_size: int | None = None,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        **kwargs: Any,
    ):
        """Initialize referential validator.

        Args:
            tables: Dictionary mapping table names to LazyFrames
            sample_size: If set, validate on a sample of this size (for large tables)
            sample_seed: Random seed for reproducible sampling
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self._tables = tables or {}
        self._sample_size = sample_size
        self._sample_seed = sample_seed

    def register_table(self, name: str, lf: pl.LazyFrame) -> None:
        """Register a table for validation.

        Args:
            name: Table identifier
            lf: LazyFrame containing table data
        """
        self._tables[name] = lf

    def get_table(self, name: str) -> pl.LazyFrame | None:
        """Get a registered table by name.

        Args:
            name: Table identifier

        Returns:
            LazyFrame if found, None otherwise
        """
        return self._tables.get(name)

    def _get_column_values(
        self, lf: pl.LazyFrame, columns: list[str], distinct: bool = True
    ) -> pl.DataFrame:
        """Extract column values from a table.

        Args:
            lf: Source LazyFrame
            columns: Columns to extract
            distinct: Whether to return only distinct values

        Returns:
            DataFrame with extracted values
        """
        expr = [pl.col(c) for c in columns]
        result = lf.select(expr)
        if distinct:
            result = result.unique()
        return result.collect()

    def _sample_lazyframe(
        self, lf: pl.LazyFrame, sample_size: int | None = None
    ) -> tuple[pl.DataFrame, int, bool]:
        """Sample a LazyFrame if needed for large dataset handling.

        Args:
            lf: LazyFrame to sample
            sample_size: Max rows to sample (uses instance default if None)

        Returns:
            Tuple of (sampled_df, original_count, was_sampled)
        """
        effective_sample_size = sample_size or self._sample_size

        # Get total count
        total_count = lf.select(pl.len()).collect().item()

        if effective_sample_size is None or total_count <= effective_sample_size:
            # No sampling needed
            return lf.collect(), total_count, False

        # Sample the data
        df = lf.collect()
        sampled = df.sample(n=effective_sample_size, seed=self._sample_seed)
        return sampled, total_count, True

    def _find_orphans(
        self,
        child_lf: pl.LazyFrame,
        child_cols: list[str],
        parent_lf: pl.LazyFrame,
        parent_cols: list[str],
        sample_size: int | None = None,
    ) -> tuple[pl.DataFrame, int, bool]:
        """Find orphan records in child table.

        Args:
            child_lf: Child table LazyFrame
            child_cols: Foreign key columns in child
            parent_lf: Parent table LazyFrame
            parent_cols: Primary key columns in parent
            sample_size: Optional sample size override

        Returns:
            Tuple of (orphan_df, original_child_count, was_sampled)
        """
        # Get parent keys (always get all unique keys - they're typically smaller)
        parent_keys = parent_lf.select([pl.col(c) for c in parent_cols]).unique()

        # Rename parent columns for join
        rename_map = {p: f"_parent_{p}" for p in parent_cols}
        parent_keys = parent_keys.rename(rename_map)

        # Sample child table if needed
        child_df, original_count, was_sampled = self._sample_lazyframe(
            child_lf.select([pl.col(c) for c in child_cols]),
            sample_size,
        )

        # Left join and filter for nulls (orphans)
        parent_df = parent_keys.collect()

        # Perform anti-join to find orphans
        orphans = child_df.join(
            parent_df,
            left_on=child_cols,
            right_on=[f"_parent_{p}" for p in parent_cols],
            how="anti",
        )

        return orphans, original_count, was_sampled

    def _calculate_severity(self, violation_ratio: float) -> Severity:
        """Calculate severity based on violation ratio.

        Args:
            violation_ratio: Ratio of violating records

        Returns:
            Appropriate severity level
        """
        if violation_ratio < 0.001:
            return Severity.LOW
        elif violation_ratio < 0.01:
            return Severity.MEDIUM
        elif violation_ratio < 0.05:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    @abstractmethod
    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Validate a specific referential constraint.

        Args:
            relation: The foreign key relation to validate

        Returns:
            List of validation issues found
        """
        pass


class MultiTableValidator(Validator):
    """Base class for validators that operate on multiple tables.

    Provides utilities for managing table collections and cross-table operations.
    """

    category = "referential"

    def __init__(
        self,
        tables: dict[str, pl.LazyFrame] | None = None,
        relations: list[ForeignKeyRelation] | None = None,
        **kwargs: Any,
    ):
        """Initialize multi-table validator.

        Args:
            tables: Dictionary mapping table names to LazyFrames
            relations: List of foreign key relationships
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self._tables = tables or {}
        self._relations = relations or []

    def add_relation(self, relation: ForeignKeyRelation) -> None:
        """Add a foreign key relation.

        Args:
            relation: The relation to add
        """
        self._relations.append(relation)

    def build_dependency_graph(self) -> dict[str, TableNode]:
        """Build a dependency graph from registered relations.

        Returns:
            Dictionary mapping table names to TableNode objects
        """
        graph: dict[str, TableNode] = {}

        # Initialize nodes for all tables
        for table_name in self._tables:
            graph[table_name] = TableNode(name=table_name)

        # Add edges from relations
        for relation in self._relations:
            child = relation.child_table
            parent = relation.parent_table

            if child not in graph:
                graph[child] = TableNode(name=child)
            if parent not in graph:
                graph[parent] = TableNode(name=parent)

            if parent not in graph[child].references:
                graph[child].references.append(parent)
            if child not in graph[parent].referenced_by:
                graph[parent].referenced_by.append(child)

        return graph

    def find_cycles(self) -> list[list[str]]:
        """Detect circular references in the dependency graph.

        Uses DFS to find all cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of table names
        """
        graph = self.build_dependency_graph()
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph[node].references:
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def get_table_order(self) -> list[str] | None:
        """Get topological order of tables (for safe deletion/insertion).

        Returns:
            List of table names in topological order, or None if cycles exist
        """
        graph = self.build_dependency_graph()
        in_degree: dict[str, int] = {node: 0 for node in graph}

        for node in graph:
            for ref in graph[node].references:
                if ref in in_degree:
                    in_degree[node] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for other in graph:
                if node in graph[other].references:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        if len(result) != len(graph):
            return None  # Cycle detected

        return result
