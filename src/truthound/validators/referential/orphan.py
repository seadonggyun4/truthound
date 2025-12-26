"""Orphan record detection validators.

This module provides validators for detecting orphan records across tables,
which are records that reference non-existent parent records or have
become disconnected from the main data structure.
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
    DEFAULT_SAMPLE_SEED,
)


@register_validator
class OrphanRecordValidator(ReferentialValidator):
    """Detects orphan records in child tables.

    Orphan records are child records whose foreign key references
    a non-existent parent record. This often indicates:
    - Improper delete operations without cascade
    - Data migration issues
    - Concurrent modification problems

    Example:
        validator = OrphanRecordValidator(
            child_table="order_items",
            child_columns=["order_id"],
            parent_table="orders",
            parent_columns=["id"],
        )
    """

    name = "orphan_record"

    def __init__(
        self,
        child_table: str,
        child_columns: list[str] | str,
        parent_table: str,
        parent_columns: list[str] | str,
        tables: dict[str, pl.LazyFrame] | None = None,
        include_null_fk: bool = False,
        max_sample_display: int = 10,
        sample_size: int | None = None,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        **kwargs: Any,
    ):
        """Initialize orphan record validator.

        Args:
            child_table: Name of child table
            child_columns: FK column(s) in child table
            parent_table: Name of parent table
            parent_columns: PK column(s) in parent table
            tables: Pre-registered tables
            include_null_fk: Whether to count NULL FKs as orphans
            max_sample_display: Number of sample orphans to include in details
            sample_size: If set, validate on a sample of this size (for large tables)
            sample_seed: Random seed for reproducible sampling
            **kwargs: Additional config
        """
        super().__init__(tables=tables, sample_size=sample_size, sample_seed=sample_seed, **kwargs)

        if isinstance(child_columns, str):
            child_columns = [child_columns]
        if isinstance(parent_columns, str):
            parent_columns = [parent_columns]

        self.relation = ForeignKeyRelation(
            child_table=child_table,
            child_columns=child_columns,
            parent_table=parent_table,
            parent_columns=parent_columns,
        )
        self.include_null_fk = include_null_fk
        self.max_sample_display = max_sample_display

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Find orphan records.

        Args:
            relation: The FK relation to check

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        child_lf = self.get_table(relation.child_table)
        parent_lf = self.get_table(relation.parent_table)

        if child_lf is None or parent_lf is None:
            return issues

        # Get total rows first
        total_rows = child_lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Find orphans (with sampling support)
        orphans, sample_count, was_sampled = self._find_orphans(
            child_lf,
            relation.child_columns,
            parent_lf,
            relation.parent_columns,
        )

        # Handle NULL FKs
        if not self.include_null_fk:
            non_null_filter = pl.lit(True)
            for col in relation.child_columns:
                non_null_filter = non_null_filter & pl.col(col).is_not_null()
            orphans = orphans.filter(non_null_filter)

        orphan_count = len(orphans)

        if orphan_count > 0:
            # Calculate ratio based on sample
            sample_orphan_ratio = orphan_count / sample_count if sample_count > 0 else 0

            # Estimate total violations if sampled
            if was_sampled:
                estimated_total_orphans = int(sample_orphan_ratio * total_rows)
                orphan_ratio = sample_orphan_ratio
            else:
                estimated_total_orphans = orphan_count
                orphan_ratio = orphan_count / total_rows if total_rows > 0 else 0

            # Get sample orphan records
            sample_orphans = orphans.head(self.max_sample_display).to_dicts()

            # Build details message
            if was_sampled:
                details = (
                    f"Found {orphan_count} orphan records in sample of {sample_count:,} rows "
                    f"({sample_orphan_ratio:.2%}). Estimated {estimated_total_orphans:,} total "
                    f"orphans in '{relation.child_table}' ({total_rows:,} rows) with no matching "
                    f"parent in '{relation.parent_table}'. Samples: {sample_orphans}"
                )
            else:
                details = (
                    f"Found {orphan_count} orphan records ({orphan_ratio:.2%}) in "
                    f"'{relation.child_table}' with no matching parent in "
                    f"'{relation.parent_table}'. Samples: {sample_orphans}"
                )

            issues.append(
                ValidationIssue(
                    column=", ".join(relation.child_columns),
                    issue_type="orphan_record_detected",
                    count=estimated_total_orphans if was_sampled else orphan_count,
                    severity=self._calculate_severity(orphan_ratio),
                    details=details,
                    expected="All child records should have valid parent references",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for orphan records.

        Args:
            lf: Child table LazyFrame

        Returns:
            List of validation issues
        """
        if self.relation.child_table not in self._tables:
            self.register_table(self.relation.child_table, lf)

        return self.validate_reference(self.relation)


@register_validator
class MultiTableOrphanValidator(MultiTableValidator):
    """Detects orphan records across multiple table relationships.

    Analyzes all registered relationships to find orphan records
    throughout the entire schema.

    Example:
        validator = MultiTableOrphanValidator(
            relations=[
                ForeignKeyRelation("orders", ["customer_id"], "customers", ["id"]),
                ForeignKeyRelation("order_items", ["order_id"], "orders", ["id"]),
                ForeignKeyRelation("order_items", ["product_id"], "products", ["id"]),
            ]
        )
    """

    name = "multi_table_orphan"

    def __init__(
        self,
        tables: dict[str, pl.LazyFrame] | None = None,
        relations: list[ForeignKeyRelation] | None = None,
        include_null_fk: bool = False,
        **kwargs: Any,
    ):
        """Initialize multi-table orphan validator.

        Args:
            tables: Pre-registered tables
            relations: List of FK relationships
            include_null_fk: Whether to count NULL FKs as orphans
            **kwargs: Additional config
        """
        super().__init__(tables=tables, relations=relations, **kwargs)
        self.include_null_fk = include_null_fk

    def _find_relation_orphans(
        self, relation: ForeignKeyRelation
    ) -> tuple[int, float, list[dict]]:
        """Find orphans for a single relation.

        Args:
            relation: The FK relation to check

        Returns:
            Tuple of (orphan_count, orphan_ratio, sample_orphans)
        """
        child_lf = self._tables.get(relation.child_table)
        parent_lf = self._tables.get(relation.parent_table)

        if child_lf is None or parent_lf is None:
            return 0, 0.0, []

        # Get parent keys
        parent_df = parent_lf.select(
            [pl.col(c) for c in relation.parent_columns]
        ).unique().collect()

        # Rename for join
        rename_map = {p: f"_p_{p}" for p in relation.parent_columns}
        parent_df = parent_df.rename(rename_map)

        # Get child records
        child_df = child_lf.collect()
        total_rows = len(child_df)

        if total_rows == 0:
            return 0, 0.0, []

        # Anti-join to find orphans
        orphans = child_df.join(
            parent_df,
            left_on=relation.child_columns,
            right_on=[f"_p_{p}" for p in relation.parent_columns],
            how="anti",
        )

        # Filter nulls if needed
        if not self.include_null_fk:
            non_null_filter = pl.lit(True)
            for col in relation.child_columns:
                non_null_filter = non_null_filter & pl.col(col).is_not_null()
            orphans = orphans.filter(non_null_filter)

        orphan_count = len(orphans)
        orphan_ratio = orphan_count / total_rows if total_rows > 0 else 0.0
        samples = orphans.head(5).to_dicts()

        return orphan_count, orphan_ratio, samples

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate all relationships for orphans.

        Args:
            lf: Primary table (optional, uses registered tables)

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        total_orphans = 0
        affected_relations: list[str] = []

        for relation in self._relations:
            orphan_count, orphan_ratio, samples = self._find_relation_orphans(relation)

            if orphan_count > 0:
                total_orphans += orphan_count
                affected_relations.append(relation.name)

                issues.append(
                    ValidationIssue(
                        column=", ".join(relation.child_columns),
                        issue_type="orphan_in_relation",
                        count=orphan_count,
                        severity=self._calculate_severity(orphan_ratio),
                        details=(
                            f"Relation '{relation.name}': {orphan_count} orphans "
                            f"({orphan_ratio:.2%}). Samples: {samples}"
                        ),
                        expected="No orphan records in any relationship",
                    )
                )

        # Summary issue if multiple relations affected
        if len(affected_relations) > 1:
            issues.append(
                ValidationIssue(
                    column="(multiple)",
                    issue_type="multi_relation_orphans",
                    count=total_orphans,
                    severity=Severity.HIGH,
                    details=(
                        f"Total {total_orphans} orphan records across "
                        f"{len(affected_relations)} relationships: {affected_relations}"
                    ),
                    expected="Referential integrity across all tables",
                )
            )

        return issues

    def _calculate_severity(self, orphan_ratio: float) -> Severity:
        """Calculate severity based on orphan ratio."""
        if orphan_ratio < 0.001:
            return Severity.LOW
        elif orphan_ratio < 0.01:
            return Severity.MEDIUM
        elif orphan_ratio < 0.05:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class DanglingReferenceValidator(ReferentialValidator):
    """Detects dangling references in reverse direction.

    While OrphanRecordValidator checks child->parent integrity,
    this validator checks for parent records that have no children
    when children are expected (detecting potential data loss).

    Example:
        # Find customers with no orders (might indicate lost data)
        validator = DanglingReferenceValidator(
            parent_table="customers",
            parent_columns=["id"],
            child_table="orders",
            child_columns=["customer_id"],
            min_expected_children=1,
        )
    """

    name = "dangling_reference"

    def __init__(
        self,
        parent_table: str,
        parent_columns: list[str] | str,
        child_table: str,
        child_columns: list[str] | str,
        tables: dict[str, pl.LazyFrame] | None = None,
        min_expected_children: int = 0,
        max_dangling_ratio: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize dangling reference validator.

        Args:
            parent_table: Name of parent table
            parent_columns: PK column(s) in parent
            child_table: Name of child table
            child_columns: FK column(s) in child
            tables: Pre-registered tables
            min_expected_children: Minimum children expected per parent
            max_dangling_ratio: Maximum acceptable ratio of dangling parents
            **kwargs: Additional config
        """
        super().__init__(tables=tables, **kwargs)

        if isinstance(parent_columns, str):
            parent_columns = [parent_columns]
        if isinstance(child_columns, str):
            child_columns = [child_columns]

        self.relation = ForeignKeyRelation(
            child_table=child_table,
            child_columns=child_columns,
            parent_table=parent_table,
            parent_columns=parent_columns,
        )
        self.min_expected_children = min_expected_children
        self.max_dangling_ratio = max_dangling_ratio

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Find dangling parent records.

        Args:
            relation: The FK relation to check

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        parent_lf = self.get_table(relation.parent_table)
        child_lf = self.get_table(relation.child_table)

        if parent_lf is None or child_lf is None:
            return issues

        # Get unique child FK values
        child_fks = (
            child_lf.select([pl.col(c) for c in relation.child_columns])
            .unique()
            .collect()
        )

        # Rename for join
        rename_map = {c: f"_c_{c}" for c in relation.child_columns}
        child_fks = child_fks.rename(rename_map)

        # Get parent records
        parent_df = parent_lf.collect()
        total_parents = len(parent_df)

        if total_parents == 0:
            return issues

        # Anti-join to find parents with no children
        dangling = parent_df.join(
            child_fks,
            left_on=relation.parent_columns,
            right_on=[f"_c_{c}" for c in relation.child_columns],
            how="anti",
        )

        dangling_count = len(dangling)
        dangling_ratio = dangling_count / total_parents

        if self.min_expected_children > 0 and dangling_count > 0:
            issues.append(
                ValidationIssue(
                    column=", ".join(relation.parent_columns),
                    issue_type="dangling_parent_detected",
                    count=dangling_count,
                    severity=(
                        Severity.HIGH
                        if dangling_ratio > self.max_dangling_ratio
                        else Severity.MEDIUM
                    ),
                    details=(
                        f"Found {dangling_count} records ({dangling_ratio:.2%}) in "
                        f"'{relation.parent_table}' with no children in "
                        f"'{relation.child_table}'. Expected at least "
                        f"{self.min_expected_children} child(ren) per parent."
                    ),
                    expected=f"Each parent should have >= {self.min_expected_children} children",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for dangling references.

        Args:
            lf: Parent table LazyFrame

        Returns:
            List of validation issues
        """
        if self.relation.parent_table not in self._tables:
            self.register_table(self.relation.parent_table, lf)

        return self.validate_reference(self.relation)
