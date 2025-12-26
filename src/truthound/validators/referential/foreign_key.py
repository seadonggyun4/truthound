"""Foreign key validation validators.

This module provides comprehensive foreign key constraint validation,
including basic FK checks, composite keys, and partial match detection.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.referential.base import (
    ForeignKeyRelation,
    ReferentialValidator,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SAMPLE_SEED,
)


@register_validator
class ForeignKeyValidator(ReferentialValidator):
    """Validates foreign key constraints between tables.

    This validator checks that all foreign key values in a child table
    exist in the referenced parent table. Supports composite keys and
    provides detailed violation reporting.

    Performance:
        For large tables, use sample_size to validate on a sample:

        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns=["customer_id"],
            parent_table="customers",
            parent_columns=["id"],
            sample_size=100000,  # Validate 100k sample rows
        )

    Example:
        # Simple FK validation
        validator = ForeignKeyValidator(
            child_table="orders",
            child_columns=["customer_id"],
            parent_table="customers",
            parent_columns=["id"],
        )

        # Composite FK validation
        validator = ForeignKeyValidator(
            child_table="order_items",
            child_columns=["order_id", "product_id"],
            parent_table="order_products",
            parent_columns=["order_id", "product_id"],
        )
    """

    name = "foreign_key"

    def __init__(
        self,
        child_table: str,
        child_columns: list[str] | str,
        parent_table: str,
        parent_columns: list[str] | str,
        tables: dict[str, pl.LazyFrame] | None = None,
        allow_null: bool = True,
        max_violations: int = 100,
        sample_size: int | None = None,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        **kwargs: Any,
    ):
        """Initialize foreign key validator.

        Args:
            child_table: Name of the table containing foreign key
            child_columns: Column(s) in child table
            parent_table: Name of the referenced table
            parent_columns: Column(s) in parent table
            tables: Pre-registered tables
            allow_null: Whether NULL values are allowed in FK columns
            max_violations: Maximum violations to report in details
            sample_size: If set, validate on a sample of this size (for large tables)
            sample_seed: Random seed for reproducible sampling
            **kwargs: Additional config
        """
        super().__init__(tables=tables, sample_size=sample_size, sample_seed=sample_seed, **kwargs)

        # Normalize to lists
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
        self.allow_null = allow_null
        self.max_violations = max_violations

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Validate the foreign key constraint.

        Args:
            relation: The FK relation to validate

        Returns:
            List of validation issues found
        """
        issues: list[ValidationIssue] = []

        child_lf = self.get_table(relation.child_table)
        parent_lf = self.get_table(relation.parent_table)

        if child_lf is None or parent_lf is None:
            return issues

        # Get child records
        child_cols = relation.child_columns
        parent_cols = relation.parent_columns

        # Get total rows first (before sampling)
        total_rows = child_lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Check for nulls if not allowed
        if not self.allow_null:
            null_filter = pl.lit(False)
            for col in child_cols:
                null_filter = null_filter | pl.col(col).is_null()
            null_count = child_lf.filter(null_filter).select(pl.len()).collect().item()
            if null_count > 0:
                issues.append(
                    ValidationIssue(
                        column=", ".join(child_cols),
                        issue_type="fk_null_violation",
                        count=null_count,
                        severity=Severity.HIGH,
                        details=(
                            f"Foreign key columns contain {null_count} NULL values "
                            f"in table '{relation.child_table}'"
                        ),
                        expected="Non-null foreign key values",
                    )
                )

        # Find orphans using anti-join (with sampling support)
        orphans, sample_count, was_sampled = self._find_orphans(
            child_lf, child_cols, parent_lf, parent_cols
        )

        # Filter nulls from orphans if allow_null
        if self.allow_null:
            non_null_filter = pl.lit(True)
            for col in child_cols:
                non_null_filter = non_null_filter & pl.col(col).is_not_null()
            orphans = orphans.filter(non_null_filter)

        orphan_count = len(orphans)

        if orphan_count > 0:
            # Calculate violation ratio based on sampled data
            sample_violation_ratio = orphan_count / sample_count if sample_count > 0 else 0

            # Estimate total violations if sampled
            if was_sampled:
                estimated_total_violations = int(sample_violation_ratio * total_rows)
                violation_ratio = sample_violation_ratio
            else:
                estimated_total_violations = orphan_count
                violation_ratio = orphan_count / total_rows if total_rows > 0 else 0

            severity = self._calculate_severity(violation_ratio)

            # Get sample violations
            sample_violations = orphans.head(self.max_violations)
            sample_list = sample_violations.to_dicts()

            # Build details message
            if was_sampled:
                details = (
                    f"Found {orphan_count} violations in sample of {sample_count:,} rows "
                    f"({sample_violation_ratio:.2%}). Estimated {estimated_total_violations:,} "
                    f"total violations in {total_rows:,} rows. "
                    f"Sample violations: {sample_list[:5]}"
                )
            else:
                details = (
                    f"Found {orphan_count} records ({violation_ratio:.2%}) in "
                    f"'{relation.child_table}' with no matching record in "
                    f"'{relation.parent_table}'. Sample violations: {sample_list[:5]}"
                )

            issues.append(
                ValidationIssue(
                    column=", ".join(child_cols),
                    issue_type="fk_constraint_violation",
                    count=estimated_total_violations if was_sampled else orphan_count,
                    severity=severity,
                    details=details,
                    expected=f"All {', '.join(child_cols)} values exist in {relation.parent_table}",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the foreign key constraint.

        Note: The lf parameter is ignored; tables must be registered.

        Args:
            lf: Ignored (uses registered tables)

        Returns:
            List of validation issues
        """
        # Register the passed LazyFrame as child table if not already registered
        if self.relation.child_table not in self._tables:
            self.register_table(self.relation.child_table, lf)

        return self.validate_reference(self.relation)


@register_validator
class CompositeForeignKeyValidator(ReferentialValidator):
    """Validates composite foreign keys with additional checks.

    Extends basic FK validation with support for:
    - Partial key matching detection
    - Key subset validation
    - Cross-database reference support

    Example:
        validator = CompositeForeignKeyValidator(
            child_table="shipments",
            child_columns=["order_id", "warehouse_id", "product_id"],
            parent_table="inventory",
            parent_columns=["order_id", "warehouse_id", "product_id"],
            check_partial_matches=True,
        )
    """

    name = "composite_foreign_key"

    def __init__(
        self,
        child_table: str,
        child_columns: list[str],
        parent_table: str,
        parent_columns: list[str],
        tables: dict[str, pl.LazyFrame] | None = None,
        check_partial_matches: bool = False,
        sample_size: int | None = None,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        **kwargs: Any,
    ):
        """Initialize composite FK validator.

        Args:
            child_table: Name of child table
            child_columns: Composite key columns in child
            parent_table: Name of parent table
            parent_columns: Composite key columns in parent
            tables: Pre-registered tables
            check_partial_matches: Check for partial key matches
            sample_size: If set, validate on a sample of this size (for large tables)
            sample_seed: Random seed for reproducible sampling
            **kwargs: Additional config
        """
        super().__init__(tables=tables, sample_size=sample_size, sample_seed=sample_seed, **kwargs)
        self.relation = ForeignKeyRelation(
            child_table=child_table,
            child_columns=child_columns,
            parent_table=parent_table,
            parent_columns=parent_columns,
        )
        self.check_partial_matches = check_partial_matches

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Validate composite foreign key.

        Args:
            relation: The FK relation to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        child_lf = self.get_table(relation.child_table)
        parent_lf = self.get_table(relation.parent_table)

        if child_lf is None or parent_lf is None:
            return issues

        child_cols = relation.child_columns
        parent_cols = relation.parent_columns

        # Get total rows first (before any sampling)
        total_rows = child_lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Standard FK validation (with sampling support)
        orphans, sample_count, was_sampled = self._find_orphans(
            child_lf, child_cols, parent_lf, parent_cols
        )

        # Filter out nulls
        non_null_filter = pl.lit(True)
        for col in child_cols:
            non_null_filter = non_null_filter & pl.col(col).is_not_null()
        orphans = orphans.filter(non_null_filter)

        orphan_count = len(orphans)

        if orphan_count > 0:
            # Calculate violation ratio based on sampled data
            sample_violation_ratio = orphan_count / sample_count if sample_count > 0 else 0

            # Estimate total violations if sampled
            if was_sampled:
                estimated_total_violations = int(sample_violation_ratio * total_rows)
                violation_ratio = sample_violation_ratio
            else:
                estimated_total_violations = orphan_count
                violation_ratio = orphan_count / total_rows if total_rows > 0 else 0

            # Build details message
            if was_sampled:
                details = (
                    f"Composite FK violation: Found {orphan_count} violations in sample of "
                    f"{sample_count:,} rows ({sample_violation_ratio:.2%}). "
                    f"Estimated {estimated_total_violations:,} total violations in "
                    f"'{relation.child_table}' ({total_rows:,} rows) with no matching "
                    f"composite key in '{relation.parent_table}'"
                )
            else:
                details = (
                    f"Composite FK violation: {orphan_count} records in "
                    f"'{relation.child_table}' have no matching composite key "
                    f"in '{relation.parent_table}'"
                )

            issues.append(
                ValidationIssue(
                    column=", ".join(child_cols),
                    issue_type="composite_fk_violation",
                    count=estimated_total_violations if was_sampled else orphan_count,
                    severity=self._calculate_severity(violation_ratio),
                    details=details,
                    expected="All composite keys must exist in parent table",
                )
            )

        # Check partial matches if enabled
        if self.check_partial_matches and orphan_count > 0 and len(child_cols) > 1:
            partial_issues = self._check_partial_matches(
                orphans, child_cols, parent_lf, parent_cols
            )
            issues.extend(partial_issues)

        return issues

    def _check_partial_matches(
        self,
        orphans: pl.DataFrame,
        child_cols: list[str],
        parent_lf: pl.LazyFrame,
        parent_cols: list[str],
    ) -> list[ValidationIssue]:
        """Check for partial key matches in orphan records.

        This helps identify data issues where some but not all key columns match.

        Args:
            orphans: Orphan records
            child_cols: Child key columns
            parent_lf: Parent table
            parent_cols: Parent key columns

        Returns:
            List of issues for partial matches
        """
        issues: list[ValidationIssue] = []
        parent_df = parent_lf.collect()

        # Check each single column for matches
        for i, (child_col, parent_col) in enumerate(zip(child_cols, parent_cols)):
            # Get unique orphan values for this column
            orphan_values = orphans.select(pl.col(child_col)).unique()

            # Check how many exist in parent
            parent_values = parent_df.select(pl.col(parent_col)).unique()

            matching = orphan_values.join(
                parent_values,
                left_on=child_col,
                right_on=parent_col,
                how="inner",
            )

            match_count = len(matching)
            orphan_unique = len(orphan_values)

            if match_count > 0 and match_count < orphan_unique:
                issues.append(
                    ValidationIssue(
                        column=child_col,
                        issue_type="partial_fk_match",
                        count=match_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Partial match: {match_count}/{orphan_unique} unique "
                            f"'{child_col}' values from orphan records exist in "
                            f"parent column '{parent_col}'"
                        ),
                        expected="Check for data synchronization issues",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate composite foreign key constraint.

        Args:
            lf: Child table LazyFrame

        Returns:
            List of validation issues
        """
        if self.relation.child_table not in self._tables:
            self.register_table(self.relation.child_table, lf)

        return self.validate_reference(self.relation)


@register_validator
class SelfReferentialFKValidator(ReferentialValidator):
    """Validates self-referential foreign keys.

    Used for hierarchical data structures like organization charts,
    category trees, or bill of materials.

    Example:
        # Employee-Manager hierarchy
        validator = SelfReferentialFKValidator(
            table="employees",
            fk_column="manager_id",
            pk_column="employee_id",
            allow_null=True,  # Top-level employees have no manager
        )
    """

    name = "self_referential_fk"

    def __init__(
        self,
        table: str,
        fk_column: str,
        pk_column: str,
        tables: dict[str, pl.LazyFrame] | None = None,
        allow_null: bool = True,
        max_depth: int = 100,
        sample_size: int | None = None,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        **kwargs: Any,
    ):
        """Initialize self-referential FK validator.

        Args:
            table: Table name
            fk_column: Foreign key column (references same table)
            pk_column: Primary key column
            tables: Pre-registered tables
            allow_null: Allow NULL FK values (for root nodes)
            max_depth: Maximum hierarchy depth to check
            sample_size: If set, validate on a sample of this size (for large tables)
            sample_seed: Random seed for reproducible sampling
            **kwargs: Additional config
        """
        super().__init__(tables=tables, sample_size=sample_size, sample_seed=sample_seed, **kwargs)
        self.table_name = table
        self.fk_column = fk_column
        self.pk_column = pk_column
        self.allow_null = allow_null
        self.max_depth = max_depth
        self.relation = ForeignKeyRelation(
            child_table=table,
            child_columns=[fk_column],
            parent_table=table,
            parent_columns=[pk_column],
            relation_name=f"self_ref_{table}",
        )

    def validate_reference(
        self, relation: ForeignKeyRelation
    ) -> list[ValidationIssue]:
        """Validate self-referential FK.

        Args:
            relation: The self-referential relation

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        table_lf = self.get_table(self.table_name)
        if table_lf is None:
            return issues

        # Get total rows first
        total_rows = table_lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Sample or collect all data
        df, sample_count, was_sampled = self._sample_lazyframe(table_lf)

        # Get all PK values (always need all PKs for reference)
        if was_sampled:
            # For self-referential, we need all PK values from full table
            pk_values = set(
                table_lf.select(pl.col(self.pk_column)).collect().to_series().to_list()
            )
        else:
            pk_values = set(df.select(pl.col(self.pk_column)).to_series().to_list())

        # Get FK values from sampled data (excluding nulls)
        fk_df = df.filter(pl.col(self.fk_column).is_not_null())
        fk_values = fk_df.select(pl.col(self.fk_column)).to_series().to_list()

        # Find orphans
        orphan_fks = [fk for fk in fk_values if fk not in pk_values]
        orphan_count = len(orphan_fks)

        if orphan_count > 0:
            # Calculate violation ratio based on sampled data
            sample_violation_ratio = orphan_count / sample_count if sample_count > 0 else 0

            # Estimate total violations if sampled
            if was_sampled:
                estimated_total_violations = int(sample_violation_ratio * total_rows)
                violation_ratio = sample_violation_ratio
                details = (
                    f"Self-referential FK violation: Found {orphan_count} violations in "
                    f"sample of {sample_count:,} rows ({sample_violation_ratio:.2%}). "
                    f"Estimated {estimated_total_violations:,} total violations in "
                    f"{total_rows:,} rows referencing non-existent {self.pk_column} values. "
                    f"Sample invalid refs: {orphan_fks[:5]}"
                )
            else:
                estimated_total_violations = orphan_count
                violation_ratio = orphan_count / total_rows if total_rows > 0 else 0
                details = (
                    f"Self-referential FK violation: {orphan_count} records "
                    f"reference non-existent {self.pk_column} values. "
                    f"Sample invalid refs: {orphan_fks[:5]}"
                )

            issues.append(
                ValidationIssue(
                    column=self.fk_column,
                    issue_type="self_ref_fk_violation",
                    count=estimated_total_violations,
                    severity=self._calculate_severity(violation_ratio),
                    details=details,
                    expected=f"All {self.fk_column} values exist as {self.pk_column}",
                )
            )

        # Check for null if not allowed (on sampled data)
        if not self.allow_null:
            null_count = df.filter(pl.col(self.fk_column).is_null()).height
            if null_count > 0:
                if was_sampled:
                    estimated_null_count = int((null_count / sample_count) * total_rows)
                    details = (
                        f"Found {null_count} NULL values in sample ({sample_count:,} rows). "
                        f"Estimated {estimated_null_count:,} total NULLs in {self.fk_column}"
                    )
                else:
                    estimated_null_count = null_count
                    details = f"Found {null_count} NULL values in {self.fk_column}"

                issues.append(
                    ValidationIssue(
                        column=self.fk_column,
                        issue_type="self_ref_null_violation",
                        count=estimated_null_count,
                        severity=Severity.MEDIUM,
                        details=details,
                        expected="Non-null self-referential FK values",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate self-referential FK.

        Args:
            lf: Table LazyFrame

        Returns:
            List of validation issues
        """
        self.register_table(self.table_name, lf)
        return self.validate_reference(self.relation)
