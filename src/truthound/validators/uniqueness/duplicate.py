"""Duplicate row validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class DuplicateValidator(Validator):
    """Detects duplicate rows in the dataset."""

    name = "duplicate"
    category = "uniqueness"

    def __init__(
        self,
        subset: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.subset = subset

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        columns = self.subset or self._get_target_columns(lf)
        if not columns:
            return issues

        # Count duplicates
        result = lf.select([
            pl.len().alias("_total"),
            pl.struct(columns).is_duplicated().sum().alias("_dup"),
        ]).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        dup_count = result["_dup"][0]
        if dup_count > 0:
            dup_pct = dup_count / total_rows
            col_desc = ", ".join(columns[:3]) + ("..." if len(columns) > 3 else "")
            issues.append(
                ValidationIssue(
                    column=f"[{col_desc}]",
                    issue_type="duplicate_row",
                    count=dup_count,
                    severity=self._calculate_severity(
                        dup_pct, thresholds=(0.3, 0.1, 0.01)
                    ),
                    details=f"{dup_count} duplicate rows ({dup_pct:.1%})",
                )
            )

        return issues


@register_validator
class DuplicateWithinGroupValidator(Validator):
    """Detects duplicate values within groups.

    Example:
        # Check for duplicate order_id within same customer
        validator = DuplicateWithinGroupValidator(
            group_by=["customer_id"],
            check_column="order_id",
        )
    """

    name = "duplicate_within_group"
    category = "uniqueness"

    def __init__(
        self,
        group_by: list[str],
        check_column: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.group_by = group_by
        self.check_column = check_column

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Count duplicates within groups
        dup_count = (
            lf.group_by(self.group_by)
            .agg(
                (pl.col(self.check_column).count() - pl.col(self.check_column).n_unique())
                .alias("_dup")
            )
            .select(pl.col("_dup").sum())
            .collect()
            .item()
        )

        if dup_count > 0:
            total_rows = lf.select(pl.len()).collect().item()
            dup_pct = dup_count / total_rows if total_rows > 0 else 0
            group_desc = ", ".join(self.group_by)

            issues.append(
                ValidationIssue(
                    column=self.check_column,
                    issue_type="duplicate_within_group",
                    count=dup_count,
                    severity=Severity.HIGH,
                    details=f"{dup_count} duplicates within [{group_desc}] groups",
                )
            )

        return issues
