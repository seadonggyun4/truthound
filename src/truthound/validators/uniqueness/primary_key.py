"""Primary key validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class PrimaryKeyValidator(Validator):
    """Validates that a column can serve as a primary key (unique and not null)."""

    name = "primary_key"
    category = "uniqueness"

    def __init__(
        self,
        column: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        result = lf.select([
            pl.len().alias("_total"),
            pl.col(self.column).null_count().alias("_null"),
            # Count unique non-null values only
            pl.col(self.column).drop_nulls().n_unique().alias("_unique"),
            pl.col(self.column).count().alias("_count"),
        ]).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        null_count = result["_null"][0]
        unique_count = result["_unique"][0]
        non_null_count = result["_count"][0]

        # Check for nulls
        if null_count > 0:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="primary_key_null",
                    count=null_count,
                    severity=Severity.CRITICAL,
                    details=f"Primary key has {null_count} null values",
                )
            )

        # Check for duplicates (non_null_count - unique gives duplicate rows)
        dup_count = non_null_count - unique_count
        if dup_count > 0:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="primary_key_duplicate",
                    count=dup_count,
                    severity=Severity.CRITICAL,
                    details=f"Primary key has {dup_count} duplicate values",
                )
            )

        return issues


@register_validator
class CompoundKeyValidator(Validator):
    """Validates that a combination of columns forms a unique key."""

    name = "compound_key"
    category = "uniqueness"

    def __init__(
        self,
        columns: list[str],
        allow_nulls: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.key_columns = columns
        self.allow_nulls = allow_nulls

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check for nulls in any key column
        if not self.allow_nulls:
            null_exprs = [
                pl.col(c).null_count().alias(f"_null_{c}")
                for c in self.key_columns
            ]
            null_result = lf.select(null_exprs).collect()

            for col in self.key_columns:
                null_count = null_result[f"_null_{col}"][0]
                if null_count > 0:
                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="compound_key_null",
                            count=null_count,
                            severity=Severity.CRITICAL,
                            details=f"Compound key column has {null_count} nulls",
                        )
                    )

        # Check for duplicates
        result = lf.select([
            pl.len().alias("_total"),
            pl.struct(self.key_columns).n_unique().alias("_unique"),
        ]).collect()

        total_rows = result["_total"][0]
        unique_count = result["_unique"][0]

        if total_rows == 0:
            return issues

        dup_count = total_rows - unique_count
        if dup_count > 0:
            key_desc = ", ".join(self.key_columns)
            issues.append(
                ValidationIssue(
                    column=f"[{key_desc}]",
                    issue_type="compound_key_duplicate",
                    count=dup_count,
                    severity=Severity.CRITICAL,
                    details=f"Compound key has {dup_count} duplicate combinations",
                )
            )

        return issues
