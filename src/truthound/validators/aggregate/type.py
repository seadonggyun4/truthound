"""Type validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class TypeValidator(Validator, StringValidatorMixin):
    """Detects mixed types in string columns (numbers stored as strings)."""

    name = "type"
    category = "aggregate"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Build aggregation expressions for all columns at once
        agg_exprs = []
        for col in columns:
            # Strip whitespace and filter non-empty strings
            stripped = pl.col(col).str.strip_chars()
            non_empty = stripped.is_not_null() & (stripped != "")

            # Check if value can be parsed as numeric using regex pattern
            # Matches: optional sign, digits, optional decimal, optional exponent
            numeric_pattern = r"^[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?$"
            is_numeric = stripped.str.contains(numeric_pattern)

            # Count numeric values (non-empty strings that match numeric pattern)
            agg_exprs.append(
                (non_empty & is_numeric).sum().alias(f"{col}__numeric")
            )
            # Count non-numeric values (non-empty strings that don't match)
            agg_exprs.append(
                (non_empty & ~is_numeric).sum().alias(f"{col}__non_numeric")
            )

        if not agg_exprs:
            return issues

        # Single collect for all columns
        result = lf.select(agg_exprs).collect()

        if result.height == 0:
            return issues

        row = result.row(0, named=True)

        for col in columns:
            numeric_count = row.get(f"{col}__numeric", 0) or 0
            string_count = row.get(f"{col}__non_numeric", 0) or 0

            total_non_null = numeric_count + string_count
            if total_non_null == 0:
                continue

            # Detect mixed types (significant portion of both)
            numeric_ratio = numeric_count / total_non_null
            string_ratio = string_count / total_non_null

            if 0.1 < numeric_ratio < 0.9:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="mixed_type",
                        count=min(numeric_count, string_count),
                        severity=Severity.MEDIUM,
                        details=f"Mixed types: {numeric_ratio:.1%} numeric, {string_ratio:.1%} string",
                    )
                )

        return issues
