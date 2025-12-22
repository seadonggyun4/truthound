"""Null value validator."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class NullValidator(Validator):
    """Detects null/missing values in columns."""

    name = "null"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for null values in each column.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with null values.
        """
        issues: list[ValidationIssue] = []

        # Single optimized query: get null counts and total rows together
        columns = lf.collect_schema().names()
        result = lf.select([
            pl.len().alias("_total_rows"),
            *[pl.col(c).null_count().alias(f"_null_{c}") for c in columns]
        ]).collect()

        total_rows = result["_total_rows"][0]
        null_counts = [result[f"_null_{c}"][0] for c in columns]

        for col, null_count in zip(columns, null_counts):
            if null_count > 0:
                # Determine severity based on null percentage
                null_pct = null_count / total_rows if total_rows > 0 else 0

                if null_pct > 0.5:
                    severity = Severity.CRITICAL
                elif null_pct > 0.2:
                    severity = Severity.HIGH
                elif null_pct > 0.05:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="null",
                        count=null_count,
                        severity=severity,
                        details=f"{null_pct:.1%} of values are null",
                    )
                )

        return issues
