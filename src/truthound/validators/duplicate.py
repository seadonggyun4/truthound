"""Duplicate value validator."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class DuplicateValidator(Validator):
    """Detects duplicate rows in the dataset."""

    name = "duplicate"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for duplicate rows.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for duplicate rows.
        """
        issues: list[ValidationIssue] = []

        # Use is_duplicated for efficient duplicate detection
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        duplicate_count = df.is_duplicated().sum()

        if duplicate_count > 0:
            dup_pct = duplicate_count / total_rows

            if dup_pct > 0.3:
                severity = Severity.CRITICAL
            elif dup_pct > 0.1:
                severity = Severity.HIGH
            elif dup_pct > 0.01:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            issues.append(
                ValidationIssue(
                    column="*",
                    issue_type="duplicate",
                    count=duplicate_count,
                    severity=severity,
                    details=f"{dup_pct:.1%} of rows are duplicates",
                )
            )

        return issues
