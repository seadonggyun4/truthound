"""Uniqueness constraint validator."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class UniqueValidator(Validator):
    """Checks for uniqueness constraint violations in ID-like columns."""

    name = "unique"

    # Column name patterns that suggest uniqueness is expected
    UNIQUE_PATTERNS: list[str] = [
        "id",
        "uuid",
        "guid",
        "key",
        "code",
        "sku",
        "ean",
        "upc",
        "isbn",
        "ssn",
        "email",
        "username",
        "user_name",
        "login",
    ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for duplicate values in columns that should be unique.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with duplicate values.
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in schema.names():
            col_lower = col.lower()

            # Check if column name suggests uniqueness
            should_be_unique = any(
                col_lower == pattern or col_lower.endswith(f"_{pattern}") or col_lower.startswith(f"{pattern}_")
                for pattern in self.UNIQUE_PATTERNS
            )

            if not should_be_unique:
                continue

            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            unique_count = col_data.n_unique()
            duplicate_count = len(col_data) - unique_count

            if duplicate_count > 0:
                dup_pct = duplicate_count / len(col_data)

                # Uniqueness violations are usually severe
                if dup_pct > 0.1:
                    severity = Severity.CRITICAL
                elif dup_pct > 0.01:
                    severity = Severity.HIGH
                else:
                    severity = Severity.MEDIUM

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="duplicate_values",
                        count=duplicate_count,
                        severity=severity,
                        details=f"{unique_count} unique values in {len(col_data)} rows",
                    )
                )

        return issues
