"""Column pair set validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnPairInSetValidator(Validator):
    """Validates that column pair values are in a specified set of tuples.

    Useful for checking valid combinations of categorical values.

    Example:
        # (country, currency) must be a valid combination
        validator = ColumnPairInSetValidator(
            column_a="country",
            column_b="currency",
            valid_pairs=[
                ("US", "USD"),
                ("UK", "GBP"),
                ("JP", "JPY"),
                ("KR", "KRW"),
            ],
        )

        # (department, role) must be an allowed combination
        validator = ColumnPairInSetValidator(
            column_a="department",
            column_b="role",
            valid_pairs=[
                ("engineering", "developer"),
                ("engineering", "manager"),
                ("sales", "representative"),
                ("sales", "manager"),
            ],
        )
    """

    name = "column_pair_in_set"
    category = "schema"

    def __init__(
        self,
        column_a: str,
        column_b: str,
        valid_pairs: list[tuple[Any, Any]],
        ignore_nulls: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column_a = column_a
        self.column_b = column_b
        self.valid_pairs = set(valid_pairs)
        self.ignore_nulls = ignore_nulls

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        # Check each row's pair
        invalid_count = 0
        sample_values = []

        for idx in range(total_rows):
            val_a = df[self.column_a][idx]
            val_b = df[self.column_b][idx]

            # Handle nulls
            if val_a is None or val_b is None:
                if self.ignore_nulls:
                    continue
                else:
                    invalid_count += 1
                    if len(sample_values) < self.config.sample_size:
                        sample_values.append(f"({val_a}, {val_b})")
                    continue

            # Check if pair is valid
            if (val_a, val_b) not in self.valid_pairs:
                invalid_count += 1
                if len(sample_values) < self.config.sample_size:
                    sample_values.append(f"({val_a}, {val_b})")

        if invalid_count > 0:
            if self._passes_mostly(invalid_count, total_rows):
                return issues

            ratio = invalid_count / total_rows

            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="invalid_column_pair",
                    count=invalid_count,
                    severity=self._calculate_severity(ratio),
                    details=f"{invalid_count} rows have invalid ({self.column_a}, {self.column_b}) pairs",
                    expected=f"Pairs in: {list(self.valid_pairs)[:5]}{'...' if len(self.valid_pairs) > 5 else ''}",
                    sample_values=sample_values,
                )
            )

        return issues


@register_validator
class ColumnPairNotInSetValidator(Validator):
    """Validates that column pair values are NOT in a specified set of tuples.

    Useful for checking forbidden combinations of values.

    Example:
        # These (status, priority) combinations should never occur
        validator = ColumnPairNotInSetValidator(
            column_a="status",
            column_b="priority",
            forbidden_pairs=[
                ("closed", "urgent"),
                ("resolved", "urgent"),
            ],
        )
    """

    name = "column_pair_not_in_set"
    category = "schema"

    def __init__(
        self,
        column_a: str,
        column_b: str,
        forbidden_pairs: list[tuple[Any, Any]],
        ignore_nulls: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column_a = column_a
        self.column_b = column_b
        self.forbidden_pairs = set(forbidden_pairs)
        self.ignore_nulls = ignore_nulls

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        # Check each row's pair
        forbidden_count = 0
        sample_values = []

        for idx in range(total_rows):
            val_a = df[self.column_a][idx]
            val_b = df[self.column_b][idx]

            # Skip nulls if configured
            if (val_a is None or val_b is None) and self.ignore_nulls:
                continue

            # Check if pair is forbidden
            if (val_a, val_b) in self.forbidden_pairs:
                forbidden_count += 1
                if len(sample_values) < self.config.sample_size:
                    sample_values.append(f"({val_a}, {val_b})")

        if forbidden_count > 0:
            if self._passes_mostly(forbidden_count, total_rows):
                return issues

            ratio = forbidden_count / total_rows

            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="forbidden_column_pair",
                    count=forbidden_count,
                    severity=self._calculate_severity(ratio),
                    details=f"{forbidden_count} rows have forbidden ({self.column_a}, {self.column_b}) pairs",
                    expected=f"Pairs not in: {list(self.forbidden_pairs)[:5]}{'...' if len(self.forbidden_pairs) > 5 else ''}",
                    sample_values=sample_values,
                )
            )

        return issues
