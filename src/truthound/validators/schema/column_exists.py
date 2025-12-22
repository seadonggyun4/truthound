"""Column existence validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnExistsValidator(Validator):
    """Validates that expected columns exist in the dataset."""

    name = "column_exists"
    category = "schema"

    def __init__(
        self,
        columns: list[str],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_columns = columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        actual_columns = set(schema.names())

        for col in self.expected_columns:
            if col not in actual_columns:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="missing_column",
                        count=1,
                        severity=Severity.CRITICAL,
                        details=f"Expected column '{col}' not found",
                        expected=col,
                    )
                )

        return issues


@register_validator
class ColumnNotExistsValidator(Validator):
    """Validates that certain columns do NOT exist (for deprecated columns)."""

    name = "column_not_exists"
    category = "schema"

    def __init__(
        self,
        columns: list[str],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.forbidden_columns = columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        actual_columns = set(schema.names())

        for col in self.forbidden_columns:
            if col in actual_columns:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="forbidden_column",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Column '{col}' should not exist",
                    )
                )

        return issues
