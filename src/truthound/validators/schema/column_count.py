"""Column and row count validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnCountValidator(Validator):
    """Validates that the number of columns is within expected range."""

    name = "column_count"
    category = "schema"

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        exact_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.exact_count = exact_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        actual_count = len(schema.names())

        if self.exact_count is not None and actual_count != self.exact_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="column_count_mismatch",
                    count=abs(actual_count - self.exact_count),
                    severity=Severity.CRITICAL,
                    details=f"Expected {self.exact_count} columns, got {actual_count}",
                    expected=self.exact_count,
                    actual=actual_count,
                )
            )
            return issues

        if self.min_count is not None and actual_count < self.min_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="too_few_columns",
                    count=self.min_count - actual_count,
                    severity=Severity.HIGH,
                    details=f"Expected >= {self.min_count} columns, got {actual_count}",
                    expected=f">= {self.min_count}",
                    actual=actual_count,
                )
            )

        if self.max_count is not None and actual_count > self.max_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="too_many_columns",
                    count=actual_count - self.max_count,
                    severity=Severity.HIGH,
                    details=f"Expected <= {self.max_count} columns, got {actual_count}",
                    expected=f"<= {self.max_count}",
                    actual=actual_count,
                )
            )

        return issues


@register_validator
class RowCountValidator(Validator):
    """Validates that the number of rows is within expected range."""

    name = "row_count"
    category = "schema"

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        exact_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count
        self.exact_count = exact_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        actual_count = lf.select(pl.len()).collect().item()

        if self.exact_count is not None and actual_count != self.exact_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="row_count_mismatch",
                    count=abs(actual_count - self.exact_count),
                    severity=Severity.CRITICAL,
                    details=f"Expected {self.exact_count} rows, got {actual_count}",
                    expected=self.exact_count,
                    actual=actual_count,
                )
            )
            return issues

        if self.min_count is not None and actual_count < self.min_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="too_few_rows",
                    count=self.min_count - actual_count,
                    severity=Severity.HIGH,
                    details=f"Expected >= {self.min_count} rows, got {actual_count}",
                    expected=f">= {self.min_count}",
                    actual=actual_count,
                )
            )

        if self.max_count is not None and actual_count > self.max_count:
            issues.append(
                ValidationIssue(
                    column="_table_",
                    issue_type="too_many_rows",
                    count=actual_count - self.max_count,
                    severity=Severity.HIGH,
                    details=f"Expected <= {self.max_count} rows, got {actual_count}",
                    expected=f"<= {self.max_count}",
                    actual=actual_count,
                )
            )

        return issues
