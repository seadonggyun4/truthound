"""Table column count validators.

Validators for checking column counts and structure.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.table.base import TableValidator
from truthound.validators.registry import register_validator


@register_validator
class TableColumnCountValidator(TableValidator):
    """Validates that table has expected number of columns.

    Example:
        # Table should have exactly 10 columns
        validator = TableColumnCountValidator(expected_count=10)

        # Table should have between 5 and 15 columns
        validator = TableColumnCountValidator(min_count=5, max_count=15)
    """

    name = "table_column_count"
    category = "table"

    def __init__(
        self,
        expected_count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
        **kwargs: Any,
    ):
        """Initialize column count validator.

        Args:
            expected_count: Exact expected column count
            min_count: Minimum column count (if expected_count not set)
            max_count: Maximum column count (if expected_count not set)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.expected_count = expected_count
        self.min_count = min_count
        self.max_count = max_count

        if expected_count is None and min_count is None and max_count is None:
            raise ValueError(
                "At least one of 'expected_count', 'min_count', or 'max_count' required"
            )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        col_count = len(schema)

        if self.expected_count is not None:
            if col_count != self.expected_count:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="column_count_mismatch",
                        count=col_count,
                        severity=Severity.HIGH,
                        details=f"Table has {col_count} columns, expected {self.expected_count}",
                        expected=f"{self.expected_count} columns",
                    )
                )
        else:
            if self.min_count is not None and col_count < self.min_count:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="column_count_below_minimum",
                        count=col_count,
                        severity=Severity.HIGH,
                        details=f"Table has {col_count} columns, expected at least {self.min_count}",
                        expected=f">= {self.min_count} columns",
                    )
                )

            if self.max_count is not None and col_count > self.max_count:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="column_count_above_maximum",
                        count=col_count,
                        severity=Severity.HIGH,
                        details=f"Table has {col_count} columns, expected at most {self.max_count}",
                        expected=f"<= {self.max_count} columns",
                    )
                )

        return issues


@register_validator
class TableRequiredColumnsValidator(TableValidator):
    """Validates that table has all required columns.

    Example:
        validator = TableRequiredColumnsValidator(
            required_columns=["id", "name", "email", "created_at"]
        )
    """

    name = "table_required_columns"
    category = "table"

    def __init__(
        self,
        required_columns: list[str],
        **kwargs: Any,
    ):
        """Initialize required columns validator.

        Args:
            required_columns: List of column names that must exist
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.required_columns = required_columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        existing_columns = set(schema.keys())
        missing_columns = set(self.required_columns) - existing_columns

        if missing_columns:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="missing_required_columns",
                    count=len(missing_columns),
                    severity=Severity.CRITICAL,
                    details=f"Missing required columns: {sorted(missing_columns)}",
                    expected=f"All of {self.required_columns}",
                )
            )

        return issues


@register_validator
class TableForbiddenColumnsValidator(TableValidator):
    """Validates that table does not have forbidden columns.

    Example:
        # Ensure PII columns are not present
        validator = TableForbiddenColumnsValidator(
            forbidden_columns=["ssn", "credit_card", "password"]
        )
    """

    name = "table_forbidden_columns"
    category = "table"

    def __init__(
        self,
        forbidden_columns: list[str],
        **kwargs: Any,
    ):
        """Initialize forbidden columns validator.

        Args:
            forbidden_columns: List of column names that must not exist
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.forbidden_columns = forbidden_columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        existing_columns = set(schema.keys())
        found_forbidden = set(self.forbidden_columns) & existing_columns

        if found_forbidden:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="forbidden_columns_found",
                    count=len(found_forbidden),
                    severity=Severity.CRITICAL,
                    details=f"Found forbidden columns: {sorted(found_forbidden)}",
                    expected=f"None of {self.forbidden_columns}",
                )
            )

        return issues
