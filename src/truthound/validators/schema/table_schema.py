"""Table schema validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class TableSchemaValidator(Validator):
    """Validates the complete table schema (columns and types)."""

    name = "table_schema"
    category = "schema"

    def __init__(
        self,
        expected_schema: dict[str, str | type[pl.DataType]],
        allow_extra_columns: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_schema = expected_schema
        self.allow_extra_columns = allow_extra_columns

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        actual_columns = set(schema.names())
        expected_columns = set(self.expected_schema.keys())

        # Check for missing columns
        missing = expected_columns - actual_columns
        for col in missing:
            issues.append(
                ValidationIssue(
                    column=col,
                    issue_type="missing_column",
                    count=1,
                    severity=Severity.CRITICAL,
                    details=f"Expected column '{col}' not found",
                )
            )

        # Check for extra columns
        if not self.allow_extra_columns:
            extra = actual_columns - expected_columns
            for col in extra:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unexpected_column",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Unexpected column '{col}'",
                    )
                )

        # Check types for existing columns
        for col, expected_type in self.expected_schema.items():
            if col not in actual_columns:
                continue

            actual_type = schema[col]

            # Normalize expected type
            if isinstance(expected_type, str):
                expected_name = expected_type.lower()
            else:
                expected_name = expected_type.__name__.lower()

            actual_name = type(actual_type).__name__.lower()

            # Simple type matching (could be extended with aliases)
            if expected_name not in actual_name and actual_name not in expected_name:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="type_mismatch",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Expected {expected_type}, got {actual_type}",
                        expected=str(expected_type),
                        actual=str(actual_type),
                    )
                )

        return issues
