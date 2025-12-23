"""Table schema validators.

Validators for checking table schema and structure.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.table.base import TableValidator
from truthound.validators.registry import register_validator


@register_validator
class TableSchemaMatchValidator(TableValidator):
    """Validates that table schema matches expected schema.

    Example:
        validator = TableSchemaMatchValidator(
            expected_schema={
                "id": pl.Int64,
                "name": pl.Utf8,
                "age": pl.Int32,
                "created_at": pl.Datetime,
            }
        )
    """

    name = "table_schema_match"
    category = "table"

    def __init__(
        self,
        expected_schema: dict[str, pl.DataType | type],
        strict: bool = True,
        check_order: bool = False,
        **kwargs: Any,
    ):
        """Initialize schema match validator.

        Args:
            expected_schema: Expected column name to type mapping
            strict: If True, extra columns in actual schema fail validation
            check_order: If True, column order must match
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.expected_schema = expected_schema
        self.strict = strict
        self.check_order = check_order

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        actual_schema = lf.collect_schema()
        expected_cols = set(self.expected_schema.keys())
        actual_cols = set(actual_schema.keys())

        # Check for missing columns
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="schema_missing_columns",
                    count=len(missing_cols),
                    severity=Severity.CRITICAL,
                    details=f"Missing columns: {sorted(missing_cols)}",
                    expected=f"All columns from expected schema",
                )
            )

        # Check for extra columns (if strict)
        if self.strict:
            extra_cols = actual_cols - expected_cols
            if extra_cols:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="schema_extra_columns",
                        count=len(extra_cols),
                        severity=Severity.MEDIUM,
                        details=f"Unexpected columns: {sorted(extra_cols)}",
                        expected=f"Only columns from expected schema",
                    )
                )

        # Check column types
        type_mismatches = []
        for col, expected_type in self.expected_schema.items():
            if col in actual_schema:
                actual_type = actual_schema[col]
                if not self._types_compatible(actual_type, expected_type):
                    type_mismatches.append(
                        f"{col}: expected {expected_type}, got {actual_type}"
                    )

        if type_mismatches:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="schema_type_mismatch",
                    count=len(type_mismatches),
                    severity=Severity.HIGH,
                    details=f"Type mismatches: {type_mismatches}",
                    expected="Matching column types",
                )
            )

        # Check column order
        if self.check_order and not missing_cols:
            expected_order = list(self.expected_schema.keys())
            actual_order = [c for c in actual_schema.keys() if c in expected_cols]

            if expected_order != actual_order:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="schema_order_mismatch",
                        count=1,
                        severity=Severity.LOW,
                        details=f"Column order mismatch. Expected: {expected_order}, got: {actual_order}",
                        expected="Matching column order",
                    )
                )

        return issues

    def _types_compatible(
        self, actual: pl.DataType, expected: pl.DataType | type
    ) -> bool:
        """Check if types are compatible."""
        # Handle type classes vs instances
        if isinstance(expected, type):
            return isinstance(actual, expected) or actual == expected
        return actual == expected or str(actual) == str(expected)


@register_validator
class TableSchemaCompareValidator(TableValidator):
    """Validates that table schema matches another table's schema.

    Example:
        validator = TableSchemaCompareValidator(
            reference_table=reference_lf,
            check_types=True,
            check_order=False,
        )
    """

    name = "table_schema_compare"
    category = "table"

    def __init__(
        self,
        reference_table: pl.LazyFrame,
        check_types: bool = True,
        check_order: bool = False,
        ignore_columns: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize schema compare validator.

        Args:
            reference_table: Reference table to compare against
            check_types: Check column types match
            check_order: Check column order matches
            ignore_columns: Columns to ignore in comparison
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.reference_table = reference_table
        self.check_types = check_types
        self.check_order = check_order
        self.ignore_columns = set(ignore_columns or [])

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        actual_schema = lf.collect_schema()
        ref_schema = self.reference_table.collect_schema()

        # Filter out ignored columns
        actual_cols = set(actual_schema.keys()) - self.ignore_columns
        ref_cols = set(ref_schema.keys()) - self.ignore_columns

        # Check for missing columns
        missing_cols = ref_cols - actual_cols
        if missing_cols:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="schema_missing_columns",
                    count=len(missing_cols),
                    severity=Severity.HIGH,
                    details=f"Columns in reference but not in table: {sorted(missing_cols)}",
                    expected="All columns from reference table",
                )
            )

        # Check for extra columns
        extra_cols = actual_cols - ref_cols
        if extra_cols:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="schema_extra_columns",
                    count=len(extra_cols),
                    severity=Severity.MEDIUM,
                    details=f"Columns in table but not in reference: {sorted(extra_cols)}",
                    expected="Only columns from reference table",
                )
            )

        # Check types
        if self.check_types:
            type_mismatches = []
            for col in ref_cols & actual_cols:
                if actual_schema[col] != ref_schema[col]:
                    type_mismatches.append(
                        f"{col}: expected {ref_schema[col]}, got {actual_schema[col]}"
                    )

            if type_mismatches:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="schema_type_mismatch",
                        count=len(type_mismatches),
                        severity=Severity.HIGH,
                        details=f"Type mismatches: {type_mismatches}",
                        expected="Matching column types",
                    )
                )

        # Check order
        if self.check_order:
            actual_order = [c for c in actual_schema.keys() if c not in self.ignore_columns]
            ref_order = [c for c in ref_schema.keys() if c not in self.ignore_columns]

            if actual_order != ref_order:
                issues.append(
                    ValidationIssue(
                        column="_table",
                        issue_type="schema_order_mismatch",
                        count=1,
                        severity=Severity.LOW,
                        details=f"Column order differs from reference",
                        expected="Matching column order",
                    )
                )

        return issues


@register_validator
class TableColumnTypesValidator(TableValidator):
    """Validates that specific columns have expected types.

    Example:
        validator = TableColumnTypesValidator(
            column_types={
                "id": [pl.Int32, pl.Int64],  # Accept either
                "name": [pl.Utf8],
                "amount": [pl.Float64, pl.Decimal],
            }
        )
    """

    name = "table_column_types"
    category = "table"

    def __init__(
        self,
        column_types: dict[str, list[pl.DataType | type]],
        **kwargs: Any,
    ):
        """Initialize column types validator.

        Args:
            column_types: Column name to list of acceptable types
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column_types = column_types

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()

        for col, acceptable_types in self.column_types.items():
            if col not in schema:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="column_not_found",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Column '{col}' not found in table",
                        expected=f"Column '{col}' exists",
                    )
                )
                continue

            actual_type = schema[col]
            type_matches = False

            for expected_type in acceptable_types:
                if isinstance(expected_type, type):
                    if isinstance(actual_type, expected_type):
                        type_matches = True
                        break
                else:
                    if actual_type == expected_type or str(actual_type) == str(expected_type):
                        type_matches = True
                        break

            if not type_matches:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="column_type_invalid",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Column '{col}' has type {actual_type}, expected one of {acceptable_types}",
                        expected=f"Type in {acceptable_types}",
                    )
                )

        return issues
