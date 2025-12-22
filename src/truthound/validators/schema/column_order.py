"""Column order validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnOrderValidator(Validator):
    """Validates that columns appear in expected order."""

    name = "column_order"
    category = "schema"

    def __init__(
        self,
        expected_order: list[str],
        strict: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_order = expected_order
        self.strict = strict  # If False, allows extra columns between

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        actual_columns = schema.names()

        if self.strict:
            # Strict mode: columns must match exactly
            if actual_columns != self.expected_order:
                mismatches = []
                for i, (exp, act) in enumerate(
                    zip(self.expected_order, actual_columns)
                ):
                    if exp != act:
                        mismatches.append(f"pos {i}: expected '{exp}', got '{act}'")

                if len(actual_columns) != len(self.expected_order):
                    mismatches.append(
                        f"count: expected {len(self.expected_order)}, got {len(actual_columns)}"
                    )

                issues.append(
                    ValidationIssue(
                        column="_table_",
                        issue_type="column_order_mismatch",
                        count=len(mismatches),
                        severity=Severity.HIGH,
                        details="; ".join(mismatches[:3]) + ("..." if len(mismatches) > 3 else ""),
                        expected=self.expected_order,
                        actual=actual_columns,
                    )
                )
        else:
            # Non-strict: expected columns must appear in order, extras allowed
            expected_idx = 0
            for col in actual_columns:
                if expected_idx < len(self.expected_order):
                    if col == self.expected_order[expected_idx]:
                        expected_idx += 1

            if expected_idx < len(self.expected_order):
                missing = self.expected_order[expected_idx:]
                issues.append(
                    ValidationIssue(
                        column="_table_",
                        issue_type="column_order_mismatch",
                        count=len(missing),
                        severity=Severity.MEDIUM,
                        details=f"Columns not in expected order: {missing}",
                        expected=self.expected_order,
                        actual=actual_columns,
                    )
                )

        return issues
