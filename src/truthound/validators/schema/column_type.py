"""Column type validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnTypeValidator(Validator):
    """Validates that columns have expected data types."""

    name = "column_type"
    category = "schema"

    # Common type mappings
    TYPE_ALIASES: dict[str, set[type[pl.DataType]]] = {
        "int": {pl.Int8, pl.Int16, pl.Int32, pl.Int64},
        "uint": {pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64},
        "float": {pl.Float32, pl.Float64},
        "numeric": {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        },
        "string": {pl.String, pl.Utf8},
        "bool": {pl.Boolean},
        "date": {pl.Date},
        "datetime": {pl.Datetime},
        "time": {pl.Time},
    }

    def __init__(
        self,
        expected_types: dict[str, str | type[pl.DataType]],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_types = expected_types

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()

        for col, expected in self.expected_types.items():
            if col not in schema.names():
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="missing_column",
                        count=1,
                        severity=Severity.CRITICAL,
                        details=f"Column '{col}' not found",
                    )
                )
                continue

            actual_type = type(schema[col])

            # Handle string type aliases
            if isinstance(expected, str):
                expected_lower = expected.lower()
                if expected_lower in self.TYPE_ALIASES:
                    if actual_type not in self.TYPE_ALIASES[expected_lower]:
                        issues.append(
                            ValidationIssue(
                                column=col,
                                issue_type="type_mismatch",
                                count=1,
                                severity=Severity.HIGH,
                                details=f"Expected {expected}, got {actual_type.__name__}",
                                expected=expected,
                                actual=actual_type.__name__,
                            )
                        )
            else:
                # Direct type comparison
                if actual_type != expected:
                    issues.append(
                        ValidationIssue(
                            column=col,
                            issue_type="type_mismatch",
                            count=1,
                            severity=Severity.HIGH,
                            details=f"Expected {expected.__name__}, got {actual_type.__name__}",
                            expected=expected.__name__,
                            actual=actual_type.__name__,
                        )
                    )

        return issues
