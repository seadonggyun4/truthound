"""Column pair relationship validators."""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnPairValidator(Validator):
    """Validates relationships between two columns.

    Example:
        # price should be less than max_price
        validator = ColumnPairValidator(
            column_a="price",
            column_b="max_price",
            relationship="<",
        )
    """

    name = "column_pair"
    category = "schema"

    OPERATORS = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(
        self,
        column_a: str,
        column_b: str,
        relationship: Literal["<", "<=", ">", ">=", "==", "!="],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column_a = column_a
        self.column_b = column_b
        self.relationship = relationship

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        op = self.OPERATORS.get(self.relationship)
        if op is None:
            return issues

        # Build violation expression
        violation_expr = ~op(pl.col(self.column_a), pl.col(self.column_b))

        result = lf.select([
            pl.len().alias("_total"),
            (
                violation_expr
                & pl.col(self.column_a).is_not_null()
                & pl.col(self.column_b).is_not_null()
            ).sum().alias("_violations"),
        ]).collect()

        violations = result["_violations"][0]

        if violations > 0:
            total = result["_total"][0]
            ratio = violations / total if total > 0 else 0

            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="column_pair_violation",
                    count=violations,
                    severity=self._calculate_severity(ratio),
                    details=f"Expected {self.column_a} {self.relationship} {self.column_b}",
                    expected=f"{self.column_a} {self.relationship} {self.column_b}",
                )
            )

        return issues
