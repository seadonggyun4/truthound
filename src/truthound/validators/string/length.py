"""String length validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class LengthValidator(Validator, StringValidatorMixin):
    """Validates that string lengths are within specified range."""

    name = "length"
    category = "string"

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        exact_length: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.exact_length = exact_length

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        exprs = [pl.len().alias("_total")]

        for col in columns:
            length_expr = pl.col(col).str.len_chars()

            if self.exact_length is not None:
                violation = (length_expr != self.exact_length) & pl.col(col).is_not_null()
            else:
                conditions = []
                if self.min_length is not None:
                    conditions.append(length_expr < self.min_length)
                if self.max_length is not None:
                    conditions.append(length_expr > self.max_length)

                if conditions:
                    violation = conditions[0]
                    for cond in conditions[1:]:
                        violation = violation | cond
                    violation = violation & pl.col(col).is_not_null()
                else:
                    continue

            exprs.append(violation.sum().alias(f"_inv_{col}"))

        if len(exprs) == 1:
            return issues

        # Use streaming for large datasets
        result = lf.select(exprs).collect(engine="streaming")
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            alias = f"_inv_{col}"
            if alias not in result.columns:
                continue

            invalid_count = result[alias][0]
            if invalid_count > 0:
                ratio = invalid_count / total_rows

                if self.exact_length is not None:
                    expected = f"exactly {self.exact_length}"
                elif self.min_length is not None and self.max_length is not None:
                    expected = f"[{self.min_length}, {self.max_length}]"
                elif self.min_length is not None:
                    expected = f">= {self.min_length}"
                else:
                    expected = f"<= {self.max_length}"

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="length_violation",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Length should be {expected}",
                        expected=expected,
                    )
                )

        return issues
