"""Quantile validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class QuantileValidator(Validator, NumericValidatorMixin):
    """Validates that quantile values are within expected range."""

    name = "quantile"
    category = "distribution"

    def __init__(
        self,
        quantile: float = 0.5,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.quantile = quantile
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.col(c).quantile(self.quantile).alias(f"_q_{c}")
            for c in columns
        ]
        result = lf.select(exprs).collect()

        for col in columns:
            q_value = result[f"_q_{col}"][0]

            if q_value is None:
                continue

            if self.min_value is not None and q_value < self.min_value:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="quantile_low",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Q{self.quantile:.0%}={q_value:.2f} < min {self.min_value}",
                        expected=f">= {self.min_value}",
                        actual=q_value,
                    )
                )

            if self.max_value is not None and q_value > self.max_value:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="quantile_high",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Q{self.quantile:.0%}={q_value:.2f} > max {self.max_value}",
                        expected=f"<= {self.max_value}",
                        actual=q_value,
                    )
                )

        return issues
