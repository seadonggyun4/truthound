"""Monotonic sequence validators."""

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
class IncreasingValidator(Validator, NumericValidatorMixin):
    """Validates that values are monotonically increasing."""

    name = "increasing"
    category = "distribution"

    def __init__(
        self,
        strict: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.strict = strict

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows < 2:
            return issues

        for col in columns:
            col_data = df.get_column(col)
            diff = col_data.diff()

            if self.strict:
                violations = (diff <= 0).sum() - 1  # First diff is null
            else:
                violations = (diff < 0).sum()

            if violations > 0:
                ratio = violations / (total_rows - 1)
                mode = "strictly increasing" if self.strict else "increasing"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_increasing",
                        count=violations,
                        severity=self._calculate_severity(ratio),
                        details=f"{violations} violations of {mode} order",
                    )
                )

        return issues


@register_validator
class DecreasingValidator(Validator, NumericValidatorMixin):
    """Validates that values are monotonically decreasing."""

    name = "decreasing"
    category = "distribution"

    def __init__(
        self,
        strict: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.strict = strict

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows < 2:
            return issues

        for col in columns:
            col_data = df.get_column(col)
            diff = col_data.diff()

            if self.strict:
                violations = (diff >= 0).sum() - 1  # First diff is null
            else:
                violations = (diff > 0).sum()

            if violations > 0:
                ratio = violations / (total_rows - 1)
                mode = "strictly decreasing" if self.strict else "decreasing"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_decreasing",
                        count=violations,
                        severity=self._calculate_severity(ratio),
                        details=f"{violations} violations of {mode} order",
                    )
                )

        return issues
