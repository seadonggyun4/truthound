"""Decorators for custom validators."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class CustomValidator(Validator):
    """Wrapper for user-defined validator functions."""

    def __init__(self, func: Callable[[Any], bool], name: str | None = None):
        """Initialize custom validator.

        Args:
            func: Validation function that takes a value and returns bool.
            name: Optional name for the validator. Defaults to function name.
        """
        self._func = func
        self.name = name or func.__name__

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run the custom validation on all columns.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues found.
        """
        issues: list[ValidationIssue] = []
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in df.columns:
            col_data = df.get_column(col).drop_nulls()
            invalid_count = 0

            for val in col_data.to_list():
                try:
                    if not self._func(val):
                        invalid_count += 1
                except (TypeError, ValueError):
                    # Skip values that can't be validated by this function
                    pass

            if invalid_count > 0:
                invalid_pct = invalid_count / len(col_data) if len(col_data) > 0 else 0

                if invalid_pct > 0.3:
                    severity = Severity.HIGH
                elif invalid_pct > 0.1:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type=self.name,
                        count=invalid_count,
                        severity=severity,
                        details=f"Failed custom validation: {self.name}",
                    )
                )

        return issues


def validator(func: Callable[[Any], bool]) -> CustomValidator:
    """Decorator to create a custom validator from a function.

    Usage:
        @th.validator
        def check_positive(value: int) -> bool:
            return value > 0

        report = th.check(df, validators=[check_positive])

    Args:
        func: Validation function that takes a value and returns bool.

    Returns:
        CustomValidator instance.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> bool:
        return func(*args, **kwargs)

    return CustomValidator(func)
