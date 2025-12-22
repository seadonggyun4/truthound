"""Extreme value validators (min, max)."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    AggregateValidator,
)
from truthound.validators.registry import register_validator


@register_validator
class MinBetweenValidator(AggregateValidator):
    """Validates that column minimum is within expected range."""

    name = "min_between"
    category = "aggregate"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def check_aggregate(
        self,
        col: str,
        stats: dict[str, Any],
        total_rows: int,
    ) -> ValidationIssue | None:
        col_min = stats.get("min")
        if col_min is None:
            return None

        if self.min_value is not None and col_min < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="min_below_expected",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Min {col_min} < expected min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=col_min,
            )

        if self.max_value is not None and col_min > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="min_above_expected",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Min {col_min} > expected max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=col_min,
            )

        return None


@register_validator
class MaxBetweenValidator(AggregateValidator):
    """Validates that column maximum is within expected range."""

    name = "max_between"
    category = "aggregate"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def check_aggregate(
        self,
        col: str,
        stats: dict[str, Any],
        total_rows: int,
    ) -> ValidationIssue | None:
        col_max = stats.get("max")
        if col_max is None:
            return None

        if self.min_value is not None and col_max < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="max_below_expected",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Max {col_max} < expected min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=col_max,
            )

        if self.max_value is not None and col_max > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="max_above_expected",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Max {col_max} > expected max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=col_max,
            )

        return None
