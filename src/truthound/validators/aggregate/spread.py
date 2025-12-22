"""Spread validators (std, variance)."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    AggregateValidator,
)
from truthound.validators.registry import register_validator


@register_validator
class StdBetweenValidator(AggregateValidator):
    """Validates that column standard deviation is within expected range."""

    name = "std_between"
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
        std = stats.get("std")
        if std is None:
            return None

        if self.min_value is not None and std < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="std_below_min",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Std {std:.4f} < min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=std,
            )

        if self.max_value is not None and std > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="std_above_max",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Std {std:.4f} > max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=std,
            )

        return None


@register_validator
class VarianceBetweenValidator(AggregateValidator):
    """Validates that column variance is within expected range."""

    name = "variance_between"
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
        std = stats.get("std")
        if std is None:
            return None

        variance = std ** 2

        if self.min_value is not None and variance < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="variance_below_min",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Variance {variance:.4f} < min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=variance,
            )

        if self.max_value is not None and variance > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="variance_above_max",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Variance {variance:.4f} > max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=variance,
            )

        return None
