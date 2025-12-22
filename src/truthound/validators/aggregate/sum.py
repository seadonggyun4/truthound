"""Sum validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    AggregateValidator,
)
from truthound.validators.registry import register_validator


@register_validator
class SumBetweenValidator(AggregateValidator):
    """Validates that column sum is within expected range."""

    name = "sum_between"
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
        col_sum = stats.get("sum")
        if col_sum is None:
            return None

        if self.min_value is not None and col_sum < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="sum_below_min",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Sum {col_sum:.4f} < min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=col_sum,
            )

        if self.max_value is not None and col_sum > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="sum_above_max",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Sum {col_sum:.4f} > max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=col_sum,
            )

        return None
