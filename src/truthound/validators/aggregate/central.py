"""Central tendency validators (mean, median)."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    AggregateValidator,
)
from truthound.validators.registry import register_validator


@register_validator
class MeanBetweenValidator(AggregateValidator):
    """Validates that column mean is within expected range."""

    name = "mean_between"
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
        mean = stats.get("mean")
        if mean is None:
            return None

        if self.min_value is not None and mean < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="mean_below_min",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Mean {mean:.4f} < min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=mean,
            )

        if self.max_value is not None and mean > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="mean_above_max",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Mean {mean:.4f} > max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=mean,
            )

        return None


@register_validator
class MedianBetweenValidator(AggregateValidator):
    """Validates that column median is within expected range."""

    name = "median_between"
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
        median = stats.get("median")
        if median is None:
            return None

        if self.min_value is not None and median < self.min_value:
            return ValidationIssue(
                column=col,
                issue_type="median_below_min",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Median {median:.4f} < min {self.min_value}",
                expected=f">= {self.min_value}",
                actual=median,
            )

        if self.max_value is not None and median > self.max_value:
            return ValidationIssue(
                column=col,
                issue_type="median_above_max",
                count=1,
                severity=Severity.MEDIUM,
                details=f"Median {median:.4f} > max {self.max_value}",
                expected=f"<= {self.max_value}",
                actual=median,
            )

        return None
