"""Cross-table row count validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class CrossTableRowCountValidator(Validator):
    """Validates that row counts between two tables match.

    Example:
        # Orders should have same count as order_details
        validator = CrossTableRowCountValidator(
            reference_data=order_details_df,
            reference_name="order_details",
        )
        issues = validator.validate(orders_df.lazy())
    """

    name = "cross_table_row_count"
    category = "cross_table"

    def __init__(
        self,
        reference_data: pl.DataFrame | pl.LazyFrame,
        reference_name: str = "reference",
        tolerance: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.reference_name = reference_name
        self.tolerance = tolerance

        # Get reference row count
        if isinstance(reference_data, pl.LazyFrame):
            self.reference_count = reference_data.select(pl.len()).collect().item()
        else:
            self.reference_count = len(reference_data)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        source_count = lf.select(pl.len()).collect().item()
        diff = abs(source_count - self.reference_count)

        if diff > self.tolerance:
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="cross_table_row_count_mismatch",
                    count=diff,
                    severity=Severity.CRITICAL,
                    details=f"Row count mismatch with {self.reference_name}",
                    expected=self.reference_count,
                    actual=source_count,
                )
            )

        return issues


@register_validator
class CrossTableRowCountFactorValidator(Validator):
    """Validates that row count equals reference count times a factor.

    Example:
        # Daily aggregates should be ~30x monthly aggregates
        validator = CrossTableRowCountFactorValidator(
            reference_data=monthly_df,
            factor=30,
            tolerance_ratio=0.1,  # Allow 10% variance
        )
    """

    name = "cross_table_row_count_factor"
    category = "cross_table"

    def __init__(
        self,
        reference_data: pl.DataFrame | pl.LazyFrame,
        factor: float,
        reference_name: str = "reference",
        tolerance_ratio: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.factor = factor
        self.reference_name = reference_name
        self.tolerance_ratio = tolerance_ratio

        if isinstance(reference_data, pl.LazyFrame):
            self.reference_count = reference_data.select(pl.len()).collect().item()
        else:
            self.reference_count = len(reference_data)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        source_count = lf.select(pl.len()).collect().item()
        expected_count = self.reference_count * self.factor
        tolerance = expected_count * self.tolerance_ratio

        if abs(source_count - expected_count) > tolerance:
            ratio = source_count / expected_count if expected_count > 0 else float("inf")
            issues.append(
                ValidationIssue(
                    column="_table",
                    issue_type="cross_table_row_count_factor_mismatch",
                    count=abs(int(source_count - expected_count)),
                    severity=Severity.HIGH,
                    details=f"Expected {self.reference_name} × {self.factor}",
                    expected=f"{expected_count:.0f} (±{self.tolerance_ratio:.0%})",
                    actual=f"{source_count} (ratio: {ratio:.2f}x)",
                )
            )

        return issues
