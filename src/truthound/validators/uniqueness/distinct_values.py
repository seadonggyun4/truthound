"""Distinct value validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class DistinctValuesInSetValidator(Validator):
    """Validates that all distinct values are within an allowed set.

    Example:
        # All status values should be in the allowed list
        validator = DistinctValuesInSetValidator(
            column="status",
            allowed_values=["pending", "active", "completed", "cancelled"],
        )
    """

    name = "distinct_values_in_set"
    category = "uniqueness"

    def __init__(
        self,
        column: str,
        allowed_values: set[Any] | list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.allowed_values = set(allowed_values)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get distinct values
        distinct_values = set(
            lf.select(pl.col(self.column).drop_nulls().unique())
            .collect()
            .to_series()
            .to_list()
        )

        # Find values not in allowed set
        unexpected = distinct_values - self.allowed_values

        if unexpected:
            samples = list(unexpected)[: self.config.sample_size]
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_values_not_in_set",
                    count=len(unexpected),
                    severity=Severity.HIGH,
                    details=f"{len(unexpected)} unexpected distinct values found",
                    expected=f"Values in {list(self.allowed_values)[:5]}{'...' if len(self.allowed_values) > 5 else ''}",
                    sample_values=samples,
                )
            )

        return issues


@register_validator
class DistinctValuesEqualSetValidator(Validator):
    """Validates that distinct values exactly match an expected set.

    Example:
        # Days of week should be exactly these 7 values
        validator = DistinctValuesEqualSetValidator(
            column="day_of_week",
            expected_values=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        )
    """

    name = "distinct_values_equal_set"
    category = "uniqueness"

    def __init__(
        self,
        column: str,
        expected_values: set[Any] | list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.expected_values = set(expected_values)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get distinct values
        distinct_values = set(
            lf.select(pl.col(self.column).drop_nulls().unique())
            .collect()
            .to_series()
            .to_list()
        )

        # Check for missing values
        missing = self.expected_values - distinct_values
        if missing:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_values_missing",
                    count=len(missing),
                    severity=Severity.HIGH,
                    details=f"Missing {len(missing)} expected distinct values",
                    expected=list(self.expected_values),
                    sample_values=list(missing)[: self.config.sample_size],
                )
            )

        # Check for unexpected values
        unexpected = distinct_values - self.expected_values
        if unexpected:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_values_unexpected",
                    count=len(unexpected),
                    severity=Severity.HIGH,
                    details=f"Found {len(unexpected)} unexpected distinct values",
                    expected=list(self.expected_values),
                    sample_values=list(unexpected)[: self.config.sample_size],
                )
            )

        return issues


@register_validator
class DistinctValuesContainSetValidator(Validator):
    """Validates that distinct values contain all values from a required set.

    Example:
        # Product categories must include these core categories
        validator = DistinctValuesContainSetValidator(
            column="category",
            required_values=["Electronics", "Clothing", "Food"],
        )
    """

    name = "distinct_values_contain_set"
    category = "uniqueness"

    def __init__(
        self,
        column: str,
        required_values: set[Any] | list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.required_values = set(required_values)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get distinct values
        distinct_values = set(
            lf.select(pl.col(self.column).drop_nulls().unique())
            .collect()
            .to_series()
            .to_list()
        )

        # Check if all required values are present
        missing = self.required_values - distinct_values

        if missing:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_values_missing_required",
                    count=len(missing),
                    severity=Severity.HIGH,
                    details=f"Missing {len(missing)} required values",
                    expected=list(self.required_values),
                    sample_values=list(missing)[: self.config.sample_size],
                )
            )

        return issues


@register_validator
class DistinctCountBetweenValidator(Validator):
    """Validates that distinct value count is within a range.

    Example:
        # Should have between 5 and 20 unique product categories
        validator = DistinctCountBetweenValidator(
            column="category",
            min_count=5,
            max_count=20,
        )
    """

    name = "distinct_count_between"
    category = "uniqueness"

    def __init__(
        self,
        column: str,
        min_count: int | None = None,
        max_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.min_count = min_count
        self.max_count = max_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        distinct_count = lf.select(pl.col(self.column).n_unique()).collect().item()

        # Check minimum
        if self.min_count is not None and distinct_count < self.min_count:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_count_too_low",
                    count=self.min_count - distinct_count,
                    severity=Severity.HIGH,
                    details=f"Distinct count {distinct_count} below minimum {self.min_count}",
                    expected=f">= {self.min_count}",
                    actual=distinct_count,
                )
            )

        # Check maximum
        if self.max_count is not None and distinct_count > self.max_count:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distinct_count_too_high",
                    count=distinct_count - self.max_count,
                    severity=Severity.HIGH,
                    details=f"Distinct count {distinct_count} above maximum {self.max_count}",
                    expected=f"<= {self.max_count}",
                    actual=distinct_count,
                )
            )

        return issues
