"""Null value validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    ValidatorConfig,
)
from truthound.validators.registry import register_validator


@register_validator
class NullValidator(Validator):
    """Detects null/missing values in columns."""

    name = "null"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        # Single optimized query
        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).null_count().alias(f"_null_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            null_count = result[f"_null_{col}"][0]
            if null_count > 0:
                null_pct = null_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="null",
                        count=null_count,
                        severity=self._calculate_severity(null_pct),
                        details=f"{null_pct:.1%} of values are null",
                    )
                )

        return issues


@register_validator
class NotNullValidator(Validator):
    """Validates that columns have no null values."""

    name = "not_null"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).null_count().alias(f"_null_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            null_count = result[f"_null_{col}"][0]
            if null_count > 0:
                null_pct = null_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_null_violation",
                        count=null_count,
                        severity=Severity.HIGH,
                        details=f"Expected no nulls, found {null_count} ({null_pct:.1%})",
                        expected=0,
                        actual=null_count,
                    )
                )

        return issues


@register_validator
class CompletenessRatioValidator(Validator):
    """Validates that columns meet a minimum completeness ratio."""

    name = "completeness_ratio"
    category = "completeness"

    def __init__(
        self,
        min_ratio: float = 0.95,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            non_null_count = result[f"_count_{col}"][0]
            ratio = non_null_count / total_rows

            if ratio < self.min_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="completeness_ratio",
                        count=total_rows - non_null_count,
                        severity=self._calculate_severity(
                            1 - ratio, thresholds=(0.5, 0.2, 0.05)
                        ),
                        details=f"Completeness {ratio:.1%} < {self.min_ratio:.1%}",
                        expected=self.min_ratio,
                        actual=ratio,
                    )
                )

        return issues
