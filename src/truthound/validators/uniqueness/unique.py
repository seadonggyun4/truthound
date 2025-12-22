"""Unique value validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class UniqueValidator(Validator):
    """Validates that column values are unique (no duplicates)."""

    name = "unique"
    category = "uniqueness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]
            duplicate_count = non_null_count - unique_count

            if duplicate_count > 0:
                dup_pct = duplicate_count / non_null_count if non_null_count > 0 else 0
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_violation",
                        count=duplicate_count,
                        severity=self._calculate_severity(
                            dup_pct, thresholds=(0.3, 0.1, 0.01)
                        ),
                        details=f"{duplicate_count} duplicate values ({dup_pct:.1%})",
                    )
                )

        return issues


@register_validator
class UniqueRatioValidator(Validator):
    """Validates that uniqueness ratio is within expected range."""

    name = "unique_ratio"
    category = "uniqueness"

    def __init__(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if non_null_count == 0:
                continue

            ratio = unique_count / non_null_count

            if self.min_ratio is not None and ratio < self.min_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_ratio_low",
                        count=int((self.min_ratio - ratio) * non_null_count),
                        severity=Severity.MEDIUM,
                        details=f"Unique ratio {ratio:.1%} < min {self.min_ratio:.1%}",
                        expected=f">= {self.min_ratio:.1%}",
                        actual=f"{ratio:.1%}",
                    )
                )

            if self.max_ratio is not None and ratio > self.max_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_ratio_high",
                        count=int((ratio - self.max_ratio) * non_null_count),
                        severity=Severity.MEDIUM,
                        details=f"Unique ratio {ratio:.1%} > max {self.max_ratio:.1%}",
                        expected=f"<= {self.max_ratio:.1%}",
                        actual=f"{ratio:.1%}",
                    )
                )

        return issues


@register_validator
class DistinctCountValidator(Validator):
    """Validates that distinct count is within expected range."""

    name = "distinct_count"
    category = "uniqueness"

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]

            if self.min_count is not None and unique_count < self.min_count:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="distinct_count_low",
                        count=self.min_count - unique_count,
                        severity=Severity.MEDIUM,
                        details=f"Distinct count {unique_count} < min {self.min_count}",
                        expected=f">= {self.min_count}",
                        actual=unique_count,
                    )
                )

            if self.max_count is not None and unique_count > self.max_count:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="distinct_count_high",
                        count=unique_count - self.max_count,
                        severity=Severity.MEDIUM,
                        details=f"Distinct count {unique_count} > max {self.max_count}",
                        expected=f"<= {self.max_count}",
                        actual=unique_count,
                    )
                )

        return issues
