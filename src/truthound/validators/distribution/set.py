"""Set membership validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class InSetValidator(Validator):
    """Validates that values are in a specified set."""

    name = "in_set"
    category = "distribution"

    def __init__(
        self,
        allowed_values: list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.allowed_values = allowed_values

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                (
                    ~pl.col(c).is_in(self.allowed_values) & pl.col(c).is_not_null()
                ).sum().alias(f"_not_in_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            not_in_count = result[f"_not_in_{col}"][0]
            if not_in_count > 0:
                ratio = not_in_count / total_rows
                allowed_str = str(self.allowed_values[:5])
                if len(self.allowed_values) > 5:
                    allowed_str = allowed_str[:-1] + ", ...]"

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_in_set",
                        count=not_in_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{not_in_count} values not in allowed set",
                        expected=allowed_str,
                    )
                )

        return issues


@register_validator
class NotInSetValidator(Validator):
    """Validates that values are NOT in a specified set (forbidden values)."""

    name = "not_in_set"
    category = "distribution"

    def __init__(
        self,
        forbidden_values: list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.forbidden_values = forbidden_values

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                pl.col(c).is_in(self.forbidden_values).sum().alias(f"_in_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            in_count = result[f"_in_{col}"][0]
            if in_count > 0:
                ratio = in_count / total_rows
                forbidden_str = str(self.forbidden_values[:5])
                if len(self.forbidden_values) > 5:
                    forbidden_str = forbidden_str[:-1] + ", ...]"

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="in_forbidden_set",
                        count=in_count,
                        severity=Severity.HIGH,
                        details=f"{in_count} forbidden values found",
                        expected=f"Not in {forbidden_str}",
                    )
                )

        return issues
