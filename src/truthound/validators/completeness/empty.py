"""Empty string validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class EmptyStringValidator(Validator, StringValidatorMixin):
    """Detects empty strings in columns."""

    name = "empty_string"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                (pl.col(c) == "").sum().alias(f"_empty_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            empty_count = result[f"_empty_{col}"][0]
            if empty_count > 0:
                empty_pct = empty_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="empty_string",
                        count=empty_count,
                        severity=self._calculate_severity(empty_pct),
                        details=f"{empty_pct:.1%} of values are empty strings",
                    )
                )

        return issues


@register_validator
class WhitespaceOnlyValidator(Validator, StringValidatorMixin):
    """Detects strings that contain only whitespace."""

    name = "whitespace_only"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                (
                    (pl.col(c).str.strip_chars() == "") & pl.col(c).is_not_null()
                ).sum().alias(f"_ws_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            ws_count = result[f"_ws_{col}"][0]
            if ws_count > 0:
                ws_pct = ws_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="whitespace_only",
                        count=ws_count,
                        severity=self._calculate_severity(ws_pct),
                        details=f"{ws_pct:.1%} of values contain only whitespace",
                    )
                )

        return issues
