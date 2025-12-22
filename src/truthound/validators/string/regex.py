"""Regex pattern validators."""

import re
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
class RegexValidator(Validator, StringValidatorMixin):
    """Validates that string values match a regex pattern."""

    name = "regex"
    category = "string"

    def __init__(
        self,
        pattern: str,
        match_full: bool = True,
        dotall: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.match_full = match_full
        # Use DOTALL flag to make . match newlines
        flags = re.DOTALL if dotall else 0
        self._compiled = re.compile(pattern, flags)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            invalid_count = 0
            samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                if self.match_full:
                    match = self._compiled.fullmatch(val)
                else:
                    match = self._compiled.search(val)

                if not match:
                    invalid_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val[:50] + "..." if len(val) > 50 else val)

            if invalid_count > 0:
                ratio = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="regex_mismatch",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Pattern: {self.pattern}",
                        expected=self.pattern,
                        sample_values=samples,
                    )
                )

        return issues
