"""Character set validators."""

import re
from typing import Any

import polars as pl

from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class AlphanumericValidator(Validator, StringValidatorMixin):
    """Validates that string values contain only alphanumeric characters."""

    name = "alphanumeric"
    category = "string"

    def __init__(
        self,
        allow_underscore: bool = False,
        allow_hyphen: bool = False,
        allow_space: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.allow_underscore = allow_underscore
        self.allow_hyphen = allow_hyphen
        self.allow_space = allow_space

        # Build pattern
        pattern = r"^[a-zA-Z0-9"
        if allow_underscore:
            pattern += "_"
        if allow_hyphen:
            pattern += "-"
        if allow_space:
            pattern += r"\s"
        pattern += r"]+$"
        self._pattern = re.compile(pattern)

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
                if isinstance(val, str) and val:
                    if not self._pattern.match(val):
                        invalid_count += 1
                        if len(samples) < self.config.sample_size:
                            samples.append(val[:50] + "..." if len(val) > 50 else val)

            if invalid_count > 0:
                ratio = invalid_count / len(col_data)
                allowed = ["alphanumeric"]
                if self.allow_underscore:
                    allowed.append("underscore")
                if self.allow_hyphen:
                    allowed.append("hyphen")
                if self.allow_space:
                    allowed.append("space")

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="non_alphanumeric",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Only {', '.join(allowed)} allowed",
                        sample_values=samples,
                    )
                )

        return issues
