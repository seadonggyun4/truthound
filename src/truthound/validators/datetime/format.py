"""Date format validators."""

from datetime import datetime
from typing import Any

import polars as pl

from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class DateFormatValidator(Validator, StringValidatorMixin):
    """Validates that string values match a date format (strftime)."""

    name = "date_format"
    category = "datetime"

    def __init__(
        self,
        format: str = "%Y-%m-%d",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.format = format

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

                try:
                    datetime.strptime(val, self.format)
                except ValueError:
                    invalid_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val)

            if invalid_count > 0:
                ratio = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="invalid_date_format",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Expected format: {self.format}",
                        expected=self.format,
                        sample_values=samples,
                    )
                )

        return issues
