"""Date range validators."""

from datetime import date, datetime
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    DatetimeValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class DateBetweenValidator(Validator, DatetimeValidatorMixin):
    """Validates that date values are within a specified range."""

    name = "date_between"
    category = "datetime"

    def __init__(
        self,
        min_date: date | datetime | str | None = None,
        max_date: date | datetime | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_date = self._parse_date(min_date) if min_date else None
        self.max_date = self._parse_date(max_date) if max_date else None

    def _parse_date(self, d: date | datetime | str) -> date:
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        return datetime.strptime(d, "%Y-%m-%d").date()

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_datetime_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()
            out_count = 0

            for val in col_data.to_list():
                if val is None:
                    continue

                # Convert to date if datetime
                if isinstance(val, datetime):
                    val = val.date()

                if self.min_date and val < self.min_date:
                    out_count += 1
                elif self.max_date and val > self.max_date:
                    out_count += 1

            if out_count > 0:
                ratio = out_count / len(col_data)
                range_str = f"[{self.min_date}, {self.max_date}]"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="date_out_of_range",
                        count=out_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Dates outside {range_str}",
                        expected=range_str,
                    )
                )

        return issues


@register_validator
class FutureDateValidator(Validator, DatetimeValidatorMixin):
    """Validates that dates are not in the future."""

    name = "future_date"
    category = "datetime"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_datetime_columns(lf)

        if not columns:
            return issues

        today = date.today()
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()
            future_count = 0

            for val in col_data.to_list():
                if val is None:
                    continue

                if isinstance(val, datetime):
                    val = val.date()

                if val > today:
                    future_count += 1

            if future_count > 0:
                ratio = future_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="future_date",
                        count=future_count,
                        severity=Severity.HIGH,
                        details=f"{future_count} dates are in the future",
                        expected=f"<= {today}",
                    )
                )

        return issues


@register_validator
class PastDateValidator(Validator, DatetimeValidatorMixin):
    """Validates that dates are not in the past (before a threshold)."""

    name = "past_date"
    category = "datetime"

    def __init__(
        self,
        min_date: date | datetime | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if min_date:
            if isinstance(min_date, str):
                self.min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
            elif isinstance(min_date, datetime):
                self.min_date = min_date.date()
            else:
                self.min_date = min_date
        else:
            self.min_date = date(1900, 1, 1)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_datetime_columns(lf)

        if not columns:
            return issues

        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_data = df.get_column(col).drop_nulls()
            past_count = 0

            for val in col_data.to_list():
                if val is None:
                    continue

                if isinstance(val, datetime):
                    val = val.date()

                if val < self.min_date:
                    past_count += 1

            if past_count > 0:
                ratio = past_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="past_date",
                        count=past_count,
                        severity=Severity.MEDIUM,
                        details=f"{past_count} dates before {self.min_date}",
                        expected=f">= {self.min_date}",
                    )
                )

        return issues
