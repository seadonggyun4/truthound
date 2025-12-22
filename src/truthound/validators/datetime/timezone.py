"""Timezone validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class TimezoneValidator(Validator):
    """Validates timezone consistency in datetime columns."""

    name = "timezone"
    category = "datetime"

    def __init__(
        self,
        expected_timezone: str | None = None,
        require_timezone: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.expected_timezone = expected_timezone
        self.require_timezone = require_timezone

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        schema = lf.collect_schema()
        columns = self._get_target_columns(lf)

        for col in columns:
            dtype = schema[col]

            # Only check Datetime columns
            if not isinstance(dtype, pl.Datetime):
                continue

            tz = dtype.time_zone

            if self.require_timezone and tz is None:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="missing_timezone",
                        count=1,
                        severity=Severity.MEDIUM,
                        details="Datetime column has no timezone",
                        expected="timezone-aware",
                        actual="timezone-naive",
                    )
                )

            if self.expected_timezone and tz != self.expected_timezone:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="wrong_timezone",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Expected {self.expected_timezone}, got {tz}",
                        expected=self.expected_timezone,
                        actual=tz,
                    )
                )

        return issues
