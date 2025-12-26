"""Date parsing validators."""

from datetime import datetime
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, StringValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class DateutilParseableValidator(Validator, StringValidatorMixin):
    """Validates that string values can be parsed as dates using flexible parsing.

    Supports various date formats without requiring explicit format specification.
    Uses dateutil library if available, otherwise falls back to common formats.

    Example:
        validator = DateutilParseableValidator(column="date_string")

        # With dateutil, these all parse correctly:
        # "2024-01-15"
        # "January 15, 2024"
        # "15/01/2024"
        # "Jan 15 2024"
    """

    name = "dateutil_parseable"
    category = "datetime"

    # Common date formats to try if dateutil is not available
    # Ordered by specificity: more specific formats first
    COMMON_FORMATS = [
        # ISO formats
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        # Day-Month-Year formats
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        # Month-Day-Year formats (US style)
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m-%d-%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        # Named month formats - with comma
        "%B %d, %Y",              # January 15, 2024
        "%b %d, %Y",              # Jan 15, 2024
        "%d %B, %Y",              # 15 January, 2024
        "%d %b, %Y",              # 15 Jan, 2024
        # Named month formats - without comma
        "%B %d %Y",               # January 15 2024
        "%b %d %Y",               # Jan 15 2024
        "%d %B %Y",               # 15 January 2024
        "%d %b %Y",               # 15 Jan 2024
        # Named month with hyphen
        "%d-%b-%Y",               # 15-Jan-2024
        "%d-%B-%Y",               # 15-January-2024
        "%b-%d-%Y",               # Jan-15-2024
        "%B-%d-%Y",               # January-15-2024
        # Named month with time - 12-hour format
        "%b %d %Y %I:%M %p",      # Jan 15 2024 10:30 AM
        "%B %d %Y %I:%M %p",      # January 15 2024 10:30 AM
        "%b %d, %Y %I:%M %p",     # Jan 15, 2024 10:30 AM
        "%B %d, %Y %I:%M %p",     # January 15, 2024 10:30 AM
        "%d %b %Y %I:%M %p",      # 15 Jan 2024 10:30 AM
        "%d %B %Y %I:%M %p",      # 15 January 2024 10:30 AM
        # Named month with time - 24-hour format
        "%b %d %Y %H:%M",         # Jan 15 2024 10:30
        "%B %d %Y %H:%M",         # January 15 2024 10:30
        "%b %d %Y %H:%M:%S",      # Jan 15 2024 10:30:00
        "%B %d %Y %H:%M:%S",      # January 15 2024 10:30:00
        # Other common formats
        "%Y%m%d",                 # 20240115
        "%d.%m.%Y",               # 15.01.2024
        "%Y.%m.%d",               # 2024.01.15
    ]

    def __init__(
        self,
        column: str | None = None,
        fuzzy: bool = True,
        dayfirst: bool = False,
        yearfirst: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.target_column = column
        self.fuzzy = fuzzy
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

        # Try to import dateutil
        self._parser = None
        try:
            from dateutil import parser
            self._parser = parser
        except ImportError:
            pass

    def _try_parse(self, value: str) -> bool:
        """Try to parse a date string."""
        if self._parser:
            try:
                self._parser.parse(
                    value,
                    fuzzy=self.fuzzy,
                    dayfirst=self.dayfirst,
                    yearfirst=self.yearfirst,
                )
                return True
            except (ValueError, TypeError):
                return False
        else:
            # Fallback: try common formats
            for fmt in self.COMMON_FORMATS:
                try:
                    datetime.strptime(value.strip(), fmt)
                    return True
                except ValueError:
                    continue
            return False

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
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

            unparseable_count = 0
            samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                if not self._try_parse(val):
                    unparseable_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val[:50] + "..." if len(val) > 50 else val)

            if unparseable_count > 0:
                if self._passes_mostly(unparseable_count, len(col_data)):
                    continue

                ratio = unparseable_count / len(col_data)
                parser_info = "dateutil" if self._parser else "standard formats"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="date_not_parseable",
                        count=unparseable_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values cannot be parsed as dates (using {parser_info})",
                        sample_values=samples,
                    )
                )

        return issues
