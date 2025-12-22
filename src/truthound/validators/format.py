"""Format validator for common patterns like email, phone, etc."""

import re

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class FormatValidator(Validator):
    """Validates format patterns for emails, phone numbers, etc."""

    name = "format"

    # Patterns for common formats
    PATTERNS: dict[str, tuple[str, re.Pattern]] = {
        "email": (
            "email",
            re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        ),
        "phone": (
            "phone",
            re.compile(r"^[\d\s\-\+\(\)]{7,20}$"),
        ),
        "url": (
            "url",
            re.compile(r"^https?://[^\s]+$"),
        ),
        "uuid": (
            "uuid",
            re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        ),
        "ip": (
            "ip_address",
            re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
        ),
        "date": (
            "date",
            re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        ),
    }

    # Column name patterns that suggest a specific format
    COLUMN_PATTERNS: dict[str, list[str]] = {
        "email": ["email", "e-mail", "mail"],
        "phone": ["phone", "tel", "mobile", "cell", "fax"],
        "url": ["url", "link", "website", "href"],
        "uuid": ["uuid", "guid", "id"],
        "ip": ["ip", "ip_address", "ipaddress"],
        "date": ["date", "dob", "birth"],
    }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for format violations in string columns.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with format violations.
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in schema.names():
            dtype = schema[col]

            if dtype not in (pl.String, pl.Utf8):
                continue

            col_lower = col.lower()

            # Determine expected format based on column name
            expected_format = None
            for fmt, patterns in self.COLUMN_PATTERNS.items():
                if any(p in col_lower for p in patterns):
                    expected_format = fmt
                    break

            if expected_format is None:
                continue

            format_name, pattern = self.PATTERNS[expected_format]
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            # Count invalid format values
            invalid_count = 0
            for val in col_data.to_list():
                if isinstance(val, str) and val.strip():
                    if not pattern.match(val):
                        invalid_count += 1

            if invalid_count > 0:
                invalid_pct = invalid_count / len(col_data)

                if invalid_pct > 0.3:
                    severity = Severity.HIGH
                elif invalid_pct > 0.1:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="invalid_format",
                        count=invalid_count,
                        severity=severity,
                        details=f"expected {format_name} format",
                    )
                )

        return issues
