"""Format validators for common patterns like email, phone, URL, etc."""

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


class BaseFormatValidator(Validator, StringValidatorMixin):
    """Base class for format validators."""

    pattern: re.Pattern[str]
    format_name: str = "format"

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
                if isinstance(val, str) and val.strip():
                    if not self.pattern.match(val):
                        invalid_count += 1
                        if len(samples) < self.config.sample_size:
                            samples.append(val[:50] + "..." if len(val) > 50 else val)

            if invalid_count > 0:
                ratio = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type=f"invalid_{self.format_name}",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Expected {self.format_name} format",
                        sample_values=samples,
                    )
                )

        return issues


@register_validator
class EmailValidator(BaseFormatValidator):
    """Validates email format."""

    name = "email"
    category = "string"
    format_name = "email"
    pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


@register_validator
class UrlValidator(BaseFormatValidator):
    """Validates URL format."""

    name = "url"
    category = "string"
    format_name = "url"
    pattern = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )


@register_validator
class PhoneValidator(BaseFormatValidator):
    """Validates phone number format (international)."""

    name = "phone"
    category = "string"
    format_name = "phone"
    pattern = re.compile(r"^[\d\s\-\+\(\)]{7,20}$")


@register_validator
class UuidValidator(BaseFormatValidator):
    """Validates UUID format."""

    name = "uuid"
    category = "string"
    format_name = "uuid"
    pattern = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )


@register_validator
class IpAddressValidator(BaseFormatValidator):
    """Validates IPv4 address format."""

    name = "ip_address"
    category = "string"
    format_name = "ip_address"
    pattern = re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )


@register_validator
class FormatValidator(Validator, StringValidatorMixin):
    """Auto-detects and validates format based on column name."""

    name = "format"
    category = "string"

    PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
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
            re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
            ),
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

    COLUMN_PATTERNS: dict[str, list[str]] = {
        "email": ["email", "e-mail", "mail"],
        "phone": ["phone", "tel", "mobile", "cell", "fax"],
        "url": ["url", "link", "website", "href"],
        "uuid": ["uuid", "guid", "id"],
        "ip": ["ip", "ip_address", "ipaddress"],
        "date": ["date", "dob", "birth"],
    }

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
            col_lower = col.lower()

            # Detect expected format
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

            invalid_count = 0
            for val in col_data.to_list():
                if isinstance(val, str) and val.strip():
                    if not pattern.match(val):
                        invalid_count += 1

            if invalid_count > 0:
                invalid_pct = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="invalid_format",
                        count=invalid_count,
                        severity=self._calculate_severity(invalid_pct),
                        details=f"Expected {format_name} format",
                    )
                )

        return issues
