"""Extended regex validators for complex pattern matching."""

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
class RegexListValidator(Validator, StringValidatorMixin):
    """Validates that values match at least one pattern from a list.

    Example:
        # Value should match at least one date format
        validator = RegexListValidator(
            patterns=[
                r"\\d{4}-\\d{2}-\\d{2}",  # YYYY-MM-DD
                r"\\d{2}/\\d{2}/\\d{4}",  # DD/MM/YYYY
                r"\\d{2}\\.\\d{2}\\.\\d{4}",  # DD.MM.YYYY
            ],
            match_mode="any",  # Match any pattern
        )
    """

    name = "regex_list"
    category = "string"

    def __init__(
        self,
        patterns: list[str],
        match_mode: str = "any",  # "any" = match at least one, "all" = match all
        match_full: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.patterns = patterns
        self.match_mode = match_mode
        self.match_full = match_full
        self._compiled = [re.compile(p, re.DOTALL) for p in patterns]

    def _matches_value(self, val: str) -> bool:
        """Check if value matches based on match_mode."""
        if self.match_mode == "any":
            # At least one pattern must match
            for pattern in self._compiled:
                if self.match_full:
                    if pattern.fullmatch(val):
                        return True
                else:
                    if pattern.search(val):
                        return True
            return False
        else:  # match_mode == "all"
            # All patterns must match
            for pattern in self._compiled:
                if self.match_full:
                    if not pattern.fullmatch(val):
                        return False
                else:
                    if not pattern.search(val):
                        return False
            return True

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

                if not self._matches_value(val):
                    invalid_count += 1
                    if len(samples) < self.config.sample_size:
                        display_val = val[:50] + "..." if len(val) > 50 else val
                        samples.append(display_val)

            if invalid_count > 0:
                # Check mostly threshold
                if self._passes_mostly(invalid_count, len(col_data)):
                    continue

                ratio = invalid_count / len(col_data)
                mode_desc = "any of" if self.match_mode == "any" else "all of"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="regex_list_mismatch",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values must match {mode_desc}: {self.patterns[:3]}{'...' if len(self.patterns) > 3 else ''}",
                        expected=f"Match {mode_desc} {len(self.patterns)} patterns",
                        sample_values=samples,
                    )
                )

        return issues


@register_validator
class NotMatchRegexValidator(Validator, StringValidatorMixin):
    """Validates that values do NOT match a regex pattern.

    Example:
        # Values should not contain PII patterns
        validator = NotMatchRegexValidator(
            pattern=r"\\d{3}-\\d{2}-\\d{4}",  # SSN pattern
        )
    """

    name = "not_match_regex"
    category = "string"

    def __init__(
        self,
        pattern: str,
        match_full: bool = False,  # Default to search mode for "not match"
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.match_full = match_full
        self._compiled = re.compile(pattern, re.DOTALL)

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

            matching_count = 0
            samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                if self.match_full:
                    match = self._compiled.fullmatch(val)
                else:
                    match = self._compiled.search(val)

                if match:
                    matching_count += 1
                    if len(samples) < self.config.sample_size:
                        # Mask matched portion for privacy
                        display_val = val[:50] + "..." if len(val) > 50 else val
                        samples.append(display_val)

            if matching_count > 0:
                # Check mostly threshold
                if self._passes_mostly(matching_count, len(col_data)):
                    continue

                ratio = matching_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="regex_unexpected_match",
                        count=matching_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values should NOT match: {self.pattern}",
                        expected=f"No match for {self.pattern}",
                        sample_values=samples,
                    )
                )

        return issues


@register_validator
class NotMatchRegexListValidator(Validator, StringValidatorMixin):
    """Validates that values do NOT match any pattern from a list.

    Example:
        # Values should not contain any PII patterns
        validator = NotMatchRegexListValidator(
            patterns=[
                r"\\d{3}-\\d{2}-\\d{4}",  # SSN
                r"\\d{16}",  # Credit card
                r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}",  # Email
            ],
        )
    """

    name = "not_match_regex_list"
    category = "string"

    def __init__(
        self,
        patterns: list[str],
        match_full: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.patterns = patterns
        self.match_full = match_full
        self._compiled = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in patterns]

    def _matches_any(self, val: str) -> tuple[bool, str | None]:
        """Check if value matches any forbidden pattern."""
        for i, pattern in enumerate(self._compiled):
            if self.match_full:
                if pattern.fullmatch(val):
                    return True, self.patterns[i]
            else:
                if pattern.search(val):
                    return True, self.patterns[i]
        return False, None

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

            matching_count = 0
            samples = []
            matched_patterns: set[str] = set()

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                matched, pattern = self._matches_any(val)
                if matched:
                    matching_count += 1
                    if pattern:
                        matched_patterns.add(pattern)
                    if len(samples) < self.config.sample_size:
                        display_val = val[:50] + "..." if len(val) > 50 else val
                        samples.append(display_val)

            if matching_count > 0:
                if self._passes_mostly(matching_count, len(col_data)):
                    continue

                ratio = matching_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="regex_list_unexpected_match",
                        count=matching_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values should NOT match any forbidden pattern",
                        expected=f"No match for any of {len(self.patterns)} patterns",
                        sample_values=samples,
                    )
                )

        return issues
