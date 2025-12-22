"""SQL LIKE pattern validators."""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, StringValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class LikePatternValidator(Validator, StringValidatorMixin):
    """Validates that string values match SQL LIKE patterns.

    Supports SQL LIKE syntax:
    - % matches any sequence of characters (including empty)
    - _ matches any single character
    - Use \\ to escape % or _

    Example:
        # Product codes should start with 'PRD-'
        validator = LikePatternValidator(
            pattern="PRD-%",
            column="product_code",
        )

        # Email domain should be company.com
        validator = LikePatternValidator(
            pattern="%@company.com",
            column="email",
        )
    """

    name = "like_pattern"
    category = "string"

    def __init__(
        self,
        pattern: str,
        column: str | None = None,
        case_sensitive: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.target_column = column
        self.case_sensitive = case_sensitive
        self._regex = self._like_to_regex(pattern)

    def _like_to_regex(self, pattern: str) -> re.Pattern:
        """Convert SQL LIKE pattern to regex."""
        # Escape regex special characters except % and _
        regex_pattern = ""
        i = 0
        while i < len(pattern):
            char = pattern[i]

            if char == "\\" and i + 1 < len(pattern):
                # Escaped character
                next_char = pattern[i + 1]
                if next_char in ("%", "_"):
                    regex_pattern += re.escape(next_char)
                    i += 2
                    continue

            if char == "%":
                regex_pattern += ".*"
            elif char == "_":
                regex_pattern += "."
            else:
                regex_pattern += re.escape(char)

            i += 1

        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(f"^{regex_pattern}$", flags)

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

            invalid_count = 0
            samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                if not self._regex.match(val):
                    invalid_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val[:50] + "..." if len(val) > 50 else val)

            if invalid_count > 0:
                if self._passes_mostly(invalid_count, len(col_data)):
                    continue

                ratio = invalid_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="like_pattern_mismatch",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values don't match LIKE pattern: {self.pattern}",
                        expected=self.pattern,
                        sample_values=samples,
                    )
                )

        return issues


@register_validator
class NotLikePatternValidator(Validator, StringValidatorMixin):
    """Validates that string values do NOT match SQL LIKE patterns.

    Example:
        # Values should not contain 'test' anywhere
        validator = NotLikePatternValidator(
            pattern="%test%",
            column="name",
        )
    """

    name = "not_like_pattern"
    category = "string"

    def __init__(
        self,
        pattern: str,
        column: str | None = None,
        case_sensitive: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.target_column = column
        self.case_sensitive = case_sensitive
        self._regex = self._like_to_regex(pattern)

    def _like_to_regex(self, pattern: str) -> re.Pattern:
        """Convert SQL LIKE pattern to regex."""
        regex_pattern = ""
        i = 0
        while i < len(pattern):
            char = pattern[i]

            if char == "\\" and i + 1 < len(pattern):
                next_char = pattern[i + 1]
                if next_char in ("%", "_"):
                    regex_pattern += re.escape(next_char)
                    i += 2
                    continue

            if char == "%":
                regex_pattern += ".*"
            elif char == "_":
                regex_pattern += "."
            else:
                regex_pattern += re.escape(char)

            i += 1

        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(f"^{regex_pattern}$", flags)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
            columns = self._get_string_columns(lf)

        if not columns:
            return issues

        df = lf.collect()

        for col in columns:
            col_data = df.get_column(col).drop_nulls()

            if len(col_data) == 0:
                continue

            matching_count = 0
            samples = []

            for val in col_data.to_list():
                if not isinstance(val, str):
                    continue

                if self._regex.match(val):
                    matching_count += 1
                    if len(samples) < self.config.sample_size:
                        samples.append(val[:50] + "..." if len(val) > 50 else val)

            if matching_count > 0:
                if self._passes_mostly(matching_count, len(col_data)):
                    continue

                ratio = matching_count / len(col_data)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="like_pattern_unexpected_match",
                        count=matching_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Values should NOT match LIKE pattern: {self.pattern}",
                        expected=f"NOT LIKE '{self.pattern}'",
                        sample_values=samples,
                    )
                )

        return issues
