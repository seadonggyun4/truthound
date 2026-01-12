"""SQL LIKE pattern validators."""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
    SampledEarlyTerminationMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class LikePatternValidator(Validator, StringValidatorMixin, SampledEarlyTerminationMixin):
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

    def _get_regex_pattern_str(self) -> str:
        """Get regex pattern string for Polars."""
        regex_pattern = ""
        i = 0
        while i < len(self.pattern):
            char = self.pattern[i]

            if char == "\\" and i + 1 < len(self.pattern):
                next_char = self.pattern[i + 1]
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

        # Add case-insensitive flag if needed
        if not self.case_sensitive:
            return f"(?i)^{regex_pattern}$"
        return f"^{regex_pattern}$"

    def _build_invalid_expr(self, col: str) -> pl.Expr:
        """Build expression that returns True for invalid values."""
        regex_pattern = self._get_regex_pattern_str()
        return pl.col(col).is_not_null() & ~pl.col(col).str.contains(regex_pattern)

    def _get_invalid_samples(self, lf: pl.LazyFrame, col: str) -> list[str]:
        """Get sample invalid values for error reporting."""
        regex_pattern = self._get_regex_pattern_str()
        samples_df = lf.filter(
            pl.col(col).is_not_null() & ~pl.col(col).str.contains(regex_pattern)
        ).select(pl.col(col)).head(self.config.sample_size).collect(engine="streaming")

        return [
            (v[:50] + "..." if len(v) > 50 else v)
            for v in samples_df[col].to_list()
            if isinstance(v, str)
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
            columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Check for early termination opportunity
        early_results = self._check_early_termination(
            lf,
            columns=columns,
            build_invalid_expr=self._build_invalid_expr,
        )

        # Separate columns into early-terminatable and full-validation needed
        early_term_cols: list[str] = []
        full_validate_cols: list[str] = []

        for col in columns:
            if early_results[col].should_terminate:
                early_term_cols.append(col)
            else:
                full_validate_cols.append(col)

        # Process early termination columns
        for col in early_term_cols:
            result = early_results[col]

            # Check mostly threshold with extrapolated values
            if self._passes_mostly(result.estimated_fail_count, result.total_rows):
                continue

            samples = self._get_invalid_samples(lf.head(self.early_termination_sample_size), col)

            issues.append(
                self._build_early_termination_issue(
                    col=col,
                    result=result,
                    issue_type="like_pattern_mismatch",
                    details=f"Values don't match LIKE pattern: {self.pattern}",
                    expected=self.pattern,
                    sample_values=samples,
                )
            )

        # Full validation for remaining columns
        if not full_validate_cols:
            return issues

        regex_pattern = self._get_regex_pattern_str()

        for col in full_validate_cols:
            # Use Polars native regex matching with streaming
            result = lf.select([
                pl.col(col).is_not_null().sum().alias("non_null_count"),
                self._build_invalid_expr(col).sum().alias("invalid_count"),
            ]).collect(engine="streaming")

            non_null_count = result["non_null_count"][0]
            invalid_count = result["invalid_count"][0]

            if non_null_count == 0 or invalid_count == 0:
                continue

            if self._passes_mostly(invalid_count, non_null_count):
                continue

            samples = self._get_invalid_samples(lf, col)

            ratio = invalid_count / non_null_count
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
class NotLikePatternValidator(Validator, StringValidatorMixin, SampledEarlyTerminationMixin):
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

    def _get_regex_pattern_str(self) -> str:
        """Get regex pattern string for Polars."""
        regex_pattern = ""
        i = 0
        while i < len(self.pattern):
            char = self.pattern[i]

            if char == "\\" and i + 1 < len(self.pattern):
                next_char = self.pattern[i + 1]
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

        # Add case-insensitive flag if needed
        if not self.case_sensitive:
            return f"(?i)^{regex_pattern}$"
        return f"^{regex_pattern}$"

    def _build_invalid_expr(self, col: str) -> pl.Expr:
        """Build expression that returns True for invalid values (matching the pattern)."""
        regex_pattern = self._get_regex_pattern_str()
        return pl.col(col).is_not_null() & pl.col(col).str.contains(regex_pattern)

    def _get_invalid_samples(self, lf: pl.LazyFrame, col: str) -> list[str]:
        """Get sample invalid values for error reporting."""
        regex_pattern = self._get_regex_pattern_str()
        samples_df = lf.filter(
            pl.col(col).is_not_null() & pl.col(col).str.contains(regex_pattern)
        ).select(pl.col(col)).head(self.config.sample_size).collect(engine="streaming")

        return [
            (v[:50] + "..." if len(v) > 50 else v)
            for v in samples_df[col].to_list()
            if isinstance(v, str)
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.target_column:
            columns = [self.target_column]
        else:
            columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Check for early termination opportunity
        early_results = self._check_early_termination(
            lf,
            columns=columns,
            build_invalid_expr=self._build_invalid_expr,
        )

        # Separate columns into early-terminatable and full-validation needed
        early_term_cols: list[str] = []
        full_validate_cols: list[str] = []

        for col in columns:
            if early_results[col].should_terminate:
                early_term_cols.append(col)
            else:
                full_validate_cols.append(col)

        # Process early termination columns
        for col in early_term_cols:
            result = early_results[col]

            # Check mostly threshold with extrapolated values
            if self._passes_mostly(result.estimated_fail_count, result.total_rows):
                continue

            samples = self._get_invalid_samples(lf.head(self.early_termination_sample_size), col)

            issues.append(
                self._build_early_termination_issue(
                    col=col,
                    result=result,
                    issue_type="like_pattern_unexpected_match",
                    details=f"Values should NOT match LIKE pattern: {self.pattern}",
                    expected=f"NOT LIKE '{self.pattern}'",
                    sample_values=samples,
                )
            )

        # Full validation for remaining columns
        if not full_validate_cols:
            return issues

        for col in full_validate_cols:
            # Use Polars native regex matching with streaming
            result = lf.select([
                pl.col(col).is_not_null().sum().alias("non_null_count"),
                self._build_invalid_expr(col).sum().alias("matching_count"),
            ]).collect(engine="streaming")

            non_null_count = result["non_null_count"][0]
            matching_count = result["matching_count"][0]

            if non_null_count == 0 or matching_count == 0:
                continue

            if self._passes_mostly(matching_count, non_null_count):
                continue

            samples = self._get_invalid_samples(lf, col)

            ratio = matching_count / non_null_count
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
