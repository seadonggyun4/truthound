"""Character set validators."""

from typing import Any

import polars as pl

from truthound.validators.base import (
    ValidationIssue,
    Validator,
    StringValidatorMixin,
    SampledEarlyTerminationMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class AlphanumericValidator(Validator, StringValidatorMixin, SampledEarlyTerminationMixin):
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

        # Build pattern for Polars regex
        # Note: hyphen must be at start or end of character class to avoid range interpretation
        pattern = r"^["
        if allow_hyphen:
            pattern += "-"  # Hyphen at start to avoid range interpretation
        pattern += "a-zA-Z0-9"
        if allow_underscore:
            pattern += "_"
        if allow_space:
            pattern += r"\s"
        pattern += r"]+$"
        self._pattern_str = pattern

    def _build_invalid_expr(self, col: str) -> pl.Expr:
        """Build expression that returns True for invalid values."""
        return (
            pl.col(col).is_not_null()
            & (pl.col(col).str.len_chars() > 0)
            & ~pl.col(col).str.contains(self._pattern_str)
        )

    def _get_allowed_description(self) -> str:
        """Get description of allowed characters."""
        allowed = ["alphanumeric"]
        if self.allow_underscore:
            allowed.append("underscore")
        if self.allow_hyphen:
            allowed.append("hyphen")
        if self.allow_space:
            allowed.append("space")
        return ", ".join(allowed)

    def _get_invalid_samples(self, lf: pl.LazyFrame, col: str) -> list[str]:
        """Get sample invalid values for error reporting."""
        samples_df = lf.filter(
            self._build_invalid_expr(col)
        ).select(pl.col(col)).head(self.config.sample_size).collect(engine="streaming")

        return [
            (v[:50] + "..." if len(v) > 50 else v)
            for v in samples_df[col].to_list()
            if isinstance(v, str)
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
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
        allowed_desc = self._get_allowed_description()
        for col in early_term_cols:
            result = early_results[col]
            samples = self._get_invalid_samples(lf.head(self.early_termination_sample_size), col)

            issues.append(
                self._build_early_termination_issue(
                    col=col,
                    result=result,
                    issue_type="non_alphanumeric",
                    details=f"Only {allowed_desc} allowed",
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
                self._build_invalid_expr(col).sum().alias("invalid_count"),
            ]).collect(engine="streaming")

            non_null_count = result["non_null_count"][0]
            invalid_count = result["invalid_count"][0]

            if non_null_count == 0 or invalid_count == 0:
                continue

            samples = self._get_invalid_samples(lf, col)

            ratio = invalid_count / non_null_count
            issues.append(
                ValidationIssue(
                    column=col,
                    issue_type="non_alphanumeric",
                    count=invalid_count,
                    severity=self._calculate_severity(ratio),
                    details=f"Only {allowed_desc} allowed",
                    sample_values=samples,
                )
            )

        return issues
