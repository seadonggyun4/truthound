"""Regex pattern validators.

Provides vectorized regex validation using Polars' native string operations
for optimal performance on large datasets.

Performance Notes:
- Uses str.contains() for vectorized regex matching
- ~100x faster than row-by-row iteration on 1M+ rows
- Supports both strict (fullmatch) and partial (search) matching
- Expression-based architecture enables single collect() across validators
"""

import re
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    ValidationExpressionSpec,
    Validator,
    StringValidatorMixin,
    RegexValidatorMixin,
    RegexValidationError,
    SampledEarlyTerminationMixin,
    ExpressionValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class RegexValidator(
    Validator,
    StringValidatorMixin,
    RegexValidatorMixin,
    SampledEarlyTerminationMixin,
    ExpressionValidatorMixin,
):
    """Validates that string values match a regex pattern.

    Uses Polars' native str.contains() for vectorized regex matching,
    providing ~100x performance improvement over row-by-row iteration.

    This validator supports expression-based architecture for batched
    validation with other validators (single collect() call).

    Pattern is validated at construction time to catch errors early.

    Example:
        # Valid pattern
        validator = RegexValidator(pattern=r"^[A-Z]{3}-\d{4}$")

        # Invalid pattern raises immediately
        try:
            validator = RegexValidator(pattern=r"[invalid(")
        except RegexValidationError as e:
            print(f"Bad pattern: {e}")

        # Batched execution with other validators
        from truthound.validators.base import ExpressionBatchExecutor
        executor = ExpressionBatchExecutor()
        executor.add_validator(RegexValidator(pattern=r"^[A-Z]+$"))
        executor.add_validator(NullValidator())
        all_issues = executor.execute(lf)  # Single collect()!

    Performance:
        - 1M rows: ~50ms (vectorized) vs ~5000ms (iteration)
        - Batched with 5 validators: ~100ms vs ~250ms sequential
    """

    name = "regex"
    category = "string"

    def __init__(
        self,
        pattern: str | None = None,
        match_full: bool = True,
        dotall: bool = True,
        case_insensitive: bool = False,
        **kwargs: Any,
    ):
        """Initialize with pattern validation.

        Args:
            pattern: Regex pattern string (required, validated at construction)
            match_full: If True, pattern must match entire string (adds ^ and $ anchors)
            dotall: If True, . matches newlines (note: Polars uses (?s) flag)
            case_insensitive: If True, ignore case (note: Polars uses (?i) flag)
            **kwargs: Additional validator config

        Raises:
            RegexValidationError: If pattern is missing or invalid

        Example YAML configuration::

            validators:
              - name: regex
                pattern: "^[A-Z]{3}-\\d{4}$"
        """
        super().__init__(**kwargs)

        # Validate required pattern parameter
        if pattern is None:
            raise RegexValidationError(
                pattern="",
                error=(
                    "The 'pattern' parameter is required for RegexValidator. "
                    "Example: RegexValidator(pattern=r'^[A-Z]+$') or in YAML: "
                    "validators: [{name: regex, pattern: '^[A-Z]+$'}]"
                ),
            )

        # Build flags for Python regex validation
        flags = 0
        if dotall:
            flags |= re.DOTALL
        if case_insensitive:
            flags |= re.IGNORECASE

        # Validate pattern at construction time using Python regex
        self._compiled = self.validate_pattern(pattern, flags)
        self.pattern = pattern
        self.match_full = match_full
        self.case_insensitive = case_insensitive
        self.dotall = dotall

        # Build Polars-compatible pattern string
        self._polars_pattern = self._build_polars_pattern()

    def _build_polars_pattern(self) -> str:
        """Build pattern string for Polars str.contains().

        Polars uses inline flags (?i), (?s), etc. instead of Python's re.IGNORECASE.
        For full match, we ensure the pattern has ^ and $ anchors.
        """
        pattern = self.pattern

        # Add anchors for full match if not already present
        if self.match_full:
            if not pattern.startswith("^"):
                pattern = "^" + pattern
            if not pattern.endswith("$"):
                pattern = pattern + "$"

        # Add inline flags for Polars
        flags = ""
        if self.case_insensitive:
            flags += "i"
        if self.dotall:
            flags += "s"

        if flags:
            pattern = f"(?{flags}){pattern}"

        return pattern

    def _build_match_expr(self, col: str) -> pl.Expr:
        """Build vectorized match expression for a column.

        Returns expression that is True for INVALID values.
        """
        col_expr = pl.col(col)

        # Skip null and empty strings
        non_empty = col_expr.is_not_null() & (col_expr.str.strip_chars() != "")

        # Use Polars native regex matching
        matches = col_expr.str.contains(self._polars_pattern)

        # Return True for invalid (non-empty but not matching)
        return non_empty & ~matches

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for regex matching.

        Args:
            lf: LazyFrame to validate
            columns: String columns to validate

        Returns:
            List of ValidationExpressionSpec for batched execution
        """
        # Escape braces in pattern for .format() compatibility
        safe_pattern = self.pattern.replace("{", "{{").replace("}", "}}")
        specs = []
        for col in columns:
            invalid_expr = self._build_match_expr(col)
            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="regex_mismatch",
                    count_expr=invalid_expr.sum(),
                    non_null_expr=pl.col(col).is_not_null().sum(),
                    details_template=f"Pattern: {safe_pattern}",
                    expected=self.pattern,
                )
            )
        return specs

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach with early termination.

        This method first attempts early termination on large datasets,
        then uses the expression-based approach for full validation.
        """
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)

        if not columns:
            return issues

        # Check for early termination opportunity
        early_results = self._check_early_termination(
            lf,
            columns=columns,
            build_invalid_expr=self._build_match_expr,
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
            samples = self._get_invalid_samples(lf.head(self.early_termination_sample_size), col)

            # Check mostly threshold with extrapolated values
            if self._passes_mostly(result.estimated_fail_count, result.total_rows):
                continue

            issues.append(
                self._build_early_termination_issue(
                    col=col,
                    result=result,
                    issue_type="regex_mismatch",
                    details=f"Pattern: {self.pattern}",
                    expected=self.pattern,
                    sample_values=samples,
                )
            )

        # Full validation for remaining columns using expression approach
        if full_validate_cols:
            expr_issues = self._validate_full(lf, columns=full_validate_cols)
            issues.extend(expr_issues)

        return issues

    def _validate_full(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationIssue]:
        """Full validation with sample values collection.

        This method performs full validation and collects sample invalid values
        for better error reporting.
        """
        if not columns:
            return []

        # Build expressions for validation
        exprs: list[pl.Expr] = [pl.len().alias("_total")]

        for col in columns:
            invalid_expr = self._build_match_expr(col)
            exprs.append(invalid_expr.sum().alias(f"_inv_{col}"))
            exprs.append(pl.col(col).is_not_null().sum().alias(f"_nn_{col}"))

        # Use streaming for large datasets
        result = lf.select(exprs).collect(engine="streaming")
        total_rows = result["_total"][0]

        if total_rows == 0:
            return []

        issues: list[ValidationIssue] = []

        for col in columns:
            invalid_count = result[f"_inv_{col}"][0]
            non_null_count = result[f"_nn_{col}"][0]

            if invalid_count > 0 and non_null_count > 0:
                # Check mostly threshold
                if self._passes_mostly(invalid_count, non_null_count):
                    continue

                ratio = invalid_count / non_null_count

                # Get sample invalid values
                samples = self._get_invalid_samples(lf, col)

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="regex_mismatch",
                        count=invalid_count,
                        severity=self._calculate_severity(ratio),
                        details=f"Pattern: {self.pattern}",
                        expected=self.pattern,
                        sample_values=samples,
                    )
                )

        return issues

    def _get_invalid_samples(self, lf: pl.LazyFrame, col: str) -> list[str]:
        """Get sample invalid values for error reporting."""
        invalid_expr = self._build_match_expr(col)

        # Use streaming for sample collection
        sample_df = (
            lf.filter(invalid_expr)
            .select(col)
            .head(self.config.sample_size)
            .collect(engine="streaming")
        )

        samples = []
        for val in sample_df.get_column(col).to_list():
            if val is not None:
                display = val[:50] + "..." if len(str(val)) > 50 else str(val)
                samples.append(display)

        return samples
