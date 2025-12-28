"""Pre-built validator templates for common patterns.

This module provides ready-to-use validator templates that cover
common validation scenarios. Extend these templates to create
custom validators with minimal boilerplate.

Templates:
    - SimpleColumnValidator: One check per column
    - SimplePatternValidator: Regex-based string validation
    - SimpleRangeValidator: Numeric range validation
    - SimpleComparisonValidator: Cross-column comparison
    - CompositeValidator: Combine multiple validators

Example:
    class EmailValidator(SimplePatternValidator):
        name = "email"
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"  # noqa: W605
        issue_type = "invalid_email"
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import polars as pl

from truthound.validators.base import (
    Validator,
    ColumnValidator,
    ValidationIssue,
    ValidatorConfig,
    StringValidatorMixin,
    NumericValidatorMixin,
    RegexValidatorMixin,
    NUMERIC_TYPES,
    STRING_TYPES,
)
from truthound.types import Severity


class SimpleColumnValidator(ColumnValidator):
    """Template for simple column-by-column validation.

    Subclass and implement `check_value` to create a validator that
    checks each column independently.

    Class Attributes:
        name: Validator name (required)
        category: Validator category (default: "custom")
        issue_type: Issue type for violations (default: "validation_failed")
        default_severity: Default severity level (default: MEDIUM)
        dtype_filter: Set of allowed data types (default: None = all)

    Example:
        class PositiveValidator(SimpleColumnValidator):
            name = "positive"
            category = "numeric"
            issue_type = "non_positive_value"
            default_severity = Severity.HIGH
            dtype_filter = NUMERIC_TYPES

            def check_column_values(self, lf: pl.LazyFrame, col: str) -> int:
                return lf.filter(pl.col(col) <= 0).select(pl.len()).collect().item()

            def get_violation_samples(self, lf: pl.LazyFrame, col: str) -> list:
                return (
                    lf.filter(pl.col(col) <= 0)
                    .select(col)
                    .head(5)
                    .collect()
                    .to_series()
                    .to_list()
                )
    """

    name: str = "simple_column"
    category: str = "custom"
    issue_type: str = "validation_failed"
    default_severity: Severity = Severity.MEDIUM
    dtype_filter: set[type[pl.DataType]] | None = None

    @abstractmethod
    def check_column_values(self, lf: pl.LazyFrame, col: str) -> int:
        """Check values in a column and return violation count.

        Args:
            lf: LazyFrame to check
            col: Column name to check

        Returns:
            Number of violations found
        """
        pass

    def get_violation_samples(
        self, lf: pl.LazyFrame, col: str
    ) -> list[Any] | None:
        """Get sample violation values (optional override).

        Args:
            lf: LazyFrame to sample from
            col: Column name

        Returns:
            List of sample values or None
        """
        return None

    def get_issue_details(self, col: str, count: int, total: int) -> str:
        """Get issue details message (optional override).

        Args:
            col: Column name
            count: Violation count
            total: Total row count

        Returns:
            Details string
        """
        pct = (count / total * 100) if total > 0 else 0
        return f"Found {count} violations ({pct:.1f}%)"

    def check_column(
        self,
        lf: pl.LazyFrame,
        col: str,
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check a single column for violations."""
        count = self.check_column_values(lf, col)

        if count == 0:
            return None

        # Check mostly threshold
        if self._passes_mostly(count, total_rows):
            return None

        samples = self.get_violation_samples(lf, col)
        details = self.get_issue_details(col, count, total_rows)
        severity = self._calculate_severity(count / total_rows if total_rows > 0 else 0)

        return ValidationIssue(
            column=col,
            issue_type=self.issue_type,
            count=count,
            severity=self.config.severity_override or severity,
            details=details,
            sample_values=samples,
        )

    def _get_target_columns(
        self,
        lf: pl.LazyFrame,
        dtype_filter: set[type[pl.DataType]] | None = None,
    ) -> list[str]:
        """Override to use class-level dtype_filter."""
        return super()._get_target_columns(lf, dtype_filter or self.dtype_filter)


class SimplePatternValidator(Validator, StringValidatorMixin, RegexValidatorMixin):
    """Template for regex pattern-based string validation.

    Subclass and set `pattern` to create a validator that checks
    string columns against a regex pattern.

    Class Attributes:
        name: Validator name (required)
        pattern: Regex pattern to match (required)
        match_full: Whether to match full string (default: True)
        invert_match: If True, flag values that DO match (default: False)
        issue_type: Issue type for violations (default: "pattern_mismatch")
        case_sensitive: Case sensitivity (default: True)

    Example:
        class PhoneValidator(SimplePatternValidator):
            name = "phone_us"
            category = "string"
            pattern = r"^\\+?1?[-.]?\\(?\\d{3}\\)?[-.]?\\d{3}[-.]?\\d{4}$"
            issue_type = "invalid_phone_format"
            match_full = True

        class NoSSNValidator(SimplePatternValidator):
            name = "no_ssn"
            category = "privacy"
            pattern = r"\\d{3}-\\d{2}-\\d{4}"
            invert_match = True  # Flag values that HAVE SSN pattern
            issue_type = "contains_ssn"
    """

    name: str = "simple_pattern"
    category: str = "string"
    pattern: str = ""
    match_full: bool = True
    invert_match: bool = False
    issue_type: str = "pattern_mismatch"
    case_sensitive: bool = True
    default_severity: Severity = Severity.MEDIUM

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)

        if not self.pattern:
            raise ValueError(f"{self.__class__.__name__} must define 'pattern'")

        # Validate pattern at construction time
        import re

        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled = self.validate_pattern(self.pattern, flags)

        # Build Polars-compatible pattern
        if self.match_full and not self.pattern.startswith("^"):
            self._polars_pattern = f"^{self.pattern}$"
        else:
            self._polars_pattern = self.pattern

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_string_columns(lf)
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        for col in columns:
            # Check for pattern match/mismatch
            if self.invert_match:
                # Flag values that DO match the pattern
                count = (
                    lf.filter(
                        pl.col(col).str.contains(self._polars_pattern)
                    )
                    .select(pl.len())
                    .collect()
                    .item()
                )
            else:
                # Flag values that DON'T match the pattern
                count = (
                    lf.filter(
                        pl.col(col).is_not_null()
                        & ~pl.col(col).str.contains(self._polars_pattern)
                    )
                    .select(pl.len())
                    .collect()
                    .item()
                )

            if count == 0:
                continue

            if self._passes_mostly(count, total_rows):
                continue

            # Get samples
            try:
                if self.invert_match:
                    samples = (
                        lf.filter(pl.col(col).str.contains(self._polars_pattern))
                        .select(col)
                        .head(self.config.sample_size)
                        .collect()
                        .to_series()
                        .to_list()
                    )
                else:
                    samples = (
                        lf.filter(
                            pl.col(col).is_not_null()
                            & ~pl.col(col).str.contains(self._polars_pattern)
                        )
                        .select(col)
                        .head(self.config.sample_size)
                        .collect()
                        .to_series()
                        .to_list()
                    )
            except Exception:
                samples = None

            severity = self._calculate_severity(count / total_rows)

            issues.append(
                ValidationIssue(
                    column=col,
                    issue_type=self.issue_type,
                    count=count,
                    severity=self.config.severity_override or severity,
                    details=f"Pattern: {self.pattern}",
                    expected=self.pattern if not self.invert_match else f"NOT {self.pattern}",
                    sample_values=samples,
                )
            )

        return issues


class SimpleRangeValidator(Validator, NumericValidatorMixin):
    """Template for numeric range validation.

    Subclass and set range bounds to create a validator that checks
    numeric columns are within specified bounds.

    Class Attributes:
        name: Validator name (required)
        min_value: Minimum allowed value (None = no minimum)
        max_value: Maximum allowed value (None = no maximum)
        inclusive_min: Include min_value in range (default: True)
        inclusive_max: Include max_value in range (default: True)
        issue_type: Issue type for violations (default: "out_of_range")

    Example:
        class PercentageValidator(SimpleRangeValidator):
            name = "percentage"
            min_value = 0
            max_value = 100
            issue_type = "invalid_percentage"

        class PositiveValidator(SimpleRangeValidator):
            name = "positive_only"
            min_value = 0
            inclusive_min = False  # Strictly > 0
            issue_type = "non_positive"

        class AgeValidator(SimpleRangeValidator):
            name = "valid_age"
            min_value = 0
            max_value = 150
            issue_type = "invalid_age"
    """

    name: str = "simple_range"
    category: str = "numeric"
    min_value: float | int | None = None
    max_value: float | int | None = None
    inclusive_min: bool = True
    inclusive_max: bool = True
    issue_type: str = "out_of_range"
    default_severity: Severity = Severity.MEDIUM

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)

        if self.min_value is None and self.max_value is None:
            raise ValueError(
                f"{self.__class__.__name__} must define at least one of "
                "'min_value' or 'max_value'"
            )

    def _build_violation_expr(self, col: str) -> pl.Expr:
        """Build expression to find violations."""
        conditions: list[pl.Expr] = []

        if self.min_value is not None:
            if self.inclusive_min:
                conditions.append(pl.col(col) < self.min_value)
            else:
                conditions.append(pl.col(col) <= self.min_value)

        if self.max_value is not None:
            if self.inclusive_max:
                conditions.append(pl.col(col) > self.max_value)
            else:
                conditions.append(pl.col(col) >= self.max_value)

        if len(conditions) == 1:
            return conditions[0]
        return conditions[0] | conditions[1]

    def _format_range(self) -> str:
        """Format the expected range for display."""
        min_bracket = "[" if self.inclusive_min else "("
        max_bracket = "]" if self.inclusive_max else ")"
        min_val = str(self.min_value) if self.min_value is not None else "-∞"
        max_val = str(self.max_value) if self.max_value is not None else "+∞"
        return f"{min_bracket}{min_val}, {max_val}{max_bracket}"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        for col in columns:
            violation_expr = self._build_violation_expr(col)
            count = lf.filter(violation_expr).select(pl.len()).collect().item()

            if count == 0:
                continue

            if self._passes_mostly(count, total_rows):
                continue

            # Get samples
            try:
                samples = (
                    lf.filter(violation_expr)
                    .select(col)
                    .head(self.config.sample_size)
                    .collect()
                    .to_series()
                    .to_list()
                )
            except Exception:
                samples = None

            severity = self._calculate_severity(count / total_rows)

            issues.append(
                ValidationIssue(
                    column=col,
                    issue_type=self.issue_type,
                    count=count,
                    severity=self.config.severity_override or severity,
                    details=f"Expected range: {self._format_range()}",
                    expected=self._format_range(),
                    sample_values=samples,
                )
            )

        return issues


class SimpleComparisonValidator(Validator):
    """Template for cross-column comparison validation.

    Subclass and set comparison parameters to create a validator
    that compares values between columns.

    Class Attributes:
        name: Validator name (required)
        left_column: Left side column name (required)
        right_column: Right side column name (required)
        operator: Comparison operator (required)
            - "eq": left == right
            - "ne": left != right
            - "lt": left < right
            - "le": left <= right
            - "gt": left > right
            - "ge": left >= right
        issue_type: Issue type for violations

    Example:
        class StartBeforeEndValidator(SimpleComparisonValidator):
            name = "start_before_end"
            left_column = "start_date"
            right_column = "end_date"
            operator = "lt"
            issue_type = "invalid_date_range"

        class AmountMatchesValidator(SimpleComparisonValidator):
            name = "amounts_match"
            left_column = "calculated_total"
            right_column = "reported_total"
            operator = "eq"
            issue_type = "amount_mismatch"
    """

    name: str = "simple_comparison"
    category: str = "cross_column"
    left_column: str = ""
    right_column: str = ""
    operator: str = "eq"  # eq, ne, lt, le, gt, ge
    issue_type: str = "comparison_failed"
    default_severity: Severity = Severity.HIGH

    OPERATORS = {
        "eq": lambda l, r: l == r,
        "ne": lambda l, r: l != r,
        "lt": lambda l, r: l < r,
        "le": lambda l, r: l <= r,
        "gt": lambda l, r: l > r,
        "ge": lambda l, r: l >= r,
    }

    OPERATOR_SYMBOLS = {
        "eq": "==",
        "ne": "!=",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
    }

    def __init__(self, config: ValidatorConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)

        if not self.left_column or not self.right_column:
            raise ValueError(
                f"{self.__class__.__name__} must define 'left_column' and 'right_column'"
            )

        if self.operator not in self.OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Valid operators: {list(self.OPERATORS.keys())}"
            )

    def _build_violation_expr(self) -> pl.Expr:
        """Build expression to find violations (rows where condition is FALSE)."""
        left = pl.col(self.left_column)
        right = pl.col(self.right_column)

        if self.operator == "eq":
            return left != right
        elif self.operator == "ne":
            return left == right
        elif self.operator == "lt":
            return left >= right
        elif self.operator == "le":
            return left > right
        elif self.operator == "gt":
            return left <= right
        else:  # ge
            return left < right

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Check columns exist
        schema = lf.collect_schema()
        available = schema.names()

        if self.left_column not in available:
            self.logger.warning(f"Column '{self.left_column}' not found")
            return issues

        if self.right_column not in available:
            self.logger.warning(f"Column '{self.right_column}' not found")
            return issues

        total_rows = lf.select(pl.len()).collect().item()
        if total_rows == 0:
            return issues

        violation_expr = self._build_violation_expr()
        count = lf.filter(violation_expr).select(pl.len()).collect().item()

        if count == 0:
            return issues

        if self._passes_mostly(count, total_rows):
            return issues

        # Get samples
        try:
            samples = (
                lf.filter(violation_expr)
                .select([self.left_column, self.right_column])
                .head(self.config.sample_size)
                .collect()
                .to_dicts()
            )
        except Exception:
            samples = None

        symbol = self.OPERATOR_SYMBOLS[self.operator]
        severity = self._calculate_severity(count / total_rows)

        issues.append(
            ValidationIssue(
                column=f"{self.left_column}, {self.right_column}",
                issue_type=self.issue_type,
                count=count,
                severity=self.config.severity_override or severity,
                details=f"Expected: {self.left_column} {symbol} {self.right_column}",
                expected=f"{self.left_column} {symbol} {self.right_column}",
                sample_values=samples,
            )
        )

        return issues


class CompositeValidator(Validator):
    """Template for combining multiple validators.

    Use this to create a validator that runs multiple validations
    and combines their results.

    Example:
        class CustomerDataValidator(CompositeValidator):
            name = "customer_data"
            category = "business"

            def get_validators(self) -> list[Validator]:
                return [
                    EmailValidator(columns=("email",)),
                    PhoneValidator(columns=("phone",)),
                    AgeValidator(columns=("age",)),
                ]

        # Or create inline:
        composite = CompositeValidator(
            validators=[
                NullValidator(columns=("id", "name")),
                UniqueValidator(columns=("id",)),
                RangeValidator(columns=("age",), min_value=0, max_value=150),
            ]
        )
    """

    name: str = "composite"
    category: str = "composite"

    def __init__(
        self,
        validators: list[Validator] | None = None,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self._validators = validators or []

    def get_validators(self) -> list[Validator]:
        """Get the list of validators to run.

        Override this method to define validators at the class level.

        Returns:
            List of Validator instances
        """
        return self._validators

    def add_validator(self, validator: Validator) -> "CompositeValidator":
        """Add a validator to the composite.

        Args:
            validator: Validator to add

        Returns:
            Self for chaining
        """
        self._validators.append(validator)
        return self

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run all validators and combine results."""
        all_issues: list[ValidationIssue] = []
        validators = self.get_validators()

        for validator in validators:
            try:
                issues = validator.validate(lf)
                all_issues.extend(issues)
            except Exception as e:
                if self.config.graceful_degradation:
                    self.logger.warning(
                        f"Validator '{validator.name}' failed: {e}"
                    )
                else:
                    raise

        return all_issues

    def validate_safe(self, lf: pl.LazyFrame) -> list[tuple[str, Any]]:
        """Run all validators with error handling.

        Returns:
            List of (validator_name, result) tuples
        """
        results: list[tuple[str, Any]] = []
        validators = self.get_validators()

        for validator in validators:
            result = validator.validate_safe(lf)
            results.append((validator.name, result))

        return results


# ============================================================================
# Factory Functions
# ============================================================================


def create_pattern_validator(
    name: str,
    pattern: str,
    issue_type: str = "pattern_mismatch",
    invert: bool = False,
    case_sensitive: bool = True,
) -> type[SimplePatternValidator]:
    """Factory to create a pattern validator class.

    Args:
        name: Validator name
        pattern: Regex pattern
        issue_type: Issue type for violations
        invert: If True, flag matches instead of non-matches
        case_sensitive: Case sensitivity

    Returns:
        New validator class

    Example:
        EmailValidator = create_pattern_validator(
            "email",
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            "invalid_email",
        )
        validator = EmailValidator()
    """

    class GeneratedPatternValidator(SimplePatternValidator):
        pass

    GeneratedPatternValidator.name = name
    GeneratedPatternValidator.pattern = pattern
    GeneratedPatternValidator.issue_type = issue_type
    GeneratedPatternValidator.invert_match = invert
    GeneratedPatternValidator.case_sensitive = case_sensitive

    return GeneratedPatternValidator


def create_range_validator(
    name: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    issue_type: str = "out_of_range",
    inclusive: bool = True,
) -> type[SimpleRangeValidator]:
    """Factory to create a range validator class.

    Args:
        name: Validator name
        min_value: Minimum value
        max_value: Maximum value
        issue_type: Issue type for violations
        inclusive: Whether bounds are inclusive

    Returns:
        New validator class

    Example:
        PercentageValidator = create_range_validator(
            "percentage", 0, 100, "invalid_percentage"
        )
        validator = PercentageValidator()
    """

    class GeneratedRangeValidator(SimpleRangeValidator):
        pass

    GeneratedRangeValidator.name = name
    GeneratedRangeValidator.min_value = min_value
    GeneratedRangeValidator.max_value = max_value
    GeneratedRangeValidator.issue_type = issue_type
    GeneratedRangeValidator.inclusive_min = inclusive
    GeneratedRangeValidator.inclusive_max = inclusive

    return GeneratedRangeValidator
