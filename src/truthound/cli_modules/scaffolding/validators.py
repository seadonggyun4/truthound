"""Validator scaffold generator.

This module provides scaffolding for creating custom validators with
various template variants:
    - basic: Minimal validator with core structure
    - column: Column-level validator with target column support
    - pattern: Pattern matching validator with regex
    - range: Numeric range validator
    - composite: Multi-validator composite
    - full: Full-featured with tests and documentation
"""

from __future__ import annotations

from typing import Any, ClassVar

from truthound.cli_modules.scaffolding.base import (
    BaseScaffold,
    ScaffoldConfig,
    ScaffoldResult,
    register_scaffold,
)


@register_scaffold(
    name="validator",
    description="Generate a custom validator",
    aliases=("val", "v"),
)
class ValidatorScaffold(BaseScaffold):
    """Scaffold generator for validators.

    Supports multiple template variants for different validation patterns.
    """

    name: ClassVar[str] = "validator"
    description: ClassVar[str] = "Generate a custom validator"
    aliases: ClassVar[tuple[str, ...]] = ("val", "v")

    TEMPLATE_VARIANTS: ClassVar[tuple[str, ...]] = (
        "basic",
        "column",
        "pattern",
        "range",
        "comparison",
        "composite",
        "full",
    )

    def get_options(self) -> dict[str, Any]:
        """Get validator-specific options."""
        return {
            "category": {
                "type": "str",
                "default": "custom",
                "description": "Validator category (e.g., numeric, string, business)",
            },
            "severity": {
                "type": "str",
                "default": "MEDIUM",
                "description": "Default severity level",
                "choices": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            },
            "columns": {
                "type": "list[str]",
                "default": None,
                "description": "Target columns for column validators",
            },
            "pattern": {
                "type": "str",
                "default": None,
                "description": "Regex pattern for pattern validators",
            },
            "min_value": {
                "type": "float",
                "default": None,
                "description": "Minimum value for range validators",
            },
            "max_value": {
                "type": "float",
                "default": None,
                "description": "Maximum value for range validators",
            },
        }

    def _generate_files(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate validator files based on variant."""
        variant = config.template_variant

        # Generate main validator file
        if variant == "basic":
            self._generate_basic(config, result)
        elif variant == "column":
            self._generate_column(config, result)
        elif variant == "pattern":
            self._generate_pattern(config, result)
        elif variant == "range":
            self._generate_range(config, result)
        elif variant == "comparison":
            self._generate_comparison(config, result)
        elif variant == "composite":
            self._generate_composite(config, result)
        elif variant == "full":
            self._generate_full(config, result)
        else:
            self._generate_basic(config, result)

        # Generate __init__.py
        self._generate_init(config, result)

        # Generate tests if requested
        if config.include_tests:
            self._generate_tests(config, result)

        # Generate docs if requested
        if config.include_docs:
            self._generate_docs(config, result)

    def _generate_init(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate __init__.py file."""
        content = f'''"""Package for {config.title_name} validator."""

from {config.name}.validator import {config.class_name}Validator

__all__ = ["{config.class_name}Validator"]
'''
        result.add_file(f"{config.name}/__init__.py", content)

    def _generate_basic(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate basic validator template."""
        severity = config.extra.get("severity", "MEDIUM")
        category = config.extra.get("category", config.category)

        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}"],
)
class {config.class_name}Validator(Validator):
    """{config.description or f'{config.title_name} validator.'}

    This validator checks data quality based on custom business rules.

    Example:
        >>> validator = {config.class_name}Validator()
        >>> issues = validator.validate(lf)
    """

    name: str = "{config.name}"
    category: str = "{category}"
    default_severity: Severity = Severity.{severity}

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Optional validator configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the data.

        Args:
            lf: Polars LazyFrame to validate

        Returns:
            List of validation issues found
        """
        issues: list[ValidationIssue] = []
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # TODO: Implement your validation logic here
        # Example:
        # for col in self._get_target_columns(lf):
        #     violations = lf.filter(pl.col(col).is_null()).collect()
        #     if violations.height > 0:
        #         issues.append(ValidationIssue(
        #             column=col,
        #             issue_type=self.name,
        #             count=violations.height,
        #             severity=self.default_severity,
        #             details=f"Found {{violations.height}} issues in '{{col}}'",
        #         ))

        return issues
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_column(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate column validator template."""
        severity = config.extra.get("severity", "MEDIUM")
        category = config.extra.get("category", config.category)

        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.validators.base import (
    ColumnValidator,
    ValidationIssue,
    ValidatorConfig,
)
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} column validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}", "column"],
)
class {config.class_name}Validator(ColumnValidator):
    """{config.description or f'{config.title_name} column validator.'}

    Validates each column that matches the configured criteria.

    Example:
        >>> validator = {config.class_name}Validator(columns=["value", "amount"])
        >>> issues = validator.validate(lf)
    """

    name: str = "{config.name}"
    category: str = "{category}"
    default_severity: Severity = Severity.{severity}

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        columns: tuple[str, ...] | list[str] | None = None,
        severity: Severity = Severity.{severity},
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Validator configuration
            columns: Columns to validate (None = all applicable)
            severity: Severity level for issues
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, columns=columns, **kwargs)
        self._severity = severity

    def check_column(
        self,
        lf: pl.LazyFrame,
        col: str,
        total_rows: int,
    ) -> ValidationIssue | None:
        """Check a single column for violations.

        Args:
            lf: LazyFrame to validate
            col: Column name to check
            total_rows: Total row count

        Returns:
            ValidationIssue if violations found, None otherwise
        """
        # TODO: Implement your column validation logic
        # Example: Check for null values
        # violations = lf.filter(pl.col(col).is_null()).collect()
        #
        # if violations.height == 0:
        #     return None
        #
        # samples = violations[col].head(5).to_list()
        # return ValidationIssue(
        #     column=col,
        #     issue_type=self.name,
        #     count=violations.height,
        #     severity=self._severity,
        #     details=f"Found {{violations.height}} issues in '{{col}}'",
        #     sample_values=samples,
        # )

        return None
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_pattern(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate pattern validator template."""
        severity = config.extra.get("severity", "MEDIUM")
        category = config.extra.get("category", config.category)
        pattern = config.extra.get("pattern", r".*")

        content = f'''{self._get_header(config)}

from __future__ import annotations

import re
from typing import Any

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    RegexValidatorMixin,
    StringValidatorMixin,
)
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} pattern validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}", "pattern", "regex"],
)
class {config.class_name}Validator(Validator, StringValidatorMixin, RegexValidatorMixin):
    """{config.description or f'{config.title_name} pattern validator.'}

    Validates string values against a regex pattern.

    Example:
        >>> validator = {config.class_name}Validator(pattern=r"^[A-Z]{{2,3}}-\\\\d{{4}}$")
        >>> issues = validator.validate(lf)
    """

    name: str = "{config.name}"
    category: str = "{category}"
    default_severity: Severity = Severity.{severity}

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        pattern: str = r"{pattern}",
        columns: tuple[str, ...] | list[str] | None = None,
        case_sensitive: bool = True,
        invert: bool = False,
        severity: Severity = Severity.{severity},
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Validator configuration
            pattern: Regex pattern to match
            columns: Columns to validate
            case_sensitive: Whether pattern matching is case sensitive
            invert: If True, match values that DON'T match the pattern
            severity: Severity level for issues
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)
        self._pattern_str = pattern
        self._columns = columns
        self._case_sensitive = case_sensitive
        self._invert = invert
        self._severity = severity

        # Compile pattern for validation
        flags = 0 if case_sensitive else re.IGNORECASE
        self._pattern = re.compile(pattern, flags)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate string columns against pattern.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Get string columns
        columns = self._columns or self._get_string_columns(lf)

        for column in columns:
            if column not in lf.collect_schema().names():
                continue

            # Check pattern match
            if self._invert:
                # Find values that match (when we want non-matches)
                violations = lf.filter(
                    pl.col(column).str.contains(self._pattern_str)
                ).collect()
            else:
                # Find values that don't match (when we want matches)
                violations = lf.filter(
                    pl.col(column).is_not_null()
                    & ~pl.col(column).str.contains(self._pattern_str)
                ).collect()

            if violations.height == 0:
                continue

            samples = violations[column].head(5).to_list()
            match_type = "match" if self._invert else "do not match"

            issues.append(ValidationIssue(
                column=column,
                issue_type=f"{{self.name}}_pattern_mismatch",
                count=violations.height,
                severity=self._severity,
                details=(
                    f"Column '{{column}}' has {{violations.height}} values that "
                    f"{{match_type}} pattern '{{self._pattern_str}}'"
                ),
                sample_values=samples,
                expected=self._pattern_str if not self._invert else f"NOT {{self._pattern_str}}",
            ))

        return issues
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_range(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate range validator template."""
        severity = config.extra.get("severity", "MEDIUM")
        category = config.extra.get("category", config.category)
        min_value = config.extra.get("min_value")
        max_value = config.extra.get("max_value")

        min_val_str = str(min_value) if min_value is not None else "None"
        max_val_str = str(max_value) if max_value is not None else "None"

        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    NumericValidatorMixin,
    NUMERIC_TYPES,
)
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} range validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}", "range", "numeric"],
)
class {config.class_name}Validator(Validator, NumericValidatorMixin):
    """{config.description or f'{config.title_name} range validator.'}

    Validates numeric values are within a specified range.

    Example:
        >>> validator = {config.class_name}Validator(min_value=0, max_value=100)
        >>> issues = validator.validate(lf)
    """

    name: str = "{config.name}"
    category: str = "{category}"
    default_severity: Severity = Severity.{severity}

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        min_value: float | int | None = {min_val_str},
        max_value: float | int | None = {max_val_str},
        columns: tuple[str, ...] | list[str] | None = None,
        inclusive: bool = True,
        severity: Severity = Severity.{severity},
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Validator configuration
            min_value: Minimum allowed value (None = no minimum)
            max_value: Maximum allowed value (None = no maximum)
            columns: Columns to validate
            inclusive: Whether bounds are inclusive
            severity: Severity level for issues
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)
        self._min_value = min_value
        self._max_value = max_value
        self._columns = columns
        self._inclusive = inclusive
        self._severity = severity

        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

    def _build_violation_expr(self, col: str) -> pl.Expr:
        """Build expression to find violations."""
        conditions: list[pl.Expr] = []

        if self._min_value is not None:
            if self._inclusive:
                conditions.append(pl.col(col) < self._min_value)
            else:
                conditions.append(pl.col(col) <= self._min_value)

        if self._max_value is not None:
            if self._inclusive:
                conditions.append(pl.col(col) > self._max_value)
            else:
                conditions.append(pl.col(col) >= self._max_value)

        if len(conditions) == 1:
            return conditions[0]
        return conditions[0] | conditions[1]

    def _format_range(self) -> str:
        """Format the expected range for display."""
        min_bracket = "[" if self._inclusive else "("
        max_bracket = "]" if self._inclusive else ")"
        min_val = str(self._min_value) if self._min_value is not None else "-∞"
        max_val = str(self._max_value) if self._max_value is not None else "+∞"
        return f"{{min_bracket}}{{min_val}}, {{max_val}}{{max_bracket}}"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate numeric values are in range.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return issues

        # Get numeric columns
        columns = self._columns or self._get_numeric_columns(lf)

        for column in columns:
            if column not in lf.collect_schema().names():
                continue

            violation_expr = self._build_violation_expr(column)
            violations = lf.filter(violation_expr).collect()

            if violations.height == 0:
                continue

            samples = violations[column].head(5).to_list()

            issues.append(ValidationIssue(
                column=column,
                issue_type=f"{{self.name}}_out_of_range",
                count=violations.height,
                severity=self._severity,
                details=(
                    f"Column '{{column}}' has {{violations.height}} values "
                    f"outside range {{self._format_range()}}"
                ),
                sample_values=samples,
                expected=self._format_range(),
            ))

        return issues
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_comparison(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate comparison validator template."""
        severity = config.extra.get("severity", "HIGH")
        category = config.extra.get("category", config.category)

        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} comparison validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}", "comparison", "cross_column"],
)
class {config.class_name}Validator(Validator):
    """{config.description or f'{config.title_name} comparison validator.'}

    Compares values between columns.

    Supported operators:
        - eq: left == right
        - ne: left != right
        - lt: left < right
        - le: left <= right
        - gt: left > right
        - ge: left >= right

    Example:
        >>> validator = {config.class_name}Validator(
        ...     left_column="start_date",
        ...     right_column="end_date",
        ...     operator="lt",
        ... )
        >>> issues = validator.validate(lf)
    """

    name: str = "{config.name}"
    category: str = "{category}"
    default_severity: Severity = Severity.{severity}

    OPERATORS = {{
        "eq": ("==", lambda l, r: l != r),  # Violation when NOT equal
        "ne": ("!=", lambda l, r: l == r),  # Violation when equal
        "lt": ("<", lambda l, r: l >= r),   # Violation when >=
        "le": ("<=", lambda l, r: l > r),   # Violation when >
        "gt": (">", lambda l, r: l <= r),   # Violation when <=
        "ge": (">=", lambda l, r: l < r),   # Violation when <
    }}

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        left_column: str = "",
        right_column: str = "",
        operator: str = "eq",
        severity: Severity = Severity.{severity},
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Validator configuration
            left_column: Left side column name
            right_column: Right side column name
            operator: Comparison operator
            severity: Severity level for issues
            **kwargs: Additional keyword arguments
        """
        super().__init__(config, **kwargs)
        self._left_column = left_column
        self._right_column = right_column
        self._operator = operator
        self._severity = severity

        if not left_column or not right_column:
            raise ValueError("Both left_column and right_column must be specified")

        if operator not in self.OPERATORS:
            raise ValueError(
                f"Invalid operator '{{operator}}'. "
                f"Valid operators: {{list(self.OPERATORS.keys())}}"
            )

    def _build_violation_expr(self) -> pl.Expr:
        """Build expression to find violations."""
        left = pl.col(self._left_column)
        right = pl.col(self._right_column)
        _, expr_builder = self.OPERATORS[self._operator]
        return expr_builder(left, right)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate column comparison.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Check columns exist
        schema = lf.collect_schema()
        available = schema.names()

        if self._left_column not in available:
            self.logger.warning(f"Column '{{self._left_column}}' not found")
            return issues

        if self._right_column not in available:
            self.logger.warning(f"Column '{{self._right_column}}' not found")
            return issues

        total_rows = lf.select(pl.len()).collect().item()
        if total_rows == 0:
            return issues

        violation_expr = self._build_violation_expr()
        violations = lf.filter(violation_expr).collect()

        if violations.height == 0:
            return issues

        samples = violations.select([self._left_column, self._right_column]).head(5).to_dicts()
        symbol, _ = self.OPERATORS[self._operator]

        issues.append(ValidationIssue(
            column=f"{{self._left_column}}, {{self._right_column}}",
            issue_type=f"{{self.name}}_comparison_failed",
            count=violations.height,
            severity=self._severity,
            details=(
                f"Expected: {{self._left_column}} {{symbol}} {{self._right_column}}. "
                f"Found {{violations.height}} violations."
            ),
            sample_values=samples,
            expected=f"{{self._left_column}} {{symbol}} {{self._right_column}}",
        ))

        return issues
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_composite(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate composite validator template."""
        severity = config.extra.get("severity", "MEDIUM")
        category = config.extra.get("category", config.category)

        content = f'''{self._get_header(config)}

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.validators.sdk import custom_validator, CompositeValidator
from truthound.types import Severity


@custom_validator(
    name="{config.name}",
    category="{category}",
    description="{config.description or f'{config.title_name} composite validator'}",
    version="{config.version}",
    author="{config.author}",
    tags=["{category}", "composite"],
)
class {config.class_name}Validator(CompositeValidator):
    """{config.description or f'{config.title_name} composite validator.'}

    Combines multiple validators into one validation pass.

    Example:
        >>> validator = {config.class_name}Validator()
        >>> issues = validator.validate(lf)

        # Or with custom validators
        >>> validator = {config.class_name}Validator(validators=[
        ...     NullValidator(columns=("id",)),
        ...     UniqueValidator(columns=("id",)),
        ... ])
    """

    name: str = "{config.name}"
    category: str = "{category}"

    def __init__(
        self,
        validators: list[Validator] | None = None,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the validator.

        Args:
            validators: List of validators to run
            config: Validator configuration
            **kwargs: Additional keyword arguments
        """
        super().__init__(validators=validators, config=config, **kwargs)

    def get_validators(self) -> list[Validator]:
        """Get the list of validators to run.

        Override this method to define validators at the class level.
        You can combine this with validators passed to __init__.

        Returns:
            List of Validator instances
        """
        validators = super().get_validators()

        # TODO: Add your default validators here
        # Example:
        # from truthound.validators import NullValidator, UniqueValidator
        # validators.extend([
        #     NullValidator(columns=("id", "name")),
        #     UniqueValidator(columns=("id",)),
        # ])

        return validators
'''
        result.add_file(f"{config.name}/validator.py", content)

    def _generate_full(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate full-featured validator template."""
        # Generate column validator as the base
        self._generate_column(config, result)

        # Always include tests and docs for full template
        self._generate_tests(config, result)
        self._generate_docs(config, result)
        self._generate_example(config, result)

    def _generate_tests(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate test file."""
        content = f'''"""Tests for {config.class_name}Validator."""

import pytest
import polars as pl

from truthound.validators.sdk.testing import (
    ValidatorTestCase,
    create_test_dataframe,
    assert_no_issues,
    assert_has_issue,
    assert_issue_count,
)
from {config.name} import {config.class_name}Validator


class Test{config.class_name}Validator(ValidatorTestCase):
    """Test cases for {config.class_name}Validator."""

    def test_valid_data_produces_no_issues(self):
        """Test that valid data produces no issues."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [1, 2, 3, 4, 5],
            "column2": ["a", "b", "c", "d", "e"],
        }})
        validator = {config.class_name}Validator()

        # Act
        issues = validator.validate(lf)

        # Assert
        assert_no_issues(issues)

    def test_invalid_data_produces_issues(self):
        """Test that invalid data produces issues."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [None, 2, None, 4, None],
            "column2": ["a", None, "c", None, "e"],
        }})
        validator = {config.class_name}Validator()

        # Act
        issues = validator.validate(lf)

        # Assert
        # TODO: Update assertion based on your validation logic
        # Example:
        # assert_has_issue(issues, issue_type="{config.name}")
        # assert_issue_count(issues, 2)
        pass

    def test_empty_dataframe_returns_no_issues(self):
        """Test with empty dataframe."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [],
            "column2": [],
        }})
        validator = {config.class_name}Validator()

        # Act
        issues = validator.validate(lf)

        # Assert
        assert_no_issues(issues)

    def test_specific_columns(self):
        """Test validation with specific columns."""
        # Arrange
        lf = pl.LazyFrame({{
            "target_col": [1, 2, 3],
            "other_col": [None, None, None],
        }})
        validator = {config.class_name}Validator(columns=["target_col"])

        # Act
        issues = validator.validate(lf)

        # Assert
        # Only target_col should be validated
        for issue in issues:
            assert issue.column == "target_col"

    def test_severity_override(self):
        """Test that severity can be overridden."""
        from truthound.types import Severity

        # Arrange
        lf = pl.LazyFrame({{"column1": [None, 2, 3]}})
        validator = {config.class_name}Validator(severity=Severity.CRITICAL)

        # Act
        issues = validator.validate(lf)

        # Assert
        for issue in issues:
            assert issue.severity == Severity.CRITICAL

    def test_configuration_options(self):
        """Test validator configuration options."""
        # TODO: Add tests for your specific configuration options
        pass

    def test_edge_cases(self):
        """Test edge cases."""
        # TODO: Add edge case tests
        pass


class Test{config.class_name}ValidatorPerformance:
    """Performance tests for {config.class_name}Validator."""

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        import time

        # Arrange
        n_rows = 1_000_000
        lf = pl.LazyFrame({{
            "column1": list(range(n_rows)),
            "column2": ["value"] * n_rows,
        }})
        validator = {config.class_name}Validator()

        # Act
        start = time.time()
        issues = validator.validate(lf)
        elapsed = time.time() - start

        # Assert - should complete in reasonable time
        assert elapsed < 10.0, f"Validation took too long: {{elapsed:.2f}}s"
'''
        result.add_file(f"{config.name}/tests/__init__.py", "")
        result.add_file(f"{config.name}/tests/test_validator.py", content)

    def _generate_docs(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate documentation file."""
        content = f'''# {config.class_name}Validator

> {config.category} validator

## Description

{config.description or 'TODO: Add description'}

## Installation

This validator is part of the Truthound package:

```bash
pip install truthound
```

## Usage

```python
from {config.name} import {config.class_name}Validator

# Create validator
validator = {config.class_name}Validator()

# Validate data
issues = validator.validate(lf)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| columns | list[str] | None | Columns to validate (None = all) |
| severity | Severity | MEDIUM | Default severity level |

## Examples

### Basic Usage

```python
import polars as pl
from {config.name} import {config.class_name}Validator

# Create sample data
lf = pl.LazyFrame({{
    "value": [1, 2, 3, 4, 5],
    "name": ["a", "b", "c", "d", "e"],
}})

# Create and run validator
validator = {config.class_name}Validator()
issues = validator.validate(lf)

print(f"Found {{len(issues)}} issues")
```

### With Specific Columns

```python
validator = {config.class_name}Validator(columns=["value"])
issues = validator.validate(lf)
```

### With Custom Severity

```python
from truthound.types import Severity

validator = {config.class_name}Validator(severity=Severity.CRITICAL)
issues = validator.validate(lf)
```

## See Also

- [Validator SDK](https://github.com/seadonggyun4/Truthound/docs/sdk.md)
- [Testing Guide](https://github.com/seadonggyun4/Truthound/docs/testing.md)

---

*Version: {config.version}*
*Author: {config.author or 'Unknown'}*
*License: {config.license_type}*
'''
        result.add_file(f"{config.name}/docs/README.md", content)

    def _generate_example(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate example usage file."""
        content = f'''#!/usr/bin/env python3
"""Example usage of {config.class_name}Validator."""

import polars as pl

from {config.name} import {config.class_name}Validator


def main():
    """Run example validation."""
    # Create sample data
    lf = pl.LazyFrame({{
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, None, 40, 50],
        "name": ["Alice", "Bob", "", "David", "Eve"],
    }})

    print("Sample data:")
    print(lf.collect())
    print()

    # Create and run validator
    validator = {config.class_name}Validator()
    issues = validator.validate(lf)

    # Print results
    print(f"Validation Results: {{len(issues)}} issues found")
    print("-" * 50)

    for issue in issues:
        print(f"  Column: {{issue.column}}")
        print(f"  Type: {{issue.issue_type}}")
        print(f"  Count: {{issue.count}}")
        print(f"  Severity: {{issue.severity.value}}")
        print(f"  Details: {{issue.details}}")
        if issue.sample_values:
            print(f"  Samples: {{issue.sample_values[:5]}}")
        print()


if __name__ == "__main__":
    main()
'''
        result.add_file(f"{config.name}/examples/basic_usage.py", content, executable=True)
