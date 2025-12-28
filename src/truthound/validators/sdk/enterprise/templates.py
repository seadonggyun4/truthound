"""Validator template CLI for scaffolding.

This module provides CLI tools for creating validator templates:
- Basic validator scaffold
- Category-specific templates
- Test file generation
- Project structure creation

Example:
    from truthound.validators.sdk.enterprise.templates import (
        TemplateCLI,
        TemplateType,
        create_validator_template,
    )

    # Create template
    cli = TemplateCLI()
    cli.create_validator(
        name="my_validator",
        category="custom",
        template_type=TemplateType.COLUMN,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of validator templates."""

    BASIC = auto()       # Minimal validator
    COLUMN = auto()      # Column-level validator
    AGGREGATE = auto()   # Aggregate validator
    PATTERN = auto()     # Pattern matching validator
    RANGE = auto()       # Range checking validator
    COMPARISON = auto()  # Column comparison validator
    COMPOSITE = auto()   # Multi-validator composite
    FULL = auto()        # Full-featured with tests


@dataclass(frozen=True)
class TemplateConfig:
    """Configuration for template generation.

    Attributes:
        name: Validator name
        category: Validator category
        template_type: Type of template
        output_dir: Output directory
        author: Author name
        version: Initial version
        description: Validator description
        include_tests: Whether to generate test file
        include_docs: Whether to generate documentation
        license_type: License type
    """

    name: str
    category: str = "custom"
    template_type: TemplateType = TemplateType.BASIC
    output_dir: Path = field(default_factory=lambda: Path("."))
    author: str = ""
    version: str = "1.0.0"
    description: str = ""
    include_tests: bool = True
    include_docs: bool = True
    license_type: str = "MIT"


@dataclass
class ValidatorTemplate:
    """Generated validator template.

    Attributes:
        name: Validator name
        source_code: Generated source code
        test_code: Generated test code
        documentation: Generated documentation
        config: Template configuration used
    """

    name: str
    source_code: str
    test_code: str = ""
    documentation: str = ""
    config: TemplateConfig | None = None


class TemplateGenerator:
    """Generates validator template code."""

    def __init__(self, config: TemplateConfig):
        """Initialize generator.

        Args:
            config: Template configuration
        """
        self.config = config

    def _snake_to_pascal(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))

    def _generate_basic(self) -> str:
        """Generate basic validator template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"

        return f'''"""
{self.config.description or f'{self.config.name} validator.'}

Author: {self.config.author or 'Unknown'}
Version: {self.config.version}
License: {self.config.license_type}
"""

from __future__ import annotations

import polars as pl

from truthound.validators.base import Validator, ValidationIssue
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{self.config.name}",
    category="{self.config.category}",
    description="{self.config.description or f'{self.config.name} validator'}",
    version="{self.config.version}",
    author="{self.config.author}",
    tags=["{self.config.category}"],
)
class {class_name}(Validator):
    """{self.config.description or f'{self.config.name} validator.'}

    Example:
        validator = {class_name}()
        issues = validator.validate(lf)
    """

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the data.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues found
        """
        issues: list[ValidationIssue] = []

        # TODO: Implement validation logic

        return issues
'''

    def _generate_column(self) -> str:
        """Generate column validator template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"

        return f'''"""
{self.config.description or f'{self.config.name} column validator.'}

Author: {self.config.author or 'Unknown'}
Version: {self.config.version}
License: {self.config.license_type}
"""

from __future__ import annotations

import polars as pl

from truthound.validators.base import (
    ColumnValidator,
    ValidationIssue,
    ValidatorConfig,
)
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{self.config.name}",
    category="{self.config.category}",
    description="{self.config.description or f'{self.config.name} column validator'}",
    version="{self.config.version}",
    author="{self.config.author}",
    tags=["{self.config.category}", "column"],
)
class {class_name}(ColumnValidator):
    """{self.config.description or f'{self.config.name} column validator.'}

    Validates each column that matches the configured criteria.

    Example:
        validator = {class_name}(columns=["value", "amount"])
        issues = validator.validate(lf)
    """

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        columns: list[str] | None = None,
        severity: Severity = Severity.MEDIUM,
        **kwargs,
    ):
        """Initialize the validator.

        Args:
            config: Validator configuration
            columns: Columns to validate (None = all applicable)
            severity: Severity level for issues
        """
        super().__init__(config, **kwargs)
        self._target_columns = columns
        self._severity = severity

    def validate_column(
        self,
        lf: pl.LazyFrame,
        column: str,
    ) -> list[ValidationIssue]:
        """Validate a single column.

        Args:
            lf: LazyFrame to validate
            column: Column name to validate

        Returns:
            List of validation issues for this column
        """
        issues: list[ValidationIssue] = []

        # TODO: Implement column validation logic
        # Example:
        # violations = lf.filter(pl.col(column).is_null()).collect()
        # if violations.height > 0:
        #     issues.append(ValidationIssue(
        #         column=column,
        #         issue_type="{self.config.name}",
        #         count=violations.height,
        #         severity=self._severity,
        #         details=f"Column '{{column}}' has {{violations.height}} issues",
        #     ))

        return issues

    def get_target_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get columns to validate.

        Args:
            lf: LazyFrame to get columns from

        Returns:
            List of column names to validate
        """
        if self._target_columns:
            return [c for c in self._target_columns if c in lf.columns]
        return lf.columns
'''

    def _generate_pattern(self) -> str:
        """Generate pattern validator template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"

        return f'''"""
{self.config.description or f'{self.config.name} pattern validator.'}

Author: {self.config.author or 'Unknown'}
Version: {self.config.version}
License: {self.config.license_type}
"""

from __future__ import annotations

import re

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorConfig,
    RegexValidatorMixin,
)
from truthound.validators.sdk import custom_validator
from truthound.types import Severity


@custom_validator(
    name="{self.config.name}",
    category="{self.config.category}",
    description="{self.config.description or f'{self.config.name} pattern validator'}",
    version="{self.config.version}",
    author="{self.config.author}",
    tags=["{self.config.category}", "pattern", "regex"],
)
class {class_name}(Validator, RegexValidatorMixin):
    """{self.config.description or f'{self.config.name} pattern validator.'}

    Validates values against a regex pattern.

    Example:
        validator = {class_name}(pattern=r"^[A-Z]{{2,3}}-\\d{{4}}$")
        issues = validator.validate(lf)
    """

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        pattern: str = r".*",
        columns: list[str] | None = None,
        case_sensitive: bool = True,
        invert: bool = False,
        severity: Severity = Severity.MEDIUM,
        **kwargs,
    ):
        """Initialize the validator.

        Args:
            config: Validator configuration
            pattern: Regex pattern to match
            columns: Columns to validate
            case_sensitive: Whether pattern matching is case sensitive
            invert: If True, match values that DON'T match the pattern
            severity: Severity level for issues
        """
        super().__init__(config, **kwargs)
        self._pattern_str = pattern
        self._columns = columns
        self._case_sensitive = case_sensitive
        self._invert = invert
        self._severity = severity

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        self._pattern = re.compile(pattern, flags)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate data against pattern.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Get string columns
        columns = self._columns or [
            col for col, dtype in zip(lf.columns, lf.dtypes)
            if dtype == pl.Utf8 or dtype == pl.String
        ]

        for column in columns:
            if column not in lf.columns:
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
                    ~pl.col(column).str.contains(self._pattern_str)
                ).collect()

            if violations.height > 0:
                samples = violations[column].head(5).to_list()
                issues.append(ValidationIssue(
                    column=column,
                    issue_type="{self.config.name}_pattern_mismatch",
                    count=violations.height,
                    severity=self._severity,
                    details=(
                        f"Column '{{column}}' has {{violations.height}} values that "
                        f"{'match' if self._invert else 'do not match'} "
                        f"pattern '{{self._pattern_str}}'"
                    ),
                    sample_values=samples,
                ))

        return issues
'''

    def _generate_range(self) -> str:
        """Generate range validator template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"

        return f'''"""
{self.config.description or f'{self.config.name} range validator.'}

Author: {self.config.author or 'Unknown'}
Version: {self.config.version}
License: {self.config.license_type}
"""

from __future__ import annotations

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
    name="{self.config.name}",
    category="{self.config.category}",
    description="{self.config.description or f'{self.config.name} range validator'}",
    version="{self.config.version}",
    author="{self.config.author}",
    tags=["{self.config.category}", "range", "numeric"],
)
class {class_name}(Validator, NumericValidatorMixin):
    """{self.config.description or f'{self.config.name} range validator.'}

    Validates numeric values are within a specified range.

    Example:
        validator = {class_name}(min_value=0, max_value=100)
        issues = validator.validate(lf)
    """

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        columns: list[str] | None = None,
        inclusive: bool = True,
        severity: Severity = Severity.MEDIUM,
        **kwargs,
    ):
        """Initialize the validator.

        Args:
            config: Validator configuration
            min_value: Minimum allowed value (None = no minimum)
            max_value: Maximum allowed value (None = no maximum)
            columns: Columns to validate
            inclusive: Whether bounds are inclusive
            severity: Severity level for issues
        """
        super().__init__(config, **kwargs)
        self._min_value = min_value
        self._max_value = max_value
        self._columns = columns
        self._inclusive = inclusive
        self._severity = severity

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate numeric values are in range.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Get numeric columns
        columns = self._columns or [
            col for col, dtype in zip(lf.columns, lf.dtypes)
            if any(isinstance(dtype, t) for t in NUMERIC_TYPES)
        ]

        for column in columns:
            if column not in lf.columns:
                continue

            # Build filter expression
            conditions = []

            if self._min_value is not None:
                if self._inclusive:
                    conditions.append(pl.col(column) < self._min_value)
                else:
                    conditions.append(pl.col(column) <= self._min_value)

            if self._max_value is not None:
                if self._inclusive:
                    conditions.append(pl.col(column) > self._max_value)
                else:
                    conditions.append(pl.col(column) >= self._max_value)

            if not conditions:
                continue

            # Combine conditions with OR
            filter_expr = conditions[0]
            for cond in conditions[1:]:
                filter_expr = filter_expr | cond

            violations = lf.filter(filter_expr).collect()

            if violations.height > 0:
                samples = violations[column].head(5).to_list()
                range_str = f"[{{self._min_value}}, {{self._max_value}}]"
                issues.append(ValidationIssue(
                    column=column,
                    issue_type="{self.config.name}_out_of_range",
                    count=violations.height,
                    severity=self._severity,
                    details=(
                        f"Column '{{column}}' has {{violations.height}} values "
                        f"outside range {{range_str}}"
                    ),
                    sample_values=samples,
                ))

        return issues
'''

    def _generate_test(self) -> str:
        """Generate test file template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"
        test_class_name = f"Test{class_name}"

        return f'''"""Tests for {class_name}."""

import pytest
import polars as pl

from truthound.validators.sdk.testing import (
    ValidatorTestCase,
    create_test_dataframe,
    assert_no_issues,
    assert_has_issue,
)
from .{self.config.name} import {class_name}


class {test_class_name}(ValidatorTestCase):
    """Test cases for {class_name}."""

    def test_valid_data(self):
        """Test that valid data produces no issues."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [1, 2, 3],
            "column2": ["a", "b", "c"],
        }})
        validator = {class_name}()

        # Act
        issues = validator.validate(lf)

        # Assert
        assert_no_issues(issues)

    def test_invalid_data(self):
        """Test that invalid data produces issues."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [None, 2, None],
            "column2": ["a", None, "c"],
        }})
        validator = {class_name}()

        # Act
        issues = validator.validate(lf)

        # Assert
        # TODO: Update assertion based on your validation logic
        # assert_has_issue(issues, issue_type="{self.config.name}")

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        # Arrange
        lf = pl.LazyFrame({{
            "column1": [],
            "column2": [],
        }})
        validator = {class_name}()

        # Act
        issues = validator.validate(lf)

        # Assert
        assert_no_issues(issues)

    def test_configuration(self):
        """Test validator configuration options."""
        # TODO: Add configuration tests
        pass

    def test_edge_cases(self):
        """Test edge cases."""
        # TODO: Add edge case tests
        pass
'''

    def _generate_documentation(self) -> str:
        """Generate documentation template."""
        class_name = self._snake_to_pascal(self.config.name) + "Validator"

        return f'''# {class_name}

> {self.config.category} validator

## Description

{self.config.description or 'TODO: Add description'}

## Installation

This validator is part of the Truthound package:

```bash
pip install truthound
```

## Usage

```python
from truthound.validators import {class_name}

# Create validator
validator = {class_name}()

# Validate data
issues = validator.validate(lf)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| TODO | TODO | TODO | TODO |

## Examples

### Basic Usage

```python
import polars as pl
from truthound.validators import {class_name}

# Create sample data
lf = pl.LazyFrame({{
    "value": [1, 2, 3, 4, 5],
}})

# Create and run validator
validator = {class_name}()
issues = validator.validate(lf)

print(f"Found {{len(issues)}} issues")
```

## See Also

- [Validator SDK]({self.config.name}_sdk.md)
- [Testing Guide]({self.config.name}_testing.md)

---

*Version: {self.config.version}*
*Author: {self.config.author or 'Unknown'}*
*License: {self.config.license_type}*
'''

    def generate(self) -> ValidatorTemplate:
        """Generate validator template.

        Returns:
            ValidatorTemplate with generated code
        """
        # Generate source code based on template type
        generators = {
            TemplateType.BASIC: self._generate_basic,
            TemplateType.COLUMN: self._generate_column,
            TemplateType.PATTERN: self._generate_pattern,
            TemplateType.RANGE: self._generate_range,
            TemplateType.AGGREGATE: self._generate_basic,  # TODO: Add specific
            TemplateType.COMPARISON: self._generate_basic,  # TODO: Add specific
            TemplateType.COMPOSITE: self._generate_basic,  # TODO: Add specific
            TemplateType.FULL: self._generate_column,
        }

        generator = generators.get(self.config.template_type, self._generate_basic)
        source_code = generator()

        # Generate test code
        test_code = ""
        if self.config.include_tests:
            test_code = self._generate_test()

        # Generate documentation
        documentation = ""
        if self.config.include_docs:
            documentation = self._generate_documentation()

        return ValidatorTemplate(
            name=self.config.name,
            source_code=source_code,
            test_code=test_code,
            documentation=documentation,
            config=self.config,
        )


class TemplateCLI:
    """CLI for creating validator templates."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize CLI.

        Args:
            output_dir: Default output directory
        """
        self.output_dir = output_dir or Path(".")

    def create_validator(
        self,
        name: str,
        category: str = "custom",
        template_type: TemplateType = TemplateType.BASIC,
        output_dir: Path | None = None,
        author: str = "",
        description: str = "",
        include_tests: bool = True,
        include_docs: bool = True,
    ) -> dict[str, Path]:
        """Create a new validator from template.

        Args:
            name: Validator name (snake_case)
            category: Validator category
            template_type: Type of template to use
            output_dir: Output directory
            author: Author name
            description: Validator description
            include_tests: Generate test file
            include_docs: Generate documentation

        Returns:
            Dictionary of file type to path
        """
        output_dir = output_dir or self.output_dir

        config = TemplateConfig(
            name=name,
            category=category,
            template_type=template_type,
            output_dir=output_dir,
            author=author,
            description=description,
            include_tests=include_tests,
            include_docs=include_docs,
        )

        generator = TemplateGenerator(config)
        template = generator.generate()

        # Create output directory
        validator_dir = output_dir / name
        validator_dir.mkdir(parents=True, exist_ok=True)

        created_files: dict[str, Path] = {}

        # Write source code
        source_path = validator_dir / f"{name}.py"
        with open(source_path, "w") as f:
            f.write(template.source_code)
        created_files["source"] = source_path

        # Write init file
        init_path = validator_dir / "__init__.py"
        class_name = generator._snake_to_pascal(name) + "Validator"
        with open(init_path, "w") as f:
            f.write(f'from .{name} import {class_name}\n\n__all__ = ["{class_name}"]\n')
        created_files["init"] = init_path

        # Write test file
        if template.test_code:
            tests_dir = validator_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_path = tests_dir / f"test_{name}.py"
            with open(test_path, "w") as f:
                f.write(template.test_code)
            created_files["test"] = test_path

            # Tests init
            tests_init = tests_dir / "__init__.py"
            with open(tests_init, "w") as f:
                f.write("")
            created_files["test_init"] = tests_init

        # Write documentation
        if template.documentation:
            docs_dir = validator_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            docs_path = docs_dir / f"{name}.md"
            with open(docs_path, "w") as f:
                f.write(template.documentation)
            created_files["docs"] = docs_path

        return created_files

    def list_templates(self) -> list[dict[str, str]]:
        """List available templates.

        Returns:
            List of template info dictionaries
        """
        return [
            {
                "type": t.name,
                "description": {
                    TemplateType.BASIC: "Minimal validator template",
                    TemplateType.COLUMN: "Column-level validator with target column support",
                    TemplateType.AGGREGATE: "Aggregate validator for statistical checks",
                    TemplateType.PATTERN: "Pattern matching validator with regex support",
                    TemplateType.RANGE: "Range checking validator for numeric columns",
                    TemplateType.COMPARISON: "Column comparison validator",
                    TemplateType.COMPOSITE: "Multi-validator composite",
                    TemplateType.FULL: "Full-featured with tests and documentation",
                }.get(t, ""),
            }
            for t in TemplateType
        ]


def create_validator_template(
    name: str,
    category: str = "custom",
    template_type: TemplateType = TemplateType.BASIC,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Create a new validator from template.

    Args:
        name: Validator name
        category: Validator category
        template_type: Template type
        output_dir: Output directory

    Returns:
        Dictionary of created file paths
    """
    cli = TemplateCLI(output_dir)
    return cli.create_validator(name, category, template_type)
