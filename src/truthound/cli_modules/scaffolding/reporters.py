"""Reporter scaffold generator.

This module provides scaffolding for creating custom reporters with
various template variants:
    - basic: Minimal reporter with core structure
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
    name="reporter",
    description="Generate a custom reporter",
    aliases=("rep", "r"),
)
class ReporterScaffold(BaseScaffold):
    """Scaffold generator for reporters.

    Creates custom reporter templates for different output formats.
    """

    name: ClassVar[str] = "reporter"
    description: ClassVar[str] = "Generate a custom reporter"
    aliases: ClassVar[tuple[str, ...]] = ("rep", "r")

    TEMPLATE_VARIANTS: ClassVar[tuple[str, ...]] = ("basic", "full")

    def get_options(self) -> dict[str, Any]:
        """Get reporter-specific options."""
        return {
            "file_extension": {
                "type": "str",
                "default": ".txt",
                "description": "File extension for output (e.g., .txt, .json, .xml)",
            },
            "content_type": {
                "type": "str",
                "default": "text/plain",
                "description": "MIME content type",
            },
        }

    def _generate_files(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate reporter files based on variant."""
        variant = config.template_variant

        if variant == "full":
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
        content = f'''"""Package for {config.title_name} reporter."""

from {config.name}.reporter import {config.class_name}Reporter, {config.class_name}ReporterConfig

__all__ = ["{config.class_name}Reporter", "{config.class_name}ReporterConfig"]
'''
        result.add_file(f"{config.name}/__init__.py", content)

    def _generate_basic(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate basic reporter template."""
        file_ext = config.extra.get("file_extension", ".txt")
        content_type = config.extra.get("content_type", "text/plain")

        content = f'''{self._get_header(config)}

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.reporters.base import ReporterConfig, ValidationReporter
from truthound.reporters.factory import register_reporter

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class {config.class_name}ReporterConfig(ReporterConfig):
    """{config.class_name} reporter configuration.

    Attributes:
        include_passed: Whether to include passed validators.
        include_samples: Whether to include sample values.
        max_issues: Maximum number of issues to include.
    """

    include_passed: bool = False
    include_samples: bool = True
    max_issues: int | None = None


@register_reporter("{config.name}")
class {config.class_name}Reporter(ValidationReporter[{config.class_name}ReporterConfig]):
    """{config.description or f'{config.title_name} reporter.'}

    Generates validation reports in custom format.

    Example:
        >>> reporter = {config.class_name}Reporter()
        >>> output = reporter.render(validation_result)
        >>> reporter.write(validation_result, "report{file_ext}")
    """

    name = "{config.name}"
    file_extension = "{file_ext}"
    content_type = "{content_type}"

    @classmethod
    def _default_config(cls) -> {config.class_name}ReporterConfig:
        """Return default configuration."""
        return {config.class_name}ReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result to string.

        Args:
            data: Validation result to render.

        Returns:
            Rendered string in custom format.
        """
        lines: list[str] = []

        # Header
        lines.append(f"Validation Report: {{data.data_asset}}")
        lines.append(f"Status: {{data.status.value}}")
        lines.append(f"Run ID: {{data.run_id}}")
        if data.run_time:
            lines.append(f"Run Time: {{data.run_time.isoformat()}}")
        lines.append("")

        # Summary
        total = len(data.results)
        passed = sum(1 for r in data.results if r.success)
        failed = total - passed

        lines.append("Summary")
        lines.append("-" * 40)
        lines.append(f"Total Validators: {{total}}")
        lines.append(f"Passed: {{passed}}")
        lines.append(f"Failed: {{failed}}")
        if total > 0:
            lines.append(f"Pass Rate: {{passed / total * 100:.1f}}%")
        lines.append("")

        # Filter results
        results = data.results
        if not self._config.include_passed:
            results = [r for r in results if not r.success]

        if self._config.max_issues:
            results = results[: self._config.max_issues]

        # Issues
        if results:
            lines.append("Issues")
            lines.append("-" * 40)

            for i, result in enumerate(results, 1):
                lines.append(f"{{i}}. {{result.validator_name}}")
                lines.append(f"   Column: {{result.column or 'N/A'}}")
                lines.append(f"   Severity: {{result.severity or 'N/A'}}")
                lines.append(f"   Message: {{result.message or 'N/A'}}")

                if self._config.include_samples and result.sample_values:
                    samples = result.sample_values[: self._config.max_sample_values]
                    lines.append(f"   Samples: {{samples}}")

                lines.append("")

        return "\\n".join(lines)
'''
        result.add_file(f"{config.name}/reporter.py", content)

    def _generate_full(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate full-featured reporter template."""
        file_ext = config.extra.get("file_extension", ".json")
        content_type = config.extra.get("content_type", "application/json")

        content = f'''{self._get_header(config)}

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from truthound.reporters.base import ReporterConfig, ValidationReporter
from truthound.reporters.factory import register_reporter
from truthound.reporters.sdk.mixins import (
    AggregationMixin,
    FilteringMixin,
    FormattingMixin,
    SerializationMixin,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


@dataclass
class {config.class_name}ReporterConfig(ReporterConfig):
    """{config.class_name} reporter configuration.

    Attributes:
        include_passed: Whether to include passed validators.
        include_samples: Whether to include sample values.
        include_metadata: Whether to include metadata section.
        include_statistics: Whether to include statistics section.
        max_issues: Maximum number of issues to include.
        sort_by: Field to sort issues by (severity, column, validator).
        sort_ascending: Sort order.
        indent: JSON indentation level.
        date_format: Format for datetime fields.
    """

    include_passed: bool = False
    include_samples: bool = True
    include_metadata: bool = True
    include_statistics: bool = True
    max_issues: int | None = None
    sort_by: str = "severity"
    sort_ascending: bool = False
    indent: int = 2
    date_format: str = "%Y-%m-%dT%H:%M:%SZ"


@register_reporter("{config.name}")
class {config.class_name}Reporter(
    AggregationMixin,
    FilteringMixin,
    FormattingMixin,
    SerializationMixin,
    ValidationReporter[{config.class_name}ReporterConfig],
):
    """{config.description or f'{config.title_name} reporter.'}

    Full-featured reporter with JSON output, filtering, and sorting.

    Example:
        >>> reporter = {config.class_name}Reporter(include_statistics=True)
        >>> output = reporter.render(validation_result)
        >>> reporter.write(validation_result, "report{file_ext}")
    """

    name = "{config.name}"
    file_extension = "{file_ext}"
    content_type = "{content_type}"

    @classmethod
    def _default_config(cls) -> {config.class_name}ReporterConfig:
        """Return default configuration."""
        return {config.class_name}ReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result to JSON.

        Args:
            data: Validation result to render.

        Returns:
            JSON formatted string.
        """
        output: dict[str, Any] = {{}}

        # Metadata section
        if self._config.include_metadata:
            output["metadata"] = self._build_metadata(data)

        # Statistics section
        if self._config.include_statistics:
            output["statistics"] = self._build_statistics(data)

        # Results section
        output["issues"] = self._build_issues(data)

        return json.dumps(
            output,
            indent=self._config.indent,
            default=self._json_serializer,
        )

    def _build_metadata(self, data: "ValidationResult") -> dict[str, Any]:
        """Build metadata section."""
        return {{
            "run_id": data.run_id,
            "data_asset": data.data_asset,
            "status": data.status.value,
            "run_time": (
                data.run_time.strftime(self._config.date_format)
                if data.run_time
                else None
            ),
            "generated_at": datetime.utcnow().strftime(self._config.date_format),
            "reporter_version": "{config.version}",
        }}

    def _build_statistics(self, data: "ValidationResult") -> dict[str, Any]:
        """Build statistics section."""
        stats = self.get_summary_stats(data)

        return {{
            "total_validators": stats["total_validators"],
            "passed": stats["passed"],
            "failed": stats["failed"],
            "pass_rate": round(stats["pass_rate"], 2),
            "by_severity": stats["by_severity"],
        }}

    def _build_issues(self, data: "ValidationResult") -> list[dict[str, Any]]:
        """Build issues section."""
        results = data.results

        # Filter
        if not self._config.include_passed:
            results = self.filter_failed(results)

        # Sort
        if self._config.sort_by == "severity":
            results = self.sort_by_severity(
                results, ascending=self._config.sort_ascending
            )
        elif self._config.sort_by == "column":
            results = self.sort_by_column(
                results, ascending=self._config.sort_ascending
            )

        # Limit
        if self._config.max_issues:
            results = results[: self._config.max_issues]

        # Convert to dictionaries
        issues = []
        for result in results:
            issue = self._result_to_dict(result)
            issues.append(issue)

        return issues

    def _result_to_dict(self, result: "ValidatorResult") -> dict[str, Any]:
        """Convert ValidatorResult to dictionary."""
        issue: dict[str, Any] = {{
            "validator": result.validator_name,
            "column": result.column,
            "success": result.success,
            "severity": result.severity,
            "issue_type": result.issue_type,
            "message": result.message,
            "count": result.count,
        }}

        if self._config.include_samples and result.sample_values:
            issue["sample_values"] = result.sample_values[
                : self._config.max_sample_values
            ]

        return issue

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.strftime(self._config.date_format)
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
'''
        result.add_file(f"{config.name}/reporter.py", content)

        # Include tests and docs for full template
        self._generate_tests(config, result)
        self._generate_docs(config, result)

    def _generate_tests(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate test file."""
        content = f'''"""Tests for {config.class_name}Reporter."""

import pytest
from datetime import datetime

from truthound.stores.results import ValidationResult, ValidatorResult
from truthound.types import ValidationStatus

from {config.name} import {config.class_name}Reporter, {config.class_name}ReporterConfig


@pytest.fixture
def sample_validation_result():
    """Create a sample validation result for testing."""
    return ValidationResult(
        run_id="test-run-123",
        data_asset="test_data.csv",
        status=ValidationStatus.FAILURE,
        run_time=datetime(2024, 1, 1, 12, 0, 0),
        results=[
            ValidatorResult(
                validator_name="not_null",
                column="email",
                success=False,
                severity="high",
                issue_type="null_values",
                message="Found 5 null values",
                count=5,
                sample_values=[None, None],
            ),
            ValidatorResult(
                validator_name="unique",
                column="id",
                success=True,
                severity=None,
                issue_type=None,
                message=None,
                count=0,
            ),
        ],
    )


class Test{config.class_name}Reporter:
    """Test cases for {config.class_name}Reporter."""

    def test_render_basic(self, sample_validation_result):
        """Test basic rendering."""
        reporter = {config.class_name}Reporter()
        output = reporter.render(sample_validation_result)

        assert output is not None
        assert len(output) > 0

    def test_render_excludes_passed_by_default(self, sample_validation_result):
        """Test that passed validators are excluded by default."""
        reporter = {config.class_name}Reporter()
        output = reporter.render(sample_validation_result)

        # Should not contain the passed validator details
        assert "unique" not in output or "passed" not in output.lower()

    def test_render_includes_passed_when_configured(self, sample_validation_result):
        """Test including passed validators."""
        config = {config.class_name}ReporterConfig(include_passed=True)
        reporter = {config.class_name}Reporter(config=config)
        output = reporter.render(sample_validation_result)

        # Both validators should be included
        assert "not_null" in output
        assert "unique" in output

    def test_write_creates_file(self, sample_validation_result, tmp_path):
        """Test that write creates a file."""
        reporter = {config.class_name}Reporter()
        output_path = tmp_path / "report{config.extra.get('file_extension', '.txt')}"

        reporter.write(sample_validation_result, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0

    def test_file_extension(self):
        """Test file extension property."""
        reporter = {config.class_name}Reporter()
        assert reporter.file_extension == "{config.extra.get('file_extension', '.txt')}"

    def test_content_type(self):
        """Test content type property."""
        reporter = {config.class_name}Reporter()
        assert reporter.content_type == "{config.extra.get('content_type', 'text/plain')}"

    def test_empty_results(self):
        """Test with empty results."""
        result = ValidationResult(
            run_id="test-run",
            data_asset="empty.csv",
            status=ValidationStatus.SUCCESS,
            results=[],
        )
        reporter = {config.class_name}Reporter()
        output = reporter.render(result)

        assert output is not None

    def test_configuration_options(self, sample_validation_result):
        """Test various configuration options."""
        config = {config.class_name}ReporterConfig(
            include_passed=True,
            include_samples=False,
            max_issues=1,
        )
        reporter = {config.class_name}Reporter(config=config)
        output = reporter.render(sample_validation_result)

        assert output is not None
'''
        result.add_file(f"{config.name}/tests/__init__.py", "")
        result.add_file(f"{config.name}/tests/test_reporter.py", content)

    def _generate_docs(self, config: ScaffoldConfig, result: ScaffoldResult) -> None:
        """Generate documentation file."""
        content = f'''# {config.class_name}Reporter

> Custom reporter for Truthound

## Description

{config.description or 'TODO: Add description'}

## Installation

```bash
pip install truthound
```

## Usage

```python
from {config.name} import {config.class_name}Reporter

# Create reporter
reporter = {config.class_name}Reporter()

# Render to string
output = reporter.render(validation_result)

# Write to file
reporter.write(validation_result, "report{config.extra.get('file_extension', '.txt')}")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| include_passed | bool | False | Include passed validators |
| include_samples | bool | True | Include sample values |
| max_issues | int | None | Maximum issues to include |

## Examples

### Basic Usage

```python
from {config.name} import {config.class_name}Reporter

reporter = {config.class_name}Reporter()
output = reporter.render(validation_result)
print(output)
```

### With Custom Configuration

```python
from {config.name} import {config.class_name}Reporter, {config.class_name}ReporterConfig

config = {config.class_name}ReporterConfig(
    include_passed=True,
    max_issues=10,
)
reporter = {config.class_name}Reporter(config=config)
output = reporter.render(validation_result)
```

---

*Version: {config.version}*
*Author: {config.author or 'Unknown'}*
*License: {config.license_type}*
'''
        result.add_file(f"{config.name}/docs/README.md", content)
