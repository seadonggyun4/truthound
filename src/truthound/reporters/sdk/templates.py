"""Pre-built reporter templates for common output formats.

This module provides ready-to-use reporters for:
- CSV: Tabular data output
- YAML: Human-readable structured data
- JUnit XML: Test framework integration
- NDJSON: Newline-delimited JSON for streaming
- Table: Formatted ASCII/Unicode tables

These can be used directly or as templates for custom reporters.

Example:
    >>> from truthound.reporters.sdk import CSVReporter, JUnitXMLReporter
    >>>
    >>> # Use CSV reporter
    >>> reporter = CSVReporter()
    >>> csv_output = reporter.render(validation_result)
    >>>
    >>> # Use JUnit XML for CI integration
    >>> reporter = JUnitXMLReporter()
    >>> reporter.write(validation_result, "test-results.xml")
"""

from __future__ import annotations

import csv
import io
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


# =============================================================================
# CSV Reporter
# =============================================================================


@dataclass
class CSVReporterConfig(ReporterConfig):
    """Configuration for CSV reporter.

    Attributes:
        delimiter: Field delimiter character.
        quote_char: Quote character for fields with special chars.
        include_header: Whether to include header row.
        include_passed: Whether to include passed validators.
        columns: List of columns to include (default: all).
        date_format: Format string for datetime fields.
    """

    delimiter: str = ","
    quote_char: str = '"'
    include_header: bool = True
    include_passed: bool = False
    columns: list[str] | None = None
    date_format: str = "%Y-%m-%d %H:%M:%S"


@register_reporter("csv")
class CSVReporter(
    FormattingMixin,
    FilteringMixin,
    SerializationMixin,
    ValidationReporter[CSVReporterConfig],
):
    """CSV format reporter for tabular output.

    Outputs validation results as CSV with configurable columns.

    Example:
        >>> reporter = CSVReporter(include_passed=True)
        >>> csv_output = reporter.render(result)
        >>> print(csv_output)
        validator,column,success,severity,message
        not_null,email,False,high,Found 5 null values
        unique,id,True,,,
    """

    name = "csv"
    file_extension = ".csv"
    content_type = "text/csv"

    # Default columns for output
    DEFAULT_COLUMNS = [
        "validator_name",
        "column",
        "success",
        "severity",
        "issue_type",
        "count",
        "message",
    ]

    @classmethod
    def _default_config(cls) -> CSVReporterConfig:
        return CSVReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as CSV.

        Args:
            data: Validation result to render.

        Returns:
            CSV formatted string.
        """
        # Filter results
        results = data.results
        if not self._config.include_passed:
            results = self.filter_failed(results)

        # Determine columns
        columns = self._config.columns or self.DEFAULT_COLUMNS

        # Build rows
        rows = []
        for result in results:
            row = self._result_to_row(result)
            rows.append(row)

        # Generate CSV
        return self.to_csv(
            rows,
            columns=columns,
            delimiter=self._config.delimiter,
            include_header=self._config.include_header,
        )

    def _result_to_row(self, result: "ValidatorResult") -> dict[str, Any]:
        """Convert ValidatorResult to row dictionary."""
        return {
            "validator_name": result.validator_name,
            "column": result.column or "",
            "success": result.success,
            "severity": result.severity or "",
            "issue_type": result.issue_type or "",
            "count": result.count or 0,
            "message": result.message or "",
            "sample_values": (
                json.dumps(
                    result.details.get("sample_values", [])[:self._config.max_sample_values]
                )
                if result.details.get("sample_values")
                else ""
            ),
        }


# =============================================================================
# YAML Reporter
# =============================================================================


@dataclass
class YAMLReporterConfig(ReporterConfig):
    """Configuration for YAML reporter.

    Attributes:
        default_flow_style: Use flow style for nested structures.
        indent: Indentation size.
        include_passed: Whether to include passed validators.
        sort_keys: Whether to sort dictionary keys.
    """

    default_flow_style: bool = False
    indent: int = 2
    include_passed: bool = False
    sort_keys: bool = False


@register_reporter("yaml")
class YAMLReporter(
    AggregationMixin,
    FilteringMixin,
    ValidationReporter[YAMLReporterConfig],
):
    """YAML format reporter for human-readable structured output.

    Example:
        >>> reporter = YAMLReporter()
        >>> yaml_output = reporter.render(result)
        >>> print(yaml_output)
        validation_result:
          status: failure
          run_id: abc123
          issues:
            - validator: not_null
              column: email
              severity: high
    """

    name = "yaml"
    file_extension = ".yaml"
    content_type = "application/yaml"

    @classmethod
    def _default_config(cls) -> YAMLReporterConfig:
        return YAMLReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as YAML.

        Args:
            data: Validation result to render.

        Returns:
            YAML formatted string.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML output. "
                "Install with: pip install pyyaml"
            )

        # Build structure
        structure = self._build_structure(data)

        # Render YAML
        return yaml.dump(
            structure,
            default_flow_style=self._config.default_flow_style,
            indent=self._config.indent,
            sort_keys=self._config.sort_keys,
            allow_unicode=True,
        )

    def _build_structure(self, data: "ValidationResult") -> dict[str, Any]:
        """Build YAML structure from validation result."""
        # Filter results
        results = data.results
        if not self._config.include_passed:
            results = self.filter_failed(results)

        # Get summary stats
        stats = self.get_summary_stats(data)

        return {
            "validation_result": {
                "run_id": data.run_id,
                "data_asset": data.data_asset,
                "status": data.status.value,
                "run_time": data.run_time.isoformat() if data.run_time else None,
                "summary": {
                    "total_validators": stats["total_validators"],
                    "passed": stats["passed"],
                    "failed": stats["failed"],
                    "pass_rate": f"{stats['pass_rate']:.1f}%",
                },
                "severity_counts": stats["by_severity"],
                "issues": [
                    {
                        "validator": r.validator_name,
                        "column": r.column,
                        "severity": r.severity,
                        "issue_type": r.issue_type,
                        "message": r.message,
                        "count": r.count,
                    }
                    for r in results
                ],
            }
        }


# =============================================================================
# JUnit XML Reporter
# =============================================================================


@dataclass
class JUnitXMLReporterConfig(ReporterConfig):
    """Configuration for JUnit XML reporter.

    Attributes:
        suite_name: Name for the test suite.
        include_passed: Whether to include passed validators as test cases.
        include_properties: Whether to include properties element.
        include_system_out: Whether to include system-out with details.
    """

    suite_name: str | None = None
    include_passed: bool = True
    include_properties: bool = True
    include_system_out: bool = True


@register_reporter("junit")
class JUnitXMLReporter(
    SerializationMixin,
    AggregationMixin,
    ValidationReporter[JUnitXMLReporterConfig],
):
    """JUnit XML format reporter for CI/CD integration.

    Produces JUnit-compatible XML that can be consumed by CI systems
    like Jenkins, GitHub Actions, GitLab CI, etc.

    Example:
        >>> reporter = JUnitXMLReporter(suite_name="data-validation")
        >>> reporter.write(result, "junit-results.xml")
    """

    name = "junit"
    file_extension = ".xml"
    content_type = "application/xml"

    @classmethod
    def _default_config(cls) -> JUnitXMLReporterConfig:
        return JUnitXMLReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as JUnit XML.

        Args:
            data: Validation result to render.

        Returns:
            JUnit XML formatted string.
        """
        # Gather statistics
        results = data.results
        failures = [r for r in results if not r.success]
        errors = 0  # We treat all failures as failures, not errors

        # Build test cases
        test_cases = []
        for result in results:
            if not self._config.include_passed and result.success:
                continue
            test_cases.append(self._render_test_case(result))

        # Build properties if enabled
        properties_element = ""
        if self._config.include_properties:
            properties = [
                self.to_xml_element("property", attributes={
                    "name": "data_asset",
                    "value": data.data_asset,
                }),
                self.to_xml_element("property", attributes={
                    "name": "run_id",
                    "value": data.run_id,
                }),
                self.to_xml_element("property", attributes={
                    "name": "status",
                    "value": data.status.value,
                }),
            ]
            properties_element = self.to_xml_element(
                "properties",
                children=properties,
            )

        # Build system-out if enabled
        system_out = ""
        if self._config.include_system_out:
            stats = self.get_summary_stats(data)
            out_content = (
                f"Validation Summary:\n"
                f"  Total: {stats['total_validators']}\n"
                f"  Passed: {stats['passed']}\n"
                f"  Failed: {stats['failed']}\n"
                f"  Pass Rate: {stats['pass_rate']:.1f}%"
            )
            system_out = f"<system-out><![CDATA[{out_content}]]></system-out>"

        # Calculate time
        time_seconds = 0.0  # We don't track execution time per validator

        # Build testsuite
        suite_name = self._config.suite_name or f"truthound.{data.data_asset}"
        suite_attrs = {
            "name": suite_name,
            "tests": str(len(results) if self._config.include_passed else len(failures)),
            "failures": str(len(failures)),
            "errors": str(errors),
            "time": f"{time_seconds:.3f}",
            "timestamp": data.run_time.isoformat() if data.run_time else datetime.now().isoformat(),
        }

        suite_content = []
        if properties_element:
            suite_content.append(properties_element)
        suite_content.extend(test_cases)
        if system_out:
            suite_content.append(system_out)

        testsuite = self.to_xml_element(
            "testsuite",
            attributes=suite_attrs,
            children=suite_content,
        )

        # Wrap in testsuites
        testsuites = self.to_xml_element(
            "testsuites",
            children=[testsuite],
        )

        return f'<?xml version="1.0" encoding="UTF-8"?>\n{testsuites}'

    def _render_test_case(self, result: "ValidatorResult") -> str:
        """Render a single test case."""
        class_name = f"truthound.validators.{result.validator_name}"
        name = result.column or "table_level"

        attrs = {
            "classname": class_name,
            "name": name,
            "time": "0.000",
        }

        if result.success:
            return self.to_xml_element("testcase", attributes=attrs)

        # Add failure element
        failure_attrs = {
            "type": result.issue_type or "ValidationFailure",
            "message": self._escape_xml(result.message or "Validation failed"),
        }

        failure_content = ""
        sample_values = result.details.get("sample_values", [])
        if sample_values:
            samples = ", ".join(str(v) for v in sample_values[:5])
            failure_content = f"Sample values: {samples}"

        failure = self.to_xml_element(
            "failure",
            value=failure_content if failure_content else None,
            attributes=failure_attrs,
        )

        return self.to_xml_element(
            "testcase",
            attributes=attrs,
            children=[failure],
        )


# =============================================================================
# NDJSON Reporter (Newline Delimited JSON)
# =============================================================================


@dataclass
class NDJSONReporterConfig(ReporterConfig):
    """Configuration for NDJSON reporter.

    Attributes:
        include_passed: Whether to include passed validators.
        include_metadata: Whether to include metadata line.
        compact: Whether to use compact JSON (no extra spaces).
    """

    include_passed: bool = False
    include_metadata: bool = True
    compact: bool = True


@register_reporter("ndjson")
class NDJSONReporter(
    SerializationMixin,
    FilteringMixin,
    ValidationReporter[NDJSONReporterConfig],
):
    """NDJSON format reporter for streaming and log processing.

    Outputs one JSON object per line, suitable for:
    - Log aggregation systems (ELK, Splunk)
    - Streaming processing
    - Line-by-line parsing

    Example:
        >>> reporter = NDJSONReporter()
        >>> ndjson_output = reporter.render(result)
        >>> for line in ndjson_output.split("\\n"):
        ...     record = json.loads(line)
        ...     print(record["validator"])
    """

    name = "ndjson"
    file_extension = ".ndjson"
    content_type = "application/x-ndjson"

    @classmethod
    def _default_config(cls) -> NDJSONReporterConfig:
        return NDJSONReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as NDJSON.

        Args:
            data: Validation result to render.

        Returns:
            NDJSON formatted string (one JSON object per line).
        """
        lines = []

        # Optionally add metadata line
        if self._config.include_metadata:
            metadata = {
                "type": "metadata",
                "run_id": data.run_id,
                "data_asset": data.data_asset,
                "status": data.status.value,
                "run_time": data.run_time.isoformat() if data.run_time else None,
                "total_validators": len(data.results),
            }
            lines.append(self._to_json_line(metadata))

        # Filter and add result lines
        results = data.results
        if not self._config.include_passed:
            results = self.filter_failed(results)

        for result in results:
            record = {
                "type": "result",
                "validator": result.validator_name,
                "column": result.column,
                "success": result.success,
                "severity": result.severity,
                "issue_type": result.issue_type,
                "count": result.count,
                "message": result.message,
            }
            lines.append(self._to_json_line(record))

        return "\n".join(lines)

    def _to_json_line(self, data: dict[str, Any]) -> str:
        """Convert dict to JSON line."""
        indent = None if self._config.compact else None
        separators = (",", ":") if self._config.compact else None
        return json.dumps(data, default=str, indent=indent, separators=separators)


# =============================================================================
# Table Reporter (ASCII/Unicode Tables)
# =============================================================================


@dataclass
class TableReporterConfig(ReporterConfig):
    """Configuration for table reporter.

    Attributes:
        style: Table style ("ascii", "markdown", "grid", "simple").
        include_passed: Whether to include passed validators.
        max_column_width: Maximum width for any column.
        columns: Columns to display.
        sort_by: Column to sort by.
        sort_ascending: Sort order.
    """

    style: str = "ascii"
    include_passed: bool = False
    max_column_width: int = 50
    columns: list[str] = field(
        default_factory=lambda: ["validator", "column", "severity", "message"]
    )
    sort_by: str | None = "severity"
    sort_ascending: bool = False


@register_reporter("table")
class TableReporter(
    FormattingMixin,
    FilteringMixin,
    AggregationMixin,
    ValidationReporter[TableReporterConfig],
):
    """ASCII/Unicode table reporter for terminal output.

    Produces formatted tables suitable for console output.

    Example:
        >>> reporter = TableReporter(style="grid")
        >>> print(reporter.render(result))
        ╔════════════╤════════╤══════════╤═══════════════════════════╗
        ║ validator  │ column │ severity │ message                   ║
        ╠════════════╪════════╪══════════╪═══════════════════════════╣
        ║ not_null   │ email  │ high     │ Found 5 null values       ║
        ║ unique     │ id     │ critical │ Found 3 duplicate values  ║
        ╚════════════╧════════╧══════════╧═══════════════════════════╝
    """

    name = "table"
    file_extension = ".txt"
    content_type = "text/plain"

    # Column headers for display
    COLUMN_HEADERS = {
        "validator": "Validator",
        "column": "Column",
        "severity": "Severity",
        "message": "Message",
        "count": "Count",
        "issue_type": "Issue Type",
    }

    @classmethod
    def _default_config(cls) -> TableReporterConfig:
        return TableReporterConfig()

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as formatted table.

        Args:
            data: Validation result to render.

        Returns:
            Formatted table string.
        """
        # Filter results
        results = data.results
        if not self._config.include_passed:
            results = self.filter_failed(results)

        # Sort if configured
        if self._config.sort_by == "severity":
            results = self.sort_by_severity(results, ascending=self._config.sort_ascending)
        elif self._config.sort_by == "column":
            results = self.sort_by_column(results, ascending=self._config.sort_ascending)

        # Build rows
        rows = []
        for result in results:
            row = {
                "validator": result.validator_name,
                "column": result.column or "",
                "severity": result.severity or "",
                "message": result.message or "",
                "count": str(result.count or ""),
                "issue_type": result.issue_type or "",
            }
            rows.append(row)

        if not rows:
            return "No issues found."

        # Render table
        return self.format_as_table(
            rows,
            columns=self._config.columns,
            headers=self.COLUMN_HEADERS,
            style=self._config.style,
            max_width=self._config.max_column_width,
        )
