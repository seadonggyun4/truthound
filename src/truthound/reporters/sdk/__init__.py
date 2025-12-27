"""Reporter SDK - Tools and utilities for building custom reporters.

This module provides a comprehensive SDK for developing custom reporters with:
- Mixins for common functionality (formatting, aggregation, filtering)
- Template-based reporter generators
- Schema validation for output formats
- Built-in reporter templates (CSV, YAML, JUnit XML)
- Testing utilities

Quick Start:
    >>> from truthound.reporters.sdk import (
    ...     create_reporter,
    ...     FormattingMixin,
    ...     AggregationMixin,
    ...     FilteringMixin,
    ... )
    >>>
    >>> # Create a simple custom reporter using the SDK
    >>> @create_reporter("my_format", extension=".myf")
    ... def render_my_format(result, config):
    ...     return f"Status: {result.status.value}"

Full Custom Reporter:
    >>> from truthound.reporters.sdk import (
    ...     ReporterBuilder,
    ...     FormattingMixin,
    ...     AggregationMixin,
    ... )
    >>> from truthound.reporters.base import ValidationReporter, ReporterConfig
    >>>
    >>> class MyReporterConfig(ReporterConfig):
    ...     custom_option: str = "default"
    >>>
    >>> class MyReporter(FormattingMixin, AggregationMixin, ValidationReporter[MyReporterConfig]):
    ...     name = "my_format"
    ...     file_extension = ".myf"
    ...
    ...     @classmethod
    ...     def _default_config(cls):
    ...         return MyReporterConfig()
    ...
    ...     def render(self, data):
    ...         # Use mixin methods
    ...         issues = self.filter_by_severity(data, min_severity="medium")
    ...         grouped = self.group_by_column(issues)
    ...         return self.format_as_table(grouped)
"""

from truthound.reporters.sdk.mixins import (
    FormattingMixin,
    AggregationMixin,
    FilteringMixin,
    SerializationMixin,
    TemplatingMixin,
    StreamingMixin,
)
from truthound.reporters.sdk.builder import (
    ReporterBuilder,
    create_reporter,
    create_validation_reporter,
)
from truthound.reporters.sdk.templates import (
    CSVReporter,
    YAMLReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,
)
from truthound.reporters.sdk.schema import (
    # Core classes
    ReportSchema,
    JSONSchema,
    XMLSchema,
    CSVSchema,
    TextSchema,
    # Validation
    ValidationResult,
    ValidationError,
    SchemaError,
    # Functions
    validate_output,
    register_schema,
    get_schema,
    unregister_schema,
    validate_reporter_output,
    infer_schema,
    merge_schemas,
)
from truthound.reporters.sdk.testing import (
    # Test case base
    ReporterTestCase,
    # Mock data generators
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    MockResultBuilder,
    MockValidationResult,
    MockValidatorResult,
    # Assertions
    assert_valid_output,
    assert_json_valid,
    assert_xml_valid,
    assert_csv_valid,
    assert_contains_patterns,
    # Fixtures
    create_sample_data,
    create_edge_case_data,
    create_stress_test_data,
    # Utilities
    capture_output,
    benchmark_reporter,
    BenchmarkResult,
)

__all__ = [
    # Mixins
    "FormattingMixin",
    "AggregationMixin",
    "FilteringMixin",
    "SerializationMixin",
    "TemplatingMixin",
    "StreamingMixin",
    # Builder
    "ReporterBuilder",
    "create_reporter",
    "create_validation_reporter",
    # Templates
    "CSVReporter",
    "YAMLReporter",
    "JUnitXMLReporter",
    "NDJSONReporter",
    "TableReporter",
    # Schema - Core
    "ReportSchema",
    "JSONSchema",
    "XMLSchema",
    "CSVSchema",
    "TextSchema",
    # Schema - Validation
    "ValidationResult",
    "ValidationError",
    "SchemaError",
    # Schema - Functions
    "validate_output",
    "register_schema",
    "get_schema",
    "unregister_schema",
    "validate_reporter_output",
    "infer_schema",
    "merge_schemas",
    # Testing - Base
    "ReporterTestCase",
    # Testing - Mock data
    "create_mock_result",
    "create_mock_results",
    "create_mock_validator_result",
    "MockResultBuilder",
    "MockValidationResult",
    "MockValidatorResult",
    # Testing - Assertions
    "assert_valid_output",
    "assert_json_valid",
    "assert_xml_valid",
    "assert_csv_valid",
    "assert_contains_patterns",
    # Testing - Fixtures
    "create_sample_data",
    "create_edge_case_data",
    "create_stress_test_data",
    # Testing - Utilities
    "capture_output",
    "benchmark_reporter",
    "BenchmarkResult",
]
