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

from truthound.reporters.sdk.builder import (
    ReporterBuilder,
    create_reporter,
    create_validation_reporter,
)
from truthound.reporters.sdk.compat import (
    build_sdk_legacy_view,
    build_sdk_presentation,
    to_validation_run_result,
)
from truthound.reporters.sdk.mixins import (
    AggregationMixin,
    FilteringMixin,
    FormattingMixin,
    SerializationMixin,
    StreamingMixin,
    TemplatingMixin,
)
from truthound.reporters.sdk.schema import (
    CSVSchema,
    JSONSchema,
    # Core classes
    ReportSchema,
    SchemaError,
    # Validation
    SchemaValidationOutcome,
    TextSchema,
    ValidationError,
    ValidationResult,
    XMLSchema,
    get_schema,
    infer_schema,
    merge_schemas,
    register_schema,
    unregister_schema,
    # Functions
    validate_output,
    validate_reporter_output,
)
from truthound.reporters.sdk.templates import (
    CSVReporter,
    JUnitXMLReporter,
    NDJSONReporter,
    TableReporter,
    YAMLReporter,
)
from truthound.reporters.sdk.testing import (
    BenchmarkResult,
    MockResultBuilder,
    MockValidationResult,
    MockValidatorResult,
    # Test case base
    ReporterTestCase,
    assert_contains_patterns,
    assert_csv_valid,
    assert_json_valid,
    # Assertions
    assert_valid_output,
    assert_xml_valid,
    benchmark_reporter,
    # Utilities
    capture_output,
    create_edge_case_data,
    # Mock data generators
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    # Fixtures
    create_sample_data,
    create_stress_test_data,
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
    "to_validation_run_result",
    "build_sdk_presentation",
    "build_sdk_legacy_view",
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
    "SchemaValidationOutcome",
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
