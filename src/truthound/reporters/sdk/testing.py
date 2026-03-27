"""Testing utilities facade for reporter SDK development.

This module keeps the long-standing public import surface stable while the
implementation is split across private helper modules:

- `_testing_models`: mock result models, builders, and sample data
- `_testing_assertions`: reusable assertions and `ReporterTestCase`
- `_testing_runtime`: capture and benchmark helpers
"""

from __future__ import annotations

from truthound.reporters.sdk._testing_assertions import (
    ReporterTestCase,
    assert_contains_patterns,
    assert_csv_valid,
    assert_json_valid,
    assert_valid_output,
    assert_xml_valid,
)
from truthound.reporters.sdk._testing_models import (
    MockResultBuilder,
    MockValidationResult,
    MockValidatorResult,
    Severity,
    create_edge_case_data,
    create_mock_result,
    create_mock_results,
    create_mock_validator_result,
    create_sample_data,
    create_stress_test_data,
)
from truthound.reporters.sdk._testing_runtime import (
    BenchmarkResult,
    CapturedOutput,
    benchmark_reporter,
    capture_output,
)

__all__ = [
    "ReporterTestCase",
    "Severity",
    "create_mock_result",
    "create_mock_results",
    "create_mock_validator_result",
    "MockResultBuilder",
    "MockValidationResult",
    "MockValidatorResult",
    "assert_valid_output",
    "assert_json_valid",
    "assert_xml_valid",
    "assert_csv_valid",
    "assert_contains_patterns",
    "create_sample_data",
    "create_edge_case_data",
    "create_stress_test_data",
    "CapturedOutput",
    "capture_output",
    "BenchmarkResult",
    "benchmark_reporter",
]
