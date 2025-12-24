"""Reporters module for generating validation reports.

This module provides a unified interface for generating reports in various
formats (JSON, HTML, Console) from validation results.

Example:
    >>> from truthound.reporters import get_reporter, JSONReporter
    >>>
    >>> # Use factory function
    >>> reporter = get_reporter("json", output_path="report.json")
    >>> reporter.report(validation_result)
    >>>
    >>> # Or use directly
    >>> reporter = JSONReporter()
    >>> json_str = reporter.render(validation_result)
"""

from truthound.reporters.base import (
    BaseReporter,
    ReporterConfig,
    ReporterError,
)
from truthound.reporters.factory import get_reporter, register_reporter

__all__ = [
    # Base classes
    "BaseReporter",
    "ReporterConfig",
    "ReporterError",
    # Factory functions
    "get_reporter",
    "register_reporter",
]
