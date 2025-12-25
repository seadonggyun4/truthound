"""Reporters module for generating validation reports.

This module provides a unified interface for generating reports in various
formats (JSON, HTML, Console, CI/CD platforms) from validation results.

Example:
    >>> from truthound.reporters import get_reporter
    >>>
    >>> # Use factory function
    >>> reporter = get_reporter("json", output_path="report.json")
    >>> reporter.report(validation_result)
    >>>
    >>> # CI reporter (auto-detect platform)
    >>> reporter = get_reporter("ci")
    >>> exit_code = reporter.report_to_ci(validation_result)
    >>>
    >>> # Specific CI platform
    >>> reporter = get_reporter("github")
    >>> reporter.report_to_ci(validation_result)

Supported CI Platforms:
    - GitHub Actions: "github"
    - GitLab CI: "gitlab"
    - Jenkins: "jenkins"
    - Azure DevOps: "azure"
    - CircleCI: "circleci"
    - Bitbucket Pipelines: "bitbucket"
    - Auto-detect: "ci"
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
