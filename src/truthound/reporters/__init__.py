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

Quality Reporters:
    Quality reporters provide specialized reporting for rule quality scores.
    See truthound.reporters.quality for the full quality reporting system.

    >>> from truthound.reporters.quality import get_quality_reporter
    >>>
    >>> reporter = get_quality_reporter("html", include_charts=True)
    >>> report = reporter.render(quality_scores)
"""

from truthound.reporters.base import (
    BaseReporter,
    ReporterConfig,
    ReporterError,
)
from truthound.reporters.factory import get_reporter, register_reporter

# Quality reporter lazy imports (avoid circular imports)
def __getattr__(name: str):
    """Lazy load quality reporter components."""
    if name in (
        "get_quality_reporter",
        "QualityReporterConfig",
        "QualityFilter",
        "QualityReportEngine",
    ):
        from truthound.reporters import quality
        return getattr(quality, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "BaseReporter",
    "ReporterConfig",
    "ReporterError",
    # Factory functions
    "get_reporter",
    "register_reporter",
    # Quality reporters (lazy loaded)
    "get_quality_reporter",
    "QualityReporterConfig",
    "QualityFilter",
    "QualityReportEngine",
]
