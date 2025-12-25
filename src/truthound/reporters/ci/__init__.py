"""CI/CD Pipeline Reporters.

This module provides reporters specialized for various CI/CD platforms.
Each reporter outputs validation results in a format native to its target platform.

Supported Platforms:
    - GitHub Actions: Annotations and job summaries
    - GitLab CI: Code quality reports and artifacts
    - Jenkins: JUnit XML format
    - Azure DevOps: VSO logging commands
    - CircleCI: Test metadata format
    - Bitbucket Pipelines: Reports and annotations

Example:
    >>> from truthound.reporters.ci import get_ci_reporter, detect_ci_platform
    >>>
    >>> # Auto-detect CI platform
    >>> platform = detect_ci_platform()
    >>> reporter = get_ci_reporter(platform)
    >>> reporter.report(validation_result)
    >>>
    >>> # Or use specific reporter
    >>> from truthound.reporters.ci import GitHubActionsReporter
    >>> reporter = GitHubActionsReporter()
    >>> reporter.report(validation_result)
"""

from truthound.reporters.ci.base import (
    BaseCIReporter,
    CIReporterConfig,
    CIPlatform,
    CIAnnotation,
    AnnotationLevel,
)
from truthound.reporters.ci.detection import detect_ci_platform, get_ci_environment
from truthound.reporters.ci.factory import get_ci_reporter, register_ci_reporter

# Import individual reporters for direct use
from truthound.reporters.ci.github import GitHubActionsReporter
from truthound.reporters.ci.gitlab import GitLabCIReporter
from truthound.reporters.ci.jenkins import JenkinsReporter
from truthound.reporters.ci.azure import AzureDevOpsReporter
from truthound.reporters.ci.circleci import CircleCIReporter
from truthound.reporters.ci.bitbucket import BitbucketPipelinesReporter

__all__ = [
    # Base classes
    "BaseCIReporter",
    "CIReporterConfig",
    "CIPlatform",
    "CIAnnotation",
    "AnnotationLevel",
    # Detection
    "detect_ci_platform",
    "get_ci_environment",
    # Factory
    "get_ci_reporter",
    "register_ci_reporter",
    # Reporters
    "GitHubActionsReporter",
    "GitLabCIReporter",
    "JenkinsReporter",
    "AzureDevOpsReporter",
    "CircleCIReporter",
    "BitbucketPipelinesReporter",
]
