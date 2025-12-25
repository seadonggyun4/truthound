"""CI/CD integration utilities for Truthound.

This module provides utilities for integrating Truthound checkpoints
with various CI/CD platforms including GitHub Actions, GitLab CI,
Jenkins, CircleCI, and more.

Example:
    >>> from truthound.checkpoint.ci import CIEnvironment, get_ci_environment
    >>>
    >>> # Auto-detect CI environment
    >>> ci = get_ci_environment()
    >>> print(f"Running on: {ci.platform}")
    >>>
    >>> # Report results
    >>> ci.report_status(checkpoint_result)
"""

from truthound.checkpoint.ci.detector import (
    CIPlatform,
    CIEnvironment,
    get_ci_environment,
    detect_ci_platform,
    is_ci_environment,
)
from truthound.checkpoint.ci.reporter import (
    CIReporter,
    GitHubActionsReporter,
    GitLabCIReporter,
    JenkinsCIReporter,
    CircleCIReporter,
    GenericCIReporter,
    get_ci_reporter,
)
from truthound.checkpoint.ci.templates import (
    generate_github_workflow,
    generate_gitlab_ci,
    generate_jenkinsfile,
    generate_circleci_config,
)

__all__ = [
    # Detection
    "CIPlatform",
    "CIEnvironment",
    "get_ci_environment",
    "detect_ci_platform",
    "is_ci_environment",
    # Reporters
    "CIReporter",
    "GitHubActionsReporter",
    "GitLabCIReporter",
    "JenkinsCIReporter",
    "CircleCIReporter",
    "GenericCIReporter",
    "get_ci_reporter",
    # Templates
    "generate_github_workflow",
    "generate_gitlab_ci",
    "generate_jenkinsfile",
    "generate_circleci_config",
]
