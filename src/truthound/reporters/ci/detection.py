"""CI Platform Detection.

This module provides utilities for automatically detecting the current
CI/CD platform based on environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from truthound.reporters.ci.base import CIPlatform


@dataclass
class CIEnvironment:
    """Information about the detected CI environment.

    Attributes:
        platform: Detected CI platform.
        is_ci: Whether running in a CI environment.
        is_pr: Whether this is a pull/merge request build.
        branch: Current branch name.
        commit: Current commit SHA.
        build_id: Build number or ID.
        build_url: URL to the build (if available).
        repository: Repository name/path.
        extra: Platform-specific extra information.
    """

    platform: CIPlatform
    is_ci: bool = False
    is_pr: bool = False
    branch: str | None = None
    commit: str | None = None
    build_id: str | None = None
    build_url: str | None = None
    repository: str | None = None
    extra: dict[str, str | None] | None = None


# Detection functions for each platform
# Each returns True if the platform is detected


def _is_github_actions() -> bool:
    """Check if running in GitHub Actions."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _is_gitlab_ci() -> bool:
    """Check if running in GitLab CI."""
    return os.environ.get("GITLAB_CI") == "true"


def _is_jenkins() -> bool:
    """Check if running in Jenkins."""
    return bool(os.environ.get("JENKINS_URL") or os.environ.get("BUILD_NUMBER"))


def _is_azure_devops() -> bool:
    """Check if running in Azure DevOps."""
    return os.environ.get("TF_BUILD") == "True"


def _is_circleci() -> bool:
    """Check if running in CircleCI."""
    return os.environ.get("CIRCLECI") == "true"


def _is_bitbucket() -> bool:
    """Check if running in Bitbucket Pipelines."""
    return bool(os.environ.get("BITBUCKET_BUILD_NUMBER"))


def _is_travis() -> bool:
    """Check if running in Travis CI."""
    return os.environ.get("TRAVIS") == "true"


def _is_drone() -> bool:
    """Check if running in Drone CI."""
    return os.environ.get("DRONE") == "true"


def _is_buildkite() -> bool:
    """Check if running in Buildkite."""
    return os.environ.get("BUILDKITE") == "true"


def _is_teamcity() -> bool:
    """Check if running in TeamCity."""
    return bool(os.environ.get("TEAMCITY_VERSION"))


def _is_codebuild() -> bool:
    """Check if running in AWS CodeBuild."""
    return bool(os.environ.get("CODEBUILD_BUILD_ID"))


def _is_generic_ci() -> bool:
    """Check if running in a generic CI environment."""
    ci_vars = ["CI", "CONTINUOUS_INTEGRATION", "BUILD_ID", "BUILD_NUMBER"]
    return any(os.environ.get(var) for var in ci_vars)


# Ordered list of detection checks (most specific first)
_PLATFORM_DETECTORS: list[tuple[CIPlatform, Callable[[], bool]]] = [
    (CIPlatform.GITHUB_ACTIONS, _is_github_actions),
    (CIPlatform.GITLAB_CI, _is_gitlab_ci),
    (CIPlatform.AZURE_DEVOPS, _is_azure_devops),
    (CIPlatform.CIRCLECI, _is_circleci),
    (CIPlatform.BITBUCKET, _is_bitbucket),
    (CIPlatform.JENKINS, _is_jenkins),
    # Generic CI check last
    (CIPlatform.GENERIC, _is_generic_ci),
]


def detect_ci_platform() -> CIPlatform | None:
    """Detect the current CI/CD platform.

    Checks environment variables to determine which CI platform
    is currently running.

    Returns:
        The detected CIPlatform, or None if not in a CI environment.

    Example:
        >>> platform = detect_ci_platform()
        >>> if platform:
        ...     print(f"Running on {platform}")
        ... else:
        ...     print("Not in CI")
    """
    for platform, detector in _PLATFORM_DETECTORS:
        if detector():
            return platform
    return None


def is_ci_environment() -> bool:
    """Check if running in any CI environment.

    Returns:
        True if running in a CI environment.
    """
    return detect_ci_platform() is not None


def get_ci_environment() -> CIEnvironment:
    """Get detailed information about the CI environment.

    Returns:
        CIEnvironment with platform and context information.

    Example:
        >>> env = get_ci_environment()
        >>> if env.is_ci:
        ...     print(f"Platform: {env.platform}")
        ...     print(f"Branch: {env.branch}")
        ...     print(f"Commit: {env.commit}")
    """
    platform = detect_ci_platform()

    if platform is None:
        return CIEnvironment(
            platform=CIPlatform.GENERIC,
            is_ci=False,
        )

    # Get platform-specific information
    if platform == CIPlatform.GITHUB_ACTIONS:
        return _get_github_environment()
    elif platform == CIPlatform.GITLAB_CI:
        return _get_gitlab_environment()
    elif platform == CIPlatform.JENKINS:
        return _get_jenkins_environment()
    elif platform == CIPlatform.AZURE_DEVOPS:
        return _get_azure_environment()
    elif platform == CIPlatform.CIRCLECI:
        return _get_circleci_environment()
    elif platform == CIPlatform.BITBUCKET:
        return _get_bitbucket_environment()
    else:
        return _get_generic_environment()


def _get_github_environment() -> CIEnvironment:
    """Get GitHub Actions environment information."""
    return CIEnvironment(
        platform=CIPlatform.GITHUB_ACTIONS,
        is_ci=True,
        is_pr=bool(os.environ.get("GITHUB_EVENT_NAME") == "pull_request"),
        branch=os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_HEAD_REF"),
        commit=os.environ.get("GITHUB_SHA"),
        build_id=os.environ.get("GITHUB_RUN_ID"),
        build_url=(
            f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}"
            f"/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
            if os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID")
            else None
        ),
        repository=os.environ.get("GITHUB_REPOSITORY"),
        extra={
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "job": os.environ.get("GITHUB_JOB"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "actor": os.environ.get("GITHUB_ACTOR"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        },
    )


def _get_gitlab_environment() -> CIEnvironment:
    """Get GitLab CI environment information."""
    return CIEnvironment(
        platform=CIPlatform.GITLAB_CI,
        is_ci=True,
        is_pr=bool(os.environ.get("CI_MERGE_REQUEST_IID")),
        branch=os.environ.get("CI_COMMIT_BRANCH") or os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"),
        commit=os.environ.get("CI_COMMIT_SHA"),
        build_id=os.environ.get("CI_PIPELINE_ID"),
        build_url=os.environ.get("CI_PIPELINE_URL"),
        repository=os.environ.get("CI_PROJECT_PATH"),
        extra={
            "job_id": os.environ.get("CI_JOB_ID"),
            "job_name": os.environ.get("CI_JOB_NAME"),
            "project_id": os.environ.get("CI_PROJECT_ID"),
            "merge_request_iid": os.environ.get("CI_MERGE_REQUEST_IID"),
        },
    )


def _get_jenkins_environment() -> CIEnvironment:
    """Get Jenkins environment information."""
    return CIEnvironment(
        platform=CIPlatform.JENKINS,
        is_ci=True,
        is_pr=bool(os.environ.get("CHANGE_ID")),
        branch=os.environ.get("BRANCH_NAME") or os.environ.get("GIT_BRANCH"),
        commit=os.environ.get("GIT_COMMIT"),
        build_id=os.environ.get("BUILD_NUMBER"),
        build_url=os.environ.get("BUILD_URL"),
        repository=os.environ.get("JOB_NAME"),
        extra={
            "job_name": os.environ.get("JOB_NAME"),
            "node_name": os.environ.get("NODE_NAME"),
            "workspace": os.environ.get("WORKSPACE"),
            "change_id": os.environ.get("CHANGE_ID"),
        },
    )


def _get_azure_environment() -> CIEnvironment:
    """Get Azure DevOps environment information."""
    collection_uri = os.environ.get("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI", "")
    project = os.environ.get("SYSTEM_TEAMPROJECT", "")
    build_id = os.environ.get("BUILD_BUILDID", "")

    return CIEnvironment(
        platform=CIPlatform.AZURE_DEVOPS,
        is_ci=True,
        is_pr=bool(os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID")),
        branch=os.environ.get("BUILD_SOURCEBRANCH"),
        commit=os.environ.get("BUILD_SOURCEVERSION"),
        build_id=build_id,
        build_url=f"{collection_uri}{project}/_build/results?buildId={build_id}" if all([collection_uri, project, build_id]) else None,
        repository=os.environ.get("BUILD_REPOSITORY_NAME"),
        extra={
            "project": project,
            "definition_name": os.environ.get("BUILD_DEFINITIONNAME"),
            "agent_name": os.environ.get("AGENT_NAME"),
            "pr_id": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID"),
        },
    )


def _get_circleci_environment() -> CIEnvironment:
    """Get CircleCI environment information."""
    return CIEnvironment(
        platform=CIPlatform.CIRCLECI,
        is_ci=True,
        is_pr=bool(os.environ.get("CIRCLE_PR_NUMBER")),
        branch=os.environ.get("CIRCLE_BRANCH"),
        commit=os.environ.get("CIRCLE_SHA1"),
        build_id=os.environ.get("CIRCLE_BUILD_NUM"),
        build_url=os.environ.get("CIRCLE_BUILD_URL"),
        repository=f"{os.environ.get('CIRCLE_PROJECT_USERNAME')}/{os.environ.get('CIRCLE_PROJECT_REPONAME')}",
        extra={
            "job": os.environ.get("CIRCLE_JOB"),
            "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
            "node_index": os.environ.get("CIRCLE_NODE_INDEX"),
            "node_total": os.environ.get("CIRCLE_NODE_TOTAL"),
            "pr_number": os.environ.get("CIRCLE_PR_NUMBER"),
        },
    )


def _get_bitbucket_environment() -> CIEnvironment:
    """Get Bitbucket Pipelines environment information."""
    workspace = os.environ.get("BITBUCKET_WORKSPACE", "")
    repo_slug = os.environ.get("BITBUCKET_REPO_SLUG", "")
    build_number = os.environ.get("BITBUCKET_BUILD_NUMBER", "")

    return CIEnvironment(
        platform=CIPlatform.BITBUCKET,
        is_ci=True,
        is_pr=bool(os.environ.get("BITBUCKET_PR_ID")),
        branch=os.environ.get("BITBUCKET_BRANCH"),
        commit=os.environ.get("BITBUCKET_COMMIT"),
        build_id=build_number,
        build_url=f"https://bitbucket.org/{workspace}/{repo_slug}/pipelines/results/{build_number}" if all([workspace, repo_slug, build_number]) else None,
        repository=f"{workspace}/{repo_slug}",
        extra={
            "pr_id": os.environ.get("BITBUCKET_PR_ID"),
            "pipeline_uuid": os.environ.get("BITBUCKET_PIPELINE_UUID"),
            "step_uuid": os.environ.get("BITBUCKET_STEP_UUID"),
            "deployment_environment": os.environ.get("BITBUCKET_DEPLOYMENT_ENVIRONMENT"),
        },
    )


def _get_generic_environment() -> CIEnvironment:
    """Get generic CI environment information."""
    return CIEnvironment(
        platform=CIPlatform.GENERIC,
        is_ci=True,
        is_pr=False,
        branch=os.environ.get("BRANCH") or os.environ.get("GIT_BRANCH"),
        commit=os.environ.get("COMMIT") or os.environ.get("GIT_COMMIT"),
        build_id=os.environ.get("BUILD_ID") or os.environ.get("BUILD_NUMBER"),
        build_url=os.environ.get("BUILD_URL"),
        repository=os.environ.get("REPOSITORY") or os.environ.get("REPO_NAME"),
    )


def get_recommended_reporter(platform: CIPlatform | None = None) -> str:
    """Get the recommended reporter name for a platform.

    Args:
        platform: The CI platform, or None to auto-detect.

    Returns:
        Reporter name suitable for get_ci_reporter().
    """
    if platform is None:
        platform = detect_ci_platform()

    if platform is None:
        return "console"

    platform_to_reporter = {
        CIPlatform.GITHUB_ACTIONS: "github",
        CIPlatform.GITLAB_CI: "gitlab",
        CIPlatform.JENKINS: "jenkins",
        CIPlatform.AZURE_DEVOPS: "azure",
        CIPlatform.CIRCLECI: "circleci",
        CIPlatform.BITBUCKET: "bitbucket",
        CIPlatform.GENERIC: "console",
    }

    return platform_to_reporter.get(platform, "console")
