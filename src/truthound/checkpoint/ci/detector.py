"""CI/CD platform detection.

This module provides utilities for detecting the current CI/CD platform
and extracting relevant environment information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CIPlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET_PIPELINES = "bitbucket_pipelines"
    TEAMCITY = "teamcity"
    BUILDKITE = "buildkite"
    DRONE = "drone"
    AWS_CODEBUILD = "aws_codebuild"
    GOOGLE_CLOUD_BUILD = "google_cloud_build"
    LOCAL = "local"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass
class CIEnvironment:
    """Information about the current CI environment.

    Attributes:
        platform: The detected CI platform.
        is_ci: Whether running in a CI environment.
        is_pr: Whether this is a pull/merge request build.
        branch: Current branch name.
        commit_sha: Current commit SHA.
        commit_message: Commit message.
        pr_number: Pull request number (if applicable).
        pr_target_branch: Target branch for PR.
        repository: Repository name (owner/repo format).
        run_id: CI run/build ID.
        run_url: URL to the CI run.
        actor: User/bot that triggered the build.
        job_name: Name of the current job.
        workflow_name: Name of the workflow/pipeline.
        environment_vars: Relevant environment variables.
    """

    platform: CIPlatform = CIPlatform.LOCAL
    is_ci: bool = False
    is_pr: bool = False
    branch: str = ""
    commit_sha: str = ""
    commit_message: str = ""
    pr_number: int | None = None
    pr_target_branch: str = ""
    repository: str = ""
    run_id: str = ""
    run_url: str = ""
    actor: str = ""
    job_name: str = ""
    workflow_name: str = ""
    environment_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform.value,
            "is_ci": self.is_ci,
            "is_pr": self.is_pr,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "commit_message": self.commit_message,
            "pr_number": self.pr_number,
            "pr_target_branch": self.pr_target_branch,
            "repository": self.repository,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "actor": self.actor,
            "job_name": self.job_name,
            "workflow_name": self.workflow_name,
        }


def detect_ci_platform() -> CIPlatform:
    """Detect the current CI platform.

    Returns:
        The detected CI platform.
    """
    # GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return CIPlatform.GITHUB_ACTIONS

    # GitLab CI
    if os.environ.get("GITLAB_CI") == "true":
        return CIPlatform.GITLAB_CI

    # Jenkins
    if os.environ.get("JENKINS_URL") or os.environ.get("BUILD_ID"):
        return CIPlatform.JENKINS

    # CircleCI
    if os.environ.get("CIRCLECI") == "true":
        return CIPlatform.CIRCLECI

    # Travis CI
    if os.environ.get("TRAVIS") == "true":
        return CIPlatform.TRAVIS_CI

    # Azure DevOps
    if os.environ.get("TF_BUILD") == "True":
        return CIPlatform.AZURE_DEVOPS

    # Bitbucket Pipelines
    if os.environ.get("BITBUCKET_BUILD_NUMBER"):
        return CIPlatform.BITBUCKET_PIPELINES

    # TeamCity
    if os.environ.get("TEAMCITY_VERSION"):
        return CIPlatform.TEAMCITY

    # Buildkite
    if os.environ.get("BUILDKITE") == "true":
        return CIPlatform.BUILDKITE

    # Drone
    if os.environ.get("DRONE") == "true":
        return CIPlatform.DRONE

    # AWS CodeBuild
    if os.environ.get("CODEBUILD_BUILD_ID"):
        return CIPlatform.AWS_CODEBUILD

    # Google Cloud Build
    if os.environ.get("BUILDER_OUTPUT"):
        return CIPlatform.GOOGLE_CLOUD_BUILD

    # Generic CI detection
    if os.environ.get("CI") == "true" or os.environ.get("CONTINUOUS_INTEGRATION"):
        return CIPlatform.UNKNOWN

    return CIPlatform.LOCAL


def is_ci_environment() -> bool:
    """Check if running in a CI environment.

    Returns:
        True if in CI environment.
    """
    return detect_ci_platform() != CIPlatform.LOCAL


def get_ci_environment() -> CIEnvironment:
    """Get information about the current CI environment.

    Returns:
        CIEnvironment with detected information.
    """
    platform = detect_ci_platform()

    if platform == CIPlatform.GITHUB_ACTIONS:
        return _get_github_environment()
    elif platform == CIPlatform.GITLAB_CI:
        return _get_gitlab_environment()
    elif platform == CIPlatform.JENKINS:
        return _get_jenkins_environment()
    elif platform == CIPlatform.CIRCLECI:
        return _get_circleci_environment()
    elif platform == CIPlatform.TRAVIS_CI:
        return _get_travis_environment()
    elif platform == CIPlatform.AZURE_DEVOPS:
        return _get_azure_environment()
    elif platform == CIPlatform.BITBUCKET_PIPELINES:
        return _get_bitbucket_environment()
    elif platform == CIPlatform.BUILDKITE:
        return _get_buildkite_environment()
    else:
        return CIEnvironment(
            platform=platform,
            is_ci=platform != CIPlatform.LOCAL,
        )


def _get_github_environment() -> CIEnvironment:
    """Get GitHub Actions environment info."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    is_pr = event_name in ("pull_request", "pull_request_target")

    # Get PR number from ref
    pr_number = None
    github_ref = os.environ.get("GITHUB_REF", "")
    if is_pr and "/pull/" in github_ref:
        try:
            pr_number = int(github_ref.split("/pull/")[1].split("/")[0])
        except (ValueError, IndexError):
            pass

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")

    return CIEnvironment(
        platform=CIPlatform.GITHUB_ACTIONS,
        is_ci=True,
        is_pr=is_pr,
        branch=os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME", ""),
        commit_sha=os.environ.get("GITHUB_SHA", ""),
        pr_number=pr_number,
        pr_target_branch=os.environ.get("GITHUB_BASE_REF", ""),
        repository=repository,
        run_id=run_id,
        run_url=f"{server_url}/{repository}/actions/runs/{run_id}" if repository and run_id else "",
        actor=os.environ.get("GITHUB_ACTOR", ""),
        job_name=os.environ.get("GITHUB_JOB", ""),
        workflow_name=os.environ.get("GITHUB_WORKFLOW", ""),
    )


def _get_gitlab_environment() -> CIEnvironment:
    """Get GitLab CI environment info."""
    return CIEnvironment(
        platform=CIPlatform.GITLAB_CI,
        is_ci=True,
        is_pr=bool(os.environ.get("CI_MERGE_REQUEST_ID")),
        branch=os.environ.get("CI_COMMIT_REF_NAME", ""),
        commit_sha=os.environ.get("CI_COMMIT_SHA", ""),
        commit_message=os.environ.get("CI_COMMIT_MESSAGE", ""),
        pr_number=int(os.environ.get("CI_MERGE_REQUEST_IID", 0)) or None,
        pr_target_branch=os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", ""),
        repository=os.environ.get("CI_PROJECT_PATH", ""),
        run_id=os.environ.get("CI_PIPELINE_ID", ""),
        run_url=os.environ.get("CI_PIPELINE_URL", ""),
        actor=os.environ.get("GITLAB_USER_LOGIN", ""),
        job_name=os.environ.get("CI_JOB_NAME", ""),
        workflow_name=os.environ.get("CI_PIPELINE_NAME", ""),
    )


def _get_jenkins_environment() -> CIEnvironment:
    """Get Jenkins environment info."""
    return CIEnvironment(
        platform=CIPlatform.JENKINS,
        is_ci=True,
        is_pr=bool(os.environ.get("CHANGE_ID")),
        branch=os.environ.get("BRANCH_NAME") or os.environ.get("GIT_BRANCH", ""),
        commit_sha=os.environ.get("GIT_COMMIT", ""),
        pr_number=int(os.environ.get("CHANGE_ID", 0)) or None,
        pr_target_branch=os.environ.get("CHANGE_TARGET", ""),
        repository=os.environ.get("GIT_URL", "").replace(".git", "").split("/")[-2:],
        run_id=os.environ.get("BUILD_ID", ""),
        run_url=os.environ.get("BUILD_URL", ""),
        actor=os.environ.get("BUILD_USER", ""),
        job_name=os.environ.get("JOB_NAME", ""),
    )


def _get_circleci_environment() -> CIEnvironment:
    """Get CircleCI environment info."""
    return CIEnvironment(
        platform=CIPlatform.CIRCLECI,
        is_ci=True,
        is_pr=bool(os.environ.get("CIRCLE_PULL_REQUEST")),
        branch=os.environ.get("CIRCLE_BRANCH", ""),
        commit_sha=os.environ.get("CIRCLE_SHA1", ""),
        pr_number=int(os.environ.get("CIRCLE_PR_NUMBER", 0)) or None,
        repository=f"{os.environ.get('CIRCLE_PROJECT_USERNAME', '')}/{os.environ.get('CIRCLE_PROJECT_REPONAME', '')}",
        run_id=os.environ.get("CIRCLE_BUILD_NUM", ""),
        run_url=os.environ.get("CIRCLE_BUILD_URL", ""),
        actor=os.environ.get("CIRCLE_USERNAME", ""),
        job_name=os.environ.get("CIRCLE_JOB", ""),
        workflow_name=os.environ.get("CIRCLE_WORKFLOW_ID", ""),
    )


def _get_travis_environment() -> CIEnvironment:
    """Get Travis CI environment info."""
    return CIEnvironment(
        platform=CIPlatform.TRAVIS_CI,
        is_ci=True,
        is_pr=os.environ.get("TRAVIS_PULL_REQUEST", "false") != "false",
        branch=os.environ.get("TRAVIS_BRANCH", ""),
        commit_sha=os.environ.get("TRAVIS_COMMIT", ""),
        commit_message=os.environ.get("TRAVIS_COMMIT_MESSAGE", ""),
        pr_number=int(os.environ.get("TRAVIS_PULL_REQUEST", 0)) or None,
        repository=os.environ.get("TRAVIS_REPO_SLUG", ""),
        run_id=os.environ.get("TRAVIS_BUILD_ID", ""),
        run_url=os.environ.get("TRAVIS_BUILD_WEB_URL", ""),
    )


def _get_azure_environment() -> CIEnvironment:
    """Get Azure DevOps environment info."""
    return CIEnvironment(
        platform=CIPlatform.AZURE_DEVOPS,
        is_ci=True,
        is_pr=os.environ.get("BUILD_REASON") == "PullRequest",
        branch=os.environ.get("BUILD_SOURCEBRANCHNAME", ""),
        commit_sha=os.environ.get("BUILD_SOURCEVERSION", ""),
        commit_message=os.environ.get("BUILD_SOURCEVERSIONMESSAGE", ""),
        pr_number=int(os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", 0)) or None,
        pr_target_branch=os.environ.get("SYSTEM_PULLREQUEST_TARGETBRANCH", ""),
        repository=os.environ.get("BUILD_REPOSITORY_NAME", ""),
        run_id=os.environ.get("BUILD_BUILDID", ""),
        run_url=f"{os.environ.get('SYSTEM_TEAMFOUNDATIONSERVERURI', '')}{os.environ.get('SYSTEM_TEAMPROJECT', '')}/_build/results?buildId={os.environ.get('BUILD_BUILDID', '')}",
        actor=os.environ.get("BUILD_REQUESTEDFOR", ""),
        job_name=os.environ.get("AGENT_JOBNAME", ""),
        workflow_name=os.environ.get("BUILD_DEFINITIONNAME", ""),
    )


def _get_bitbucket_environment() -> CIEnvironment:
    """Get Bitbucket Pipelines environment info."""
    return CIEnvironment(
        platform=CIPlatform.BITBUCKET_PIPELINES,
        is_ci=True,
        is_pr=bool(os.environ.get("BITBUCKET_PR_ID")),
        branch=os.environ.get("BITBUCKET_BRANCH", ""),
        commit_sha=os.environ.get("BITBUCKET_COMMIT", ""),
        pr_number=int(os.environ.get("BITBUCKET_PR_ID", 0)) or None,
        pr_target_branch=os.environ.get("BITBUCKET_PR_DESTINATION_BRANCH", ""),
        repository=os.environ.get("BITBUCKET_REPO_FULL_NAME", ""),
        run_id=os.environ.get("BITBUCKET_BUILD_NUMBER", ""),
    )


def _get_buildkite_environment() -> CIEnvironment:
    """Get Buildkite environment info."""
    return CIEnvironment(
        platform=CIPlatform.BUILDKITE,
        is_ci=True,
        is_pr=os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false",
        branch=os.environ.get("BUILDKITE_BRANCH", ""),
        commit_sha=os.environ.get("BUILDKITE_COMMIT", ""),
        commit_message=os.environ.get("BUILDKITE_MESSAGE", ""),
        pr_number=int(os.environ.get("BUILDKITE_PULL_REQUEST", 0)) or None,
        pr_target_branch=os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", ""),
        repository=os.environ.get("BUILDKITE_REPO", ""),
        run_id=os.environ.get("BUILDKITE_BUILD_ID", ""),
        run_url=os.environ.get("BUILDKITE_BUILD_URL", ""),
        actor=os.environ.get("BUILDKITE_BUILD_CREATOR", ""),
        job_name=os.environ.get("BUILDKITE_LABEL", ""),
        workflow_name=os.environ.get("BUILDKITE_PIPELINE_NAME", ""),
    )
