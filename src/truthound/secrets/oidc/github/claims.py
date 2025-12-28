"""GitHub Actions OIDC Claims Parsing and Validation.

This module provides comprehensive parsing and validation of GitHub Actions
OIDC token claims, including all standard and GitHub-specific claims.

GitHub Actions OIDC Token Claims Reference:
https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect

Standard Claims:
    - iss: Token issuer (https://token.actions.githubusercontent.com)
    - sub: Subject identifier (repo:owner/repo:ref:refs/heads/main)
    - aud: Token audience (configured by user)
    - exp: Expiration time
    - iat: Issued at time
    - nbf: Not before time
    - jti: JWT ID (unique identifier)

GitHub-Specific Claims:
    - repository: Full repository name (owner/repo)
    - repository_owner: Repository owner (user or org)
    - repository_owner_id: Owner's numeric ID
    - repository_id: Repository's numeric ID
    - repository_visibility: public, private, or internal
    - actor: User who triggered the workflow
    - actor_id: Actor's numeric ID
    - workflow: Workflow name
    - workflow_ref: Full workflow reference
    - workflow_sha: Workflow file's commit SHA
    - head_ref: Source branch for pull requests
    - base_ref: Target branch for pull requests
    - event_name: Triggering event (push, pull_request, etc.)
    - ref: Git ref (refs/heads/main, refs/tags/v1.0)
    - ref_type: branch or tag
    - sha: Commit SHA
    - run_id: Workflow run ID
    - run_number: Workflow run number
    - run_attempt: Workflow run attempt
    - job_workflow_ref: Reusable workflow reference
    - job_workflow_sha: Reusable workflow SHA
    - runner_environment: github-hosted or self-hosted
    - environment: Deployment environment name
    - environment_node_id: Environment's node ID
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """GitHub Actions event types."""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_TARGET = "pull_request_target"
    WORKFLOW_DISPATCH = "workflow_dispatch"
    WORKFLOW_CALL = "workflow_call"
    SCHEDULE = "schedule"
    RELEASE = "release"
    DEPLOYMENT = "deployment"
    DEPLOYMENT_STATUS = "deployment_status"
    CREATE = "create"
    DELETE = "delete"
    FORK = "fork"
    ISSUE_COMMENT = "issue_comment"
    ISSUES = "issues"
    MERGE_GROUP = "merge_group"
    PAGE_BUILD = "page_build"
    REGISTRY_PACKAGE = "registry_package"
    REPOSITORY_DISPATCH = "repository_dispatch"
    STATUS = "status"
    WATCH = "watch"
    UNKNOWN = "unknown"


class RefType(str, Enum):
    """Git reference types."""

    BRANCH = "branch"
    TAG = "tag"
    UNKNOWN = "unknown"


class RepositoryVisibility(str, Enum):
    """Repository visibility levels."""

    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class RunnerEnvironment(str, Enum):
    """Runner environment types."""

    GITHUB_HOSTED = "github-hosted"
    SELF_HOSTED = "self-hosted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GitHubActionsClaims:
    """Parsed GitHub Actions OIDC token claims.

    This dataclass provides a strongly-typed representation of all claims
    available in a GitHub Actions OIDC token.

    Attributes:
        # Standard OIDC Claims
        issuer: Token issuer URL.
        subject: Subject identifier.
        audience: Token audience(s).
        expiration: Token expiration time.
        issued_at: Token issue time.
        not_before: Token not-before time.
        jwt_id: Unique JWT identifier.

        # Repository Claims
        repository: Full repository name (owner/repo).
        repository_owner: Repository owner.
        repository_owner_id: Owner's numeric ID.
        repository_id: Repository's numeric ID.
        repository_visibility: Repository visibility.

        # Actor Claims
        actor: User who triggered the workflow.
        actor_id: Actor's numeric ID.

        # Workflow Claims
        workflow: Workflow name.
        workflow_ref: Full workflow reference.
        workflow_sha: Workflow file's commit SHA.
        job_workflow_ref: Reusable workflow reference.
        job_workflow_sha: Reusable workflow SHA.

        # Git Reference Claims
        ref: Git reference.
        ref_type: Reference type (branch/tag).
        sha: Commit SHA.
        head_ref: Source branch for PRs.
        base_ref: Target branch for PRs.

        # Run Claims
        run_id: Workflow run ID.
        run_number: Workflow run number.
        run_attempt: Workflow run attempt.
        event_name: Triggering event type.

        # Environment Claims
        environment: Deployment environment name.
        environment_node_id: Environment's node ID.
        runner_environment: Runner type.

        # Extra Claims
        extra: Additional claims not covered above.
    """

    # Standard OIDC Claims
    issuer: str
    subject: str
    audience: str | list[str]
    expiration: datetime
    issued_at: datetime
    not_before: datetime | None = None
    jwt_id: str | None = None

    # Repository Claims
    repository: str = ""
    repository_owner: str = ""
    repository_owner_id: str = ""
    repository_id: str = ""
    repository_visibility: RepositoryVisibility = RepositoryVisibility.UNKNOWN

    # Actor Claims
    actor: str = ""
    actor_id: str = ""

    # Workflow Claims
    workflow: str = ""
    workflow_ref: str = ""
    workflow_sha: str = ""
    job_workflow_ref: str | None = None
    job_workflow_sha: str | None = None

    # Git Reference Claims
    ref: str = ""
    ref_type: RefType = RefType.UNKNOWN
    sha: str = ""
    head_ref: str | None = None
    base_ref: str | None = None

    # Run Claims
    run_id: str = ""
    run_number: str = ""
    run_attempt: str = ""
    event_name: EventType = EventType.UNKNOWN

    # Environment Claims
    environment: str | None = None
    environment_node_id: str | None = None
    runner_environment: RunnerEnvironment = RunnerEnvironment.UNKNOWN

    # Extra Claims
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now() >= self.expiration

    @property
    def is_pull_request(self) -> bool:
        """Check if triggered by a pull request."""
        return self.event_name in (
            EventType.PULL_REQUEST,
            EventType.PULL_REQUEST_TARGET,
        )

    @property
    def is_main_branch(self) -> bool:
        """Check if running on main/master branch."""
        return self.ref in (
            "refs/heads/main",
            "refs/heads/master",
        )

    @property
    def is_tag(self) -> bool:
        """Check if triggered by a tag."""
        return self.ref_type == RefType.TAG or self.ref.startswith("refs/tags/")

    @property
    def is_release(self) -> bool:
        """Check if triggered by a release event."""
        return self.event_name == EventType.RELEASE

    @property
    def is_scheduled(self) -> bool:
        """Check if triggered by schedule."""
        return self.event_name == EventType.SCHEDULE

    @property
    def is_workflow_dispatch(self) -> bool:
        """Check if manually triggered."""
        return self.event_name == EventType.WORKFLOW_DISPATCH

    @property
    def is_reusable_workflow(self) -> bool:
        """Check if running as a reusable workflow."""
        return self.job_workflow_ref is not None

    @property
    def has_environment(self) -> bool:
        """Check if running in a deployment environment."""
        return self.environment is not None

    @property
    def is_github_hosted(self) -> bool:
        """Check if running on GitHub-hosted runner."""
        return self.runner_environment == RunnerEnvironment.GITHUB_HOSTED

    @property
    def branch_name(self) -> str | None:
        """Extract branch name from ref."""
        if self.ref.startswith("refs/heads/"):
            return self.ref[len("refs/heads/"):]
        return None

    @property
    def tag_name(self) -> str | None:
        """Extract tag name from ref."""
        if self.ref.startswith("refs/tags/"):
            return self.ref[len("refs/tags/"):]
        return None

    @property
    def short_sha(self) -> str:
        """Get short commit SHA (7 characters)."""
        return self.sha[:7] if self.sha else ""

    def matches_repository(self, pattern: str) -> bool:
        """Check if repository matches a pattern.

        Args:
            pattern: Repository pattern (supports wildcards).
                Examples: "owner/repo", "owner/*", "*/repo"

        Returns:
            True if repository matches the pattern.
        """
        if not self.repository:
            return False

        # Convert pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", self.repository))

    def matches_ref(self, pattern: str) -> bool:
        """Check if ref matches a pattern.

        Args:
            pattern: Ref pattern (supports wildcards).
                Examples: "refs/heads/main", "refs/heads/*", "refs/tags/v*"

        Returns:
            True if ref matches the pattern.
        """
        if not self.ref:
            return False

        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", self.ref))

    def to_dict(self) -> dict[str, Any]:
        """Convert claims to dictionary."""
        return {
            "iss": self.issuer,
            "sub": self.subject,
            "aud": self.audience,
            "exp": int(self.expiration.timestamp()),
            "iat": int(self.issued_at.timestamp()),
            "nbf": int(self.not_before.timestamp()) if self.not_before else None,
            "jti": self.jwt_id,
            "repository": self.repository,
            "repository_owner": self.repository_owner,
            "repository_owner_id": self.repository_owner_id,
            "repository_id": self.repository_id,
            "repository_visibility": self.repository_visibility.value,
            "actor": self.actor,
            "actor_id": self.actor_id,
            "workflow": self.workflow,
            "workflow_ref": self.workflow_ref,
            "workflow_sha": self.workflow_sha,
            "job_workflow_ref": self.job_workflow_ref,
            "job_workflow_sha": self.job_workflow_sha,
            "ref": self.ref,
            "ref_type": self.ref_type.value,
            "sha": self.sha,
            "head_ref": self.head_ref,
            "base_ref": self.base_ref,
            "run_id": self.run_id,
            "run_number": self.run_number,
            "run_attempt": self.run_attempt,
            "event_name": self.event_name.value,
            "environment": self.environment,
            "environment_node_id": self.environment_node_id,
            "runner_environment": self.runner_environment.value,
            **self.extra,
        }


@dataclass
class GitHubActionsContext:
    """GitHub Actions workflow context from environment variables.

    This provides access to GitHub Actions context information that
    is available through environment variables rather than the OIDC token.

    Attributes:
        github_token: GITHUB_TOKEN for API access.
        github_api_url: GitHub API URL.
        github_graphql_url: GitHub GraphQL API URL.
        github_server_url: GitHub server URL.
        github_workspace: Workspace directory path.
        github_output: Output file path.
        github_env: Environment file path.
        github_step_summary: Step summary file path.
        github_path: Path file.
        github_action: Current action name.
        github_action_path: Action's directory.
        github_action_repository: Action's repository.
        github_action_ref: Action's ref.
        github_retention_days: Artifact retention days.
        runner_name: Runner name.
        runner_os: Runner OS (Linux, Windows, macOS).
        runner_arch: Runner architecture.
        runner_temp: Temp directory path.
        runner_tool_cache: Tool cache path.
    """

    # GitHub Environment
    github_token: str = ""
    github_api_url: str = ""
    github_graphql_url: str = ""
    github_server_url: str = ""
    github_workspace: str = ""
    github_output: str = ""
    github_env: str = ""
    github_step_summary: str = ""
    github_path: str = ""

    # Action Context
    github_action: str = ""
    github_action_path: str = ""
    github_action_repository: str = ""
    github_action_ref: str = ""
    github_retention_days: int = 0

    # Runner Context
    runner_name: str = ""
    runner_os: str = ""
    runner_arch: str = ""
    runner_temp: str = ""
    runner_tool_cache: str = ""

    @classmethod
    def from_environment(cls) -> "GitHubActionsContext":
        """Create context from current environment variables."""
        import os

        retention_days = os.environ.get("GITHUB_RETENTION_DAYS", "0")
        try:
            retention_days_int = int(retention_days)
        except ValueError:
            retention_days_int = 0

        return cls(
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            github_api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            github_graphql_url=os.environ.get(
                "GITHUB_GRAPHQL_URL", "https://api.github.com/graphql"
            ),
            github_server_url=os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
            github_workspace=os.environ.get("GITHUB_WORKSPACE", ""),
            github_output=os.environ.get("GITHUB_OUTPUT", ""),
            github_env=os.environ.get("GITHUB_ENV", ""),
            github_step_summary=os.environ.get("GITHUB_STEP_SUMMARY", ""),
            github_path=os.environ.get("GITHUB_PATH", ""),
            github_action=os.environ.get("GITHUB_ACTION", ""),
            github_action_path=os.environ.get("GITHUB_ACTION_PATH", ""),
            github_action_repository=os.environ.get("GITHUB_ACTION_REPOSITORY", ""),
            github_action_ref=os.environ.get("GITHUB_ACTION_REF", ""),
            github_retention_days=retention_days_int,
            runner_name=os.environ.get("RUNNER_NAME", ""),
            runner_os=os.environ.get("RUNNER_OS", ""),
            runner_arch=os.environ.get("RUNNER_ARCH", ""),
            runner_temp=os.environ.get("RUNNER_TEMP", ""),
            runner_tool_cache=os.environ.get("RUNNER_TOOL_CACHE", ""),
        )


def parse_github_claims(payload: dict[str, Any]) -> GitHubActionsClaims:
    """Parse GitHub Actions OIDC token claims.

    Args:
        payload: Decoded JWT payload.

    Returns:
        GitHubActionsClaims instance.
    """
    # Standard claims
    issuer = payload.get("iss", "")
    subject = payload.get("sub", "")
    audience = payload.get("aud", "")

    # Time claims
    exp = payload.get("exp", 0)
    iat = payload.get("iat", 0)
    nbf = payload.get("nbf")

    expiration = datetime.fromtimestamp(exp) if exp else datetime.now()
    issued_at = datetime.fromtimestamp(iat) if iat else datetime.now()
    not_before = datetime.fromtimestamp(nbf) if nbf else None

    # Parse repository visibility
    visibility_str = payload.get("repository_visibility", "unknown")
    try:
        visibility = RepositoryVisibility(visibility_str)
    except ValueError:
        visibility = RepositoryVisibility.UNKNOWN

    # Parse ref type
    ref_type_str = payload.get("ref_type", "unknown")
    try:
        ref_type = RefType(ref_type_str)
    except ValueError:
        ref_type = RefType.UNKNOWN

    # Parse event name
    event_str = payload.get("event_name", "unknown")
    try:
        event_name = EventType(event_str)
    except ValueError:
        event_name = EventType.UNKNOWN

    # Parse runner environment
    runner_str = payload.get("runner_environment", "unknown")
    try:
        runner_env = RunnerEnvironment(runner_str)
    except ValueError:
        runner_env = RunnerEnvironment.UNKNOWN

    # Known claim names
    known_claims = {
        "iss", "sub", "aud", "exp", "iat", "nbf", "jti",
        "repository", "repository_owner", "repository_owner_id",
        "repository_id", "repository_visibility",
        "actor", "actor_id",
        "workflow", "workflow_ref", "workflow_sha",
        "job_workflow_ref", "job_workflow_sha",
        "ref", "ref_type", "sha", "head_ref", "base_ref",
        "run_id", "run_number", "run_attempt", "event_name",
        "environment", "environment_node_id", "runner_environment",
    }

    # Collect extra claims
    extra = {k: v for k, v in payload.items() if k not in known_claims}

    return GitHubActionsClaims(
        issuer=issuer,
        subject=subject,
        audience=audience,
        expiration=expiration,
        issued_at=issued_at,
        not_before=not_before,
        jwt_id=payload.get("jti"),
        repository=payload.get("repository", ""),
        repository_owner=payload.get("repository_owner", ""),
        repository_owner_id=str(payload.get("repository_owner_id", "")),
        repository_id=str(payload.get("repository_id", "")),
        repository_visibility=visibility,
        actor=payload.get("actor", ""),
        actor_id=str(payload.get("actor_id", "")),
        workflow=payload.get("workflow", ""),
        workflow_ref=payload.get("workflow_ref", ""),
        workflow_sha=payload.get("workflow_sha", ""),
        job_workflow_ref=payload.get("job_workflow_ref"),
        job_workflow_sha=payload.get("job_workflow_sha"),
        ref=payload.get("ref", ""),
        ref_type=ref_type,
        sha=payload.get("sha", ""),
        head_ref=payload.get("head_ref"),
        base_ref=payload.get("base_ref"),
        run_id=str(payload.get("run_id", "")),
        run_number=str(payload.get("run_number", "")),
        run_attempt=str(payload.get("run_attempt", "")),
        event_name=event_name,
        environment=payload.get("environment"),
        environment_node_id=payload.get("environment_node_id"),
        runner_environment=runner_env,
        extra=extra,
    )


@dataclass
class ClaimsValidationResult:
    """Result of claims validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class ClaimsValidationPolicy:
    """Policy for validating GitHub Actions claims.

    Attributes:
        allowed_repositories: List of allowed repository patterns.
        allowed_branches: List of allowed branch patterns.
        allowed_tags: List of allowed tag patterns.
        allowed_actors: List of allowed actors.
        allowed_events: List of allowed event types.
        allowed_environments: List of allowed environment names.
        require_environment: Require a deployment environment.
        allow_pull_requests: Allow pull request events.
        allow_forks: Allow forked repositories.
        require_github_hosted: Require GitHub-hosted runners.
        allowed_workflows: List of allowed workflow patterns.
    """

    allowed_repositories: list[str] | None = None
    allowed_branches: list[str] | None = None
    allowed_tags: list[str] | None = None
    allowed_actors: list[str] | None = None
    allowed_events: list[EventType] | None = None
    allowed_environments: list[str] | None = None
    require_environment: bool = False
    allow_pull_requests: bool = True
    allow_forks: bool = False
    require_github_hosted: bool = False
    allowed_workflows: list[str] | None = None


def validate_claims(
    claims: GitHubActionsClaims,
    policy: ClaimsValidationPolicy | None = None,
) -> ClaimsValidationResult:
    """Validate GitHub Actions claims against a policy.

    Args:
        claims: Parsed claims to validate.
        policy: Validation policy (uses defaults if None).

    Returns:
        ClaimsValidationResult with errors and warnings.
    """
    if policy is None:
        policy = ClaimsValidationPolicy()

    errors: list[str] = []
    warnings: list[str] = []

    # Check expiration
    if claims.is_expired:
        errors.append("Token is expired")

    # Check issuer
    if claims.issuer != "https://token.actions.githubusercontent.com":
        errors.append(f"Invalid issuer: {claims.issuer}")

    # Check repository
    if policy.allowed_repositories:
        if not any(
            claims.matches_repository(pattern)
            for pattern in policy.allowed_repositories
        ):
            errors.append(
                f"Repository '{claims.repository}' not in allowed list: "
                f"{policy.allowed_repositories}"
            )

    # Check branches
    if policy.allowed_branches and claims.branch_name:
        branch_patterns = [f"refs/heads/{b}" for b in policy.allowed_branches]
        if not any(claims.matches_ref(pattern) for pattern in branch_patterns):
            errors.append(
                f"Branch '{claims.branch_name}' not in allowed list: "
                f"{policy.allowed_branches}"
            )

    # Check tags
    if policy.allowed_tags and claims.tag_name:
        tag_patterns = [f"refs/tags/{t}" for t in policy.allowed_tags]
        if not any(claims.matches_ref(pattern) for pattern in tag_patterns):
            errors.append(
                f"Tag '{claims.tag_name}' not in allowed list: "
                f"{policy.allowed_tags}"
            )

    # Check actors
    if policy.allowed_actors:
        if claims.actor and claims.actor not in policy.allowed_actors:
            errors.append(
                f"Actor '{claims.actor}' not in allowed list: "
                f"{policy.allowed_actors}"
            )

    # Check events
    if policy.allowed_events:
        if claims.event_name not in policy.allowed_events:
            errors.append(
                f"Event '{claims.event_name.value}' not in allowed list: "
                f"{[e.value for e in policy.allowed_events]}"
            )

    # Check environments
    if policy.require_environment and not claims.environment:
        errors.append("Deployment environment is required but not present")

    if policy.allowed_environments and claims.environment:
        if claims.environment not in policy.allowed_environments:
            errors.append(
                f"Environment '{claims.environment}' not in allowed list: "
                f"{policy.allowed_environments}"
            )

    # Check pull requests
    if not policy.allow_pull_requests and claims.is_pull_request:
        errors.append("Pull request events are not allowed")

    # Check forks
    if not policy.allow_forks:
        if claims.repository_visibility == RepositoryVisibility.PUBLIC:
            # For public repos, check if head_ref suggests a fork
            if claims.head_ref and ":" in claims.head_ref:
                errors.append("Forked repository events are not allowed")

    # Check runner environment
    if policy.require_github_hosted and not claims.is_github_hosted:
        errors.append("GitHub-hosted runners are required")

    # Check workflows
    if policy.allowed_workflows:
        workflow_match = False
        for pattern in policy.allowed_workflows:
            if claims.workflow_ref:
                regex_pattern = pattern.replace("*", ".*")
                if re.match(f"^{regex_pattern}$", claims.workflow_ref):
                    workflow_match = True
                    break
        if not workflow_match:
            errors.append(
                f"Workflow '{claims.workflow_ref}' not in allowed list: "
                f"{policy.allowed_workflows}"
            )

    # Add warnings
    if claims.is_pull_request:
        warnings.append("Token was issued for a pull request event")

    if claims.is_reusable_workflow:
        warnings.append(
            f"Token was issued for a reusable workflow: {claims.job_workflow_ref}"
        )

    return ClaimsValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
