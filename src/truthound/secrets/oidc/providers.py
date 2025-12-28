"""OIDC Identity Provider Implementations.

This module provides OIDC token retrieval for various CI/CD platforms:
- GitHub Actions
- GitLab CI/CD
- CircleCI
- Bitbucket Pipelines

Each provider handles the platform-specific token request mechanism.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from truthound.secrets.oidc.base import (
    BaseOIDCProvider,
    CIProvider,
    OIDCToken,
    OIDCProviderNotAvailableError,
    OIDCTokenError,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# GitHub Actions OIDC Provider
# =============================================================================


@dataclass
class GitHubActionsConfig:
    """Configuration for GitHub Actions OIDC provider.

    Attributes:
        audience: Token audience (default: based on cloud provider).
        request_timeout: HTTP request timeout in seconds.
        enable_cache: Whether to cache tokens.
        cache_ttl_seconds: Token cache TTL.
    """

    audience: str | None = None
    request_timeout: float = 30.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


class GitHubActionsOIDCProvider(BaseOIDCProvider):
    """GitHub Actions OIDC token provider.

    Retrieves OIDC tokens from GitHub Actions environment using the
    ACTIONS_ID_TOKEN_REQUEST_URL and ACTIONS_ID_TOKEN_REQUEST_TOKEN
    environment variables.

    Note: The workflow must have `id-token: write` permission:

    ```yaml
    permissions:
      id-token: write
      contents: read
    ```

    Example:
        >>> provider = GitHubActionsOIDCProvider(
        ...     audience="sts.amazonaws.com",
        ... )
        >>> token = provider.get_token()
        >>> print(token.claims.repository)  # "owner/repo"

    Token Claims (GitHub-specific):
        - iss: https://token.actions.githubusercontent.com
        - sub: repo:owner/repo:ref:refs/heads/main
        - aud: configured audience
        - repository: owner/repo
        - repository_owner: owner
        - ref: refs/heads/main
        - sha: commit SHA
        - actor: triggering user
        - workflow: workflow name
        - run_id: run ID
        - environment: deployment environment (if any)
    """

    # Environment variables
    TOKEN_URL_VAR = "ACTIONS_ID_TOKEN_REQUEST_URL"
    TOKEN_VAR = "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
    GITHUB_ACTIONS_VAR = "GITHUB_ACTIONS"

    def __init__(
        self,
        audience: str | None = None,
        *,
        config: GitHubActionsConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GitHub Actions OIDC provider.

        Args:
            audience: Default token audience.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or GitHubActionsConfig(audience=audience)
        if audience:
            self._config.audience = audience

        super().__init__(
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            enable_cache=self._config.enable_cache,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "github_actions"

    def is_available(self) -> bool:
        """Check if running in GitHub Actions with OIDC support."""
        return (
            os.environ.get(self.GITHUB_ACTIONS_VAR) == "true"
            and self.TOKEN_URL_VAR in os.environ
            and self.TOKEN_VAR in os.environ
        )

    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch OIDC token from GitHub Actions.

        Args:
            audience: Token audience (falls back to config).

        Returns:
            Raw JWT token string.
        """
        import urllib.request
        import urllib.error
        import urllib.parse

        token_url = os.environ.get(self.TOKEN_URL_VAR)
        bearer_token = os.environ.get(self.TOKEN_VAR)

        if not token_url or not bearer_token:
            raise OIDCProviderNotAvailableError(
                self.name,
                f"Missing {self.TOKEN_URL_VAR} or {self.TOKEN_VAR}",
            )

        # Use provided audience or config default
        effective_audience = audience or self._config.audience

        # Build URL with audience parameter
        if effective_audience:
            parsed = urllib.parse.urlparse(token_url)
            query = urllib.parse.parse_qs(parsed.query)
            query["audience"] = [effective_audience]
            new_query = urllib.parse.urlencode(query, doseq=True)
            token_url = urllib.parse.urlunparse(
                parsed._replace(query=new_query)
            )

        # Make request
        request = urllib.request.Request(
            token_url,
            headers={
                "Authorization": f"bearer {bearer_token}",
                "Accept": "application/json; api-version=2.0",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())
                return data["value"]

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCTokenError(
                f"HTTP {e.code}: {error_body or e.reason}",
                provider=self.name,
            ) from e
        except urllib.error.URLError as e:
            raise OIDCTokenError(
                f"Network error: {e.reason}",
                provider=self.name,
            ) from e
        except KeyError:
            raise OIDCTokenError(
                "Response missing 'value' field",
                provider=self.name,
            )

    def get_environment_info(self) -> dict[str, str | None]:
        """Get GitHub Actions environment information."""
        return {
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "ref": os.environ.get("GITHUB_REF"),
            "sha": os.environ.get("GITHUB_SHA"),
            "actor": os.environ.get("GITHUB_ACTOR"),
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "job": os.environ.get("GITHUB_JOB"),
        }


# =============================================================================
# GitLab CI OIDC Provider
# =============================================================================


@dataclass
class GitLabCIConfig:
    """Configuration for GitLab CI OIDC provider.

    Attributes:
        audience: Token audience.
        request_timeout: HTTP request timeout in seconds.
        enable_cache: Whether to cache tokens.
        cache_ttl_seconds: Token cache TTL.
    """

    audience: str | None = None
    request_timeout: float = 30.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


class GitLabCIOIDCProvider(BaseOIDCProvider):
    """GitLab CI OIDC token provider.

    Retrieves OIDC tokens from GitLab CI using the CI_JOB_JWT_V2
    environment variable (or CI_JOB_JWT for older versions).

    Note: Must be enabled in CI/CD settings and job must use:

    ```yaml
    job:
      id_tokens:
        GITLAB_OIDC_TOKEN:
          aud: https://your-audience.example.com
    ```

    Token Claims (GitLab-specific):
        - iss: https://gitlab.com (or self-hosted URL)
        - sub: project_path:{group}/{project}:ref_type:{type}:ref:{ref}
        - aud: configured audience
        - project_path: group/project
        - ref: branch/tag name
        - ref_path: refs/heads/main or refs/tags/v1.0
        - pipeline_id: pipeline ID
        - pipeline_source: trigger source (push, schedule, etc.)
        - user_id: triggering user ID
        - user_login: triggering user login
        - namespace_path: group path
        - environment: deployment environment (if any)
    """

    # Environment variables
    JWT_V2_VAR = "CI_JOB_JWT_V2"
    JWT_V1_VAR = "CI_JOB_JWT"
    GITLAB_CI_VAR = "GITLAB_CI"
    OIDC_TOKEN_VAR = "GITLAB_OIDC_TOKEN"

    def __init__(
        self,
        audience: str | None = None,
        *,
        token_variable: str | None = None,
        config: GitLabCIConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GitLab CI OIDC provider.

        Args:
            audience: Default token audience.
            token_variable: Custom environment variable for the token.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or GitLabCIConfig(audience=audience)
        if audience:
            self._config.audience = audience
        self._token_variable = token_variable

        super().__init__(
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            enable_cache=self._config.enable_cache,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "gitlab_ci"

    def is_available(self) -> bool:
        """Check if running in GitLab CI with OIDC support."""
        if os.environ.get(self.GITLAB_CI_VAR) != "true":
            return False

        # Check for any of the token sources
        if self._token_variable and os.environ.get(self._token_variable):
            return True
        if os.environ.get(self.OIDC_TOKEN_VAR):
            return True
        if os.environ.get(self.JWT_V2_VAR):
            return True
        if os.environ.get(self.JWT_V1_VAR):
            return True

        return False

    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch OIDC token from GitLab CI environment.

        GitLab provides the token directly in environment variables,
        no HTTP request needed.

        Args:
            audience: Token audience (not used, set in CI config).

        Returns:
            Raw JWT token string.
        """
        # Priority order for token sources
        token_sources = [
            self._token_variable,
            self.OIDC_TOKEN_VAR,
            self.JWT_V2_VAR,
            self.JWT_V1_VAR,
        ]

        for source in token_sources:
            if source and (token := os.environ.get(source)):
                return token

        raise OIDCProviderNotAvailableError(
            self.name,
            "No OIDC token found in environment. "
            "Ensure id_tokens is configured in your .gitlab-ci.yml",
        )

    def get_environment_info(self) -> dict[str, str | None]:
        """Get GitLab CI environment information."""
        return {
            "project_path": os.environ.get("CI_PROJECT_PATH"),
            "project_id": os.environ.get("CI_PROJECT_ID"),
            "ref": os.environ.get("CI_COMMIT_REF_NAME"),
            "sha": os.environ.get("CI_COMMIT_SHA"),
            "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
            "pipeline_source": os.environ.get("CI_PIPELINE_SOURCE"),
            "job_id": os.environ.get("CI_JOB_ID"),
            "job_name": os.environ.get("CI_JOB_NAME"),
            "user_login": os.environ.get("GITLAB_USER_LOGIN"),
            "user_id": os.environ.get("GITLAB_USER_ID"),
            "environment": os.environ.get("CI_ENVIRONMENT_NAME"),
        }


# =============================================================================
# CircleCI OIDC Provider
# =============================================================================


@dataclass
class CircleCIConfig:
    """Configuration for CircleCI OIDC provider.

    Attributes:
        audience: Token audience.
        request_timeout: HTTP request timeout in seconds.
        enable_cache: Whether to cache tokens.
        cache_ttl_seconds: Token cache TTL.
    """

    audience: str | None = None
    request_timeout: float = 30.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


class CircleCIOIDCProvider(BaseOIDCProvider):
    """CircleCI OIDC token provider.

    Retrieves OIDC tokens from CircleCI using the CIRCLE_OIDC_TOKEN
    environment variable (v2) or CIRCLE_OIDC_TOKEN_V2 for newer versions.

    Note: Must have OIDC enabled in project settings. The token is
    automatically injected for jobs with a configured context.

    Token Claims (CircleCI-specific):
        - iss: https://oidc.circleci.com/org/{org-id}
        - sub: org/{org-id}/project/{project-id}/user/{user-id}
        - aud: configured audience
        - oidc.circleci.com/project-id: project UUID
        - oidc.circleci.com/context-ids: list of context IDs
        - oidc.circleci.com/vcs-origin: VCS origin URL
    """

    # Environment variables
    OIDC_TOKEN_VAR = "CIRCLE_OIDC_TOKEN"
    OIDC_TOKEN_V2_VAR = "CIRCLE_OIDC_TOKEN_V2"
    CIRCLECI_VAR = "CIRCLECI"

    def __init__(
        self,
        audience: str | None = None,
        *,
        config: CircleCIConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CircleCI OIDC provider.

        Args:
            audience: Default token audience.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or CircleCIConfig(audience=audience)
        if audience:
            self._config.audience = audience

        super().__init__(
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            enable_cache=self._config.enable_cache,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "circleci"

    def is_available(self) -> bool:
        """Check if running in CircleCI with OIDC support."""
        if os.environ.get(self.CIRCLECI_VAR) != "true":
            return False

        return bool(
            os.environ.get(self.OIDC_TOKEN_V2_VAR)
            or os.environ.get(self.OIDC_TOKEN_VAR)
        )

    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch OIDC token from CircleCI environment.

        Args:
            audience: Token audience (not used, set in project settings).

        Returns:
            Raw JWT token string.
        """
        token = os.environ.get(self.OIDC_TOKEN_V2_VAR) or os.environ.get(
            self.OIDC_TOKEN_VAR
        )

        if not token:
            raise OIDCProviderNotAvailableError(
                self.name,
                "CIRCLE_OIDC_TOKEN not found. Ensure OIDC is enabled "
                "in project settings and a context is configured.",
            )

        return token

    def get_environment_info(self) -> dict[str, str | None]:
        """Get CircleCI environment information."""
        return {
            "project_reponame": os.environ.get("CIRCLE_PROJECT_REPONAME"),
            "project_username": os.environ.get("CIRCLE_PROJECT_USERNAME"),
            "branch": os.environ.get("CIRCLE_BRANCH"),
            "sha": os.environ.get("CIRCLE_SHA1"),
            "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
            "job": os.environ.get("CIRCLE_JOB"),
            "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
            "workflow_job_id": os.environ.get("CIRCLE_WORKFLOW_JOB_ID"),
            "username": os.environ.get("CIRCLE_USERNAME"),
        }


# =============================================================================
# Bitbucket Pipelines OIDC Provider
# =============================================================================


@dataclass
class BitbucketPipelinesConfig:
    """Configuration for Bitbucket Pipelines OIDC provider.

    Attributes:
        audience: Token audience.
        identity_provider: Identity provider name configured in Bitbucket.
        request_timeout: HTTP request timeout in seconds.
        enable_cache: Whether to cache tokens.
        cache_ttl_seconds: Token cache TTL.
    """

    audience: str | None = None
    identity_provider: str | None = None
    request_timeout: float = 30.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


class BitbucketPipelinesOIDCProvider(BaseOIDCProvider):
    """Bitbucket Pipelines OIDC token provider.

    Retrieves OIDC tokens from Bitbucket Pipelines using the
    bitbucket-request-oidc-token pipe or environment variable.

    Note: Must configure OIDC identity provider in repository settings.

    Token Claims (Bitbucket-specific):
        - iss: https://api.bitbucket.org/2.0/workspaces/{workspace}/pipelines-config/identity/oidc
        - sub: {repository-uuid}:{step-uuid}
        - aud: configured audience
        - repository_uuid: repository UUID
        - branch_name: branch name
        - workspace: workspace slug
    """

    # Environment variables
    OIDC_TOKEN_VAR = "BITBUCKET_STEP_OIDC_TOKEN"
    BITBUCKET_VAR = "BITBUCKET_BUILD_NUMBER"

    def __init__(
        self,
        audience: str | None = None,
        *,
        identity_provider: str | None = None,
        config: BitbucketPipelinesConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Bitbucket Pipelines OIDC provider.

        Args:
            audience: Default token audience.
            identity_provider: Identity provider name.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or BitbucketPipelinesConfig(
            audience=audience,
            identity_provider=identity_provider,
        )
        if audience:
            self._config.audience = audience
        if identity_provider:
            self._config.identity_provider = identity_provider

        super().__init__(
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            enable_cache=self._config.enable_cache,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "bitbucket_pipelines"

    def is_available(self) -> bool:
        """Check if running in Bitbucket Pipelines with OIDC support."""
        return (
            os.environ.get(self.BITBUCKET_VAR) is not None
            and os.environ.get(self.OIDC_TOKEN_VAR) is not None
        )

    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch OIDC token from Bitbucket Pipelines environment.

        Args:
            audience: Token audience.

        Returns:
            Raw JWT token string.
        """
        token = os.environ.get(self.OIDC_TOKEN_VAR)

        if not token:
            raise OIDCProviderNotAvailableError(
                self.name,
                f"{self.OIDC_TOKEN_VAR} not found. "
                "Ensure OIDC is configured in repository settings "
                "and bitbucket-request-oidc-token pipe is used.",
            )

        return token

    def get_environment_info(self) -> dict[str, str | None]:
        """Get Bitbucket Pipelines environment information."""
        return {
            "repo_slug": os.environ.get("BITBUCKET_REPO_SLUG"),
            "repo_uuid": os.environ.get("BITBUCKET_REPO_UUID"),
            "workspace": os.environ.get("BITBUCKET_WORKSPACE"),
            "branch": os.environ.get("BITBUCKET_BRANCH"),
            "commit": os.environ.get("BITBUCKET_COMMIT"),
            "build_number": os.environ.get("BITBUCKET_BUILD_NUMBER"),
            "pipeline_uuid": os.environ.get("BITBUCKET_PIPELINE_UUID"),
            "step_uuid": os.environ.get("BITBUCKET_STEP_UUID"),
        }


# =============================================================================
# Generic OIDC Provider
# =============================================================================


@dataclass
class GenericOIDCConfig:
    """Configuration for generic OIDC provider.

    Attributes:
        name: Provider name.
        token_url: URL to fetch token from.
        token_header_name: Authorization header name.
        token_header_value_env: Env var containing header value.
        token_response_field: JSON field containing the token.
        audience_param: Query parameter name for audience.
        request_timeout: HTTP request timeout.
        enable_cache: Whether to cache tokens.
        cache_ttl_seconds: Token cache TTL.
    """

    name: str = "generic"
    token_url_env: str = ""
    token_header_name: str = "Authorization"
    token_header_value_env: str = ""
    token_response_field: str = "token"
    audience_param: str = "audience"
    request_timeout: float = 30.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300


class GenericOIDCProvider(BaseOIDCProvider):
    """Generic OIDC token provider for custom CI platforms.

    Allows configuration of a custom OIDC token endpoint for CI platforms
    not directly supported.

    Example:
        >>> provider = GenericOIDCProvider(
        ...     config=GenericOIDCConfig(
        ...         name="my-ci",
        ...         token_url_env="MY_CI_TOKEN_URL",
        ...         token_header_value_env="MY_CI_TOKEN",
        ...         token_response_field="id_token",
        ...     ),
        ... )
    """

    def __init__(
        self,
        config: GenericOIDCConfig,
        **kwargs: Any,
    ) -> None:
        """Initialize generic OIDC provider.

        Args:
            config: Provider configuration.
            **kwargs: Additional base class arguments.
        """
        self._config = config

        super().__init__(
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            enable_cache=self._config.enable_cache,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return self._config.name

    def is_available(self) -> bool:
        """Check if the configured environment variables are present."""
        if self._config.token_url_env:
            return os.environ.get(self._config.token_url_env) is not None
        return False

    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch OIDC token from configured endpoint.

        Args:
            audience: Token audience.

        Returns:
            Raw JWT token string.
        """
        import urllib.request
        import urllib.error
        import urllib.parse

        token_url = os.environ.get(self._config.token_url_env, "")
        if not token_url:
            raise OIDCProviderNotAvailableError(
                self.name,
                f"Missing {self._config.token_url_env}",
            )

        # Add audience parameter if provided
        if audience and self._config.audience_param:
            parsed = urllib.parse.urlparse(token_url)
            query = urllib.parse.parse_qs(parsed.query)
            query[self._config.audience_param] = [audience]
            new_query = urllib.parse.urlencode(query, doseq=True)
            token_url = urllib.parse.urlunparse(
                parsed._replace(query=new_query)
            )

        # Build headers
        headers = {"Accept": "application/json"}
        if self._config.token_header_value_env:
            header_value = os.environ.get(self._config.token_header_value_env, "")
            if header_value:
                headers[self._config.token_header_name] = header_value

        request = urllib.request.Request(token_url, headers=headers)

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())
                return data[self._config.token_response_field]

        except urllib.error.HTTPError as e:
            raise OIDCTokenError(
                f"HTTP {e.code}: {e.reason}",
                provider=self.name,
            ) from e
        except KeyError:
            raise OIDCTokenError(
                f"Response missing '{self._config.token_response_field}' field",
                provider=self.name,
            )


# =============================================================================
# Detection Functions
# =============================================================================


def detect_ci_oidc_provider() -> BaseOIDCProvider | None:
    """Detect and return the OIDC provider for the current CI environment.

    Returns:
        Appropriate OIDC provider instance, or None if not in a supported CI.

    Example:
        >>> provider = detect_ci_oidc_provider()
        >>> if provider:
        ...     token = provider.get_token(audience="sts.amazonaws.com")
    """
    # Try each provider in order of likelihood
    providers: list[type[BaseOIDCProvider]] = [
        GitHubActionsOIDCProvider,
        GitLabCIOIDCProvider,
        CircleCIOIDCProvider,
        BitbucketPipelinesOIDCProvider,
    ]

    for provider_cls in providers:
        try:
            provider = provider_cls()
            if provider.is_available():
                logger.info(f"Detected OIDC provider: {provider.name}")
                return provider
        except Exception as e:
            logger.debug(f"Provider {provider_cls.__name__} check failed: {e}")

    logger.debug("No OIDC provider detected in current environment")
    return None


def is_oidc_available() -> bool:
    """Check if OIDC authentication is available in the current environment.

    Returns:
        True if an OIDC provider is available.

    Example:
        >>> if is_oidc_available():
        ...     # Use OIDC authentication
        ...     creds = get_oidc_credentials()
        ... else:
        ...     # Fall back to traditional credentials
        ...     creds = get_static_credentials()
    """
    provider = detect_ci_oidc_provider()
    return provider is not None


def get_ci_provider_type() -> CIProvider:
    """Detect the CI provider type from environment.

    Returns:
        CIProvider enum value.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return CIProvider.GITHUB_ACTIONS
    if os.environ.get("GITLAB_CI") == "true":
        return CIProvider.GITLAB_CI
    if os.environ.get("CIRCLECI") == "true":
        return CIProvider.CIRCLECI
    if os.environ.get("BITBUCKET_BUILD_NUMBER"):
        return CIProvider.BITBUCKET
    if os.environ.get("JENKINS_URL"):
        return CIProvider.JENKINS

    return CIProvider.UNKNOWN
