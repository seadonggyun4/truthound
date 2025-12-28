"""Enhanced GitHub Actions OIDC Provider.

This module provides an enhanced GitHub Actions OIDC provider with:
- Policy-based access control
- Environment and workflow validation
- Automatic claims parsing
- Integration with cloud providers
- Caching with intelligent refresh
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from truthound.secrets.oidc.base import (
    AWSCredentials,
    AzureCredentials,
    CloudCredentials,
    CloudProvider,
    GCPCredentials,
    OIDCConfigurationError,
    OIDCError,
    OIDCProviderNotAvailableError,
    OIDCToken,
)
from truthound.secrets.oidc.providers import GitHubActionsOIDCProvider
from truthound.secrets.oidc.exchangers import (
    AWSTokenExchanger,
    AzureTokenExchanger,
    GCPTokenExchanger,
    VaultTokenExchanger,
)
from truthound.secrets.oidc.github.claims import (
    ClaimsValidationPolicy,
    EventType,
    GitHubActionsClaims,
    GitHubActionsContext,
    parse_github_claims,
    validate_claims,
)

if TYPE_CHECKING:
    from truthound.secrets.oidc.exchangers import VaultCredentials


logger = logging.getLogger(__name__)


# =============================================================================
# Policy Classes
# =============================================================================


@dataclass
class EnvironmentPolicy:
    """Policy for deployment environment requirements.

    Attributes:
        require: Require a deployment environment.
        allowed: List of allowed environment names.
        deny: List of denied environment names.
        production_only: Only allow production environment.
    """

    require: bool = False
    allowed: list[str] | None = None
    deny: list[str] | None = None
    production_only: bool = False

    def validate(self, environment: str | None) -> tuple[bool, str | None]:
        """Validate environment against policy.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.require and not environment:
            return False, "Deployment environment is required"

        if environment:
            if self.production_only and environment.lower() != "production":
                return False, f"Only 'production' environment allowed, got: {environment}"

            if self.allowed and environment not in self.allowed:
                return False, f"Environment '{environment}' not in allowed list"

            if self.deny and environment in self.deny:
                return False, f"Environment '{environment}' is explicitly denied"

        return True, None


@dataclass
class WorkflowPolicy:
    """Policy for workflow requirements.

    Attributes:
        allowed_repositories: List of allowed repository patterns.
        allowed_branches: List of allowed branch names.
        allowed_tags: List of allowed tag patterns.
        allowed_actors: List of allowed actor usernames.
        allowed_events: List of allowed event types.
        deny_pull_requests: Deny pull request events.
        deny_forks: Deny forked repository access.
        require_github_hosted: Require GitHub-hosted runners.
        allowed_workflows: List of allowed workflow file patterns.
        max_run_attempts: Maximum allowed run attempts.
    """

    allowed_repositories: list[str] | None = None
    allowed_branches: list[str] | None = None
    allowed_tags: list[str] | None = None
    allowed_actors: list[str] | None = None
    allowed_events: list[EventType] | None = None
    deny_pull_requests: bool = False
    deny_forks: bool = False
    require_github_hosted: bool = False
    allowed_workflows: list[str] | None = None
    max_run_attempts: int | None = None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GitHubActionsOIDCConfig:
    """Configuration for enhanced GitHub Actions OIDC provider.

    Attributes:
        audience: Default token audience.
        enable_cache: Enable token caching.
        cache_ttl_seconds: Token cache TTL.
        cache_refresh_margin_seconds: Refresh margin before expiry.
        retry_attempts: Number of retry attempts.
        retry_delay_seconds: Base delay between retries.
        request_timeout: HTTP request timeout.
        validate_claims: Enable claims validation.
        environment_policy: Deployment environment policy.
        workflow_policy: Workflow access policy.
        verify_token: Enable token signature verification.
    """

    audience: str | None = None
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    cache_refresh_margin_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    request_timeout: float = 30.0
    validate_claims: bool = True
    environment_policy: EnvironmentPolicy = field(default_factory=EnvironmentPolicy)
    workflow_policy: WorkflowPolicy = field(default_factory=WorkflowPolicy)
    verify_token: bool = False  # Requires cryptography package


# =============================================================================
# Enhanced Provider
# =============================================================================


class GitHubActionsOIDC:
    """Enhanced GitHub Actions OIDC integration.

    Provides a high-level interface for GitHub Actions OIDC with:
    - Policy-based access control
    - Automatic claims validation
    - Easy cloud credential retrieval
    - Intelligent caching

    Example:
        >>> # Basic usage
        >>> oidc = GitHubActionsOIDC(audience="sts.amazonaws.com")
        >>> creds = oidc.get_aws_credentials(
        ...     role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
        >>>
        >>> # With policies
        >>> oidc = GitHubActionsOIDC(
        ...     audience="sts.amazonaws.com",
        ...     config=GitHubActionsOIDCConfig(
        ...         environment_policy=EnvironmentPolicy(
        ...             require=True,
        ...             allowed=["production", "staging"],
        ...         ),
        ...         workflow_policy=WorkflowPolicy(
        ...             allowed_repositories=["owner/repo"],
        ...             allowed_branches=["main"],
        ...             deny_pull_requests=True,
        ...         ),
        ...     ),
        ... )

    Example with decorator:
        >>> @oidc.with_credentials(
        ...     cloud_provider="aws",
        ...     role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
        ... def my_function(credentials: AWSCredentials):
        ...     # Use credentials
        ...     pass
    """

    # GitHub Actions OIDC issuer
    ISSUER = "https://token.actions.githubusercontent.com"

    def __init__(
        self,
        audience: str | None = None,
        *,
        config: GitHubActionsOIDCConfig | None = None,
        require_environment: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GitHub Actions OIDC provider.

        Args:
            audience: Token audience.
            config: Full configuration.
            require_environment: Shortcut to require specific environment.
            **kwargs: Additional config options.
        """
        # Build config
        if config is None:
            config = GitHubActionsOIDCConfig(audience=audience)
        elif audience:
            config.audience = audience

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Handle require_environment shortcut
        if require_environment:
            config.environment_policy.require = True
            config.environment_policy.allowed = [require_environment]

        self._config = config

        # Create underlying provider with config
        from truthound.secrets.oidc.providers import GitHubActionsConfig

        provider_config = GitHubActionsConfig(
            audience=config.audience,
            cache_ttl_seconds=config.cache_ttl_seconds,
            enable_cache=config.enable_cache,
        )
        self._provider = GitHubActionsOIDCProvider(config=provider_config)

        # Cached claims
        self._cached_claims: GitHubActionsClaims | None = None
        self._context: GitHubActionsContext | None = None

    @property
    def is_available(self) -> bool:
        """Check if GitHub Actions OIDC is available."""
        return self._provider.is_available()

    @property
    def claims(self) -> GitHubActionsClaims:
        """Get parsed claims from current token."""
        if self._cached_claims is None:
            token = self.get_token()
            self._cached_claims = parse_github_claims(token.claims.__dict__)
        return self._cached_claims

    @property
    def context(self) -> GitHubActionsContext:
        """Get GitHub Actions context from environment."""
        if self._context is None:
            self._context = GitHubActionsContext.from_environment()
        return self._context

    @property
    def repository(self) -> str:
        """Get repository name (owner/repo)."""
        return self.claims.repository

    @property
    def branch(self) -> str | None:
        """Get current branch name."""
        return self.claims.branch_name

    @property
    def commit_sha(self) -> str:
        """Get commit SHA."""
        return self.claims.sha

    @property
    def environment(self) -> str | None:
        """Get deployment environment name."""
        return self.claims.environment

    @property
    def actor(self) -> str:
        """Get actor who triggered the workflow."""
        return self.claims.actor

    def get_token(self, audience: str | None = None) -> OIDCToken:
        """Get OIDC token with validation.

        Args:
            audience: Token audience (uses config default if not provided).

        Returns:
            OIDC token.

        Raises:
            OIDCError: If token retrieval or validation fails.
        """
        effective_audience = audience or self._config.audience
        token = self._provider.get_token(effective_audience)

        # Parse and validate claims
        if self._config.validate_claims:
            claims = parse_github_claims(token.claims.__dict__)
            self._validate_claims(claims)
            self._cached_claims = claims

        return token

    def _validate_claims(self, claims: GitHubActionsClaims) -> None:
        """Validate claims against configured policies."""
        errors: list[str] = []

        # Validate environment policy
        env_policy = self._config.environment_policy
        is_valid, error = env_policy.validate(claims.environment)
        if not is_valid and error:
            errors.append(error)

        # Validate workflow policy
        wf_policy = self._config.workflow_policy

        # Check repositories
        if wf_policy.allowed_repositories:
            if not any(
                claims.matches_repository(pattern)
                for pattern in wf_policy.allowed_repositories
            ):
                errors.append(
                    f"Repository '{claims.repository}' not in allowed list"
                )

        # Check branches
        if wf_policy.allowed_branches and claims.branch_name:
            if claims.branch_name not in wf_policy.allowed_branches:
                errors.append(
                    f"Branch '{claims.branch_name}' not in allowed list"
                )

        # Check tags
        if wf_policy.allowed_tags and claims.tag_name:
            tag_match = False
            for pattern in wf_policy.allowed_tags:
                if claims.matches_ref(f"refs/tags/{pattern}"):
                    tag_match = True
                    break
            if not tag_match:
                errors.append(f"Tag '{claims.tag_name}' not in allowed list")

        # Check actors
        if wf_policy.allowed_actors:
            if claims.actor not in wf_policy.allowed_actors:
                errors.append(f"Actor '{claims.actor}' not in allowed list")

        # Check events
        if wf_policy.allowed_events:
            if claims.event_name not in wf_policy.allowed_events:
                errors.append(
                    f"Event '{claims.event_name.value}' not in allowed list"
                )

        # Check pull requests
        if wf_policy.deny_pull_requests and claims.is_pull_request:
            errors.append("Pull request events are denied by policy")

        # Check GitHub-hosted requirement
        if wf_policy.require_github_hosted and not claims.is_github_hosted:
            errors.append("GitHub-hosted runners are required by policy")

        # Check workflows
        if wf_policy.allowed_workflows:
            workflow_match = False
            for pattern in wf_policy.allowed_workflows:
                if claims.matches_ref(pattern):
                    workflow_match = True
                    break
            if not workflow_match:
                errors.append(
                    f"Workflow '{claims.workflow_ref}' not in allowed list"
                )

        # Check run attempts
        if wf_policy.max_run_attempts:
            try:
                run_attempt = int(claims.run_attempt)
                if run_attempt > wf_policy.max_run_attempts:
                    errors.append(
                        f"Run attempt {run_attempt} exceeds maximum "
                        f"{wf_policy.max_run_attempts}"
                    )
            except ValueError:
                pass

        if errors:
            raise OIDCConfigurationError(
                f"Claims validation failed: {'; '.join(errors)}"
            )

    def get_aws_credentials(
        self,
        role_arn: str | None = None,
        *,
        session_name: str = "github-actions-oidc",
        session_duration_seconds: int = 3600,
        region: str | None = None,
    ) -> AWSCredentials:
        """Get AWS credentials using OIDC.

        Args:
            role_arn: IAM role ARN to assume.
            session_name: Session name for assumed role.
            session_duration_seconds: Credential validity duration.
            region: AWS region.

        Returns:
            AWS credentials.
        """
        role_arn = role_arn or os.environ.get("AWS_ROLE_ARN")
        if not role_arn:
            raise OIDCConfigurationError("AWS role ARN is required")

        # Get token with AWS STS audience
        token = self.get_token(audience="sts.amazonaws.com")

        # Create exchanger and get credentials
        exchanger = AWSTokenExchanger(
            role_arn=role_arn,
            session_name=session_name,
            session_duration_seconds=session_duration_seconds,
            region=region,
            enable_cache=self._config.enable_cache,
        )

        return exchanger.exchange(token)  # type: ignore

    def get_gcp_credentials(
        self,
        project_number: str | None = None,
        pool_id: str | None = None,
        provider_id: str | None = None,
        service_account_email: str | None = None,
    ) -> GCPCredentials:
        """Get GCP credentials using OIDC.

        Args:
            project_number: GCP project number.
            pool_id: Workload Identity Pool ID.
            provider_id: Workload Identity Provider ID.
            service_account_email: Service account to impersonate.

        Returns:
            GCP credentials.
        """
        project_number = project_number or os.environ.get("GCP_PROJECT_NUMBER")
        pool_id = pool_id or os.environ.get("GCP_POOL_ID")
        provider_id = provider_id or os.environ.get("GCP_PROVIDER_ID")
        service_account_email = service_account_email or os.environ.get(
            "GCP_SERVICE_ACCOUNT"
        )

        if not all([project_number, pool_id, provider_id, service_account_email]):
            raise OIDCConfigurationError(
                "GCP project_number, pool_id, provider_id, and "
                "service_account_email are required"
            )

        # Build audience for GCP
        audience = (
            f"//iam.googleapis.com/projects/{project_number}/"
            f"locations/global/workloadIdentityPools/{pool_id}/"
            f"providers/{provider_id}"
        )

        token = self.get_token(audience=audience)

        exchanger = GCPTokenExchanger(
            project_number=project_number,
            pool_id=pool_id,
            provider_id=provider_id,
            service_account_email=service_account_email,
            enable_cache=self._config.enable_cache,
        )

        return exchanger.exchange(token)  # type: ignore

    def get_azure_credentials(
        self,
        tenant_id: str | None = None,
        client_id: str | None = None,
        *,
        subscription_id: str | None = None,
    ) -> AzureCredentials:
        """Get Azure credentials using OIDC.

        Args:
            tenant_id: Azure tenant ID.
            client_id: Azure client/app ID.
            subscription_id: Azure subscription ID.

        Returns:
            Azure credentials.
        """
        tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
        subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")

        if not tenant_id or not client_id:
            raise OIDCConfigurationError(
                "Azure tenant_id and client_id are required"
            )

        # Get token with Azure-specific audience
        token = self.get_token(
            audience=f"api://AzureADTokenExchange"
        )

        exchanger = AzureTokenExchanger(
            tenant_id=tenant_id,
            client_id=client_id,
            subscription_id=subscription_id,
            enable_cache=self._config.enable_cache,
        )

        return exchanger.exchange(token)  # type: ignore

    def get_vault_credentials(
        self,
        vault_url: str | None = None,
        role: str | None = None,
        *,
        jwt_auth_path: str = "jwt",
    ) -> "VaultCredentials":
        """Get HashiCorp Vault credentials using OIDC.

        Args:
            vault_url: Vault server URL.
            role: Vault role name.
            jwt_auth_path: JWT auth backend path.

        Returns:
            Vault credentials.
        """
        vault_url = vault_url or os.environ.get("VAULT_ADDR")
        role = role or os.environ.get("VAULT_ROLE")

        if not vault_url or not role:
            raise OIDCConfigurationError(
                "Vault URL and role are required"
            )

        # Get token with Vault-specific audience
        token = self.get_token(audience=vault_url)

        exchanger = VaultTokenExchanger(
            vault_url=vault_url,
            role=role,
            jwt_auth_path=jwt_auth_path,
            enable_cache=self._config.enable_cache,
        )

        return exchanger.exchange(token)  # type: ignore

    def get_credentials(
        self,
        cloud_provider: CloudProvider | str,
        **kwargs: Any,
    ) -> CloudCredentials:
        """Get credentials for the specified cloud provider.

        Args:
            cloud_provider: Target cloud provider.
            **kwargs: Provider-specific configuration.

        Returns:
            Cloud credentials.
        """
        if isinstance(cloud_provider, str):
            cloud_provider = CloudProvider(cloud_provider.lower())

        if cloud_provider == CloudProvider.AWS:
            return self.get_aws_credentials(**kwargs)
        elif cloud_provider == CloudProvider.GCP:
            return self.get_gcp_credentials(**kwargs)
        elif cloud_provider == CloudProvider.AZURE:
            return self.get_azure_credentials(**kwargs)
        elif cloud_provider == CloudProvider.VAULT:
            return self.get_vault_credentials(**kwargs)
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")

    def validate_environment(self, environment: str) -> bool:
        """Validate that current environment matches expected.

        Args:
            environment: Expected environment name.

        Returns:
            True if environment matches.
        """
        return self.claims.environment == environment

    def validate_repository(self, repository: str) -> bool:
        """Validate that current repository matches pattern.

        Args:
            repository: Repository pattern.

        Returns:
            True if repository matches.
        """
        return self.claims.matches_repository(repository)

    def validate_branch(self, branch: str) -> bool:
        """Validate that current branch matches.

        Args:
            branch: Branch name.

        Returns:
            True if branch matches.
        """
        return self.claims.branch_name == branch

    def require_environment(self, environment: str) -> None:
        """Assert that current environment matches.

        Args:
            environment: Required environment name.

        Raises:
            OIDCError: If environment doesn't match.
        """
        if not self.validate_environment(environment):
            raise OIDCError(
                f"Required environment '{environment}', "
                f"but got '{self.claims.environment}'"
            )

    def require_repository(self, repository: str) -> None:
        """Assert that current repository matches.

        Args:
            repository: Required repository pattern.

        Raises:
            OIDCError: If repository doesn't match.
        """
        if not self.validate_repository(repository):
            raise OIDCError(
                f"Required repository '{repository}', "
                f"but got '{self.claims.repository}'"
            )

    def require_branch(self, branch: str) -> None:
        """Assert that current branch matches.

        Args:
            branch: Required branch name.

        Raises:
            OIDCError: If branch doesn't match.
        """
        if not self.validate_branch(branch):
            raise OIDCError(
                f"Required branch '{branch}', "
                f"but got '{self.claims.branch_name}'"
            )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._provider.clear_cache()
        self._cached_claims = None

    def __repr__(self) -> str:
        available = "available" if self.is_available else "unavailable"
        return f"GitHubActionsOIDC(status={available})"
