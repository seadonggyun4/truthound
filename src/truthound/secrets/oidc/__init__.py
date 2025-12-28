"""OpenID Connect (OIDC) Authentication Module.

This module provides OIDC-based authentication for CI/CD environments,
enabling keyless authentication to cloud providers.

Key Features:
    - GitHub Actions OIDC integration
    - GitLab CI OIDC integration
    - Cloud provider token exchange (AWS, GCP, Azure)
    - Token caching and automatic refresh
    - Configurable audience and claims

Supported Identity Providers:
    - GitHub Actions (actions.githubusercontent.com)
    - GitLab CI/CD (gitlab.com)
    - Bitbucket Pipelines
    - CircleCI OIDC

Supported Cloud Providers:
    - AWS STS (AssumeRoleWithWebIdentity)
    - Google Cloud Workload Identity
    - Azure Federated Credentials
    - HashiCorp Vault JWT Auth

Example (GitHub Actions to AWS):
    >>> from truthound.secrets.oidc import (
    ...     GitHubActionsOIDCProvider,
    ...     AWSTokenExchanger,
    ...     OIDCCredentialProvider,
    ... )
    >>>
    >>> # Create OIDC provider for GitHub Actions
    >>> oidc_provider = GitHubActionsOIDCProvider(
    ...     audience="sts.amazonaws.com",
    ... )
    >>>
    >>> # Create AWS token exchanger
    >>> aws_exchanger = AWSTokenExchanger(
    ...     role_arn="arn:aws:iam::123456789012:role/my-github-role",
    ...     session_name="truthound-ci",
    ... )
    >>>
    >>> # Get AWS credentials
    >>> credentials = OIDCCredentialProvider(
    ...     oidc_provider=oidc_provider,
    ...     token_exchanger=aws_exchanger,
    ... ).get_credentials()
    >>>
    >>> print(credentials.access_key_id)

Example (GitHub Actions to GCP):
    >>> from truthound.secrets.oidc import (
    ...     GitHubActionsOIDCProvider,
    ...     GCPTokenExchanger,
    ... )
    >>>
    >>> oidc_provider = GitHubActionsOIDCProvider(
    ...     audience="https://iam.googleapis.com/projects/123/locations/global/"
    ...              "workloadIdentityPools/my-pool/providers/github",
    ... )
    >>>
    >>> gcp_exchanger = GCPTokenExchanger(
    ...     project_number="123456789",
    ...     pool_id="my-pool",
    ...     provider_id="github",
    ...     service_account_email="sa@project.iam.gserviceaccount.com",
    ... )
    >>>
    >>> credentials = OIDCCredentialProvider(
    ...     oidc_provider=oidc_provider,
    ...     token_exchanger=gcp_exchanger,
    ... ).get_credentials()
"""

from truthound.secrets.oidc.base import (
    # Core types
    OIDCToken,
    OIDCClaims,
    CloudCredentials,
    AWSCredentials,
    GCPCredentials,
    AzureCredentials,
    # Protocols
    OIDCProvider,
    TokenExchanger,
    # Base classes
    BaseOIDCProvider,
    BaseTokenExchanger,
    # Exceptions
    OIDCError,
    OIDCTokenError,
    OIDCExchangeError,
    OIDCConfigurationError,
    OIDCProviderNotAvailableError,
)

from truthound.secrets.oidc.providers import (
    # CI Providers
    GitHubActionsOIDCProvider,
    GitLabCIOIDCProvider,
    CircleCIOIDCProvider,
    BitbucketPipelinesOIDCProvider,
    # Generic provider
    GenericOIDCProvider,
    # Detection
    detect_ci_oidc_provider,
    is_oidc_available,
)

from truthound.secrets.oidc.exchangers import (
    # Cloud Exchangers
    AWSTokenExchanger,
    GCPTokenExchanger,
    AzureTokenExchanger,
    VaultTokenExchanger,
    # Factory
    create_token_exchanger,
)

from truthound.secrets.oidc.credential_provider import (
    # Main credential provider
    OIDCCredentialProvider,
    OIDCCredentialConfig,
    # Secret provider integration
    OIDCSecretProvider,
    # Utilities
    get_oidc_credentials,
    with_oidc_credentials,
)


__all__ = [
    # Core types
    "OIDCToken",
    "OIDCClaims",
    "CloudCredentials",
    "AWSCredentials",
    "GCPCredentials",
    "AzureCredentials",
    # Protocols
    "OIDCProvider",
    "TokenExchanger",
    # Base classes
    "BaseOIDCProvider",
    "BaseTokenExchanger",
    # Exceptions
    "OIDCError",
    "OIDCTokenError",
    "OIDCExchangeError",
    "OIDCConfigurationError",
    "OIDCProviderNotAvailableError",
    # CI Providers
    "GitHubActionsOIDCProvider",
    "GitLabCIOIDCProvider",
    "CircleCIOIDCProvider",
    "BitbucketPipelinesOIDCProvider",
    "GenericOIDCProvider",
    "detect_ci_oidc_provider",
    "is_oidc_available",
    # Cloud Exchangers
    "AWSTokenExchanger",
    "GCPTokenExchanger",
    "AzureTokenExchanger",
    "VaultTokenExchanger",
    "create_token_exchanger",
    # Credential Provider
    "OIDCCredentialProvider",
    "OIDCCredentialConfig",
    "OIDCSecretProvider",
    "get_oidc_credentials",
    "with_oidc_credentials",
]
