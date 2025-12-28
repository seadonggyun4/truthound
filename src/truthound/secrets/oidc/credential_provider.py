"""OIDC Credential Provider - High-level API.

This module provides a high-level interface for obtaining cloud credentials
using OIDC authentication, integrating with the Truthound secrets framework.

Features:
    - Automatic CI provider detection
    - Unified credential management
    - Integration with SecretProvider interface
    - Context manager and decorator support
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeVar

from truthound.secrets.oidc.base import (
    CloudCredentials,
    CloudProvider,
    OIDCError,
    OIDCProviderNotAvailableError,
    OIDCToken,
)
from truthound.secrets.oidc.providers import (
    BaseOIDCProvider,
    detect_ci_oidc_provider,
    is_oidc_available,
)
from truthound.secrets.oidc.exchangers import (
    AWSTokenExchanger,
    AzureTokenExchanger,
    BaseTokenExchanger,
    GCPTokenExchanger,
    VaultTokenExchanger,
    create_token_exchanger,
)

if TYPE_CHECKING:
    from truthound.secrets.base import SecretValue


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OIDCCredentialConfig:
    """Configuration for OIDC credential provider.

    Attributes:
        cloud_provider: Target cloud provider (aws, gcp, azure, vault).
        audience: OIDC token audience.
        auto_detect_ci: Automatically detect CI provider.
        enable_cache: Cache credentials.
        cache_ttl_seconds: Credential cache TTL.
        fallback_to_env: Fall back to environment credentials if OIDC fails.

        # AWS-specific
        aws_role_arn: IAM role ARN to assume.
        aws_session_name: Session name for assumed role.
        aws_region: AWS region.

        # GCP-specific
        gcp_project_number: GCP project number.
        gcp_pool_id: Workload Identity Pool ID.
        gcp_provider_id: Workload Identity Provider ID.
        gcp_service_account: Service account to impersonate.

        # Azure-specific
        azure_tenant_id: Azure tenant ID.
        azure_client_id: Azure client/app ID.
        azure_subscription_id: Azure subscription ID.

        # Vault-specific
        vault_url: Vault server URL.
        vault_role: Vault role name.
        vault_jwt_path: JWT auth backend path.
    """

    cloud_provider: CloudProvider | str = CloudProvider.AWS
    audience: str | None = None
    auto_detect_ci: bool = True
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    fallback_to_env: bool = True

    # AWS
    aws_role_arn: str = ""
    aws_session_name: str = "truthound-oidc"
    aws_region: str = ""

    # GCP
    gcp_project_number: str = ""
    gcp_pool_id: str = ""
    gcp_provider_id: str = ""
    gcp_service_account: str = ""

    # Azure
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_subscription_id: str = ""

    # Vault
    vault_url: str = ""
    vault_role: str = ""
    vault_jwt_path: str = "jwt"

    def __post_init__(self) -> None:
        if isinstance(self.cloud_provider, str):
            self.cloud_provider = CloudProvider(self.cloud_provider.lower())


# =============================================================================
# OIDC Credential Provider
# =============================================================================


class OIDCCredentialProvider:
    """High-level OIDC credential provider.

    Combines OIDC token retrieval and cloud provider token exchange
    into a single, easy-to-use interface.

    Example:
        >>> # Simple usage with auto-detection
        >>> provider = OIDCCredentialProvider(
        ...     cloud_provider="aws",
        ...     aws_role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
        >>> creds = provider.get_credentials()
        >>>
        >>> # Use credentials
        >>> import boto3
        >>> session = boto3.Session(**creds.to_boto3_session_credentials())

    Example with config:
        >>> config = OIDCCredentialConfig(
        ...     cloud_provider="gcp",
        ...     gcp_project_number="123456789",
        ...     gcp_pool_id="my-pool",
        ...     gcp_provider_id="github",
        ...     gcp_service_account="sa@project.iam.gserviceaccount.com",
        ... )
        >>> provider = OIDCCredentialProvider(config=config)
        >>> creds = provider.get_credentials()
    """

    def __init__(
        self,
        cloud_provider: CloudProvider | str | None = None,
        *,
        oidc_provider: BaseOIDCProvider | None = None,
        token_exchanger: BaseTokenExchanger | None = None,
        config: OIDCCredentialConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OIDC credential provider.

        Args:
            cloud_provider: Target cloud provider.
            oidc_provider: OIDC identity provider (auto-detected if None).
            token_exchanger: Token exchanger (created from config if None).
            config: Full configuration object.
            **kwargs: Provider-specific configuration.
        """
        # Build config from parameters
        if config is None:
            config = OIDCCredentialConfig(**kwargs)
            if cloud_provider:
                config.cloud_provider = (
                    CloudProvider(cloud_provider.lower())
                    if isinstance(cloud_provider, str)
                    else cloud_provider
                )

        self._config = config
        self._oidc_provider = oidc_provider
        self._token_exchanger = token_exchanger
        self._cached_credentials: CloudCredentials | None = None

    @property
    def cloud_provider(self) -> CloudProvider:
        """Get target cloud provider."""
        return (
            CloudProvider(self._config.cloud_provider)
            if isinstance(self._config.cloud_provider, str)
            else self._config.cloud_provider
        )

    @property
    def oidc_provider(self) -> BaseOIDCProvider:
        """Get OIDC identity provider (lazy initialization)."""
        if self._oidc_provider is None:
            if self._config.auto_detect_ci:
                self._oidc_provider = detect_ci_oidc_provider()
                if self._oidc_provider is None:
                    raise OIDCProviderNotAvailableError(
                        "auto-detect",
                        "No supported CI environment detected",
                    )
            else:
                raise OIDCError("OIDC provider not configured")
        return self._oidc_provider

    @property
    def token_exchanger(self) -> BaseTokenExchanger:
        """Get token exchanger (lazy initialization)."""
        if self._token_exchanger is None:
            self._token_exchanger = self._create_token_exchanger()
        return self._token_exchanger

    def _create_token_exchanger(self) -> BaseTokenExchanger:
        """Create token exchanger from configuration."""
        cloud = self.cloud_provider

        if cloud == CloudProvider.AWS:
            return AWSTokenExchanger(
                role_arn=self._config.aws_role_arn,
                session_name=self._config.aws_session_name,
                region=self._config.aws_region or None,
                cache_ttl_seconds=self._config.cache_ttl_seconds,
                enable_cache=self._config.enable_cache,
            )

        elif cloud == CloudProvider.GCP:
            return GCPTokenExchanger(
                project_number=self._config.gcp_project_number,
                pool_id=self._config.gcp_pool_id,
                provider_id=self._config.gcp_provider_id,
                service_account_email=self._config.gcp_service_account,
                cache_ttl_seconds=self._config.cache_ttl_seconds,
                enable_cache=self._config.enable_cache,
            )

        elif cloud == CloudProvider.AZURE:
            return AzureTokenExchanger(
                tenant_id=self._config.azure_tenant_id,
                client_id=self._config.azure_client_id,
                subscription_id=self._config.azure_subscription_id,
                cache_ttl_seconds=self._config.cache_ttl_seconds,
                enable_cache=self._config.enable_cache,
            )

        elif cloud == CloudProvider.VAULT:
            return VaultTokenExchanger(
                vault_url=self._config.vault_url,
                role=self._config.vault_role,
                jwt_auth_path=self._config.vault_jwt_path,
                cache_ttl_seconds=self._config.cache_ttl_seconds,
                enable_cache=self._config.enable_cache,
            )

        else:
            raise ValueError(f"Unsupported cloud provider: {cloud}")

    def is_available(self) -> bool:
        """Check if OIDC authentication is available."""
        try:
            return self.oidc_provider.is_available()
        except OIDCError:
            return False

    def get_oidc_token(self, audience: str | None = None) -> OIDCToken:
        """Get OIDC token from CI provider.

        Args:
            audience: Token audience (uses config default if not provided).

        Returns:
            OIDC token.
        """
        effective_audience = audience or self._config.audience
        return self.oidc_provider.get_token(effective_audience)

    def get_credentials(self, force_refresh: bool = False) -> CloudCredentials:
        """Get cloud credentials using OIDC.

        Args:
            force_refresh: Force credential refresh even if cached.

        Returns:
            Cloud credentials.

        Raises:
            OIDCError: If credential retrieval fails.
        """
        # Check cache
        if not force_refresh and self._cached_credentials:
            if not self._cached_credentials.is_expired:
                remaining = self._cached_credentials.time_until_expiry
                if remaining and remaining.total_seconds() > 60:
                    logger.debug(
                        f"Using cached {self.cloud_provider} credentials "
                        f"(expires in {remaining.total_seconds():.0f}s)"
                    )
                    return self._cached_credentials

        # Get OIDC token
        token = self.get_oidc_token()

        # Exchange for cloud credentials
        credentials = self.token_exchanger.exchange(token)

        # Cache credentials
        if self._config.enable_cache:
            self._cached_credentials = credentials

        return credentials

    def clear_cache(self) -> None:
        """Clear cached credentials."""
        self._cached_credentials = None
        if self._oidc_provider:
            self._oidc_provider.clear_cache()
        if self._token_exchanger:
            self._token_exchanger.clear_cache()

    def __repr__(self) -> str:
        provider_name = (
            self._oidc_provider.name if self._oidc_provider else "auto-detect"
        )
        return (
            f"OIDCCredentialProvider(cloud={self.cloud_provider}, "
            f"ci={provider_name})"
        )


# =============================================================================
# Secret Provider Integration
# =============================================================================


class OIDCSecretProvider:
    """Secret provider that uses OIDC credentials to access secrets.

    Implements the SecretProvider protocol, allowing OIDC-based credentials
    to be used seamlessly with the Truthound secrets framework.

    Example:
        >>> from truthound.secrets import SecretManager
        >>> from truthound.secrets.oidc import OIDCSecretProvider
        >>>
        >>> # Create OIDC-backed secret provider
        >>> oidc_provider = OIDCSecretProvider(
        ...     cloud_provider="aws",
        ...     aws_role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
        >>>
        >>> # Register with secret manager
        >>> manager = SecretManager()
        >>> manager.add_provider(oidc_provider, priority=100)
        >>>
        >>> # Get secrets using OIDC credentials
        >>> secret = manager.get("my-secret")
    """

    def __init__(
        self,
        cloud_provider: CloudProvider | str = CloudProvider.AWS,
        *,
        credential_provider: OIDCCredentialProvider | None = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize OIDC secret provider.

        Args:
            cloud_provider: Target cloud provider.
            credential_provider: Pre-configured credential provider.
            prefix: Key prefix to add.
            **kwargs: Passed to OIDCCredentialProvider.
        """
        self._prefix = prefix
        self._credential_provider = credential_provider or OIDCCredentialProvider(
            cloud_provider=cloud_provider,
            **kwargs,
        )
        self._cloud_provider = (
            CloudProvider(cloud_provider)
            if isinstance(cloud_provider, str)
            else cloud_provider
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return f"oidc-{self._cloud_provider}"

    def get(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> "SecretValue":
        """Retrieve a secret using OIDC credentials.

        Args:
            key: Secret key.
            version: Secret version.
            field: Specific field.

        Returns:
            SecretValue containing the secret.
        """
        from truthound.secrets.base import SecretValue, SecretNotFoundError

        # Get credentials
        credentials = self._credential_provider.get_credentials()

        # Normalize key
        full_key = f"{self._prefix}{key}" if self._prefix else key

        # Retrieve based on cloud provider
        if self._cloud_provider == CloudProvider.AWS:
            value = self._get_aws_secret(credentials, full_key, version, field)
        elif self._cloud_provider == CloudProvider.GCP:
            value = self._get_gcp_secret(credentials, full_key, version, field)
        elif self._cloud_provider == CloudProvider.AZURE:
            value = self._get_azure_secret(credentials, full_key, version, field)
        elif self._cloud_provider == CloudProvider.VAULT:
            value = self._get_vault_secret(credentials, full_key, version, field)
        else:
            raise SecretNotFoundError(key, self.name)

        return SecretValue(
            value=value,
            provider=self.name,
            key=key,
            metadata={
                "cloud_provider": str(self._cloud_provider),
                "version": version,
                "field": field,
            },
        )

    def _get_aws_secret(
        self,
        credentials: CloudCredentials,
        key: str,
        version: str | None,
        field: str | None,
    ) -> str:
        """Get secret from AWS Secrets Manager."""
        # This would use boto3 with the OIDC credentials
        # For now, placeholder that documents the approach
        raise NotImplementedError(
            "AWS Secrets Manager integration requires boto3. "
            "Use AWSSecretsManagerProvider with OIDC credentials instead."
        )

    def _get_gcp_secret(
        self,
        credentials: CloudCredentials,
        key: str,
        version: str | None,
        field: str | None,
    ) -> str:
        """Get secret from GCP Secret Manager."""
        raise NotImplementedError(
            "GCP Secret Manager integration requires google-cloud-secret-manager. "
            "Use GCPSecretManagerProvider with OIDC credentials instead."
        )

    def _get_azure_secret(
        self,
        credentials: CloudCredentials,
        key: str,
        version: str | None,
        field: str | None,
    ) -> str:
        """Get secret from Azure Key Vault."""
        raise NotImplementedError(
            "Azure Key Vault integration requires azure-keyvault-secrets. "
            "Use AzureKeyVaultProvider with OIDC credentials instead."
        )

    def _get_vault_secret(
        self,
        credentials: CloudCredentials,
        key: str,
        version: str | None,
        field: str | None,
    ) -> str:
        """Get secret from HashiCorp Vault."""
        raise NotImplementedError(
            "Vault secret retrieval requires hvac. "
            "Use VaultProvider with OIDC credentials instead."
        )

    def supports_key(self, key: str) -> bool:
        """Check if this provider can handle the given key."""
        if self._prefix:
            return key.startswith(self._prefix)
        return True


# =============================================================================
# Utility Functions
# =============================================================================


def get_oidc_credentials(
    cloud_provider: CloudProvider | str = CloudProvider.AWS,
    **kwargs: Any,
) -> CloudCredentials:
    """Get cloud credentials using OIDC (convenience function).

    Args:
        cloud_provider: Target cloud provider.
        **kwargs: Provider-specific configuration.

    Returns:
        Cloud credentials.

    Example:
        >>> creds = get_oidc_credentials(
        ...     cloud_provider="aws",
        ...     aws_role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
        >>> print(creds.access_key_id)
    """
    provider = OIDCCredentialProvider(
        cloud_provider=cloud_provider,
        **kwargs,
    )
    return provider.get_credentials()


@contextmanager
def with_oidc_credentials(
    cloud_provider: CloudProvider | str = CloudProvider.AWS,
    set_environment: bool = True,
    **kwargs: Any,
) -> Generator[CloudCredentials, None, None]:
    """Context manager that provides OIDC credentials.

    Optionally sets environment variables for the duration of the context.

    Args:
        cloud_provider: Target cloud provider.
        set_environment: Whether to set environment variables.
        **kwargs: Provider-specific configuration.

    Yields:
        Cloud credentials.

    Example:
        >>> with with_oidc_credentials(
        ...     cloud_provider="aws",
        ...     aws_role_arn="arn:aws:iam::123456789012:role/my-role",
        ... ) as creds:
        ...     # Environment variables are set
        ...     import boto3
        ...     s3 = boto3.client("s3")  # Uses OIDC credentials
    """
    credentials = get_oidc_credentials(cloud_provider, **kwargs)

    if set_environment:
        # Save original environment
        original_env: dict[str, str | None] = {}
        env_vars = credentials.to_environment_variables()

        for key in env_vars:
            original_env[key] = os.environ.get(key)

        # Set new environment
        os.environ.update(env_vars)

    try:
        yield credentials
    finally:
        if set_environment:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def oidc_credentials(
    cloud_provider: CloudProvider | str = CloudProvider.AWS,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Decorator that injects OIDC credentials as first argument.

    Args:
        cloud_provider: Target cloud provider.
        **kwargs: Provider-specific configuration.

    Returns:
        Decorator function.

    Example:
        >>> @oidc_credentials("aws", aws_role_arn="arn:aws:iam::...")
        ... def my_function(credentials: AWSCredentials) -> None:
        ...     print(credentials.access_key_id)
    """
    provider = OIDCCredentialProvider(
        cloud_provider=cloud_provider,
        **kwargs,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            credentials = provider.get_credentials()
            return func(credentials, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
