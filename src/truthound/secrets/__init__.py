"""Enterprise-grade secret management for Truthound.

This module provides a unified interface for retrieving secrets from various
backends including environment variables, files, and cloud secret managers.

Security Features:
    - Lazy loading: Secrets are fetched only when needed
    - Caching with TTL: Reduces API calls to secret managers
    - Automatic redaction: Secrets are masked in logs/repr
    - Rotation support: Cached secrets can be refreshed
    - Audit logging: All secret access is logged (optional)
    - Constant-time comparison: Prevents timing attacks

Supported Backends:
    - Environment variables (default)
    - .env files
    - JSON/YAML configuration files
    - AWS Secrets Manager
    - HashiCorp Vault
    - Azure Key Vault
    - Google Cloud Secret Manager

Usage:
    >>> from truthound.secrets import SecretManager, get_secret
    >>>
    >>> # Simple usage with environment variables
    >>> api_key = get_secret("API_KEY")
    >>>
    >>> # Using secret references in configuration
    >>> from truthound.secrets import resolve_config
    >>> config = {
    ...     "webhook_url": "https://api.example.com",
    ...     "api_key": "${secrets:API_KEY}",  # Resolved at runtime
    ...     "db_password": "${vault:database/credentials#password}",
    ... }
    >>> resolved_config = resolve_config(config)
    >>>
    >>> # Multi-provider setup
    >>> from truthound.secrets import SecretManager, EnvironmentProvider
    >>> from truthound.secrets.cloud import VaultProvider, AWSSecretsManagerProvider
    >>>
    >>> manager = SecretManager()
    >>> manager.add_provider(EnvironmentProvider(), priority=10)
    >>> manager.add_provider(VaultProvider(url="https://vault.example.com"), priority=20)
    >>> manager.add_provider(AWSSecretsManagerProvider(region_name="us-east-1"), priority=30)
    >>>
    >>> # Retrieve with automatic fallback chain
    >>> secret = manager.get("database/password")
    >>>
    >>> # Configuration-based setup
    >>> manager = SecretManager.from_config("secrets.yaml")
"""

from truthound.secrets.base import (
    # Protocol and ABC
    SecretProvider,
    BaseSecretProvider,
    # Value container
    SecretReference,
    SecretValue,
    # Exceptions
    SecretError,
    SecretNotFoundError,
    SecretAccessError,
    SecretProviderError,
)
from truthound.secrets.providers import (
    EnvironmentProvider,
    DotEnvProvider,
    FileProvider,
    ChainedProvider,
)
from truthound.secrets.manager import (
    # Manager
    SecretManager,
    SecretManagerConfig,
    ProviderConfig,
    # Global functions
    get_secret_manager,
    set_secret_manager,
    get_secret,
)
from truthound.secrets.resolver import (
    # Resolver
    SecretResolver,
    ResolverConfig,
    # Global functions
    get_resolver,
    set_resolver,
    resolve_template,
    resolve_config,
    resolve_file,
    # Decorator
    with_secrets,
)
from truthound.secrets.integration import (
    # Mixin
    SecretResolutionMixin,
    SecretAwareConfig,
    # Utilities
    secret_field,
    with_secret_resolution,
    # Credential Helper
    CredentialHelper,
    get_credential_helper,
    get_bearer_token,
    get_basic_auth,
    get_api_key,
)

# Lazy imports for cloud providers (avoid import errors when SDKs not installed)
def __getattr__(name: str):
    """Lazy import cloud providers."""
    cloud_providers = {
        "AWSSecretsManagerProvider",
        "VaultProvider",
        "AzureKeyVaultProvider",
        "GCPSecretManagerProvider",
    }
    if name in cloud_providers:
        from truthound.secrets import cloud
        return getattr(cloud, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "SecretProvider",
    "BaseSecretProvider",
    "SecretReference",
    "SecretValue",
    # Exceptions
    "SecretError",
    "SecretNotFoundError",
    "SecretAccessError",
    "SecretProviderError",
    # Built-in Providers
    "EnvironmentProvider",
    "DotEnvProvider",
    "FileProvider",
    "ChainedProvider",
    # Cloud Providers (lazy loaded)
    "AWSSecretsManagerProvider",
    "VaultProvider",
    "AzureKeyVaultProvider",
    "GCPSecretManagerProvider",
    # Manager
    "SecretManager",
    "SecretManagerConfig",
    "ProviderConfig",
    "get_secret_manager",
    "set_secret_manager",
    "get_secret",
    # Resolver
    "SecretResolver",
    "ResolverConfig",
    "get_resolver",
    "set_resolver",
    "resolve_template",
    "resolve_config",
    "resolve_file",
    "with_secrets",
    # Integration
    "SecretResolutionMixin",
    "SecretAwareConfig",
    "secret_field",
    "with_secret_resolution",
    "CredentialHelper",
    "get_credential_helper",
    "get_bearer_token",
    "get_basic_auth",
    "get_api_key",
]
