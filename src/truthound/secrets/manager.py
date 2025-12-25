"""Secret Manager - Unified interface for secret management.

This module provides SecretManager, the main entry point for secret operations.
It orchestrates multiple providers and handles secret resolution, caching,
and lifecycle management.

Design Principles:
    1. Single Entry Point: One manager for all secret operations
    2. Provider Agnostic: Swap backends without code changes
    3. Configuration-driven: YAML/JSON config or programmatic setup
    4. Audit Ready: Comprehensive logging of secret access
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from truthound.secrets.base import (
    BaseSecretProvider,
    SecretNotFoundError,
    SecretProviderError,
    SecretReference,
    SecretValue,
)
from truthound.secrets.providers import (
    ChainedProvider,
    DotEnvProvider,
    EnvironmentProvider,
    FileProvider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Manager Configuration
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for a secret provider.

    Attributes:
        type: Provider type (env, dotenv, file, vault, aws, azure, gcp).
        name: Optional name override.
        priority: Provider priority (lower = higher priority).
        enabled: Whether provider is enabled.
        options: Provider-specific options.
    """

    type: str
    name: str | None = None
    priority: int = 100
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecretManagerConfig:
    """Configuration for SecretManager.

    Attributes:
        providers: List of provider configurations.
        default_provider: Name of default provider.
        enable_cache: Global cache toggle.
        cache_ttl_seconds: Default cache TTL.
        strict_mode: Raise on missing secrets vs return None.
        audit_enabled: Enable access auditing.
    """

    providers: list[ProviderConfig] = field(default_factory=list)
    default_provider: str | None = None
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    strict_mode: bool = True
    audit_enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecretManagerConfig":
        """Create config from dictionary."""
        providers = []
        for p in data.get("providers", []):
            if isinstance(p, dict):
                providers.append(
                    ProviderConfig(
                        type=p["type"],
                        name=p.get("name"),
                        priority=p.get("priority", 100),
                        enabled=p.get("enabled", True),
                        options=p.get("options", {}),
                    )
                )
        return cls(
            providers=providers,
            default_provider=data.get("default_provider"),
            enable_cache=data.get("enable_cache", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 300),
            strict_mode=data.get("strict_mode", True),
            audit_enabled=data.get("audit_enabled", True),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SecretManagerConfig":
        """Load config from JSON or YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = path.read_text()
        suffix = path.suffix.lower()

        if suffix == ".json":
            data = json.loads(content)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        return cls.from_dict(data)


# =============================================================================
# Secret Manager
# =============================================================================


class SecretManager:
    """Unified secret management interface.

    SecretManager provides a single point of access for all secret operations,
    managing multiple providers with automatic fallback and caching.

    Features:
        - Multiple provider support with priority ordering
        - Automatic fallback chain
        - Reference resolution (${secrets:KEY})
        - Caching with configurable TTL
        - Audit logging
        - Lazy provider initialization

    Example:
        >>> # Simple setup with environment and .env
        >>> manager = SecretManager.create_default()
        >>> secret = manager.get("DATABASE_URL")

        >>> # Multi-provider setup
        >>> manager = SecretManager()
        >>> manager.add_provider(EnvironmentProvider(), priority=10)
        >>> manager.add_provider(VaultProvider(url="..."), priority=20)
        >>> manager.add_provider(AWSSecretsManagerProvider(), priority=30)
        >>>
        >>> # Retrieves from first provider that has the secret
        >>> secret = manager.get("api/credentials")

        >>> # Config-based setup
        >>> manager = SecretManager.from_config("secrets.yaml")
    """

    def __init__(
        self,
        config: SecretManagerConfig | None = None,
        *,
        audit_callback: Callable[[str, str, str, bool], None] | None = None,
    ) -> None:
        """Initialize secret manager.

        Args:
            config: Manager configuration.
            audit_callback: Callback for audit logs (key, provider, action, success).
        """
        self._config = config or SecretManagerConfig()
        self._providers: dict[str, tuple[int, BaseSecretProvider]] = {}
        self._default_provider: str | None = self._config.default_provider
        self._audit_callback = audit_callback
        self._initialized = False

        # Initialize providers from config
        if self._config.providers:
            self._init_from_config()

    def _init_from_config(self) -> None:
        """Initialize providers from configuration."""
        from truthound.secrets.cloud import (
            AWSSecretsManagerProvider,
            AzureKeyVaultProvider,
            GCPSecretManagerProvider,
            VaultProvider,
        )

        provider_classes = {
            "env": EnvironmentProvider,
            "environment": EnvironmentProvider,
            "dotenv": DotEnvProvider,
            "file": FileProvider,
            "vault": VaultProvider,
            "aws": AWSSecretsManagerProvider,
            "azure": AzureKeyVaultProvider,
            "gcp": GCPSecretManagerProvider,
        }

        for pconfig in self._config.providers:
            if not pconfig.enabled:
                continue

            provider_class = provider_classes.get(pconfig.type.lower())
            if not provider_class:
                logger.warning(f"Unknown provider type: {pconfig.type}")
                continue

            try:
                # Apply global settings
                options = pconfig.options.copy()
                if "cache_ttl_seconds" not in options:
                    options["cache_ttl_seconds"] = self._config.cache_ttl_seconds
                if "enable_cache" not in options:
                    options["enable_cache"] = self._config.enable_cache

                provider = provider_class(**options)
                name = pconfig.name or provider.name
                self.add_provider(provider, name=name, priority=pconfig.priority)

            except Exception as e:
                logger.error(f"Failed to initialize provider {pconfig.type}: {e}")
                if self._config.strict_mode:
                    raise

        self._initialized = True

    @classmethod
    def create_default(
        cls,
        *,
        include_env: bool = True,
        include_dotenv: bool = True,
        dotenv_paths: list[str | Path] | None = None,
    ) -> "SecretManager":
        """Create manager with default provider chain.

        Args:
            include_env: Include environment provider.
            include_dotenv: Include .env file provider.
            dotenv_paths: Custom .env file paths.

        Returns:
            Configured SecretManager.
        """
        manager = cls()

        if include_env:
            manager.add_provider(EnvironmentProvider(), priority=10)

        if include_dotenv:
            if dotenv_paths:
                manager.add_provider(
                    DotEnvProvider(paths=dotenv_paths), priority=20
                )
            else:
                manager.add_provider(DotEnvProvider(), priority=20)

        return manager

    @classmethod
    def from_config(cls, path: str | Path) -> "SecretManager":
        """Create manager from configuration file.

        Args:
            path: Path to configuration file (JSON or YAML).

        Returns:
            Configured SecretManager.
        """
        config = SecretManagerConfig.from_file(path)
        return cls(config)

    def add_provider(
        self,
        provider: BaseSecretProvider,
        *,
        name: str | None = None,
        priority: int = 100,
    ) -> None:
        """Add a provider to the manager.

        Args:
            provider: The provider to add.
            name: Optional name override.
            priority: Provider priority (lower = tried first).
        """
        provider_name = name or provider.name

        if provider_name in self._providers:
            logger.warning(f"Replacing existing provider: {provider_name}")

        self._providers[provider_name] = (priority, provider)

        # Set default if first provider
        if self._default_provider is None:
            self._default_provider = provider_name

        logger.debug(
            f"Added provider '{provider_name}' with priority {priority}"
        )

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name.

        Args:
            name: Provider name.

        Returns:
            True if removed, False if not found.
        """
        if name in self._providers:
            del self._providers[name]
            if self._default_provider == name:
                self._default_provider = None
            return True
        return False

    def get_provider(self, name: str) -> BaseSecretProvider | None:
        """Get a provider by name.

        Args:
            name: Provider name.

        Returns:
            The provider or None.
        """
        entry = self._providers.get(name)
        return entry[1] if entry else None

    def list_providers(self) -> list[tuple[str, int]]:
        """List all providers with their priorities.

        Returns:
            List of (name, priority) tuples, sorted by priority.
        """
        return sorted(
            [(name, priority) for name, (priority, _) in self._providers.items()],
            key=lambda x: x[1],
        )

    def _sorted_providers(self) -> Iterator[tuple[str, BaseSecretProvider]]:
        """Iterate providers in priority order."""
        sorted_items = sorted(
            self._providers.items(), key=lambda x: x[1][0]
        )
        for name, (_, provider) in sorted_items:
            yield name, provider

    def get(
        self,
        key: str,
        *,
        provider: str | None = None,
        version: str | None = None,
        field: str | None = None,
        default: str | None = None,
    ) -> SecretValue | None:
        """Retrieve a secret.

        Args:
            key: Secret key or path.
            provider: Specific provider to use.
            version: Secret version.
            field: Field for structured secrets.
            default: Default value if not found.

        Returns:
            SecretValue or None (if default not set and not strict).

        Raises:
            SecretNotFoundError: If strict mode and secret not found.
        """
        # Use specific provider if requested
        if provider:
            entry = self._providers.get(provider)
            if not entry:
                if self._config.strict_mode and default is None:
                    raise SecretProviderError(
                        "manager", f"Provider not found: {provider}"
                    )
                return self._wrap_default(key, default)

            _, prov = entry
            try:
                secret = prov.get(key, version, field)
                self._audit(key, provider, "get", True)
                return secret
            except SecretNotFoundError:
                self._audit(key, provider, "get", False)
                if default is not None:
                    return self._wrap_default(key, default)
                if self._config.strict_mode:
                    raise
                return None

        # Try providers in priority order
        for prov_name, prov in self._sorted_providers():
            try:
                if prov.supports_key(key):
                    secret = prov.get(key, version, field)
                    self._audit(key, prov_name, "get", True)
                    return secret
            except SecretNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Provider {prov_name} error for '{key}': {e}")
                continue

        # Not found in any provider
        self._audit(key, "all", "get", False)

        if default is not None:
            return self._wrap_default(key, default)

        if self._config.strict_mode:
            raise SecretNotFoundError(key, "manager")

        return None

    def get_value(
        self,
        key: str,
        *,
        provider: str | None = None,
        version: str | None = None,
        field: str | None = None,
        default: str | None = None,
    ) -> str | None:
        """Get secret value as string (convenience method).

        Args:
            key: Secret key.
            provider: Specific provider.
            version: Secret version.
            field: Field for structured secrets.
            default: Default value.

        Returns:
            Secret value string or None.
        """
        secret = self.get(
            key, provider=provider, version=version, field=field, default=default
        )
        return secret.get_value() if secret else None

    def resolve_reference(self, ref: SecretReference) -> SecretValue | None:
        """Resolve a secret reference.

        Args:
            ref: SecretReference to resolve.

        Returns:
            Resolved SecretValue or None.
        """
        return self.get(
            ref.key,
            provider=ref.provider,
            version=ref.version,
            field=ref.field,
            default=ref.default,
        )

    def resolve_string(self, text: str) -> str:
        """Resolve all secret references in a string.

        Replaces ${secrets:KEY}, ${env:VAR}, etc. with actual values.

        Args:
            text: String potentially containing references.

        Returns:
            String with references resolved.
        """
        refs = SecretReference.find_all(text)

        if not refs:
            return text

        result = text
        for ref in refs:
            try:
                secret = self.resolve_reference(ref)
                if secret:
                    # Replace the reference in the string
                    result = result.replace(ref.to_string(), secret.get_value())
                elif ref.default:
                    result = result.replace(ref.to_string(), ref.default)
            except SecretNotFoundError:
                if ref.default:
                    result = result.replace(ref.to_string(), ref.default)
                elif self._config.strict_mode:
                    raise

        return result

    def resolve_dict(
        self,
        data: dict[str, Any],
        *,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Resolve secret references in a dictionary.

        Args:
            data: Dictionary potentially containing references.
            recursive: Recursively process nested dicts/lists.

        Returns:
            Dictionary with references resolved.
        """

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                return self.resolve_string(value)
            elif isinstance(value, dict) and recursive:
                return self.resolve_dict(value, recursive=True)
            elif isinstance(value, list) and recursive:
                return [resolve_value(item) for item in value]
            return value

        return {key: resolve_value(val) for key, val in data.items()}

    def _wrap_default(self, key: str, default: str) -> SecretValue:
        """Wrap a default value as SecretValue."""
        return SecretValue(
            value=default,
            provider="default",
            key=key,
            metadata={"is_default": True},
        )

    def _audit(
        self, key: str, provider: str, action: str, success: bool
    ) -> None:
        """Log audit event."""
        if not self._config.audit_enabled:
            return

        logger.debug(
            f"Secret access: key={key}, provider={provider}, "
            f"action={action}, success={success}"
        )

        if self._audit_callback:
            try:
                self._audit_callback(key, provider, action, success)
            except Exception:
                pass

    def clear_cache(self, provider: str | None = None) -> None:
        """Clear cached secrets.

        Args:
            provider: Specific provider to clear, or None for all.
        """
        if provider:
            entry = self._providers.get(provider)
            if entry:
                entry[1].clear_cache()
        else:
            for _, prov in self._sorted_providers():
                prov.clear_cache()


# =============================================================================
# Global Manager Instance
# =============================================================================

_global_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance.

    Creates a default manager if none exists.

    Returns:
        Global SecretManager instance.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = SecretManager.create_default()
    return _global_manager


def set_secret_manager(manager: SecretManager) -> None:
    """Set the global secret manager instance.

    Args:
        manager: SecretManager to use globally.
    """
    global _global_manager
    _global_manager = manager


def get_secret(
    key: str,
    *,
    provider: str | None = None,
    version: str | None = None,
    field: str | None = None,
    default: str | None = None,
) -> str | None:
    """Convenience function to get a secret value.

    Uses the global secret manager.

    Args:
        key: Secret key.
        provider: Specific provider.
        version: Secret version.
        field: Field for structured secrets.
        default: Default value.

    Returns:
        Secret value or None.
    """
    return get_secret_manager().get_value(
        key, provider=provider, version=version, field=field, default=default
    )
