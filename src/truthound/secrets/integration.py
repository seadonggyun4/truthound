"""Secret integration utilities for checkpoint actions.

This module provides mixins and utilities for integrating secret management
with checkpoint actions, enabling secure credential handling in webhooks,
notifications, and other external integrations.

Design Notes:
    - Mixin-based approach for easy integration with existing actions
    - Lazy resolution to avoid unnecessary secret fetches
    - Support for both config-time and runtime resolution
    - Backward compatible with existing action implementations
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Generic, TYPE_CHECKING

from truthound.secrets.base import SecretReference, SecretNotFoundError
from truthound.secrets.manager import SecretManager, get_secret_manager
from truthound.secrets.resolver import SecretResolver, get_resolver, ResolverConfig

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionConfig

ConfigT = TypeVar("ConfigT", bound="ActionConfig")


# =============================================================================
# Secret Resolution Mixin
# =============================================================================


class SecretResolutionMixin:
    """Mixin to add secret resolution capabilities to actions.

    This mixin adds methods for resolving secrets in configuration fields,
    supporting both ${secrets:KEY} references and direct secret lookups.

    Usage:
        >>> class MyAction(SecretResolutionMixin, BaseAction[MyConfig]):
        ...     def _execute(self, result):
        ...         # Resolve secrets in config before use
        ...         api_key = self.resolve_secret(self._config.api_key)
        ...         url = self.resolve_secret(self._config.webhook_url)
        ...         ...
    """

    _secret_manager: SecretManager | None = None
    _secret_resolver: SecretResolver | None = None

    def set_secret_manager(self, manager: SecretManager) -> None:
        """Set a custom secret manager for this action.

        Args:
            manager: SecretManager instance to use.
        """
        self._secret_manager = manager
        self._secret_resolver = None  # Reset resolver to use new manager

    def get_secret_manager(self) -> SecretManager:
        """Get the secret manager for this action.

        Returns:
            SecretManager instance.
        """
        if self._secret_manager is None:
            self._secret_manager = get_secret_manager()
        return self._secret_manager

    def get_secret_resolver(self) -> SecretResolver:
        """Get the secret resolver for this action.

        Returns:
            SecretResolver instance.
        """
        if self._secret_resolver is None:
            self._secret_resolver = SecretResolver(
                manager=self.get_secret_manager(),
                config=ResolverConfig(strict=False),
            )
        return self._secret_resolver

    def resolve_secret(
        self,
        value: str | None,
        *,
        default: str | None = None,
    ) -> str | None:
        """Resolve a potential secret reference.

        If the value is a secret reference (${secrets:KEY}), resolves it.
        Otherwise, returns the value as-is.

        Args:
            value: Value that may be a secret reference.
            default: Default if secret not found.

        Returns:
            Resolved value or default.
        """
        if value is None:
            return default

        if not isinstance(value, str):
            return str(value)

        if "${" not in value:
            return value

        try:
            return self.get_secret_resolver().resolve_template(value)
        except SecretNotFoundError:
            return default if default is not None else value

    def resolve_secrets_in_dict(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve all secret references in a dictionary.

        Args:
            data: Dictionary potentially containing secret references.

        Returns:
            Dictionary with secrets resolved.
        """
        return self.get_secret_resolver().resolve_config(data)

    def get_secret_value(
        self,
        key: str,
        *,
        provider: str | None = None,
        default: str | None = None,
    ) -> str | None:
        """Get a secret value directly by key.

        Args:
            key: Secret key.
            provider: Specific provider to use.
            default: Default if not found.

        Returns:
            Secret value or default.
        """
        return self.get_secret_manager().get_value(
            key, provider=provider, default=default
        )


# =============================================================================
# Config Resolution Utilities
# =============================================================================


@dataclass
class SecretAwareConfig:
    """Base configuration with secret resolution markers.

    Extend this class to create configurations with fields that
    support secret references. Mark sensitive fields with the
    `secret_field` decorator.

    Example:
        >>> @dataclass
        ... class MyWebhookConfig(SecretAwareConfig):
        ...     url: str = ""
        ...     api_key: str = secret_field("")  # Marked as secret
        ...     auth_token: str = secret_field("")
    """

    _secret_fields: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def resolve_secrets(
        self,
        resolver: SecretResolver | None = None,
    ) -> "SecretAwareConfig":
        """Resolve all secret references in this config.

        Args:
            resolver: SecretResolver to use. Uses global if None.

        Returns:
            Self with secrets resolved.
        """
        if resolver is None:
            resolver = get_resolver()

        for field_name in self._secret_fields:
            value = getattr(self, field_name, None)
            if value and isinstance(value, str) and "${" in value:
                try:
                    resolved = resolver.resolve_template(value)
                    object.__setattr__(self, field_name, resolved)
                except SecretNotFoundError:
                    pass  # Keep original value

        return self


def secret_field(default: Any = None) -> Any:
    """Marker for fields that may contain secret references.

    Use this in dataclass definitions to indicate that a field
    should be resolved for secrets.

    Args:
        default: Default value for the field.

    Returns:
        Field descriptor.
    """
    return field(
        default=default,
        metadata={"secret": True},
    )


# =============================================================================
# Action Decorator
# =============================================================================


def with_secret_resolution(cls: type) -> type:
    """Class decorator to add secret resolution to an action.

    Automatically resolves secrets in configuration before execution.

    Example:
        >>> @with_secret_resolution
        ... class MyWebhookAction(BaseAction[WebhookConfig]):
        ...     def _execute(self, result):
        ...         # self._config already has secrets resolved
        ...         api_key = self._config.api_key
        ...         ...
    """
    original_execute = cls._execute if hasattr(cls, "_execute") else None

    def new_execute(self: Any, checkpoint_result: Any) -> Any:
        # Resolve secrets in config before execution
        config = getattr(self, "_config", None)
        if config:
            resolver = get_resolver()

            # Resolve string fields that may contain secrets
            for attr_name in dir(config):
                if attr_name.startswith("_"):
                    continue

                try:
                    value = getattr(config, attr_name)
                    if isinstance(value, str) and "${" in value:
                        resolved = resolver.resolve_template(value)
                        object.__setattr__(config, attr_name, resolved)
                    elif isinstance(value, dict):
                        resolved = resolver.resolve_config(value)
                        object.__setattr__(config, attr_name, resolved)
                except (AttributeError, SecretNotFoundError):
                    pass

        if original_execute:
            return original_execute(self, checkpoint_result)

    if original_execute:
        cls._execute = new_execute

    return cls


# =============================================================================
# Credential Helper
# =============================================================================


class CredentialHelper:
    """Helper for managing common credential patterns.

    Provides utilities for handling authentication credentials
    stored in secret managers.

    Example:
        >>> helper = CredentialHelper()
        >>>
        >>> # Get bearer token
        >>> token = helper.get_bearer_token("api/token")
        >>>
        >>> # Get basic auth credentials
        >>> username, password = helper.get_basic_auth("api/credentials")
        >>>
        >>> # Get API key
        >>> api_key = helper.get_api_key("api/key")
    """

    def __init__(self, manager: SecretManager | None = None) -> None:
        """Initialize credential helper.

        Args:
            manager: SecretManager to use. Uses global if None.
        """
        self._manager = manager

    @property
    def manager(self) -> SecretManager:
        """Get the secret manager."""
        if self._manager is None:
            self._manager = get_secret_manager()
        return self._manager

    def get_bearer_token(
        self,
        key: str,
        *,
        provider: str | None = None,
    ) -> str | None:
        """Get a bearer token from secrets.

        Args:
            key: Secret key.
            provider: Specific provider.

        Returns:
            Bearer token or None.
        """
        return self.manager.get_value(key, provider=provider)

    def get_basic_auth(
        self,
        key: str,
        *,
        provider: str | None = None,
        username_field: str = "username",
        password_field: str = "password",
    ) -> tuple[str, str] | None:
        """Get basic auth credentials from a structured secret.

        Args:
            key: Secret key for structured secret.
            provider: Specific provider.
            username_field: Field name for username.
            password_field: Field name for password.

        Returns:
            Tuple of (username, password) or None.
        """
        username = self.manager.get_value(
            key, provider=provider, field=username_field
        )
        password = self.manager.get_value(
            key, provider=provider, field=password_field
        )

        if username and password:
            return (username, password)
        return None

    def get_api_key(
        self,
        key: str,
        *,
        provider: str | None = None,
    ) -> str | None:
        """Get an API key from secrets.

        Args:
            key: Secret key.
            provider: Specific provider.

        Returns:
            API key or None.
        """
        return self.manager.get_value(key, provider=provider)

    def get_oauth_credentials(
        self,
        key: str,
        *,
        provider: str | None = None,
    ) -> dict[str, str] | None:
        """Get OAuth credentials from a structured secret.

        Args:
            key: Secret key for structured secret.
            provider: Specific provider.

        Returns:
            Dict with client_id, client_secret, etc.
        """
        secret = self.manager.get(key, provider=provider)
        if secret:
            import json
            try:
                return json.loads(secret.get_value())
            except json.JSONDecodeError:
                return None
        return None

    def build_auth_header(
        self,
        auth_type: str,
        key: str,
        *,
        provider: str | None = None,
    ) -> dict[str, str]:
        """Build authentication header from secrets.

        Args:
            auth_type: Type of auth ("bearer", "basic", "api_key").
            key: Secret key.
            provider: Specific provider.

        Returns:
            Dict of headers.
        """
        headers: dict[str, str] = {}

        if auth_type == "bearer":
            token = self.get_bearer_token(key, provider=provider)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "basic":
            import base64
            creds = self.get_basic_auth(key, provider=provider)
            if creds:
                encoded = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif auth_type == "api_key":
            api_key = self.get_api_key(key, provider=provider)
            if api_key:
                headers["X-API-Key"] = api_key

        return headers


# =============================================================================
# Global Credential Helper
# =============================================================================

_credential_helper: CredentialHelper | None = None


def get_credential_helper() -> CredentialHelper:
    """Get the global credential helper.

    Returns:
        Global CredentialHelper instance.
    """
    global _credential_helper
    if _credential_helper is None:
        _credential_helper = CredentialHelper()
    return _credential_helper


def get_bearer_token(key: str, *, provider: str | None = None) -> str | None:
    """Get a bearer token from secrets.

    Uses the global credential helper.

    Args:
        key: Secret key.
        provider: Specific provider.

    Returns:
        Bearer token or None.
    """
    return get_credential_helper().get_bearer_token(key, provider=provider)


def get_basic_auth(
    key: str,
    *,
    provider: str | None = None,
) -> tuple[str, str] | None:
    """Get basic auth credentials from secrets.

    Uses the global credential helper.

    Args:
        key: Secret key.
        provider: Specific provider.

    Returns:
        Tuple of (username, password) or None.
    """
    return get_credential_helper().get_basic_auth(key, provider=provider)


def get_api_key(key: str, *, provider: str | None = None) -> str | None:
    """Get an API key from secrets.

    Uses the global credential helper.

    Args:
        key: Secret key.
        provider: Specific provider.

    Returns:
        API key or None.
    """
    return get_credential_helper().get_api_key(key, provider=provider)
