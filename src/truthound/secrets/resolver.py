"""Secret Reference Resolver - Template and configuration resolution.

This module provides utilities for resolving secret references in various
contexts: strings, configuration files, and data structures.

Supports:
    - Template strings with ${secrets:KEY} syntax
    - YAML/JSON config files with embedded references
    - Environment variable expansion
    - Default value fallbacks
    - Nested reference resolution
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from truthound.secrets.base import SecretNotFoundError, SecretReference
from truthound.secrets.manager import SecretManager, get_secret_manager

T = TypeVar("T")


# =============================================================================
# Resolver Configuration
# =============================================================================


@dataclass
class ResolverConfig:
    """Configuration for SecretResolver.

    Attributes:
        strict: Raise on unresolved references.
        max_depth: Maximum nesting depth for recursive resolution.
        preserve_unresolved: Keep ${...} format for unresolved refs.
        resolve_env: Resolve ${env:VAR} references.
        resolve_file: Resolve ${file:path} references.
    """

    strict: bool = True
    max_depth: int = 10
    preserve_unresolved: bool = False
    resolve_env: bool = True
    resolve_file: bool = True


# =============================================================================
# Secret Resolver
# =============================================================================


class SecretResolver:
    """Resolves secret references in templates and configurations.

    SecretResolver handles the resolution of secret references embedded in
    strings, configuration files, and data structures.

    Reference Formats:
        - ${secrets:KEY} - Default provider
        - ${secrets:provider/KEY} - Specific provider
        - ${env:VAR_NAME} - Environment variable
        - ${file:/path/to/file} - File contents
        - ${secrets:KEY|default} - With default value

    Example:
        >>> resolver = SecretResolver()
        >>> config = {
        ...     "database": {
        ...         "url": "${secrets:DATABASE_URL}",
        ...         "password": "${vault:db/password}",
        ...     },
        ...     "api_key": "${env:API_KEY|default-key}",
        ... }
        >>> resolved = resolver.resolve_config(config)

        >>> # Template string
        >>> template = "postgresql://${secrets:DB_USER}:${secrets:DB_PASS}@localhost/db"
        >>> connection_string = resolver.resolve_template(template)
    """

    # Pattern for all supported reference formats
    REFERENCE_PATTERN = re.compile(
        r"\$\{(?P<type>[a-zA-Z_]+):(?P<content>[^}]+)\}"
    )

    def __init__(
        self,
        manager: SecretManager | None = None,
        config: ResolverConfig | None = None,
    ) -> None:
        """Initialize resolver.

        Args:
            manager: SecretManager to use. Uses global if None.
            config: Resolver configuration.
        """
        self._manager = manager
        self._config = config or ResolverConfig()
        self._resolution_cache: dict[str, str] = {}

    @property
    def manager(self) -> SecretManager:
        """Get the secret manager."""
        if self._manager is None:
            self._manager = get_secret_manager()
        return self._manager

    def resolve_template(
        self,
        template: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Resolve all references in a template string.

        Args:
            template: String with ${...} references.
            context: Additional context variables for resolution.

        Returns:
            String with references resolved.
        """
        if not template or "${" not in template:
            return template

        result = template
        depth = 0

        # Iterate until no more references or max depth
        while "${" in result and depth < self._config.max_depth:
            new_result = self._resolve_pass(result, context)
            if new_result == result:
                break
            result = new_result
            depth += 1

        return result

    def _resolve_pass(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Single pass of reference resolution."""

        def replacer(match: re.Match[str]) -> str:
            ref_type = match.group("type")
            content = match.group("content")
            full_ref = match.group(0)

            try:
                return self._resolve_reference(ref_type, content, context)
            except SecretNotFoundError:
                if self._config.preserve_unresolved:
                    return full_ref
                if self._config.strict:
                    raise
                return full_ref

        return self.REFERENCE_PATTERN.sub(replacer, text)

    def _resolve_reference(
        self,
        ref_type: str,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Resolve a single reference."""
        # Parse content for key and default
        if "|" in content:
            key_part, default = content.split("|", 1)
        else:
            key_part = content
            default = None

        # Handle different reference types
        if ref_type == "secrets":
            return self._resolve_secrets_ref(key_part, default)

        elif ref_type == "env":
            return self._resolve_env_ref(key_part, default)

        elif ref_type == "vault":
            return self._resolve_provider_ref("vault", key_part, default)

        elif ref_type == "aws":
            return self._resolve_provider_ref("aws", key_part, default)

        elif ref_type == "azure":
            return self._resolve_provider_ref("azure", key_part, default)

        elif ref_type == "gcp":
            return self._resolve_provider_ref("gcp", key_part, default)

        elif ref_type == "file":
            return self._resolve_file_ref(key_part, default)

        elif ref_type == "ctx" and context:
            # Context variable resolution
            if key_part in context:
                return str(context[key_part])
            if default is not None:
                return default
            raise SecretNotFoundError(key_part, "context")

        else:
            # Unknown type, try as provider name
            return self._resolve_provider_ref(ref_type, key_part, default)

    def _resolve_secrets_ref(self, key: str, default: str | None) -> str:
        """Resolve ${secrets:...} reference."""
        # Check for provider prefix (e.g., vault/key)
        if "/" in key:
            parts = key.split("/", 1)
            if len(parts) == 2 and not parts[0].startswith("."):
                provider, actual_key = parts
                return self._resolve_provider_ref(provider, actual_key, default)

        # Use default provider chain
        secret = self.manager.get(key, default=default)
        if secret:
            return secret.get_value()

        raise SecretNotFoundError(key, "manager")

    def _resolve_env_ref(self, key: str, default: str | None) -> str:
        """Resolve ${env:...} reference."""
        import os

        if not self._config.resolve_env:
            raise SecretNotFoundError(key, "env (disabled)")

        value = os.environ.get(key)
        if value is not None:
            return value

        if default is not None:
            return default

        raise SecretNotFoundError(key, "env")

    def _resolve_provider_ref(
        self,
        provider: str,
        key: str,
        default: str | None,
    ) -> str:
        """Resolve provider-specific reference."""
        # Handle field extraction (key#field)
        field = None
        if "#" in key:
            key, field = key.split("#", 1)

        secret = self.manager.get(key, provider=provider, field=field, default=default)
        if secret:
            return secret.get_value()

        raise SecretNotFoundError(key, provider)

    def _resolve_file_ref(self, path: str, default: str | None) -> str:
        """Resolve ${file:...} reference."""
        if not self._config.resolve_file:
            raise SecretNotFoundError(path, "file (disabled)")

        file_path = Path(path)

        if not file_path.exists():
            if default is not None:
                return default
            raise SecretNotFoundError(path, "file")

        try:
            return file_path.read_text().strip()
        except Exception as e:
            if default is not None:
                return default
            raise SecretNotFoundError(path, f"file ({e})")

    def resolve_config(
        self,
        config: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve references in a configuration dictionary.

        Args:
            config: Configuration dictionary.
            context: Additional context variables.

        Returns:
            Resolved configuration dictionary.
        """
        return self._resolve_value(config, context, depth=0)

    def _resolve_value(
        self,
        value: Any,
        context: dict[str, Any] | None,
        depth: int,
    ) -> Any:
        """Recursively resolve a value."""
        if depth > self._config.max_depth:
            return value

        if isinstance(value, str):
            return self.resolve_template(value, context=context)

        elif isinstance(value, dict):
            return {
                k: self._resolve_value(v, context, depth + 1)
                for k, v in value.items()
            }

        elif isinstance(value, list):
            return [
                self._resolve_value(item, context, depth + 1) for item in value
            ]

        return value

    def resolve_file(
        self,
        path: str | Path,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve references in a configuration file.

        Args:
            path: Path to JSON or YAML configuration file.
            context: Additional context variables.

        Returns:
            Resolved configuration dictionary.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        content = path.read_text()
        suffix = path.suffix.lower()

        if suffix == ".json":
            config = json.loads(content)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml

                config = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        return self.resolve_config(config, context=context)

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._resolution_cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

_global_resolver: SecretResolver | None = None


def get_resolver() -> SecretResolver:
    """Get the global secret resolver.

    Returns:
        Global SecretResolver instance.
    """
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = SecretResolver()
    return _global_resolver


def set_resolver(resolver: SecretResolver) -> None:
    """Set the global secret resolver.

    Args:
        resolver: SecretResolver to use globally.
    """
    global _global_resolver
    _global_resolver = resolver


def resolve_template(
    template: str,
    *,
    context: dict[str, Any] | None = None,
) -> str:
    """Resolve references in a template string.

    Uses the global resolver.

    Args:
        template: Template string with ${...} references.
        context: Additional context variables.

    Returns:
        Resolved string.
    """
    return get_resolver().resolve_template(template, context=context)


def resolve_config(
    config: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve references in a configuration dictionary.

    Uses the global resolver.

    Args:
        config: Configuration dictionary.
        context: Additional context variables.

    Returns:
        Resolved configuration.
    """
    return get_resolver().resolve_config(config, context=context)


def resolve_file(
    path: str | Path,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve references in a configuration file.

    Uses the global resolver.

    Args:
        path: Path to configuration file.
        context: Additional context variables.

    Returns:
        Resolved configuration.
    """
    return get_resolver().resolve_file(path, context=context)


# =============================================================================
# Decorator for Configuration Classes
# =============================================================================


def with_secrets(cls: type[T]) -> type[T]:
    """Decorator to resolve secrets in dataclass fields.

    Automatically resolves ${...} references in string fields
    when the dataclass is initialized.

    Example:
        >>> @with_secrets
        ... @dataclass
        ... class DatabaseConfig:
        ...     host: str
        ...     password: str = "${secrets:DB_PASSWORD}"
        ...
        >>> config = DatabaseConfig(host="localhost")
        >>> # config.password is now the actual secret value
    """
    original_init = cls.__init__

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        resolver = get_resolver()

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            value = getattr(self, attr_name)
            if isinstance(value, str) and "${" in value:
                try:
                    resolved = resolver.resolve_template(value)
                    object.__setattr__(self, attr_name, resolved)
                except SecretNotFoundError:
                    pass  # Keep original value if resolution fails

    cls.__init__ = new_init
    return cls
