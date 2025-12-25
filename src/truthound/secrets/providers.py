"""Built-in secret providers.

This module provides implementations for common secret backends:
- Environment variables
- .env files
- JSON/YAML configuration files
- Chained/fallback providers
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from truthound.secrets.base import (
    BaseSecretProvider,
    SecretNotFoundError,
    SecretProviderError,
    SecretValue,
)


# =============================================================================
# Environment Variable Provider
# =============================================================================


class EnvironmentProvider(BaseSecretProvider):
    """Secret provider that reads from environment variables.

    This is the simplest and most common provider, suitable for
    containerized deployments and CI/CD pipelines.

    Key Format:
        - Direct: API_KEY -> $API_KEY
        - With prefix: api/key -> $API_KEY (normalized)
        - Nested: database/password -> $DATABASE_PASSWORD

    Example:
        >>> provider = EnvironmentProvider()
        >>> secret = provider.get("DATABASE_PASSWORD")
        >>> print(secret.get_value())  # Value from $DATABASE_PASSWORD

        >>> # With prefix
        >>> provider = EnvironmentProvider(prefix="MYAPP_")
        >>> secret = provider.get("api_key")  # Reads $MYAPP_API_KEY
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        normalize_keys: bool = True,
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize environment provider.

        Args:
            prefix: Prefix for environment variable names.
            normalize_keys: Convert keys to uppercase and replace / with _.
            allow_empty: Allow empty string values.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(prefix=prefix, **kwargs)
        self._normalize_keys = normalize_keys
        self._allow_empty = allow_empty

    @property
    def name(self) -> str:
        return "env"

    def _normalize_key(self, key: str) -> str:
        """Normalize key to environment variable format."""
        normalized = key
        if self._normalize_keys:
            # Replace path separators with underscores
            normalized = normalized.replace("/", "_").replace("-", "_")
            # Convert to uppercase
            normalized = normalized.upper()

        # Add prefix
        if self._prefix and not normalized.startswith(self._prefix.upper()):
            normalized = f"{self._prefix.upper()}{normalized}"

        return normalized

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch from environment variable."""
        value = os.environ.get(key)

        if value is None:
            raise SecretNotFoundError(key, self.name)

        if not value and not self._allow_empty:
            raise SecretNotFoundError(key, self.name)

        return value

    def supports_key(self, key: str) -> bool:
        """Check if the key exists as an environment variable."""
        normalized = self._normalize_key(key)
        return normalized in os.environ


# =============================================================================
# DotEnv File Provider
# =============================================================================


class DotEnvProvider(BaseSecretProvider):
    """Secret provider that reads from .env files.

    Parses .env files in the standard format:
        KEY=value
        # Comment
        QUOTED="value with spaces"
        MULTI_LINE="line1\\nline2"

    Example:
        >>> provider = DotEnvProvider(path=".env")
        >>> secret = provider.get("DATABASE_URL")

        >>> # Multiple env files with precedence
        >>> provider = DotEnvProvider(
        ...     paths=[".env.local", ".env"],  # .env.local takes precedence
        ... )
    """

    def __init__(
        self,
        path: str | Path | None = None,
        paths: list[str | Path] | None = None,
        *,
        encoding: str = "utf-8",
        interpolate: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize .env provider.

        Args:
            path: Single .env file path.
            paths: Multiple .env file paths (later files take precedence).
            encoding: File encoding.
            interpolate: Interpolate ${VAR} references.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._encoding = encoding
        self._interpolate = interpolate
        self._values: dict[str, str] = {}

        # Determine paths to load
        if paths:
            self._paths = [Path(p) for p in paths]
        elif path:
            self._paths = [Path(path)]
        else:
            # Default paths
            self._paths = [
                Path(".env.local"),
                Path(".env"),
            ]

        self._load_files()

    @property
    def name(self) -> str:
        return "dotenv"

    def _load_files(self) -> None:
        """Load all .env files."""
        for path in self._paths:
            if path.exists():
                self._parse_file(path)

    def _parse_file(self, path: Path) -> None:
        """Parse a single .env file."""
        try:
            content = path.read_text(encoding=self._encoding)
        except Exception as e:
            raise SecretProviderError(
                self.name, f"Failed to read {path}: {e}", e
            )

        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=value
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]

            # Handle escape sequences
            value = value.replace("\\n", "\n").replace("\\t", "\t")

            self._values[key] = value

        # Interpolate references after loading all values
        if self._interpolate:
            self._interpolate_values()

    def _interpolate_values(self) -> None:
        """Interpolate ${VAR} references in values."""
        import re

        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            # Check local values first, then environment
            return self._values.get(var_name, os.environ.get(var_name, ""))

        # Multiple passes to handle nested references
        for _ in range(10):  # Max 10 levels of nesting
            changed = False
            for key, value in list(self._values.items()):
                new_value = pattern.sub(replace, value)
                if new_value != value:
                    self._values[key] = new_value
                    changed = True
            if not changed:
                break

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch from loaded values."""
        if key not in self._values:
            raise SecretNotFoundError(key, self.name)
        return self._values[key]

    def supports_key(self, key: str) -> bool:
        """Check if key exists in loaded values."""
        return key in self._values

    def reload(self) -> None:
        """Reload all .env files."""
        self._values.clear()
        self._load_files()
        self.clear_cache()


# =============================================================================
# File-Based Provider (JSON/YAML)
# =============================================================================


class FileProvider(BaseSecretProvider):
    """Secret provider that reads from JSON or YAML files.

    Supports structured secret files with nested keys:
        {
            "database": {
                "host": "localhost",
                "password": "secret123"
            },
            "api": {
                "key": "abc123"
            }
        }

    Access nested keys with / separator:
        provider.get("database/password")  -> "secret123"

    Example:
        >>> provider = FileProvider(path="secrets.json")
        >>> secret = provider.get("database/password")

        >>> # YAML file
        >>> provider = FileProvider(path="secrets.yaml")
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        key_separator: str = "/",
        **kwargs: Any,
    ) -> None:
        """Initialize file provider.

        Args:
            path: Path to secrets file (JSON or YAML).
            encoding: File encoding.
            key_separator: Separator for nested keys.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._path = Path(path)
        self._encoding = encoding
        self._key_separator = key_separator
        self._data: dict[str, Any] = {}
        self._load_file()

    @property
    def name(self) -> str:
        return "file"

    def _load_file(self) -> None:
        """Load and parse the secrets file."""
        if not self._path.exists():
            raise SecretProviderError(
                self.name, f"File not found: {self._path}"
            )

        try:
            content = self._path.read_text(encoding=self._encoding)
        except Exception as e:
            raise SecretProviderError(
                self.name, f"Failed to read {self._path}: {e}", e
            )

        suffix = self._path.suffix.lower()

        if suffix == ".json":
            try:
                self._data = json.loads(content)
            except json.JSONDecodeError as e:
                raise SecretProviderError(
                    self.name, f"Invalid JSON in {self._path}: {e}", e
                )

        elif suffix in (".yaml", ".yml"):
            try:
                import yaml

                self._data = yaml.safe_load(content) or {}
            except ImportError:
                raise SecretProviderError(
                    self.name, "PyYAML is required for YAML files"
                )
            except Exception as e:
                raise SecretProviderError(
                    self.name, f"Invalid YAML in {self._path}: {e}", e
                )

        else:
            raise SecretProviderError(
                self.name,
                f"Unsupported file format: {suffix}. Use .json or .yaml",
            )

    def _get_nested(self, data: dict[str, Any], keys: list[str]) -> Any:
        """Get nested value from dict."""
        current = data
        for key in keys:
            if isinstance(current, dict):
                if key not in current:
                    raise SecretNotFoundError(
                        self._key_separator.join(keys), self.name
                    )
                current = current[key]
            else:
                raise SecretNotFoundError(
                    self._key_separator.join(keys), self.name
                )
        return current

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch from loaded data."""
        keys = key.split(self._key_separator)
        value = self._get_nested(self._data, keys)

        # Handle field extraction for dict values
        if field and isinstance(value, dict):
            if field not in value:
                raise SecretNotFoundError(f"{key}#{field}", self.name)
            value = value[field]

        # Convert to string
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def supports_key(self, key: str) -> bool:
        """Check if key exists in loaded data."""
        try:
            keys = key.split(self._key_separator)
            self._get_nested(self._data, keys)
            return True
        except SecretNotFoundError:
            return False

    def reload(self) -> None:
        """Reload the secrets file."""
        self._data.clear()
        self._load_file()
        self.clear_cache()


# =============================================================================
# Chained Provider
# =============================================================================


class ChainedProvider(BaseSecretProvider):
    """Provider that chains multiple providers with fallback.

    Tries each provider in order until one succeeds. Useful for:
    - Development vs production secrets
    - Local overrides
    - Fallback chains

    Example:
        >>> provider = ChainedProvider([
        ...     EnvironmentProvider(),  # Try env first
        ...     DotEnvProvider(),       # Then .env
        ...     FileProvider("secrets.json"),  # Then file
        ... ])
        >>> secret = provider.get("API_KEY")  # First successful wins
    """

    def __init__(
        self,
        providers: list[BaseSecretProvider],
        *,
        stop_on_first: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize chained provider.

        Args:
            providers: List of providers to chain.
            stop_on_first: Stop on first successful provider.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        # Disable caching in chained provider - let individual providers cache
        super().__init__(enable_cache=False, **kwargs)
        self._providers = providers
        self._stop_on_first = stop_on_first

    @property
    def name(self) -> str:
        return "chain"

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Try each provider in order."""
        errors = []

        for provider in self._providers:
            try:
                if provider.supports_key(key):
                    secret = provider.get(key, version, field)
                    return secret.get_value()
            except SecretNotFoundError:
                continue
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                continue

        # All providers failed
        if errors:
            raise SecretProviderError(
                self.name,
                f"All providers failed for '{key}': {'; '.join(errors)}",
            )
        raise SecretNotFoundError(key, self.name)

    def supports_key(self, key: str) -> bool:
        """Check if any provider supports the key."""
        return any(p.supports_key(key) for p in self._providers)

    def add_provider(self, provider: BaseSecretProvider) -> None:
        """Add a provider to the chain."""
        self._providers.append(provider)

    def insert_provider(
        self, index: int, provider: BaseSecretProvider
    ) -> None:
        """Insert a provider at specific position."""
        self._providers.insert(index, provider)
