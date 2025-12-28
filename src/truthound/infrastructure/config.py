"""Enterprise configuration management system for Truthound.

This module provides environment-aware configuration management with support for:
- Multiple environments (dev, staging, prod)
- Multiple configuration sources (files, env vars, Vault, AWS Secrets)
- Configuration validation
- Hot reloading
- Type-safe configuration access

Architecture:
    ConfigSource[] (ordered by priority)
         |
         +---> EnvConfigSource (environment variables)
         +---> FileConfigSource (YAML, JSON, TOML)
         +---> VaultConfigSource (HashiCorp Vault)
         +---> AwsSecretsSource (AWS Secrets Manager)
         |
         v
    ConfigManager
         |
         +---> Merge & Validate
         |
         v
    ConfigProfile (typed access)

Usage:
    >>> from truthound.infrastructure.config import (
    ...     get_config, load_config, Environment,
    ... )
    >>>
    >>> # Load configuration for production
    >>> config = load_config(
    ...     environment=Environment.PRODUCTION,
    ...     config_path="config/",
    ...     use_vault=True,
    ... )
    >>>
    >>> # Access configuration
    >>> db_host = config.get("database.host")
    >>> log_level = config.get("logging.level", default="INFO")
    >>>
    >>> # Type-safe access
    >>> max_workers = config.get_int("workers.max", default=4)
    >>> debug = config.get_bool("debug", default=False)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar, overload

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomllib
    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False


# =============================================================================
# Environment Types
# =============================================================================


class Environment(Enum):
    """Application environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"

    @classmethod
    def from_string(cls, value: str) -> "Environment":
        """Convert string to Environment.

        Args:
            value: Environment string (case-insensitive).

        Returns:
            Environment enum value.
        """
        mapping = {
            "dev": cls.DEVELOPMENT,
            "development": cls.DEVELOPMENT,
            "test": cls.TESTING,
            "testing": cls.TESTING,
            "stage": cls.STAGING,
            "staging": cls.STAGING,
            "prod": cls.PRODUCTION,
            "production": cls.PRODUCTION,
            "local": cls.LOCAL,
        }
        return mapping.get(value.lower(), cls.DEVELOPMENT)

    @classmethod
    def current(cls) -> "Environment":
        """Get current environment from ENV variable.

        Checks: ENVIRONMENT, ENV, TRUTHOUND_ENV
        """
        for var in ("TRUTHOUND_ENV", "ENVIRONMENT", "ENV"):
            value = os.getenv(var)
            if value:
                return cls.from_string(value)
        return cls.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if this is a production environment."""
        return self in (Environment.PRODUCTION, Environment.STAGING)

    @property
    def is_development(self) -> bool:
        """Check if this is a development environment."""
        return self in (Environment.DEVELOPMENT, Environment.LOCAL, Environment.TESTING)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(Exception):
    """Base configuration error."""

    pass


class ConfigValidationError(ConfigError):
    """Configuration validation error."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed: {', '.join(errors)}")


class ConfigSourceError(ConfigError):
    """Configuration source error."""

    pass


# =============================================================================
# Configuration Sources
# =============================================================================


class ConfigSource(ABC):
    """Abstract base class for configuration sources.

    Configuration sources provide key-value pairs from various backends.
    Sources are processed in priority order (highest first).
    """

    def __init__(self, priority: int = 0) -> None:
        """Initialize config source.

        Args:
            priority: Source priority (higher = processed later, overrides earlier).
        """
        self._priority = priority

    @property
    def priority(self) -> int:
        """Get source priority."""
        return self._priority

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load configuration from source.

        Returns:
            Dictionary of configuration values.
        """
        pass

    def reload(self) -> dict[str, Any]:
        """Reload configuration (default: same as load)."""
        return self.load()

    @property
    def supports_reload(self) -> bool:
        """Check if source supports hot reload."""
        return False


class EnvConfigSource(ConfigSource):
    """Environment variable configuration source.

    Reads configuration from environment variables with prefix.

    Example:
        TRUTHOUND_DATABASE_HOST=localhost
        TRUTHOUND_DATABASE_PORT=5432

        Will produce:
        {"database": {"host": "localhost", "port": "5432"}}
    """

    def __init__(
        self,
        prefix: str = "TRUTHOUND",
        separator: str = "_",
        priority: int = 100,
    ) -> None:
        """Initialize environment source.

        Args:
            prefix: Environment variable prefix.
            separator: Separator for nested keys.
            priority: Source priority.
        """
        super().__init__(priority)
        self._prefix = prefix
        self._separator = separator

    def load(self) -> dict[str, Any]:
        """Load configuration from environment."""
        result: dict[str, Any] = {}
        prefix = f"{self._prefix}{self._separator}"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix) :].lower()
                parts = config_key.split(self._separator)

                # Navigate/create nested structure
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set value (with type inference)
                current[parts[-1]] = self._parse_value(value)

        return result

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # None
        if value.lower() in ("null", "none", ""):
            return None

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # JSON array/object
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value


class FileConfigSource(ConfigSource):
    """File-based configuration source.

    Supports YAML, JSON, and TOML formats.
    Automatically detects format from file extension.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        required: bool = False,
        priority: int = 50,
        watch: bool = False,
    ) -> None:
        """Initialize file source.

        Args:
            path: Path to configuration file.
            required: Raise error if file not found.
            priority: Source priority.
            watch: Enable file watching for hot reload.
        """
        super().__init__(priority)
        self._path = Path(path)
        self._required = required
        self._watch = watch
        self._last_modified: float = 0
        self._cached: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self._path.exists():
            if self._required:
                raise ConfigSourceError(f"Configuration file not found: {self._path}")
            return {}

        try:
            content = self._path.read_text(encoding="utf-8")
            self._last_modified = self._path.stat().st_mtime

            suffix = self._path.suffix.lower()

            if suffix in (".yaml", ".yml"):
                if not HAS_YAML:
                    raise ConfigSourceError("PyYAML not installed")
                self._cached = yaml.safe_load(content) or {}

            elif suffix == ".json":
                self._cached = json.loads(content)

            elif suffix == ".toml":
                if not HAS_TOML:
                    raise ConfigSourceError("tomllib/tomli not installed")
                self._cached = tomllib.loads(content)

            else:
                raise ConfigSourceError(f"Unsupported file format: {suffix}")

            return self._cached

        except Exception as e:
            if self._required:
                raise ConfigSourceError(f"Failed to load config: {e}")
            return {}

    def reload(self) -> dict[str, Any]:
        """Reload if file has changed."""
        if self._path.exists():
            mtime = self._path.stat().st_mtime
            if mtime > self._last_modified:
                return self.load()
        return self._cached

    @property
    def supports_reload(self) -> bool:
        return True


class VaultConfigSource(ConfigSource):
    """HashiCorp Vault configuration source.

    Reads secrets from Vault KV v2 engine.
    """

    def __init__(
        self,
        url: str,
        path: str,
        *,
        token: str | None = None,
        role: str | None = None,
        mount_point: str = "secret",
        priority: int = 200,
    ) -> None:
        """Initialize Vault source.

        Args:
            url: Vault server URL.
            path: Secret path.
            token: Vault token (or use VAULT_TOKEN env var).
            role: AppRole for authentication.
            mount_point: KV mount point.
            priority: Source priority.
        """
        super().__init__(priority)
        self._url = url.rstrip("/")
        self._path = path
        self._token = token or os.getenv("VAULT_TOKEN")
        self._role = role
        self._mount_point = mount_point
        self._cached: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load secrets from Vault."""
        try:
            import urllib.request
            import urllib.error

            url = f"{self._url}/v1/{self._mount_point}/data/{self._path}"
            headers = {"X-Vault-Token": self._token}

            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                self._cached = data.get("data", {}).get("data", {})
                return self._cached

        except Exception:
            return {}

    def reload(self) -> dict[str, Any]:
        """Reload from Vault."""
        return self.load()

    @property
    def supports_reload(self) -> bool:
        return True


class AwsSecretsSource(ConfigSource):
    """AWS Secrets Manager configuration source.

    Reads secrets from AWS Secrets Manager.
    """

    def __init__(
        self,
        secret_name: str,
        *,
        region: str | None = None,
        priority: int = 200,
    ) -> None:
        """Initialize AWS Secrets source.

        Args:
            secret_name: Secret name or ARN.
            region: AWS region.
            priority: Source priority.
        """
        super().__init__(priority)
        self._secret_name = secret_name
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._cached: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load secrets from AWS Secrets Manager."""
        try:
            import boto3

            client = boto3.client(
                "secretsmanager",
                region_name=self._region,
            )

            response = client.get_secret_value(SecretId=self._secret_name)
            secret_string = response.get("SecretString", "{}")
            self._cached = json.loads(secret_string)
            return self._cached

        except Exception:
            return {}

    def reload(self) -> dict[str, Any]:
        """Reload from AWS."""
        return self.load()

    @property
    def supports_reload(self) -> bool:
        return True


# =============================================================================
# Configuration Schema & Validation
# =============================================================================


@dataclass
class ConfigField:
    """Configuration field definition for validation."""

    name: str
    type: type | tuple[type, ...] = str
    required: bool = False
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    choices: list[Any] | None = None
    description: str = ""


@dataclass
class ConfigSchema:
    """Configuration schema for validation.

    Example:
        >>> schema = ConfigSchema(
        ...     fields=[
        ...         ConfigField("database.host", str, required=True),
        ...         ConfigField("database.port", int, default=5432, min_value=1, max_value=65535),
        ...         ConfigField("logging.level", str, choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ...     ]
        ... )
    """

    fields: list[ConfigField] = field(default_factory=list)

    def add_field(
        self,
        name: str,
        type: type = str,
        **kwargs: Any,
    ) -> "ConfigSchema":
        """Add a field to the schema."""
        self.fields.append(ConfigField(name=name, type=type, **kwargs))
        return self


class ConfigValidator:
    """Configuration validator.

    Validates configuration against a schema.
    """

    def __init__(self, schema: ConfigSchema) -> None:
        """Initialize validator.

        Args:
            schema: Configuration schema.
        """
        self._schema = schema

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        for field_def in self._schema.fields:
            value = self._get_nested(config, field_def.name)

            # Required check
            if value is None:
                if field_def.required:
                    errors.append(f"Required field '{field_def.name}' is missing")
                continue

            # Type check
            if not isinstance(value, field_def.type):
                errors.append(
                    f"Field '{field_def.name}' should be {field_def.type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Range check
            if isinstance(value, (int, float)):
                if field_def.min_value is not None and value < field_def.min_value:
                    errors.append(
                        f"Field '{field_def.name}' must be >= {field_def.min_value}"
                    )
                if field_def.max_value is not None and value > field_def.max_value:
                    errors.append(
                        f"Field '{field_def.name}' must be <= {field_def.max_value}"
                    )

            # Pattern check
            if isinstance(value, str) and field_def.pattern:
                if not re.match(field_def.pattern, value):
                    errors.append(
                        f"Field '{field_def.name}' must match pattern '{field_def.pattern}'"
                    )

            # Choices check
            if field_def.choices and value not in field_def.choices:
                errors.append(
                    f"Field '{field_def.name}' must be one of {field_def.choices}"
                )

        return errors

    def _get_nested(self, config: dict[str, Any], key: str) -> Any:
        """Get nested configuration value."""
        parts = key.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
        return current


# =============================================================================
# Configuration Profile
# =============================================================================


class ConfigProfile:
    """Typed configuration access with environment awareness.

    Provides type-safe access to configuration values with defaults.

    Example:
        >>> profile = ConfigProfile(config_dict, environment=Environment.PRODUCTION)
        >>> host = profile.get("database.host", default="localhost")
        >>> port = profile.get_int("database.port", default=5432)
        >>> debug = profile.get_bool("debug", default=False)
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        environment: Environment = Environment.DEVELOPMENT,
    ) -> None:
        """Initialize configuration profile.

        Args:
            config: Configuration dictionary.
            environment: Current environment.
        """
        self._config = config
        self._environment = environment
        self._cache: dict[str, Any] = {}

    @property
    def environment(self) -> Environment:
        """Get current environment."""
        return self._environment

    @property
    def is_production(self) -> bool:
        """Check if production environment."""
        return self._environment.is_production

    @property
    def is_development(self) -> bool:
        """Check if development environment."""
        return self._environment.is_development

    def get(
        self,
        key: str,
        default: Any = None,
        *,
        required: bool = False,
    ) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (dot-separated for nesting).
            default: Default value if not found.
            required: Raise error if not found.

        Returns:
            Configuration value.
        """
        if key in self._cache:
            return self._cache[key]

        value = self._get_nested(key)

        if value is None:
            if required:
                raise ConfigError(f"Required configuration '{key}' not found")
            return default

        self._cache[key] = value
        return value

    def get_str(self, key: str, default: str = "") -> str:
        """Get string configuration value."""
        value = self.get(key, default)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        if isinstance(value, float):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")
        return bool(value)

    def get_list(self, key: str, default: list[Any] | None = None) -> list[Any]:
        """Get list configuration value."""
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        if value is None:
            return default or []
        return [value]

    def get_dict(
        self, key: str, default: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get dictionary configuration value."""
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default or {}

    def _get_nested(self, key: str) -> Any:
        """Get nested configuration value."""
        parts = key.split(".")
        current = self._config

        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None

        return current

    def to_dict(self) -> dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_nested(key) is not None

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigManager:
    """Central configuration manager.

    Manages multiple configuration sources and provides unified access.

    Example:
        >>> manager = ConfigManager(environment=Environment.PRODUCTION)
        >>> manager.add_source(FileConfigSource("config/base.yaml"))
        >>> manager.add_source(FileConfigSource("config/production.yaml"))
        >>> manager.add_source(EnvConfigSource())
        >>>
        >>> config = manager.load()
        >>> print(config.get("database.host"))
    """

    def __init__(
        self,
        environment: Environment | None = None,
        *,
        auto_reload: bool = False,
        reload_interval: float = 60.0,
    ) -> None:
        """Initialize configuration manager.

        Args:
            environment: Application environment.
            auto_reload: Enable automatic reloading.
            reload_interval: Reload check interval in seconds.
        """
        self._environment = environment or Environment.current()
        self._sources: list[ConfigSource] = []
        self._config: dict[str, Any] = {}
        self._profile: ConfigProfile | None = None
        self._schema: ConfigSchema | None = None
        self._lock = threading.RLock()
        self._auto_reload = auto_reload
        self._reload_interval = reload_interval
        self._reload_thread: threading.Thread | None = None
        self._running = False
        self._callbacks: list[Callable[[ConfigProfile], None]] = []

    @property
    def environment(self) -> Environment:
        """Get current environment."""
        return self._environment

    def add_source(self, source: ConfigSource) -> "ConfigManager":
        """Add a configuration source.

        Args:
            source: Configuration source.

        Returns:
            Self for chaining.
        """
        with self._lock:
            self._sources.append(source)
            self._sources.sort(key=lambda s: s.priority)
        return self

    def set_schema(self, schema: ConfigSchema) -> "ConfigManager":
        """Set configuration schema for validation.

        Args:
            schema: Configuration schema.

        Returns:
            Self for chaining.
        """
        self._schema = schema
        return self

    def on_reload(self, callback: Callable[[ConfigProfile], None]) -> "ConfigManager":
        """Register reload callback.

        Args:
            callback: Function to call on reload.

        Returns:
            Self for chaining.
        """
        self._callbacks.append(callback)
        return self

    def load(self, validate: bool = True) -> ConfigProfile:
        """Load configuration from all sources.

        Args:
            validate: Validate against schema.

        Returns:
            ConfigProfile instance.
        """
        with self._lock:
            self._config = {}

            # Load from sources in priority order
            for source in self._sources:
                try:
                    source_config = source.load()
                    self._merge_config(self._config, source_config)
                except Exception:
                    pass  # Skip failed sources

            # Validate
            if validate and self._schema:
                validator = ConfigValidator(self._schema)
                errors = validator.validate(self._config)
                if errors:
                    raise ConfigValidationError(errors)

            self._profile = ConfigProfile(
                self._config,
                environment=self._environment,
            )

            # Start auto-reload if enabled
            if self._auto_reload and not self._running:
                self._start_reload_thread()

            return self._profile

    def reload(self) -> ConfigProfile:
        """Reload configuration from sources that support it.

        Returns:
            Updated ConfigProfile.
        """
        with self._lock:
            changed = False

            for source in self._sources:
                if source.supports_reload:
                    try:
                        old_config = self._config.copy()
                        source_config = source.reload()
                        self._merge_config(self._config, source_config)
                        if self._config != old_config:
                            changed = True
                    except Exception:
                        pass

            if changed:
                self._profile = ConfigProfile(
                    self._config,
                    environment=self._environment,
                )

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(self._profile)
                    except Exception:
                        pass

            return self._profile or ConfigProfile({})

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _start_reload_thread(self) -> None:
        """Start background reload thread."""
        self._running = True
        self._reload_thread = threading.Thread(
            target=self._reload_loop,
            daemon=True,
            name="config-reload",
        )
        self._reload_thread.start()

    def _reload_loop(self) -> None:
        """Background reload loop."""
        while self._running:
            time.sleep(self._reload_interval)
            try:
                self.reload()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop auto-reload."""
        self._running = False
        if self._reload_thread:
            self._reload_thread.join(timeout=5)
            self._reload_thread = None

    @property
    def config(self) -> ConfigProfile:
        """Get current configuration profile."""
        if self._profile is None:
            return self.load()
        return self._profile


# =============================================================================
# Default Schema
# =============================================================================


def create_default_schema() -> ConfigSchema:
    """Create default Truthound configuration schema."""
    schema = ConfigSchema()

    # Logging
    schema.add_field("logging.level", str, default="INFO",
                     choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    schema.add_field("logging.format", str, default="console",
                     choices=["console", "json", "logfmt"])

    # Metrics
    schema.add_field("metrics.enabled", bool, default=True)
    schema.add_field("metrics.port", int, default=9090, min_value=1, max_value=65535)

    # Database
    schema.add_field("database.host", str, default="localhost")
    schema.add_field("database.port", int, default=5432, min_value=1, max_value=65535)
    schema.add_field("database.pool_size", int, default=10, min_value=1, max_value=100)

    # Validation
    schema.add_field("validation.timeout", int, default=300, min_value=1)
    schema.add_field("validation.max_workers", int, default=4, min_value=1, max_value=32)

    return schema


# =============================================================================
# Global Configuration
# =============================================================================

_global_manager: ConfigManager | None = None
_lock = threading.Lock()


def load_config(
    *,
    environment: Environment | str | None = None,
    config_path: str | Path | None = None,
    env_prefix: str = "TRUTHOUND",
    use_vault: bool = False,
    vault_url: str = "",
    vault_path: str = "",
    use_aws_secrets: bool = False,
    aws_secret_name: str = "",
    auto_reload: bool = False,
    validate: bool = True,
) -> ConfigProfile:
    """Load configuration.

    Args:
        environment: Application environment.
        config_path: Path to configuration files.
        env_prefix: Environment variable prefix.
        use_vault: Enable HashiCorp Vault.
        vault_url: Vault server URL.
        vault_path: Vault secret path.
        use_aws_secrets: Enable AWS Secrets Manager.
        aws_secret_name: AWS secret name.
        auto_reload: Enable auto-reload.
        validate: Validate configuration.

    Returns:
        ConfigProfile instance.
    """
    global _global_manager

    with _lock:
        if isinstance(environment, str):
            environment = Environment.from_string(environment)
        elif environment is None:
            environment = Environment.current()

        manager = ConfigManager(
            environment=environment,
            auto_reload=auto_reload,
        )

        # Add file sources
        if config_path:
            path = Path(config_path)

            # Base config
            for ext in (".yaml", ".yml", ".json", ".toml"):
                base_file = path / f"base{ext}"
                if base_file.exists():
                    manager.add_source(FileConfigSource(base_file, priority=10))
                    break

            # Environment-specific config
            for ext in (".yaml", ".yml", ".json", ".toml"):
                env_file = path / f"{environment.value}{ext}"
                if env_file.exists():
                    manager.add_source(FileConfigSource(env_file, priority=20))
                    break

            # Local overrides
            for ext in (".yaml", ".yml", ".json", ".toml"):
                local_file = path / f"local{ext}"
                if local_file.exists():
                    manager.add_source(FileConfigSource(local_file, priority=30))
                    break

        # Add environment variables
        manager.add_source(EnvConfigSource(prefix=env_prefix, priority=100))

        # Add Vault
        if use_vault and vault_url and vault_path:
            manager.add_source(
                VaultConfigSource(vault_url, vault_path, priority=200)
            )

        # Add AWS Secrets
        if use_aws_secrets and aws_secret_name:
            manager.add_source(
                AwsSecretsSource(aws_secret_name, priority=200)
            )

        # Set default schema
        manager.set_schema(create_default_schema())

        _global_manager = manager
        return manager.load(validate=validate)


def get_config() -> ConfigProfile:
    """Get the global configuration.

    Returns:
        ConfigProfile instance.
    """
    global _global_manager

    with _lock:
        if _global_manager is None:
            return load_config()
        return _global_manager.config


def reload_config() -> ConfigProfile:
    """Reload the global configuration.

    Returns:
        Updated ConfigProfile.
    """
    global _global_manager

    with _lock:
        if _global_manager is None:
            return load_config()
        return _global_manager.reload()
