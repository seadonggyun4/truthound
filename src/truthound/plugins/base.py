"""Base classes and types for Truthound plugins.

This module defines the core abstractions for the plugin system:
- Plugin: Abstract base class for all plugins
- PluginConfig: Configuration dataclass
- PluginInfo: Metadata about a plugin
- PluginType: Enumeration of plugin categories
- PluginState: Plugin lifecycle states
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Generic
from pathlib import Path
import re

if TYPE_CHECKING:
    from truthound.plugins.manager import PluginManager


# =============================================================================
# Plugin Types and States
# =============================================================================


class PluginType(str, Enum):
    """Categories of plugins supported by Truthound."""

    VALIDATOR = "validator"
    REPORTER = "reporter"
    DATASOURCE = "datasource"
    PROFILER = "profiler"
    HOOK = "hook"
    TRANSFORMER = "transformer"  # Data transformation plugins
    EXPORTER = "exporter"  # Export to external systems
    CUSTOM = "custom"  # Generic extension


class PluginState(str, Enum):
    """Lifecycle states for a plugin."""

    DISCOVERED = "discovered"  # Found but not loaded
    LOADING = "loading"  # Currently being loaded
    LOADED = "loaded"  # Successfully loaded
    ACTIVE = "active"  # Enabled and running
    INACTIVE = "inactive"  # Loaded but disabled
    ERROR = "error"  # Failed to load/run
    UNLOADING = "unloading"  # Currently being unloaded


# =============================================================================
# Exceptions
# =============================================================================


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    def __init__(self, message: str, plugin_name: str | None = None):
        self.plugin_name = plugin_name
        super().__init__(message)


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""

    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""

    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be satisfied."""

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        missing_deps: list[str] | None = None,
    ):
        self.missing_deps = missing_deps or []
        super().__init__(message, plugin_name)


class PluginCompatibilityError(PluginError):
    """Raised when a plugin is incompatible with the current version."""

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        required_version: str | None = None,
        current_version: str | None = None,
    ):
        self.required_version = required_version
        self.current_version = current_version
        super().__init__(message, plugin_name)


# =============================================================================
# Plugin Configuration
# =============================================================================


@dataclass
class PluginConfig:
    """Configuration for a plugin.

    Attributes:
        enabled: Whether the plugin is enabled.
        priority: Load order priority (lower = earlier).
        settings: Plugin-specific settings.
        auto_load: Whether to automatically load on discovery.
    """

    enabled: bool = True
    priority: int = 100
    settings: dict[str, Any] = field(default_factory=dict)
    auto_load: bool = True

    def merge(self, other: "PluginConfig | dict[str, Any]") -> "PluginConfig":
        """Merge with another config, other takes precedence.

        Args:
            other: Either a PluginConfig instance or a dict with config fields.
        """
        if isinstance(other, dict):
            other_settings = other.get("settings", {})
            merged_settings = {**self.settings, **other_settings}
            return PluginConfig(
                enabled=other.get("enabled", self.enabled),
                priority=other.get("priority", self.priority),
                settings=merged_settings,
                auto_load=other.get("auto_load", self.auto_load),
            )
        merged_settings = {**self.settings, **other.settings}
        return PluginConfig(
            enabled=other.enabled,
            priority=other.priority,
            settings=merged_settings,
            auto_load=other.auto_load,
        )


@dataclass(frozen=True)
class PluginInfo:
    """Metadata about a plugin.

    This is immutable to ensure plugin metadata remains consistent.

    Attributes:
        name: Unique identifier for the plugin.
        version: Semantic version string.
        plugin_type: Category of plugin.
        description: Human-readable description.
        author: Plugin author.
        homepage: URL to plugin homepage/repository.
        license: License identifier.
        min_truthound_version: Minimum compatible Truthound version.
        max_truthound_version: Maximum compatible Truthound version.
        dependencies: List of required plugin names.
        python_dependencies: List of required Python packages.
        tags: Searchable tags.
    """

    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    min_truthound_version: str | None = None
    max_truthound_version: str | None = None
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    python_dependencies: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate plugin info after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not re.match(r"^[a-z][a-z0-9_-]*$", self.name):
            raise ValueError(
                f"Invalid plugin name: {self.name}. "
                "Must start with lowercase letter and contain only "
                "lowercase letters, numbers, hyphens, and underscores."
            )
        if not self.version:
            raise ValueError("Plugin version cannot be empty")

    def is_compatible(self, truthound_version: str) -> bool:
        """Check if plugin is compatible with given Truthound version."""
        from packaging.version import Version, InvalidVersion

        try:
            current = Version(truthound_version)

            if self.min_truthound_version:
                min_ver = Version(self.min_truthound_version)
                if current < min_ver:
                    return False

            if self.max_truthound_version:
                max_ver = Version(self.max_truthound_version)
                if current > max_ver:
                    return False

            return True
        except InvalidVersion:
            # If versions can't be parsed, assume compatible
            return True


# =============================================================================
# Plugin Base Class
# =============================================================================

ConfigT = TypeVar("ConfigT", bound=PluginConfig)


class Plugin(ABC, Generic[ConfigT]):
    """Abstract base class for all Truthound plugins.

    Plugins must implement:
    - info property: Return PluginInfo metadata
    - setup(): Initialize the plugin
    - teardown(): Cleanup on unload

    Plugins may optionally implement:
    - register(): Register components (validators, reporters, etc.)
    - on_config_change(): Handle configuration updates
    - health_check(): Verify plugin is functioning

    Example:
        >>> class MyValidatorPlugin(Plugin):
        ...     @property
        ...     def info(self) -> PluginInfo:
        ...         return PluginInfo(
        ...             name="my-validator",
        ...             version="1.0.0",
        ...             plugin_type=PluginType.VALIDATOR,
        ...             description="Custom validation rules",
        ...         )
        ...
        ...     def setup(self) -> None:
        ...         # Initialize resources
        ...         pass
        ...
        ...     def teardown(self) -> None:
        ...         # Cleanup resources
        ...         pass
        ...
        ...     def register(self, manager: PluginManager) -> None:
        ...         # Register validators
        ...         from truthound.validators.registry import registry
        ...         registry.register(MyCustomValidator)
    """

    def __init__(self, config: ConfigT | None = None):
        """Initialize the plugin.

        Args:
            config: Optional plugin configuration.
        """
        self._config: ConfigT = config or self._default_config()  # type: ignore
        self._state: PluginState = PluginState.DISCOVERED
        self._manager: PluginManager | None = None
        self._error: Exception | None = None

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin metadata.

        Returns:
            PluginInfo with plugin name, version, type, etc.
        """
        ...

    @abstractmethod
    def setup(self) -> None:
        """Initialize the plugin.

        Called when the plugin is loaded. Use this to:
        - Initialize resources
        - Establish connections
        - Load configuration
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup plugin resources.

        Called when the plugin is unloaded. Use this to:
        - Close connections
        - Release resources
        - Save state
        """
        ...

    def register(self, manager: "PluginManager") -> None:
        """Register plugin components.

        Override this to register validators, reporters, etc.
        Called after setup() completes successfully.

        Args:
            manager: The plugin manager instance.
        """
        pass

    def unregister(self, manager: "PluginManager") -> None:
        """Unregister plugin components.

        Override this to unregister validators, reporters, etc.
        Called before teardown().

        Args:
            manager: The plugin manager instance.
        """
        pass

    def on_config_change(self, old_config: ConfigT, new_config: ConfigT) -> None:
        """Handle configuration changes.

        Override to respond to configuration updates at runtime.

        Args:
            old_config: Previous configuration.
            new_config: New configuration.
        """
        pass

    def health_check(self) -> bool:
        """Verify plugin is functioning correctly.

        Returns:
            True if healthy, False otherwise.
        """
        return self._state == PluginState.ACTIVE

    @property
    def config(self) -> ConfigT:
        """Get plugin configuration."""
        return self._config

    @config.setter
    def config(self, value: ConfigT) -> None:
        """Set plugin configuration with change notification."""
        old_config = self._config
        self._config = value
        if self._state == PluginState.ACTIVE:
            self.on_config_change(old_config, value)

    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state

    @property
    def name(self) -> str:
        """Get plugin name (convenience property)."""
        return self.info.name

    @property
    def version(self) -> str:
        """Get plugin version (convenience property)."""
        return self.info.version

    @property
    def plugin_type(self) -> PluginType:
        """Get plugin type (convenience property)."""
        return self.info.plugin_type

    @property
    def error(self) -> Exception | None:
        """Get error if plugin is in error state."""
        return self._error

    def _default_config(self) -> PluginConfig:
        """Return default configuration.

        Override in subclasses that use custom config types.
        """
        return PluginConfig()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} "
            f"version={self.version!r} "
            f"state={self.state.value!r}>"
        )


# =============================================================================
# Specialized Plugin Base Classes
# =============================================================================


class ValidatorPlugin(Plugin[PluginConfig]):
    """Base class for validator plugins.

    Provides convenience methods for registering validators.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self._get_plugin_name(),
            version=self._get_plugin_version(),
            plugin_type=PluginType.VALIDATOR,
            description=self._get_description(),
        )

    def _get_plugin_name(self) -> str:
        """Override to provide plugin name."""
        return self.__class__.__name__.lower().replace("plugin", "")

    def _get_plugin_version(self) -> str:
        """Override to provide plugin version."""
        return "1.0.0"

    def _get_description(self) -> str:
        """Override to provide description."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def get_validators(self) -> list[type]:
        """Return list of validator classes to register.

        Returns:
            List of Validator subclasses.
        """
        ...

    def register(self, manager: "PluginManager") -> None:
        """Register validators with the validator registry."""
        from truthound.validators.registry import registry

        for validator_cls in self.get_validators():
            registry.register(validator_cls)

    def setup(self) -> None:
        """Default setup - no action needed."""
        pass

    def teardown(self) -> None:
        """Default teardown - no action needed."""
        pass


class ReporterPlugin(Plugin[PluginConfig]):
    """Base class for reporter plugins.

    Provides convenience methods for registering reporters.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self._get_plugin_name(),
            version=self._get_plugin_version(),
            plugin_type=PluginType.REPORTER,
            description=self._get_description(),
        )

    def _get_plugin_name(self) -> str:
        """Override to provide plugin name."""
        return self.__class__.__name__.lower().replace("plugin", "")

    def _get_plugin_version(self) -> str:
        """Override to provide plugin version."""
        return "1.0.0"

    def _get_description(self) -> str:
        """Override to provide description."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def get_reporters(self) -> dict[str, type]:
        """Return dict of format name to reporter class.

        Returns:
            Dict mapping format names to Reporter subclasses.
        """
        ...

    def register(self, manager: "PluginManager") -> None:
        """Register reporters with the reporter factory."""
        from truthound.reporters.factory import register_reporter

        for format_name, reporter_cls in self.get_reporters().items():
            register_reporter(format_name)(reporter_cls)

    def setup(self) -> None:
        """Default setup - no action needed."""
        pass

    def teardown(self) -> None:
        """Default teardown - no action needed."""
        pass


class DataSourcePlugin(Plugin[PluginConfig]):
    """Base class for data source plugins.

    Provides convenience methods for registering data sources.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self._get_plugin_name(),
            version=self._get_plugin_version(),
            plugin_type=PluginType.DATASOURCE,
            description=self._get_description(),
        )

    def _get_plugin_name(self) -> str:
        """Override to provide plugin name."""
        return self.__class__.__name__.lower().replace("plugin", "")

    def _get_plugin_version(self) -> str:
        """Override to provide plugin version."""
        return "1.0.0"

    def _get_description(self) -> str:
        """Override to provide description."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def get_datasource_types(self) -> dict[str, type]:
        """Return dict of type name to datasource class.

        Returns:
            Dict mapping type names to DataSource subclasses.
        """
        ...

    def setup(self) -> None:
        """Default setup - no action needed."""
        pass

    def teardown(self) -> None:
        """Default teardown - no action needed."""
        pass


class HookPlugin(Plugin[PluginConfig]):
    """Base class for hook plugins.

    Hook plugins provide event handlers that are called at specific
    points in the validation/profiling lifecycle.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self._get_plugin_name(),
            version=self._get_plugin_version(),
            plugin_type=PluginType.HOOK,
            description=self._get_description(),
        )

    def _get_plugin_name(self) -> str:
        """Override to provide plugin name."""
        return self.__class__.__name__.lower().replace("plugin", "")

    def _get_plugin_version(self) -> str:
        """Override to provide plugin version."""
        return "1.0.0"

    def _get_description(self) -> str:
        """Override to provide description."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def get_hooks(self) -> dict[str, Callable]:
        """Return dict of hook name to handler function.

        Returns:
            Dict mapping hook names to handler functions.
        """
        ...

    def register(self, manager: "PluginManager") -> None:
        """Register hooks with the hook manager."""
        for hook_name, handler in self.get_hooks().items():
            manager.hooks.register(hook_name, handler, source=self.name)

    def unregister(self, manager: "PluginManager") -> None:
        """Unregister hooks."""
        for hook_name in self.get_hooks().keys():
            manager.hooks.unregister(hook_name, source=self.name)

    def setup(self) -> None:
        """Default setup - no action needed."""
        pass

    def teardown(self) -> None:
        """Default teardown - no action needed."""
        pass
