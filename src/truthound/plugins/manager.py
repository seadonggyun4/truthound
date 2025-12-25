"""Plugin manager for lifecycle management.

This module provides the central PluginManager that handles:
- Plugin discovery and loading
- Lifecycle management (load, activate, deactivate, unload)
- Dependency resolution
- Configuration management
- Hook coordination

Example:
    >>> from truthound.plugins import PluginManager
    >>>
    >>> manager = PluginManager()
    >>> manager.discover_plugins()
    >>> manager.load_plugin("my-plugin")
    >>> manager.activate_plugin("my-plugin")
    >>>
    >>> # Or load all discovered plugins
    >>> manager.load_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
import threading

from truthound.plugins.base import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginDependencyError,
    PluginCompatibilityError,
)
from truthound.plugins.registry import PluginRegistry
from truthound.plugins.hooks import HookManager, HookType
from truthound.plugins.discovery import PluginDiscovery


logger = logging.getLogger(__name__)


@dataclass
class PluginManagerConfig:
    """Configuration for PluginManager.

    Attributes:
        plugin_dirs: Directories to scan for plugins.
        scan_entrypoints: Whether to scan Python entry points.
        auto_load: Whether to auto-load discovered plugins.
        auto_activate: Whether to auto-activate loaded plugins.
        strict_dependencies: Fail if dependencies are missing.
        check_compatibility: Verify Truthound version compatibility.
    """

    plugin_dirs: list[Path] = field(default_factory=list)
    scan_entrypoints: bool = True
    auto_load: bool = False
    auto_activate: bool = True
    strict_dependencies: bool = True
    check_compatibility: bool = True


class PluginManager:
    """Central manager for plugin lifecycle.

    The PluginManager is responsible for:
    - Discovering plugins from entry points and directories
    - Loading and instantiating plugins
    - Managing plugin lifecycle (activate, deactivate)
    - Resolving dependencies between plugins
    - Coordinating hooks
    - Configuration management

    Example:
        >>> manager = PluginManager()
        >>>
        >>> # Discover plugins from entry points
        >>> discovered = manager.discover_plugins()
        >>> print(f"Found {len(discovered)} plugins")
        >>>
        >>> # Load a specific plugin
        >>> manager.load_plugin("my-validator-plugin")
        >>>
        >>> # Load all plugins
        >>> manager.load_all()
        >>>
        >>> # Get active validators
        >>> validators = manager.get_plugins_by_type(PluginType.VALIDATOR)
    """

    def __init__(
        self,
        config: PluginManagerConfig | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            config: Manager configuration.
        """
        self._config = config or PluginManagerConfig()
        self._registry = PluginRegistry()
        self._hooks = HookManager()
        self._discovery = PluginDiscovery(
            plugin_dirs=self._config.plugin_dirs,
            scan_entrypoints=self._config.scan_entrypoints,
        )
        self._discovered_classes: dict[str, type[Plugin]] = {}
        self._plugin_configs: dict[str, PluginConfig] = {}
        self._lock = threading.RLock()
        self._initialized = False

    @property
    def registry(self) -> PluginRegistry:
        """Get the plugin registry."""
        return self._registry

    @property
    def hooks(self) -> HookManager:
        """Get the hook manager."""
        return self._hooks

    @property
    def discovery(self) -> PluginDiscovery:
        """Get the plugin discovery instance."""
        return self._discovery

    # =========================================================================
    # Discovery
    # =========================================================================

    def discover_plugins(self) -> dict[str, type[Plugin]]:
        """Discover available plugins.

        Scans entry points and configured directories for plugins.

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        with self._lock:
            self._discovered_classes = self._discovery.discover_all()

            logger.info(f"Discovered {len(self._discovered_classes)} plugins")

            # Auto-load if configured
            if self._config.auto_load:
                self.load_all()

            return self._discovered_classes

    def add_plugin_directory(self, directory: Path | str) -> None:
        """Add a directory to scan for plugins.

        Args:
            directory: Directory path.
        """
        self._discovery.add_plugin_directory(directory)

    def list_discovered(self) -> list[str]:
        """List discovered plugin names.

        Returns:
            List of plugin names.
        """
        return list(self._discovered_classes.keys())

    # =========================================================================
    # Loading
    # =========================================================================

    def load_plugin(
        self,
        name: str,
        config: PluginConfig | None = None,
        activate: bool | None = None,
    ) -> Plugin:
        """Load a discovered plugin.

        Args:
            name: Plugin name.
            config: Optional configuration.
            activate: Whether to activate after loading.

        Returns:
            Loaded plugin instance.

        Raises:
            PluginNotFoundError: If plugin is not discovered.
            PluginLoadError: If plugin fails to load.
            PluginDependencyError: If dependencies are missing.
        """
        with self._lock:
            # Check if already loaded
            existing = self._registry.get_or_none(name)
            if existing:
                if activate is True or (activate is None and self._config.auto_activate):
                    self.activate_plugin(name)
                return existing

            # Get plugin class
            if name not in self._discovered_classes:
                raise PluginNotFoundError(
                    f"Plugin '{name}' has not been discovered. "
                    f"Available: {', '.join(self._discovered_classes.keys())}",
                    plugin_name=name,
                )

            plugin_cls = self._discovered_classes[name]

            # Merge config
            final_config = config or self._plugin_configs.get(name) or PluginConfig()

            # Create instance
            try:
                plugin = plugin_cls(final_config)
            except Exception as e:
                raise PluginLoadError(
                    f"Failed to instantiate plugin '{name}': {e}",
                    plugin_name=name,
                ) from e

            # Update state
            plugin._state = PluginState.LOADING
            plugin._manager = self

            # Check compatibility
            if self._config.check_compatibility:
                self._check_compatibility(plugin)

            # Check dependencies
            if self._config.strict_dependencies:
                self._check_dependencies(plugin)

            # Run setup
            try:
                plugin.setup()
            except Exception as e:
                plugin._state = PluginState.ERROR
                plugin._error = e
                raise PluginLoadError(
                    f"Plugin '{name}' setup failed: {e}",
                    plugin_name=name,
                ) from e

            # Register plugin
            plugin._state = PluginState.LOADED
            self._registry.register(plugin)

            # Trigger hook
            self._hooks.trigger(
                HookType.ON_PLUGIN_LOAD,
                plugin=plugin,
                manager=self,
            )

            logger.info(f"Loaded plugin: {name} v{plugin.version}")

            # Auto-activate
            should_activate = activate if activate is not None else self._config.auto_activate
            if should_activate and final_config.enabled:
                self.activate_plugin(name)

            return plugin

    def load_all(self, activate: bool | None = None) -> list[Plugin]:
        """Load all discovered plugins.

        Args:
            activate: Whether to activate after loading.

        Returns:
            List of loaded plugins.
        """
        loaded: list[Plugin] = []

        # Sort by priority
        plugins_by_priority = sorted(
            self._discovered_classes.items(),
            key=lambda x: self._plugin_configs.get(x[0], PluginConfig()).priority,
        )

        for name, _ in plugins_by_priority:
            try:
                plugin = self.load_plugin(name, activate=activate)
                loaded.append(plugin)
            except PluginError as e:
                logger.error(f"Failed to load plugin {name}: {e}")

        return loaded

    def load_from_class(
        self,
        plugin_cls: type[Plugin],
        config: PluginConfig | None = None,
        activate: bool | None = None,
    ) -> Plugin:
        """Load a plugin directly from a class.

        Args:
            plugin_cls: Plugin class.
            config: Optional configuration.
            activate: Whether to activate.

        Returns:
            Loaded plugin instance.
        """
        # Create instance to get name
        temp = plugin_cls(config)
        name = temp.info.name

        # Add to discovered classes
        self._discovered_classes[name] = plugin_cls

        return self.load_plugin(name, config, activate)

    # =========================================================================
    # Activation
    # =========================================================================

    def activate_plugin(self, name: str) -> None:
        """Activate a loaded plugin.

        Calls the plugin's register() method to add its components.

        Args:
            name: Plugin name.

        Raises:
            PluginNotFoundError: If plugin is not loaded.
            PluginError: If activation fails.
        """
        with self._lock:
            plugin = self._registry.get(name)

            if plugin.state == PluginState.ACTIVE:
                return  # Already active

            if plugin.state != PluginState.LOADED and plugin.state != PluginState.INACTIVE:
                raise PluginError(
                    f"Cannot activate plugin '{name}' in state {plugin.state.value}",
                    plugin_name=name,
                )

            try:
                plugin.register(self)
                plugin._state = PluginState.ACTIVE
                self._registry.update_state(name, PluginState.ACTIVE)
                logger.info(f"Activated plugin: {name}")
            except Exception as e:
                plugin._state = PluginState.ERROR
                plugin._error = e
                raise PluginError(
                    f"Plugin '{name}' activation failed: {e}",
                    plugin_name=name,
                ) from e

    def deactivate_plugin(self, name: str) -> None:
        """Deactivate an active plugin.

        Calls the plugin's unregister() method to remove its components.

        Args:
            name: Plugin name.
        """
        with self._lock:
            plugin = self._registry.get(name)

            if plugin.state != PluginState.ACTIVE:
                return  # Not active

            try:
                plugin.unregister(self)
                plugin._state = PluginState.INACTIVE
                self._registry.update_state(name, PluginState.INACTIVE)
                logger.info(f"Deactivated plugin: {name}")
            except Exception as e:
                logger.error(f"Error deactivating plugin {name}: {e}")

    # =========================================================================
    # Unloading
    # =========================================================================

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin.

        Deactivates if active, runs teardown, and removes from registry.

        Args:
            name: Plugin name.
        """
        with self._lock:
            plugin = self._registry.get(name)

            # Deactivate first
            if plugin.state == PluginState.ACTIVE:
                self.deactivate_plugin(name)

            plugin._state = PluginState.UNLOADING

            # Trigger hook before teardown
            self._hooks.trigger(
                HookType.ON_PLUGIN_UNLOAD,
                plugin=plugin,
                manager=self,
            )

            # Run teardown
            try:
                plugin.teardown()
            except Exception as e:
                logger.error(f"Error in plugin {name} teardown: {e}")

            # Remove from registry
            self._registry.unregister(name)
            plugin._manager = None

            logger.info(f"Unloaded plugin: {name}")

    def unload_all(self) -> None:
        """Unload all plugins."""
        # Unload in reverse priority order
        plugins = sorted(
            self._registry.list_all().items(),
            key=lambda x: x[1].config.priority,
            reverse=True,
        )

        for name, _ in plugins:
            try:
                self.unload_plugin(name)
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_plugin_config(self, name: str, config: PluginConfig) -> None:
        """Set configuration for a plugin.

        If plugin is already loaded, notifies it of the change.

        Args:
            name: Plugin name.
            config: New configuration.
        """
        with self._lock:
            old_config = self._plugin_configs.get(name)
            self._plugin_configs[name] = config

            # Update loaded plugin if exists
            plugin = self._registry.get_or_none(name)
            if plugin:
                plugin.config = config

    def get_plugin_config(self, name: str) -> PluginConfig:
        """Get configuration for a plugin.

        Args:
            name: Plugin name.

        Returns:
            Plugin configuration.
        """
        return self._plugin_configs.get(name, PluginConfig())

    def enable_plugin(self, name: str) -> None:
        """Enable a plugin.

        Args:
            name: Plugin name.
        """
        config = self.get_plugin_config(name)
        config.enabled = True
        self.set_plugin_config(name, config)

        # Activate if loaded
        plugin = self._registry.get_or_none(name)
        if plugin and plugin.state == PluginState.INACTIVE:
            self.activate_plugin(name)

    def disable_plugin(self, name: str) -> None:
        """Disable a plugin.

        Args:
            name: Plugin name.
        """
        config = self.get_plugin_config(name)
        config.enabled = False
        self.set_plugin_config(name, config)

        # Deactivate if loaded
        plugin = self._registry.get_or_none(name)
        if plugin and plugin.state == PluginState.ACTIVE:
            self.deactivate_plugin(name)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_plugin(self, name: str) -> Plugin:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance.

        Raises:
            PluginNotFoundError: If not found.
        """
        return self._registry.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[Plugin]:
        """Get all plugins of a type.

        Args:
            plugin_type: Plugin type.

        Returns:
            List of plugins.
        """
        return self._registry.get_by_type(plugin_type)

    def get_active_plugins(self) -> list[Plugin]:
        """Get all active plugins.

        Returns:
            List of active plugins.
        """
        return self._registry.get_active()

    def list_plugins(self) -> list[PluginInfo]:
        """List all loaded plugins.

        Returns:
            List of plugin info.
        """
        return self._registry.get_all_info()

    def is_plugin_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded.

        Args:
            name: Plugin name.

        Returns:
            True if loaded.
        """
        return name in self._registry

    def is_plugin_active(self, name: str) -> bool:
        """Check if a plugin is active.

        Args:
            name: Plugin name.

        Returns:
            True if active.
        """
        plugin = self._registry.get_or_none(name)
        return plugin is not None and plugin.state == PluginState.ACTIVE

    # =========================================================================
    # Dependency Resolution
    # =========================================================================

    def _check_dependencies(self, plugin: Plugin) -> None:
        """Check if plugin dependencies are satisfied.

        Args:
            plugin: Plugin to check.

        Raises:
            PluginDependencyError: If dependencies are missing.
        """
        missing: list[str] = []

        for dep in plugin.info.dependencies:
            # Check if dependency is loaded or discovered
            if dep not in self._registry and dep not in self._discovered_classes:
                missing.append(dep)

        if missing:
            raise PluginDependencyError(
                f"Plugin '{plugin.name}' has missing dependencies: {', '.join(missing)}",
                plugin_name=plugin.name,
                missing_deps=missing,
            )

        # Check Python dependencies
        missing_python: list[str] = []
        for dep in plugin.info.python_dependencies:
            try:
                __import__(dep.split("[")[0])  # Handle extras like pkg[extra]
            except ImportError:
                missing_python.append(dep)

        if missing_python:
            raise PluginDependencyError(
                f"Plugin '{plugin.name}' requires Python packages: {', '.join(missing_python)}. "
                f"Install with: pip install {' '.join(missing_python)}",
                plugin_name=plugin.name,
                missing_deps=missing_python,
            )

    def _check_compatibility(self, plugin: Plugin) -> None:
        """Check if plugin is compatible with current Truthound version.

        Args:
            plugin: Plugin to check.

        Raises:
            PluginCompatibilityError: If incompatible.
        """
        from truthound import __version__

        if not plugin.info.is_compatible(__version__):
            raise PluginCompatibilityError(
                f"Plugin '{plugin.name}' is not compatible with Truthound {__version__}. "
                f"Requires: {plugin.info.min_truthound_version} - {plugin.info.max_truthound_version}",
                plugin_name=plugin.name,
                required_version=plugin.info.min_truthound_version,
                current_version=__version__,
            )

    def resolve_load_order(self) -> list[str]:
        """Resolve the order in which plugins should be loaded.

        Performs topological sort based on dependencies.

        Returns:
            List of plugin names in load order.
        """
        # Build dependency graph
        graph: dict[str, set[str]] = {}
        for name, cls in self._discovered_classes.items():
            temp = cls()
            deps = set(temp.info.dependencies)
            graph[name] = deps

        # Topological sort
        result: list[str] = []
        visited: set[str] = set()
        temp_visited: set[str] = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise PluginDependencyError(
                    f"Circular dependency detected involving '{name}'",
                    plugin_name=name,
                )
            if name in visited:
                return

            temp_visited.add(name)

            for dep in graph.get(name, set()):
                if dep in graph:  # Only visit if we have the dependency
                    visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for name in graph:
            if name not in visited:
                visit(name)

        return result

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def shutdown(self) -> None:
        """Shutdown the plugin manager.

        Unloads all plugins and clears state.
        """
        logger.info("Shutting down plugin manager")
        self.unload_all()
        self._hooks.clear()
        self._discovered_classes.clear()
        self._plugin_configs.clear()

    def __enter__(self) -> "PluginManager":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"<PluginManager "
            f"discovered={len(self._discovered_classes)} "
            f"loaded={len(self._registry)} "
            f"active={len(self._registry.get_active())}>"
        )


# =============================================================================
# Global Manager
# =============================================================================

_global_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global PluginManager.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = PluginManager()
    return _global_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager (mainly for testing)."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.shutdown()
    _global_manager = None
