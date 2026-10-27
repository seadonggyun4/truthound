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
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - public configuration annotation introspection
from typing import TYPE_CHECKING, Any

from truthound.plugins.base import (
    Plugin,
    PluginCompatibilityError,
    PluginConfig,
    PluginDependencyError,
    PluginError,
    PluginInfo,
    PluginLoadError,
    PluginNotFoundError,
    PluginState,
    PluginType,
)
from truthound.plugins.discovery import PluginDiscovery
from truthound.plugins.hooks import HookManager, HookType
from truthound.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from truthound.core.contracts import CheckSpecFactory, DataAssetProvider


logger = logging.getLogger(__name__)


def _default_host_version() -> str:
    from truthound import __version__

    return __version__


@dataclass(frozen=True)
class PluginCleanupResult:
    """Completion of native cleanup callbacks, not proof of external delivery."""

    name: str
    deactivated: bool
    teardown_completed: bool
    removed: bool
    hook_failed: bool = False
    error_types: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return (
            self.deactivated and self.teardown_completed and self.removed and not self.hook_failed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "deactivated": self.deactivated,
            "teardown_completed": self.teardown_completed,
            "removed": self.removed,
            "hook_failed": self.hook_failed,
            "error_types": list(self.error_types),
            "success": self.success,
        }


@dataclass
class PluginManagerConfig:
    """Configuration for PluginManager.

    The 2.0 manager keeps the original lifecycle flags while allowing optional
    capability services such as security, versioning, hot reload, and stable
    extension ports.
    """

    plugin_dirs: list[Path] = field(default_factory=list)
    scan_entrypoints: bool = True
    auto_load: bool = False
    auto_activate: bool = True
    strict_dependencies: bool = True
    check_compatibility: bool = True
    host_version: str = field(default_factory=_default_host_version)
    strict_version_check: bool = True
    allow_missing_optional: bool = True
    trust_store_path: Path | None = None
    enable_hot_reload: bool = False
    watch_for_changes: bool = False
    strict_discovery: bool = True


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
        self._config = deepcopy(config) if config is not None else PluginManagerConfig()
        self._registry = PluginRegistry()
        self._hooks = HookManager()
        self._discovery = PluginDiscovery(
            plugin_dirs=self._config.plugin_dirs,
            scan_entrypoints=self._config.scan_entrypoints,
            strict=self._config.strict_discovery,
        )
        self._discovered_classes: dict[str, type[Plugin]] = {}
        self._plugin_configs: dict[str, PluginConfig] = {}
        self._check_factories: dict[str, Any] = {}
        self._data_asset_providers: dict[str, Any] = {}
        self._reporter_bindings: dict[str, Any] = {}
        self._reporter_contracts: dict[str, int] = {}
        self._global_reporter_bindings: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}
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
                self._load_all_plugins()

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
    # Stable extension ports
    # =========================================================================

    def register_check_factory(self, name: str, factory: CheckSpecFactory | Any) -> None:
        """Register a validation check factory exposed by a plugin."""
        self._check_factories[name] = factory

    def unregister_check_factory(self, name: str) -> bool:
        """Remove a manager-local check-factory lookup."""
        missing = object()
        return self._check_factories.pop(name, missing) is not missing

    def get_check_factory(self, name: str) -> Any | None:
        """Return a registered validation check factory."""
        return self._check_factories.get(name)

    def iter_check_factories(self) -> dict[str, Any]:
        """Return a snapshot of registered validation check factories."""
        return dict(self._check_factories)

    async def validate_async(self, check_factories, data=None, **settings):
        """Run manager-local native providers/factories and detailed hooks."""
        from truthound.plugins.workflows import validate_async

        return await validate_async(self, check_factories, data, **settings)

    def set_execution_guard(self, guard) -> None:
        """Guard local provider/factory/reporter workflows and each native hook."""
        self._hooks.set_execution_guard(guard)
        self._execution_guard = guard

    def validate(self, check_factories, data=None, **settings):
        """Synchronous counterpart; use validate_async inside an event loop."""
        from truthound.plugins.workflows import synchronous

        return synchronous(self.validate_async(check_factories, data, **settings))

    async def render_async(self, reporter_name, run):
        """Render a canonical run through this manager's registered v3 port."""
        from truthound.plugins.workflows import render_async

        return await render_async(self, reporter_name, run)

    def render(self, reporter_name, run):
        """Synchronous registered-reporter rendering without path writes."""
        from truthound.plugins.workflows import synchronous

        return synchronous(self.render_async(reporter_name, run))

    def register_validator(self, validator_cls: type[Any]) -> None:
        """Compatibility wrapper for validator-class plugins."""
        from truthound.validators.registry import registry

        registry.register(validator_cls)
        validator_name = getattr(validator_cls, "name", validator_cls.__name__.lower())
        self.register_check_factory(validator_name, validator_cls)

    def register_data_asset_provider(
        self,
        name: str,
        provider: DataAssetProvider | Any,
    ) -> None:
        """Register a data asset provider exposed by a plugin."""
        self._data_asset_providers[name] = provider

    def get_data_asset_provider(self, name: str) -> Any | None:
        """Return a registered data asset provider."""
        return self._data_asset_providers.get(name)

    def list_data_asset_providers(self) -> list[str]:
        """List registered data asset providers."""
        return sorted(self._data_asset_providers.keys())

    def unregister_data_asset_provider(self, name: str) -> bool:
        """Remove a provider lookup; the provider's owner retains cleanup authority."""
        missing = object()
        return self._data_asset_providers.pop(name, missing) is not missing

    def register_reporter(self, name: str, reporter: Any) -> None:
        """Register a reporter through the manager's stable port."""
        contract = int(getattr(reporter, "contract_version", 1))
        with self._lock:
            self._remove_global_reporter(name)
            self._reporter_bindings[name] = reporter
            self._reporter_contracts[name] = contract
            # Configured objects belong to this manager. The global factory accepts
            # constructors; placing an instance there breaks unrelated get_reporter calls.
            if not isinstance(reporter, type) and callable(getattr(reporter, "render", None)):
                return
            try:
                from truthound.reporters.factory import register_reporter

                register_reporter(name)(reporter)
                self._global_reporter_bindings[name] = reporter
            except Exception:
                logger.debug("Reporter '%s' could not be added to the global factory.", name)

    def _remove_global_reporter(self, name: str) -> None:
        from truthound.reporters.factory import unregister_reporter

        constructor = self._global_reporter_bindings.pop(name, None)
        if constructor is not None:
            unregister_reporter(name, expected_constructor=constructor)

    def unregister_reporter(self, name: str) -> bool:
        """Remove the local port and only its own matching global constructor."""
        with self._lock:
            self._remove_global_reporter(name)
            self._reporter_contracts.pop(name, None)
            missing = object()
            return self._reporter_bindings.pop(name, missing) is not missing

    def register_reporter_binding(self, name: str, reporter: Any) -> None:
        """Backward-compatible alias for reporter registration."""
        self.register_reporter(name, reporter)

    def get_reporter(self, name: str) -> Any | None:
        """Return a registered reporter binding."""
        return self._reporter_bindings.get(name)

    def get_reporter_binding(self, name: str) -> Any | None:
        """Backward-compatible alias for reporter lookup."""
        return self.get_reporter(name)

    def get_reporter_contract_version(self, name: str) -> int | None:
        """Return the registered reporter contract version."""
        return self._reporter_contracts.get(name)

    def list_reporter_contracts(self) -> dict[str, int]:
        """Return reporter contract versions by reporter name."""
        return dict(self._reporter_contracts)

    def register_hook(
        self,
        hook_type: HookType | str,
        handler: Any,
        *,
        priority: int = 100,
        source: str = "plugin",
    ) -> Any:
        """Register a lifecycle hook through the manager facade."""
        return self._hooks.register(
            hook_type,
            handler,
            priority=priority,
            source=source,
        )

    def register_capability(self, name: str, capability: Any) -> None:
        """Register an optional capability service."""
        self._capabilities[name] = capability

    def unregister_capability(self, name: str) -> bool:
        """Remove a local lookup without closing a possibly shared provider."""
        missing = object()
        return self._capabilities.pop(name, missing) is not missing

    def get_capability(self, name: str) -> Any | None:
        """Return a registered capability service."""
        return self._capabilities.get(name)

    def list_capabilities(self) -> list[str]:
        """List capability services attached to the manager."""
        return sorted(self._capabilities.keys())

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

            if plugin.name != name:
                raise PluginLoadError("Discovered name does not match plugin identity", plugin_name=name)

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
                try:
                    plugin.teardown()
                except Exception as cleanup_error:
                    raise PluginLoadError(
                        "Plugin setup and cleanup failed",
                        plugin_name=name,
                    ) from ExceptionGroup("Plugin setup/cleanup failures", [e, cleanup_error])
                raise PluginLoadError(
                    "Plugin setup failed; teardown completed",
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
        """Load all plugins, preserving individual failures in an ExceptionGroup."""
        return self._load_all_plugins(activate)

    def _load_all_plugins(self, activate: bool | None = None) -> list[Plugin]:
        """Load all discovered plugins.

        Args:
            activate: Whether to activate after loading.

        Returns:
            List of loaded plugins.
        """
        loaded: list[Plugin] = []
        errors: list[Exception] = []

        # Sort by priority
        plugins_by_priority = sorted(
            self._discovered_classes.items(),
            key=lambda x: self._plugin_configs.get(x[0], PluginConfig()).priority,
        )

        for name, _ in plugins_by_priority:
            try:
                plugin = self.load_plugin(name, activate=activate)
                loaded.append(plugin)
            except Exception as error:
                errors.append(error)

        if errors:
            raise ExceptionGroup(
                "Some plugins failed to load; successful loads remain registered", errors
            )
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
            except Exception as error:
                plugin._error = error
                logger.error("Plugin deactivation failed (%s)", type(error).__name__)

    # =========================================================================
    # Unloading
    # =========================================================================

    def unload_plugin(self, name: str, *, detailed: bool = False) -> PluginCleanupResult | None:
        """Deactivate, attempt teardown, and remove an owned plugin.

        ``detailed=True`` returns callback outcomes without exception bodies.
        The legacy default remains None; an authorization guard exception still
        propagates after cleanup. Teardown is attempted even if an unload hook
        is denied. Callback completion cannot guarantee provider-side cleanup.
        """
        with self._lock:
            plugin = self._registry.get(name)
            errors = []
            if plugin.state == PluginState.ACTIVE:
                self.deactivate_plugin(name)
            deactivated = plugin.state != PluginState.ACTIVE
            if not deactivated:
                errors.append(type(plugin.error).__name__ if plugin.error else "PluginError")
            plugin._state = PluginState.UNLOADING
            hook_error = None
            hook_failed = False
            teardown_completed = False
            try:
                outcomes = self._hooks.trigger_detailed(
                    HookType.ON_PLUGIN_UNLOAD, plugin=plugin, manager=self
                )
                hook_failed = any(not row.success for row in outcomes)
                errors.extend(
                    row.error_type for row in outcomes if not row.success and row.error_type
                )
            except Exception as error:
                hook_error = error
                hook_failed = True
                errors.append(type(error).__name__)
            finally:
                try:
                    plugin.teardown()
                    teardown_completed = True
                except Exception as error:
                    plugin._error = error
                    errors.append(type(error).__name__)
                    logger.error("Plugin teardown failed (%s)", type(error).__name__)
                finally:
                    self._registry.unregister(name)
                    plugin._manager = None
            receipt = PluginCleanupResult(
                name,
                deactivated,
                teardown_completed,
                name not in self._registry,
                hook_failed,
                tuple(errors),
            )
            if hook_error is not None and not detailed:
                raise hook_error
            logger.info("Unloaded plugin: %s", name)
            return receipt if detailed else None

    def unload_all(self) -> None:
        """Unload every plugin and report all cleanup failures."""
        self._unload_all_plugins()

    def _unload_all_plugins(self) -> None:
        """Synchronous lifecycle port shared with async facades."""
        # Unload in reverse priority order
        plugins = sorted(
            self._registry.list_all().items(),
            key=lambda x: x[1].config.priority,
            reverse=True,
        )

        errors: list[Exception] = []
        for name, _ in plugins:
            try:
                receipt = self.unload_plugin(name, detailed=True)
                if not receipt.success:
                    errors.append(PluginError("Plugin cleanup callbacks failed", plugin_name=name))
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Plugin cleanup failed", errors)

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
        from packaging.version import Version

        from truthound.plugins._dependencies import distribution_satisfied, plugin_requirements

        missing = []
        try:
            for requirement, optional in plugin_requirements(
                plugin.info, self._config.allow_missing_optional
            ):
                dependency = self._registry.get_or_none(requirement.name)
                if dependency is None:
                    if not optional:
                        missing.append(str(requirement))
                    continue
                if (
                    dependency.state
                    not in {PluginState.LOADED, PluginState.ACTIVE, PluginState.INACTIVE}
                    or self._config.strict_version_check
                    and not requirement.specifier.contains(
                        Version(dependency.version), prereleases=True
                    )
                ):
                    missing.append(str(requirement))
            for requirement in plugin.info.python_dependencies:
                if not distribution_satisfied(requirement, self._config.strict_version_check):
                    missing.append(requirement)
        except ValueError as error:
            raise PluginDependencyError(
                "Invalid dependency requirement or version", plugin_name=plugin.name
            ) from error
        if missing:
            raise PluginDependencyError(
                "Plugin dependencies are missing or incompatible",
                plugin_name=plugin.name,
                missing_deps=missing,
            )

    def _check_compatibility(self, plugin: Plugin) -> None:
        """Check if plugin is compatible with current Truthound version.

        Args:
            plugin: Plugin to check.

        Raises:
            PluginCompatibilityError: If incompatible.
        """
        host_version = self._config.host_version

        if not plugin.info.is_compatible(host_version):
            if not self._config.strict_version_check:
                logger.warning(
                    "Plugin host compatibility is not established; strict version checks disabled"
                )
                return
            raise PluginCompatibilityError(
                f"Plugin '{plugin.name}' is not compatible with Truthound {host_version}. "
                f"Requires: {plugin.info.min_truthound_version} - {plugin.info.max_truthound_version}",
                plugin_name=plugin.name,
                required_version=plugin.info.min_truthound_version,
                current_version=host_version,
            )

    def resolve_load_order(self) -> list[str]:
        """Resolve the order in which plugins should be loaded.

        Performs topological sort based on dependencies.

        Returns:
            List of plugin names in load order.
        """
        from truthound.plugins._dependencies import plugin_requirements

        graph = {}
        for name, cls in self._discovered_classes.items():
            try:
                requirements = plugin_requirements(cls().info, self._config.allow_missing_optional)
            except ValueError as error:
                raise PluginDependencyError(
                    "Invalid dependency requirement", plugin_name=name
                ) from error
            dependencies = set()
            for requirement, optional in requirements:
                if requirement.name in self._discovered_classes:
                    dependencies.add(requirement.name)
                elif (
                    requirement.name not in self._registry
                    and not optional
                    and self._config.strict_dependencies
                ):
                    raise PluginDependencyError(
                        "Required plugin is not discovered or loaded",
                        plugin_name=name,
                        missing_deps=[str(requirement)],
                    )
            graph[name] = dependencies
        result = []
        while graph:
            ready = sorted(name for name, dependencies in graph.items() if not dependencies)
            if not ready:
                raise PluginDependencyError("Circular plugin dependency detected")
            result.extend(ready)
            for name in ready:
                del graph[name]
            for dependencies in graph.values():
                dependencies.difference_update(ready)
        return result

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def shutdown(self) -> None:
        """Shutdown the plugin manager.

        Unloads all plugins and clears state.
        """
        logger.info("Shutting down plugin manager")
        self._unload_all_plugins()
        self._hooks.clear()
        self._discovered_classes.clear()
        self._plugin_configs.clear()
        self._discovery.close()
        self._check_factories.clear()
        self._data_asset_providers.clear()
        self._reporter_bindings.clear()
        self._reporter_contracts.clear()
        self._capabilities.clear()

    def __enter__(self) -> PluginManager:
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
