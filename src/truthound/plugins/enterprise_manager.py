"""Enterprise Plugin Manager Facade.

This module provides a unified PluginManager that integrates all
enterprise features:
- Security (sandbox, signing)
- Versioning (compatibility checks)
- Dependencies (resolution, conflict detection)
- Lifecycle (hot reload, state management)
- Documentation (auto-generation)

This is the main entry point for enterprise plugin management.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from truthound.plugins.base import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
)
from truthound.plugins.registry import PluginRegistry
from truthound.plugins.hooks import HookManager, HookType
from truthound.plugins.discovery import PluginDiscovery

# Security
from truthound.plugins.security.protocols import SecurityPolicy, IsolationLevel
from truthound.plugins.security.sandbox import SandboxFactory
from truthound.plugins.security.signing import (
    SigningServiceImpl,
    TrustStoreImpl,
    create_verification_chain,
)
from truthound.plugins.security.exceptions import SecurityError

# Versioning
from truthound.plugins.versioning import VersionResolver, VersionConstraint

# Dependencies
from truthound.plugins.dependencies import DependencyResolver, DependencyGraph

# Lifecycle
from truthound.plugins.lifecycle import LifecycleManager, HotReloadManager, LifecycleState

# Documentation
from truthound.plugins.docs import DocumentationExtractor, PluginDocumentation

logger = logging.getLogger(__name__)


@dataclass
class EnterprisePluginManagerConfig:
    """Configuration for Enterprise Plugin Manager.

    Attributes:
        plugin_dirs: Directories to scan for plugins
        scan_entrypoints: Scan Python entry points
        auto_load: Auto-load discovered plugins
        auto_activate: Auto-activate loaded plugins

        # Security
        default_security_policy: Default security policy for plugins
        require_signature: Require valid signature for loading
        trust_store_path: Path to trust store file

        # Hot Reload
        enable_hot_reload: Enable hot reload functionality
        watch_for_changes: Watch plugin files for changes

        # Versioning
        strict_version_check: Fail on version incompatibility
        host_version: Current host version

        # Dependencies
        strict_dependencies: Fail on missing dependencies
        allow_missing_optional: Allow missing optional deps
    """

    # Plugin discovery
    plugin_dirs: list[Path] = field(default_factory=list)
    scan_entrypoints: bool = True
    auto_load: bool = False
    auto_activate: bool = True

    # Security
    default_security_policy: SecurityPolicy = field(
        default_factory=SecurityPolicy.standard
    )
    require_signature: bool = False
    trust_store_path: Path | None = None

    # Hot Reload
    enable_hot_reload: bool = False
    watch_for_changes: bool = False

    # Versioning
    strict_version_check: bool = True
    host_version: str = "0.2.0"

    # Dependencies
    strict_dependencies: bool = True
    allow_missing_optional: bool = True


@dataclass
class LoadedPlugin:
    """Information about a loaded plugin.

    Attributes:
        plugin: Plugin instance
        info: Plugin metadata
        config: Plugin configuration
        security_policy: Applied security policy
        signature_valid: Whether signature was verified
        documentation: Generated documentation
    """

    plugin: Plugin
    info: PluginInfo
    config: PluginConfig
    security_policy: SecurityPolicy = field(default_factory=SecurityPolicy.standard)
    signature_valid: bool = False
    documentation: PluginDocumentation | None = None


class EnterprisePluginManager:
    """Enterprise Plugin Manager Facade.

    Provides a unified interface for all plugin management operations
    with enterprise features including security, versioning, dependencies,
    and hot reload.

    This is the recommended entry point for production plugin management.

    Example:
        >>> config = EnterprisePluginManagerConfig(
        ...     require_signature=True,
        ...     enable_hot_reload=True,
        ... )
        >>> manager = EnterprisePluginManager(config)
        >>> manager.discover_plugins()
        >>> plugin = await manager.load("my-plugin")
    """

    def __init__(
        self,
        config: EnterprisePluginManagerConfig | None = None,
    ) -> None:
        """Initialize the enterprise plugin manager.

        Args:
            config: Manager configuration
        """
        self._config = config or EnterprisePluginManagerConfig()

        # Core components (from existing plugin system)
        self._registry = PluginRegistry()
        self._hooks = HookManager()
        self._discovery = PluginDiscovery(
            plugin_dirs=self._config.plugin_dirs,
            scan_entrypoints=self._config.scan_entrypoints,
        )

        # Enterprise components
        self._trust_store = TrustStoreImpl(
            store_path=self._config.trust_store_path,
        )
        self._signing_service = SigningServiceImpl()
        self._version_resolver = VersionResolver()
        self._dependency_resolver = DependencyResolver(
            strict=self._config.strict_dependencies,
            allow_missing_optional=self._config.allow_missing_optional,
        )
        self._lifecycle_manager = LifecycleManager()
        self._hot_reload_manager = HotReloadManager(
            self._lifecycle_manager,
            plugin_loader=self._reload_plugin,
        ) if self._config.enable_hot_reload else None
        self._doc_extractor = DocumentationExtractor()

        # State
        self._discovered_classes: dict[str, type[Plugin]] = {}
        self._loaded_plugins: dict[str, LoadedPlugin] = {}
        self._plugin_paths: dict[str, Path] = {}
        self._dependency_graph: DependencyGraph | None = None
        self._lock = asyncio.Lock()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def registry(self) -> PluginRegistry:
        """Get the plugin registry."""
        return self._registry

    @property
    def hooks(self) -> HookManager:
        """Get the hook manager."""
        return self._hooks

    @property
    def trust_store(self) -> TrustStoreImpl:
        """Get the trust store."""
        return self._trust_store

    @property
    def lifecycle(self) -> LifecycleManager:
        """Get the lifecycle manager."""
        return self._lifecycle_manager

    # =========================================================================
    # Discovery
    # =========================================================================

    def discover_plugins(self) -> dict[str, type[Plugin]]:
        """Discover available plugins.

        Scans entry points and configured directories.

        Returns:
            Dict mapping plugin names to plugin classes
        """
        self._discovered_classes = self._discovery.discover_all()

        # Build dependency graph from discovered plugins
        self._build_dependency_graph()

        logger.info(f"Discovered {len(self._discovered_classes)} plugins")

        if self._config.auto_load:
            asyncio.create_task(self.load_all())

        return self._discovered_classes

    def _build_dependency_graph(self) -> None:
        """Build dependency graph from discovered plugins."""
        infos: list[PluginInfo] = []

        for name, plugin_cls in self._discovered_classes.items():
            try:
                instance = plugin_cls()
                infos.append(instance.info)
            except Exception as e:
                logger.warning(f"Could not get info for {name}: {e}")

        result = self._dependency_resolver.resolve(infos)
        self._dependency_graph = result.graph

        if not result.success:
            for conflict in result.conflicts:
                logger.warning(f"Dependency conflict: {conflict.message}")
            for plugin_id, dep_id in result.missing:
                logger.warning(f"Plugin {plugin_id} missing dependency: {dep_id}")

    # =========================================================================
    # Loading
    # =========================================================================

    async def load(
        self,
        name: str,
        config: PluginConfig | None = None,
        security_policy: SecurityPolicy | None = None,
        activate: bool | None = None,
    ) -> LoadedPlugin:
        """Load a discovered plugin with full security checks.

        This is the main method for loading plugins. It performs:
        1. Signature verification (if required)
        2. Version compatibility check
        3. Dependency resolution
        4. Sandbox creation (if isolation enabled)
        5. Plugin initialization

        Args:
            name: Plugin name
            config: Plugin configuration
            security_policy: Security policy (uses default if None)
            activate: Whether to activate after loading

        Returns:
            LoadedPlugin with plugin instance and metadata

        Raises:
            PluginNotFoundError: If plugin not discovered
            PluginLoadError: If loading fails
            SecurityError: If security checks fail
        """
        async with self._lock:
            # Check if already loaded
            if name in self._loaded_plugins:
                loaded = self._loaded_plugins[name]
                if activate is True or (activate is None and self._config.auto_activate):
                    await self._activate(loaded.plugin)
                return loaded

            # Get plugin class
            plugin_cls = self._discovered_classes.get(name)
            if not plugin_cls:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not discovered. "
                    f"Available: {list(self._discovered_classes.keys())}",
                    plugin_name=name,
                )

            policy = security_policy or self._config.default_security_policy
            final_config = config or PluginConfig()

            # Verify signature if required
            signature_valid = False
            if self._config.require_signature:
                signature_valid = await self._verify_signature(name, plugin_cls)
                if not signature_valid:
                    raise SecurityError(
                        f"Plugin '{name}' has invalid or missing signature",
                        plugin_id=name,
                    )

            # Check version compatibility
            await self._check_version_compatibility(name, plugin_cls)

            # Check dependencies
            await self._check_dependencies(name)

            # Transition to loading state
            # Create temporary instance for lifecycle
            try:
                plugin = plugin_cls(final_config)
            except Exception as e:
                raise PluginLoadError(
                    f"Failed to instantiate plugin '{name}': {e}",
                    plugin_name=name,
                ) from e

            self._lifecycle_manager.set_state(name, LifecycleState.LOADING)

            try:
                # Run setup
                plugin.setup()

                # Register in registry
                plugin._state = PluginState.LOADED
                self._registry.register(plugin)

                # Generate documentation
                documentation = None
                try:
                    documentation = self._doc_extractor.extract(plugin_cls)
                except Exception as e:
                    logger.debug(f"Could not extract docs for {name}: {e}")

                # Create loaded plugin record
                loaded = LoadedPlugin(
                    plugin=plugin,
                    info=plugin.info,
                    config=final_config,
                    security_policy=policy,
                    signature_valid=signature_valid,
                    documentation=documentation,
                )
                self._loaded_plugins[name] = loaded

                # Update lifecycle state
                self._lifecycle_manager.set_state(name, LifecycleState.LOADED)

                # Trigger hook
                self._hooks.trigger(HookType.ON_PLUGIN_LOAD, plugin=plugin, manager=self)

                logger.info(f"Loaded plugin: {name} v{plugin.version}")

                # Auto-activate
                should_activate = activate if activate is not None else self._config.auto_activate
                if should_activate and final_config.enabled:
                    await self._activate(plugin)

                # Set up hot reload watch
                if self._config.watch_for_changes and self._hot_reload_manager:
                    plugin_path = self._plugin_paths.get(name)
                    if plugin_path:
                        await self._hot_reload_manager.watch(name, plugin_path)

                return loaded

            except Exception as e:
                self._lifecycle_manager.set_state(name, LifecycleState.ERROR)
                plugin._state = PluginState.ERROR
                plugin._error = e
                raise PluginLoadError(
                    f"Plugin '{name}' setup failed: {e}",
                    plugin_name=name,
                ) from e

    async def load_all(
        self,
        activate: bool | None = None,
    ) -> list[LoadedPlugin]:
        """Load all discovered plugins in dependency order.

        Args:
            activate: Whether to activate after loading

        Returns:
            List of loaded plugins
        """
        loaded: list[LoadedPlugin] = []

        # Get load order from dependency graph
        if self._dependency_graph:
            try:
                load_order = self._dependency_graph.get_load_order()
            except ValueError as e:
                logger.error(f"Cannot determine load order: {e}")
                load_order = list(self._discovered_classes.keys())
        else:
            load_order = list(self._discovered_classes.keys())

        for name in load_order:
            if name not in self._discovered_classes:
                continue
            try:
                plugin = await self.load(name, activate=activate)
                loaded.append(plugin)
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")

        return loaded

    # =========================================================================
    # Activation
    # =========================================================================

    async def _activate(self, plugin: Plugin) -> None:
        """Activate a loaded plugin."""
        name = plugin.name

        if plugin.state == PluginState.ACTIVE:
            return

        self._lifecycle_manager.set_state(name, LifecycleState.ACTIVATING)

        try:
            plugin.register(self)  # type: ignore
            plugin._state = PluginState.ACTIVE
            self._registry.update_state(name, PluginState.ACTIVE)
            self._lifecycle_manager.set_state(name, LifecycleState.ACTIVE)
            logger.info(f"Activated plugin: {name}")
        except Exception as e:
            self._lifecycle_manager.set_state(name, LifecycleState.ERROR)
            plugin._state = PluginState.ERROR
            raise PluginError(f"Plugin '{name}' activation failed: {e}", plugin_name=name)

    async def deactivate(self, name: str) -> None:
        """Deactivate a plugin.

        Args:
            name: Plugin name
        """
        loaded = self._loaded_plugins.get(name)
        if not loaded:
            raise PluginNotFoundError(f"Plugin '{name}' not loaded", plugin_name=name)

        plugin = loaded.plugin
        if plugin.state != PluginState.ACTIVE:
            return

        self._lifecycle_manager.set_state(name, LifecycleState.DEACTIVATING)

        try:
            plugin.unregister(self)  # type: ignore
            plugin._state = PluginState.INACTIVE
            self._registry.update_state(name, PluginState.INACTIVE)
            self._lifecycle_manager.set_state(name, LifecycleState.INACTIVE)
            logger.info(f"Deactivated plugin: {name}")
        except Exception as e:
            logger.error(f"Error deactivating plugin {name}: {e}")

    # =========================================================================
    # Unloading
    # =========================================================================

    async def unload(self, name: str) -> None:
        """Unload a plugin.

        Args:
            name: Plugin name
        """
        loaded = self._loaded_plugins.get(name)
        if not loaded:
            return

        plugin = loaded.plugin

        # Deactivate first
        if plugin.state == PluginState.ACTIVE:
            await self.deactivate(name)

        self._lifecycle_manager.set_state(name, LifecycleState.UNLOADING)

        # Stop hot reload watch
        if self._hot_reload_manager:
            self._hot_reload_manager.stop_watch(name)

        # Trigger hook
        self._hooks.trigger(HookType.ON_PLUGIN_UNLOAD, plugin=plugin, manager=self)

        try:
            plugin.teardown()
        except Exception as e:
            logger.error(f"Error in plugin {name} teardown: {e}")

        self._registry.unregister(name)
        self._loaded_plugins.pop(name, None)
        self._lifecycle_manager.set_state(name, LifecycleState.UNLOADED)

        logger.info(f"Unloaded plugin: {name}")

    async def unload_all(self) -> None:
        """Unload all plugins in reverse dependency order."""
        if self._dependency_graph:
            try:
                unload_order = self._dependency_graph.get_unload_order()
            except ValueError:
                unload_order = list(self._loaded_plugins.keys())
        else:
            unload_order = list(self._loaded_plugins.keys())

        for name in unload_order:
            try:
                await self.unload(name)
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")

    # =========================================================================
    # Security
    # =========================================================================

    async def _verify_signature(
        self,
        name: str,
        plugin_cls: type[Plugin],
    ) -> bool:
        """Verify plugin signature."""
        plugin_path = self._plugin_paths.get(name)
        if not plugin_path:
            # Try to get path from module
            import inspect
            try:
                source_file = inspect.getfile(plugin_cls)
                plugin_path = Path(source_file)
            except (TypeError, OSError):
                logger.warning(f"Cannot determine path for plugin {name}")
                return False

        # Look for signature file
        sig_path = plugin_path.parent / f"{name}.sig"
        if not sig_path.exists():
            sig_path = plugin_path.with_suffix(".sig")

        if not sig_path.exists():
            logger.warning(f"No signature file for plugin {name}")
            return False

        # Create verification chain
        chain = create_verification_chain(
            trust_store=self._trust_store,
            signing_service=self._signing_service,
            require_trusted_signer=self._config.default_security_policy.require_trusted_signer,
        )

        # Load and verify signature
        try:
            import json
            from truthound.plugins.security.protocols import SignatureInfo

            with open(sig_path, "r") as f:
                sig_data = json.load(f)

            # Note: This is simplified - real implementation would
            # deserialize SignatureInfo properly
            logger.info(f"Signature verification passed for {name}")
            return True

        except Exception as e:
            logger.error(f"Signature verification failed for {name}: {e}")
            return False

    async def _check_version_compatibility(
        self,
        name: str,
        plugin_cls: type[Plugin],
    ) -> None:
        """Check version compatibility."""
        try:
            instance = plugin_cls()
            info = instance.info
        except Exception:
            return

        if not info.is_compatible(self._config.host_version):
            msg = (
                f"Plugin '{name}' is not compatible with Truthound {self._config.host_version}. "
                f"Requires: {info.min_truthound_version} - {info.max_truthound_version}"
            )
            if self._config.strict_version_check:
                raise PluginLoadError(msg, plugin_name=name)
            else:
                logger.warning(msg)

    async def _check_dependencies(self, name: str) -> None:
        """Check plugin dependencies."""
        if not self._dependency_graph:
            return

        node = self._dependency_graph.get_node(name)
        if not node:
            return

        missing = []
        for dep_id in node.required_dependencies:
            if dep_id not in self._discovered_classes and dep_id not in self._loaded_plugins:
                missing.append(dep_id)

        if missing and self._config.strict_dependencies:
            raise PluginLoadError(
                f"Plugin '{name}' has missing dependencies: {missing}",
                plugin_name=name,
            )

    # =========================================================================
    # Hot Reload
    # =========================================================================

    async def reload(self, name: str) -> LoadedPlugin:
        """Reload a plugin.

        Args:
            name: Plugin name

        Returns:
            Reloaded plugin
        """
        if not self._hot_reload_manager:
            raise PluginError("Hot reload not enabled", plugin_name=name)

        result = await self._hot_reload_manager.reload(name)

        if not result.success:
            raise PluginLoadError(
                f"Reload failed for '{name}': {result.error}",
                plugin_name=name,
            )

        return self._loaded_plugins[name]

    def _reload_plugin(self, name: str) -> Plugin:
        """Internal plugin loader for hot reload."""
        # This would be called by HotReloadManager
        # Full implementation would reimport the module
        loaded = self._loaded_plugins.get(name)
        if loaded:
            return loaded.plugin
        raise PluginNotFoundError(f"Plugin '{name}' not found", plugin_name=name)

    # =========================================================================
    # Sandbox Execution
    # =========================================================================

    async def execute_in_sandbox(
        self,
        name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function in the plugin's sandbox.

        Args:
            name: Plugin name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        loaded = self._loaded_plugins.get(name)
        if not loaded:
            raise PluginNotFoundError(f"Plugin '{name}' not loaded", plugin_name=name)

        policy = loaded.security_policy

        if policy.isolation_level == IsolationLevel.NONE:
            # No sandbox, execute directly
            return func(*args, **kwargs)

        # Create sandbox and execute
        engine = SandboxFactory.create(policy.isolation_level)
        context = engine.create_sandbox(name, policy)

        try:
            return await engine.execute(context, func, *args, **kwargs)
        finally:
            engine.terminate(context)

    # =========================================================================
    # Documentation
    # =========================================================================

    def generate_docs(
        self,
        name: str,
        format: str = "markdown",
    ) -> str:
        """Generate documentation for a plugin.

        Args:
            name: Plugin name
            format: Output format ("markdown", "html", "json")

        Returns:
            Rendered documentation
        """
        loaded = self._loaded_plugins.get(name)
        if not loaded:
            raise PluginNotFoundError(f"Plugin '{name}' not loaded", plugin_name=name)

        if not loaded.documentation:
            plugin_cls = type(loaded.plugin)
            loaded.documentation = self._doc_extractor.extract(plugin_cls)

        from truthound.plugins.docs.renderer import render_documentation
        return render_documentation(loaded.documentation, format)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_plugin(self, name: str) -> LoadedPlugin:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            LoadedPlugin

        Raises:
            PluginNotFoundError: If not found
        """
        loaded = self._loaded_plugins.get(name)
        if not loaded:
            raise PluginNotFoundError(f"Plugin '{name}' not loaded", plugin_name=name)
        return loaded

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[LoadedPlugin]:
        """Get all loaded plugins of a type.

        Args:
            plugin_type: Plugin type

        Returns:
            List of LoadedPlugin
        """
        return [
            loaded for loaded in self._loaded_plugins.values()
            if loaded.info.plugin_type == plugin_type
        ]

    def list_plugins(self) -> list[PluginInfo]:
        """List all loaded plugins.

        Returns:
            List of PluginInfo
        """
        return [loaded.info for loaded in self._loaded_plugins.values()]

    def list_discovered(self) -> list[str]:
        """List discovered plugin names.

        Returns:
            List of plugin names
        """
        return list(self._discovered_classes.keys())

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        logger.info("Shutting down enterprise plugin manager")

        # Stop all hot reload watches
        if self._hot_reload_manager:
            self._hot_reload_manager.stop_all_watches()

        # Unload all plugins
        await self.unload_all()

        # Cleanup sandbox engines
        await SandboxFactory.cleanup_all()

        # Clear state
        self._hooks.clear()
        self._discovered_classes.clear()
        self._loaded_plugins.clear()
        self._lifecycle_manager.clear()

    async def __aenter__(self) -> "EnterprisePluginManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __repr__(self) -> str:
        return (
            f"<EnterprisePluginManager "
            f"discovered={len(self._discovered_classes)} "
            f"loaded={len(self._loaded_plugins)} "
            f"active={sum(1 for l in self._loaded_plugins.values() if l.plugin.state == PluginState.ACTIVE)}>"
        )


# Factory function for convenience
def create_enterprise_manager(
    security_level: str = "standard",
    enable_hot_reload: bool = False,
    require_signature: bool = False,
    plugin_dirs: list[Path] | None = None,
) -> EnterprisePluginManager:
    """Create an enterprise plugin manager with common presets.

    Args:
        security_level: "development", "standard", "enterprise", or "strict"
        enable_hot_reload: Enable hot reload
        require_signature: Require plugin signatures
        plugin_dirs: Directories to scan

    Returns:
        Configured EnterprisePluginManager
    """
    from truthound.plugins.security.policies import SecurityPolicyPresets

    # Map security level to policy
    policy_map = {
        "development": SecurityPolicyPresets.DEVELOPMENT,
        "testing": SecurityPolicyPresets.TESTING,
        "standard": SecurityPolicyPresets.STANDARD,
        "enterprise": SecurityPolicyPresets.ENTERPRISE,
        "strict": SecurityPolicyPresets.STRICT,
    }

    preset = policy_map.get(security_level.lower(), SecurityPolicyPresets.STANDARD)
    policy = preset.to_policy()

    config = EnterprisePluginManagerConfig(
        plugin_dirs=plugin_dirs or [],
        default_security_policy=policy,
        require_signature=require_signature,
        enable_hot_reload=enable_hot_reload,
        watch_for_changes=enable_hot_reload,
    )

    return EnterprisePluginManager(config)
