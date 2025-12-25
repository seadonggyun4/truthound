"""Plugin registry for managing registered plugins.

This module provides a thread-safe registry for plugins that allows:
- Plugin registration and lookup
- Filtering by type, state, and tags
- Dependency resolution
"""

from __future__ import annotations

import threading
from typing import Iterator, TYPE_CHECKING
from collections import defaultdict

from truthound.plugins.base import (
    Plugin,
    PluginInfo,
    PluginType,
    PluginState,
    PluginError,
    PluginNotFoundError,
)

if TYPE_CHECKING:
    pass


class PluginRegistry:
    """Thread-safe registry for plugins.

    The registry maintains a collection of plugins and provides
    various ways to query and filter them.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(my_plugin)
        >>> plugin = registry.get("my-plugin")
        >>> validators = registry.get_by_type(PluginType.VALIDATOR)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._plugins: dict[str, Plugin] = {}
        self._by_type: dict[PluginType, dict[str, Plugin]] = defaultdict(dict)
        self._by_state: dict[PluginState, set[str]] = defaultdict(set)
        self._by_tag: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def register(self, plugin: Plugin) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance to register.

        Raises:
            PluginError: If a plugin with the same name is already registered.
        """
        with self._lock:
            name = plugin.name
            if name in self._plugins:
                raise PluginError(
                    f"Plugin '{name}' is already registered",
                    plugin_name=name,
                )

            self._plugins[name] = plugin
            self._by_type[plugin.plugin_type][name] = plugin
            self._by_state[plugin.state].add(name)

            for tag in plugin.info.tags:
                self._by_tag[tag].add(name)

    def unregister(self, name: str) -> Plugin:
        """Unregister a plugin by name.

        Args:
            name: Plugin name to unregister.

        Returns:
            The unregistered plugin.

        Raises:
            PluginNotFoundError: If plugin is not found.
        """
        with self._lock:
            if name not in self._plugins:
                raise PluginNotFoundError(
                    f"Plugin '{name}' is not registered",
                    plugin_name=name,
                )

            plugin = self._plugins.pop(name)

            # Remove from type index
            self._by_type[plugin.plugin_type].pop(name, None)

            # Remove from state index
            for state_set in self._by_state.values():
                state_set.discard(name)

            # Remove from tag index
            for tag_set in self._by_tag.values():
                tag_set.discard(name)

            return plugin

    def get(self, name: str) -> Plugin:
        """Get a plugin by name.

        Args:
            name: Plugin name.

        Returns:
            The plugin instance.

        Raises:
            PluginNotFoundError: If plugin is not found.
        """
        with self._lock:
            if name not in self._plugins:
                raise PluginNotFoundError(
                    f"Plugin '{name}' not found. "
                    f"Available: {', '.join(sorted(self._plugins.keys()))}",
                    plugin_name=name,
                )
            return self._plugins[name]

    def get_or_none(self, name: str) -> Plugin | None:
        """Get a plugin by name, returning None if not found.

        Args:
            name: Plugin name.

        Returns:
            The plugin instance or None.
        """
        with self._lock:
            return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> list[Plugin]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: Type of plugins to get.

        Returns:
            List of plugins of that type.
        """
        with self._lock:
            return list(self._by_type[plugin_type].values())

    def get_by_state(self, state: PluginState) -> list[Plugin]:
        """Get all plugins in a specific state.

        Args:
            state: State to filter by.

        Returns:
            List of plugins in that state.
        """
        with self._lock:
            names = self._by_state[state]
            return [self._plugins[name] for name in names if name in self._plugins]

    def get_by_tag(self, tag: str) -> list[Plugin]:
        """Get all plugins with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of plugins with that tag.
        """
        with self._lock:
            names = self._by_tag[tag]
            return [self._plugins[name] for name in names if name in self._plugins]

    def get_active(self) -> list[Plugin]:
        """Get all active plugins.

        Returns:
            List of active plugins.
        """
        return self.get_by_state(PluginState.ACTIVE)

    def filter(
        self,
        plugin_type: PluginType | None = None,
        state: PluginState | None = None,
        tag: str | None = None,
        enabled: bool | None = None,
    ) -> list[Plugin]:
        """Filter plugins by multiple criteria.

        Args:
            plugin_type: Filter by type.
            state: Filter by state.
            tag: Filter by tag.
            enabled: Filter by enabled status in config.

        Returns:
            List of matching plugins.
        """
        with self._lock:
            plugins = list(self._plugins.values())

            if plugin_type is not None:
                plugins = [p for p in plugins if p.plugin_type == plugin_type]

            if state is not None:
                plugins = [p for p in plugins if p.state == state]

            if tag is not None:
                tag_names = self._by_tag.get(tag, set())
                plugins = [p for p in plugins if p.name in tag_names]

            if enabled is not None:
                plugins = [p for p in plugins if p.config.enabled == enabled]

            return plugins

    def update_state(self, name: str, new_state: PluginState) -> None:
        """Update the state of a plugin.

        Args:
            name: Plugin name.
            new_state: New state to set.
        """
        with self._lock:
            if name not in self._plugins:
                return

            plugin = self._plugins[name]
            old_state = plugin.state

            # Remove from old state set
            self._by_state[old_state].discard(name)

            # Update plugin state
            plugin._state = new_state

            # Add to new state set
            self._by_state[new_state].add(name)

    def list_all(self) -> dict[str, Plugin]:
        """Get all registered plugins.

        Returns:
            Dict of plugin name to plugin instance.
        """
        with self._lock:
            return dict(self._plugins)

    def list_names(self) -> list[str]:
        """Get names of all registered plugins.

        Returns:
            List of plugin names.
        """
        with self._lock:
            return list(self._plugins.keys())

    def list_types(self) -> list[PluginType]:
        """Get all plugin types that have registered plugins.

        Returns:
            List of plugin types with at least one plugin.
        """
        with self._lock:
            return [t for t, plugins in self._by_type.items() if plugins]

    def list_tags(self) -> list[str]:
        """Get all tags used by registered plugins.

        Returns:
            List of tags.
        """
        with self._lock:
            return [t for t, names in self._by_tag.items() if names]

    def contains(self, name: str) -> bool:
        """Check if a plugin is registered.

        Args:
            name: Plugin name to check.

        Returns:
            True if registered.
        """
        with self._lock:
            return name in self._plugins

    def get_info(self, name: str) -> PluginInfo:
        """Get plugin info without full plugin instance.

        Args:
            name: Plugin name.

        Returns:
            PluginInfo for the plugin.
        """
        return self.get(name).info

    def get_all_info(self) -> list[PluginInfo]:
        """Get info for all registered plugins.

        Returns:
            List of PluginInfo instances.
        """
        with self._lock:
            return [p.info for p in self._plugins.values()]

    def clear(self) -> None:
        """Remove all registered plugins."""
        with self._lock:
            self._plugins.clear()
            self._by_type.clear()
            self._by_state.clear()
            self._by_tag.clear()

    def __len__(self) -> int:
        """Return number of registered plugins."""
        with self._lock:
            return len(self._plugins)

    def __iter__(self) -> Iterator[Plugin]:
        """Iterate over registered plugins."""
        with self._lock:
            return iter(list(self._plugins.values()))

    def __contains__(self, name: str) -> bool:
        """Check if plugin is registered."""
        return self.contains(name)

    def __repr__(self) -> str:
        with self._lock:
            type_counts = {t.value: len(p) for t, p in self._by_type.items() if p}
            return f"<PluginRegistry total={len(self._plugins)} types={type_counts}>"
