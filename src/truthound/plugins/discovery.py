"""Plugin discovery mechanisms.

This module provides multiple ways to discover plugins:
- Entry point discovery (pip installed packages)
- Directory scanning (local plugin directories)
- Module importing (programmatic registration)

Plugin packages should use the 'truthound.plugins' entry point group:

    [project.entry-points."truthound.plugins"]
    my-plugin = "my_package.plugin:MyPlugin"
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from truthound.plugins.base import (
    Plugin,
    PluginInfo,
    PluginType,
    PluginError,
    PluginLoadError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Entry point group for Truthound plugins
ENTRY_POINT_GROUP = "truthound.plugins"


class PluginDiscovery:
    """Discovers plugins from various sources.

    The discovery process finds plugin classes but does not instantiate
    or load them. Use PluginManager for the full lifecycle.

    Example:
        >>> discovery = PluginDiscovery()
        >>>
        >>> # Discover from installed packages
        >>> plugins = discovery.discover_entrypoints()
        >>>
        >>> # Discover from a directory
        >>> plugins = discovery.discover_directory("./plugins")
        >>>
        >>> # Discover all
        >>> plugins = discovery.discover_all()
    """

    def __init__(
        self,
        plugin_dirs: list[Path | str] | None = None,
        scan_entrypoints: bool = True,
    ) -> None:
        """Initialize plugin discovery.

        Args:
            plugin_dirs: Directories to scan for plugins.
            scan_entrypoints: Whether to scan Python entry points.
        """
        self._plugin_dirs = [
            Path(d) if isinstance(d, str) else d for d in (plugin_dirs or [])
        ]
        self._scan_entrypoints = scan_entrypoints
        self._discovered: dict[str, type[Plugin]] = {}

    def discover_all(self) -> dict[str, type[Plugin]]:
        """Discover plugins from all sources.

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        self._discovered.clear()

        # Discover from entry points
        if self._scan_entrypoints:
            self._discovered.update(self.discover_entrypoints())

        # Discover from configured directories
        for plugin_dir in self._plugin_dirs:
            self._discovered.update(self.discover_directory(plugin_dir))

        return self._discovered

    def discover_entrypoints(self) -> dict[str, type[Plugin]]:
        """Discover plugins from Python entry points.

        Packages can register plugins in pyproject.toml:

            [project.entry-points."truthound.plugins"]
            my-plugin = "my_package.plugin:MyPlugin"

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        discovered: dict[str, type[Plugin]] = {}

        try:
            entry_points = importlib.metadata.entry_points()

            # Python 3.10+ uses select(), 3.9 uses get()
            if hasattr(entry_points, "select"):
                plugin_eps = entry_points.select(group=ENTRY_POINT_GROUP)
            else:
                plugin_eps = entry_points.get(ENTRY_POINT_GROUP, [])

            for ep in plugin_eps:
                try:
                    plugin_cls = ep.load()
                    if self._is_valid_plugin_class(plugin_cls):
                        # Get name from entry point or class
                        name = ep.name
                        discovered[name] = plugin_cls
                        logger.debug(f"Discovered plugin from entry point: {name}")
                    else:
                        logger.warning(
                            f"Entry point {ep.name} does not point to a valid Plugin class"
                        )
                except ModuleNotFoundError as e:
                    # Module not found - likely stale entry point from uninstalled package
                    # Get package info if available
                    pkg_info = f" (from {ep.dist.name})" if hasattr(ep, "dist") else ""
                    logger.debug(
                        f"Skipping entry point '{ep.name}'{pkg_info}: {e}. "
                        f"Consider uninstalling the package or reinstalling it."
                    )
                except Exception as e:
                    # Other errors - log as warning with package info
                    pkg_info = f" (from {ep.dist.name})" if hasattr(ep, "dist") else ""
                    logger.warning(f"Failed to load plugin '{ep.name}'{pkg_info}: {e}")

        except Exception as e:
            logger.error(f"Error discovering entry points: {e}")

        return discovered

    def discover_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
    ) -> dict[str, type[Plugin]]:
        """Discover plugins from a directory.

        Scans for Python files containing Plugin subclasses.

        Args:
            directory: Directory to scan.
            recursive: Whether to scan subdirectories.

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        directory = Path(directory)
        discovered: dict[str, type[Plugin]] = {}

        if not directory.exists():
            logger.debug(f"Plugin directory does not exist: {directory}")
            return discovered

        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory.glob(pattern):
            if py_file.name.startswith("_"):
                continue

            try:
                plugins = self._discover_from_file(py_file)
                discovered.update(plugins)
            except Exception as e:
                logger.error(f"Failed to discover plugins from {py_file}: {e}")

        return discovered

    def discover_module(self, module_name: str) -> dict[str, type[Plugin]]:
        """Discover plugins from a Python module.

        Args:
            module_name: Fully qualified module name.

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        discovered: dict[str, type[Plugin]] = {}

        try:
            module = importlib.import_module(module_name)

            for name in dir(module):
                obj = getattr(module, name)
                if self._is_valid_plugin_class(obj):
                    # Create temporary instance to get info
                    try:
                        temp = obj()
                        plugin_name = temp.info.name
                        discovered[plugin_name] = obj
                        logger.debug(f"Discovered plugin from module: {plugin_name}")
                    except Exception:
                        # Use class name as fallback
                        discovered[name.lower()] = obj

        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")

        return discovered

    def _discover_from_file(self, file_path: Path) -> dict[str, type[Plugin]]:
        """Discover plugins from a single Python file.

        Args:
            file_path: Path to Python file.

        Returns:
            Dict mapping plugin names to plugin classes.
        """
        discovered: dict[str, type[Plugin]] = {}

        # Create a unique module name
        module_name = f"_truthound_plugin_{file_path.stem}_{id(file_path)}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return discovered

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            try:
                spec.loader.exec_module(module)

                for name in dir(module):
                    obj = getattr(module, name)
                    if self._is_valid_plugin_class(obj):
                        try:
                            temp = obj()
                            plugin_name = temp.info.name
                            discovered[plugin_name] = obj
                            logger.debug(
                                f"Discovered plugin from file {file_path}: {plugin_name}"
                            )
                        except Exception:
                            discovered[name.lower()] = obj

            finally:
                # Cleanup module
                sys.modules.pop(module_name, None)

        except Exception as e:
            logger.debug(f"Could not load {file_path} as plugin: {e}")

        return discovered

    def _is_valid_plugin_class(self, obj: Any) -> bool:
        """Check if object is a valid Plugin subclass.

        Args:
            obj: Object to check.

        Returns:
            True if it's a valid Plugin subclass.
        """
        try:
            return (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and not getattr(obj, "__abstractmethods__", None)
            )
        except TypeError:
            return False

    def add_plugin_directory(self, directory: Path | str) -> None:
        """Add a directory to scan for plugins.

        Args:
            directory: Directory path to add.
        """
        path = Path(directory) if isinstance(directory, str) else directory
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)

    def remove_plugin_directory(self, directory: Path | str) -> bool:
        """Remove a plugin directory.

        Args:
            directory: Directory to remove.

        Returns:
            True if directory was removed.
        """
        path = Path(directory) if isinstance(directory, str) else directory
        try:
            self._plugin_dirs.remove(path)
            return True
        except ValueError:
            return False

    @property
    def plugin_directories(self) -> list[Path]:
        """Get configured plugin directories."""
        return list(self._plugin_dirs)

    @property
    def discovered_plugins(self) -> dict[str, type[Plugin]]:
        """Get last discovered plugins."""
        return dict(self._discovered)

    def clear_cache(self) -> None:
        """Clear discovered plugins cache."""
        self._discovered.clear()

    def __repr__(self) -> str:
        return (
            f"<PluginDiscovery dirs={len(self._plugin_dirs)} "
            f"discovered={len(self._discovered)}>"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def discover_plugins(
    plugin_dirs: list[Path | str] | None = None,
    scan_entrypoints: bool = True,
) -> dict[str, type[Plugin]]:
    """Discover all plugins.

    Convenience function for one-off discovery.

    Args:
        plugin_dirs: Directories to scan.
        scan_entrypoints: Whether to scan entry points.

    Returns:
        Dict mapping plugin names to classes.
    """
    discovery = PluginDiscovery(
        plugin_dirs=plugin_dirs,
        scan_entrypoints=scan_entrypoints,
    )
    return discovery.discover_all()


def load_plugin_from_module(module_name: str, class_name: str) -> type[Plugin]:
    """Load a specific plugin class from a module.

    Args:
        module_name: Fully qualified module name.
        class_name: Name of the plugin class.

    Returns:
        Plugin class.

    Raises:
        PluginLoadError: If plugin cannot be loaded.
    """
    try:
        module = importlib.import_module(module_name)
        plugin_cls = getattr(module, class_name)

        if not issubclass(plugin_cls, Plugin):
            raise PluginLoadError(
                f"{class_name} is not a Plugin subclass",
                plugin_name=class_name,
            )

        return plugin_cls

    except ImportError as e:
        raise PluginLoadError(
            f"Could not import module {module_name}: {e}",
            plugin_name=class_name,
        )
    except AttributeError as e:
        raise PluginLoadError(
            f"Module {module_name} has no class {class_name}: {e}",
            plugin_name=class_name,
        )
