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
import types
import uuid
from pathlib import Path

from truthound.plugins.base import (
    Plugin,
    PluginLoadError,
)

logger = logging.getLogger(__name__)

# Entry point group for Truthound plugins
ENTRY_POINT_GROUP = "truthound.plugins"


class PluginDiscovery:
    """Import operator-selected plugins and retain explicit discovery failures.

    Discovery executes module code and metadata constructors. It is not a
    sandbox. Owned file modules remain registered until close() for inspection.
    """

    def __init__(self, plugin_dirs=None, scan_entrypoints=True, *, strict=True):
        if type(scan_entrypoints) is not bool or type(strict) is not bool:
            raise ValueError("Discovery switches must be boolean")
        self._plugin_dirs = [Path(path) for path in plugin_dirs or []]
        self._scan_entrypoints = scan_entrypoints
        self._discovered = {}
        self._failures = []
        self._strict = strict
        self._namespace = "_truthound_plugins_" + uuid.uuid4().hex
        self._roots = {}
        self._failed_packages = set()
        self._generation = 0
        from truthound.plugins._discovery_imports import SourceFinder

        self._finder = SourceFinder()

    def _begin(self):
        self._failures = []
        self._roots = {}
        self._failed_packages = set()
        self._generation += 1

    @property
    def failures(self):
        return [dict(item) for item in self._failures]

    def _failure(self, source, error):
        self._failures.append({"source": str(source), "error_type": type(error).__name__})

    def _merge(self, result, name, plugin):
        if name in result and result[name] is not plugin:
            raise ValueError("Conflicting plugin names")
        result[name] = plugin

    def _finish(self, result):
        self._discovered = dict(result)
        if self._strict and self._failures:
            raise ExceptionGroup(
                "Plugin discovery failed",
                [
                    PluginLoadError("Discovery failed (" + item["error_type"] + ")")
                    for item in self._failures
                ],
            )
        return dict(result)

    def discover_all(self):
        self._begin()
        result = {}
        if self._scan_entrypoints:
            self._entrypoints(result)
        for directory in self._plugin_dirs:
            self._directory(directory, True, result)
        return self._finish(result)

    def _entrypoints(self, result):
        try:
            entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as error:
            self._failure("entrypoints", error)
            return
        for entry in sorted(entries, key=lambda item: item.name):
            try:
                plugin = entry.load()
                if not self._is_valid_plugin_class(plugin):
                    raise TypeError("Entry point must resolve to a concrete Plugin")
                self._merge(result, entry.name, plugin)
            except Exception as error:
                self._failure(entry.name, error)

    def discover_entrypoints(self):
        self._begin()
        result = {}
        self._entrypoints(result)
        return self._finish(result)

    def _package(self, name, directory):
        if name in self._failed_packages:
            raise RuntimeError("Package initialization already failed during this discovery")
        if name in sys.modules:
            return
        initializer = directory / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(
                name, initializer, submodule_search_locations=[str(directory)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                exec(compile(initializer.read_bytes(), str(initializer), "exec"), module.__dict__)
            except BaseException:
                self._failed_packages.add(name)
                sys.modules.pop(name, None)
                raise
        else:
            module = types.ModuleType(name)
            module.__package__ = name
            module.__path__ = [str(directory)]
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module

    def _file(self, path, root):
        path = path.resolve(strict=True)
        relative = path.relative_to(root)
        if any(not part.isidentifier() for part in (*relative.parts[:-1], path.stem)):
            raise ValueError("Plugin module path must contain Python identifiers")
        if root not in self._roots:
            self._roots[root] = (
                self._namespace + "_" + str(self._generation) + "_" + str(len(self._roots))
            )
        package = self._roots[root]
        self._finder.roots[package] = root
        if self._finder not in sys.meta_path:
            sys.meta_path.insert(0, self._finder)
        self._package(package, root)
        directory = root
        for part in relative.parts[:-1]:
            directory /= part
            package += "." + part
            self._package(package, directory)
        if path.name == "__init__.py":
            return self._classes(sys.modules[package])
        name = package + "." + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(name)
        sys.modules[name] = module
        try:
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            return self._classes(module)
        except BaseException:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
            raise

    def _classes(self, module):
        result = {}
        for value in tuple(vars(module).values()):
            if self._is_valid_plugin_class(value) and value.__module__ == module.__name__:
                info = value().info
                self._merge(result, info.name, value)
        return result

    def _directory(self, directory, recursive, result):
        try:
            root = Path(directory).resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError("Plugin source must be a directory")
            paths = sorted(root.glob("**/*.py" if recursive else "*.py"))
        except Exception as error:
            self._failure(directory, error)
            return
        for path in paths:
            if path.name.startswith("_") and path.name != "__init__.py":
                continue
            try:
                if path.is_symlink():
                    raise ValueError("Discovery does not follow source symlinks")
                for name, plugin in self._file(path, root).items():
                    self._merge(result, name, plugin)
            except Exception as error:
                self._failure(path, error)

    def discover_directory(self, directory, recursive=True):
        self._begin()
        result = {}
        self._directory(directory, recursive, result)
        return self._finish(result)

    def discover_module(self, module_name):
        self._begin()
        result = {}
        try:
            result = self._classes(importlib.import_module(module_name))
        except Exception as error:
            self._failure(module_name, error)
        return self._finish(result)

    def _discover_from_file(self, file_path):
        path = Path(file_path).resolve(strict=True)
        return self._file(path, path.parent)

    def _is_valid_plugin_class(self, value):
        return (
            isinstance(value, type)
            and issubclass(value, Plugin)
            and value is not Plugin
            and not getattr(value, "__abstractmethods__", None)
        )

    def add_plugin_directory(self, directory):
        path = Path(directory)
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)

    def remove_plugin_directory(self, directory):
        try:
            self._plugin_dirs.remove(Path(directory))
            return True
        except ValueError:
            return False

    @property
    def plugin_directories(self):
        return list(self._plugin_dirs)

    @property
    def discovered_plugins(self):
        return dict(self._discovered)

    def clear_cache(self):
        self._discovered.clear()
        self._failures.clear()

    def close(self):
        self.clear_cache()
        if self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)
        self._finder.roots.clear()
        self._finder.failed.clear()
        for name in tuple(sys.modules):
            if name.startswith(self._namespace + "_"):
                del sys.modules[name]
        self._roots.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f"<PluginDiscovery dirs={len(self._plugin_dirs)} discovered={len(self._discovered)} failures={len(self._failures)}>"


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

    except ImportError:
        raise PluginLoadError("Could not import selected module", plugin_name=class_name) from None
    except AttributeError:
        raise PluginLoadError("Selected class does not exist", plugin_name=class_name) from None
