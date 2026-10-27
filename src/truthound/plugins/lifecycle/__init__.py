"""Plugin lifecycle management module.

This module provides lifecycle management for plugins including:
- Loading and unloading
- Hot reload support
- State management
- File watching

Example:
    >>> from truthound.plugins.lifecycle import (
    ...     HotReloadManager,
    ...     ReloadStrategy,
    ... )
    >>>
    >>> from truthound.plugins.manager import PluginManager
    >>> lifecycle = PluginManager()
    >>> hot_reload = HotReloadManager(lifecycle)
    >>> await hot_reload.watch("my-plugin", plugin_path, auto_reload=False)
"""

from __future__ import annotations

from truthound.plugins.lifecycle.manager import (
    LifecycleManager,
    LifecycleEvent,
    LifecycleState,
)
from truthound.plugins.lifecycle.hot_reload import (
    HotReloadManager,
    ReloadStrategy,
    ReloadResult,
    FileWatcher,
)

__all__ = [
    "LifecycleManager",
    "LifecycleEvent",
    "LifecycleState",
    "HotReloadManager",
    "ReloadStrategy",
    "ReloadResult",
    "FileWatcher",
]
