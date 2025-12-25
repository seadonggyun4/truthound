"""Plugin architecture for Truthound.

Phase 9: Plugin Architecture

This module provides a comprehensive plugin system that allows extending
Truthound's functionality through external packages.

Plugin Types:
    - validators: Custom data validation rules
    - reporters: Output format handlers
    - datasources: Data connection backends
    - profilers: Data profiling analyzers
    - hooks: Event-based extension points

Example:
    >>> from truthound.plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.discover_plugins()
    >>> manager.load_plugin("my-custom-plugin")
"""

from __future__ import annotations

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
    ValidatorPlugin,
    ReporterPlugin,
    DataSourcePlugin,
    HookPlugin,
)
from truthound.plugins.manager import PluginManager
from truthound.plugins.registry import PluginRegistry
from truthound.plugins.hooks import (
    HookManager,
    Hook,
    HookType,
    hook,
    before_validation,
    after_validation,
    before_profile,
    after_profile,
    on_report_generate,
)
from truthound.plugins.discovery import PluginDiscovery
from truthound.plugins.manager import get_plugin_manager

__all__ = [
    # Core classes
    "Plugin",
    "PluginConfig",
    "PluginInfo",
    "PluginType",
    "PluginState",
    # Specialized plugin base classes
    "ValidatorPlugin",
    "ReporterPlugin",
    "DataSourcePlugin",
    "HookPlugin",
    # Manager
    "PluginManager",
    "PluginRegistry",
    "PluginDiscovery",
    "get_plugin_manager",
    # Hooks
    "HookManager",
    "Hook",
    "HookType",
    "hook",
    "before_validation",
    "after_validation",
    "before_profile",
    "after_profile",
    "on_report_generate",
    # Errors
    "PluginError",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginDependencyError",
    "PluginCompatibilityError",
]
