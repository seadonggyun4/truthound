"""CLI module registry and auto-discovery system.

This module provides centralized registration and discovery of CLI
modules, enabling extensible command organization.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol

import typer

logger = logging.getLogger(__name__)


# =============================================================================
# Module Protocol
# =============================================================================


class CLIModuleProtocol(Protocol):
    """Protocol for CLI modules that can be registered."""

    def register_commands(self, parent_app: typer.Typer) -> None:
        """Register module commands with the parent app.

        Args:
            parent_app: Parent Typer app to register commands to
        """
        ...


# =============================================================================
# Module Metadata
# =============================================================================


@dataclass
class ModuleMetadata:
    """Metadata about a CLI module.

    Attributes:
        name: Module name
        description: Human-readable description
        module_path: Python import path
        priority: Load priority (lower = earlier)
        enabled: Whether the module is enabled
        dependencies: Required modules that must load first
    """

    name: str
    description: str
    module_path: str
    priority: int = 100
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)


# =============================================================================
# Module Registry
# =============================================================================


class CLIModuleRegistry:
    """Central registry for CLI modules.

    This class manages CLI module registration, discovery, and loading.
    It supports both automatic discovery from known paths and manual
    registration.

    Example:
        registry = CLIModuleRegistry()

        # Register built-in modules
        registry.register_builtin_modules()

        # Register with app
        app = typer.Typer()
        registry.register_all(app)
    """

    _instance: ClassVar["CLIModuleRegistry | None"] = None

    # Built-in module definitions
    BUILTIN_MODULES: ClassVar[list[ModuleMetadata]] = [
        ModuleMetadata(
            name="core",
            description="Core data quality commands",
            module_path="truthound.cli_modules.core",
            priority=10,
        ),
        ModuleMetadata(
            name="checkpoint",
            description="Checkpoint and CI/CD integration",
            module_path="truthound.cli_modules.checkpoint",
            priority=20,
        ),
        ModuleMetadata(
            name="profiler",
            description="Auto-profiling and rule generation",
            module_path="truthound.cli_modules.profiler",
            priority=30,
        ),
        ModuleMetadata(
            name="advanced",
            description="Advanced features (docs, ml, lineage, realtime)",
            module_path="truthound.cli_modules.advanced",
            priority=40,
        ),
        ModuleMetadata(
            name="scaffolding",
            description="Code scaffolding (th new ...)",
            module_path="truthound.cli_modules.scaffolding.commands",
            priority=50,
        ),
        ModuleMetadata(
            name="plugins",
            description="Plugin management",
            module_path="truthound.plugins.cli",
            priority=60,
        ),
    ]

    def __init__(self) -> None:
        """Initialize the registry."""
        self._modules: dict[str, ModuleMetadata] = {}
        self._loaded: set[str] = set()
        self._failed: dict[str, str] = {}
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> "CLIModuleRegistry":
        """Get the singleton registry instance.

        Returns:
            The global registry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, metadata: ModuleMetadata) -> None:
        """Register a CLI module.

        Args:
            metadata: Module metadata

        Raises:
            ValueError: If module name is already registered
        """
        name = metadata.name.lower()

        if name in self._modules:
            raise ValueError(f"Module '{name}' is already registered")

        self._modules[name] = metadata
        self.logger.debug(f"Registered CLI module: {name}")

    def register_builtin_modules(self) -> None:
        """Register all built-in modules."""
        for metadata in self.BUILTIN_MODULES:
            if metadata.name not in self._modules:
                self.register(metadata)

    def unregister(self, name: str) -> bool:
        """Unregister a CLI module.

        Args:
            name: Module name to unregister

        Returns:
            True if unregistered, False if not found
        """
        name_lower = name.lower()

        if name_lower not in self._modules:
            return False

        del self._modules[name_lower]
        self._loaded.discard(name_lower)
        self._failed.pop(name_lower, None)

        return True

    def get(self, name: str) -> ModuleMetadata | None:
        """Get module metadata by name.

        Args:
            name: Module name

        Returns:
            Module metadata or None if not found
        """
        return self._modules.get(name.lower())

    def list_modules(self) -> list[ModuleMetadata]:
        """List all registered modules.

        Returns:
            List of module metadata, sorted by priority
        """
        return sorted(self._modules.values(), key=lambda m: m.priority)

    def list_enabled(self) -> list[ModuleMetadata]:
        """List all enabled modules.

        Returns:
            List of enabled module metadata, sorted by priority
        """
        return [m for m in self.list_modules() if m.enabled]

    def _load_module(self, metadata: ModuleMetadata) -> Any | None:
        """Load a module by its metadata.

        Args:
            metadata: Module metadata

        Returns:
            Loaded module or None if failed
        """
        if metadata.name in self._loaded:
            return importlib.import_module(metadata.module_path)

        # Check dependencies
        for dep in metadata.dependencies:
            if dep not in self._loaded:
                dep_meta = self.get(dep)
                if dep_meta:
                    self._load_module(dep_meta)

        try:
            module = importlib.import_module(metadata.module_path)
            self._loaded.add(metadata.name)
            self.logger.debug(f"Loaded CLI module: {metadata.name}")
            return module
        except ImportError as e:
            self._failed[metadata.name] = str(e)
            self.logger.warning(
                f"Failed to load CLI module '{metadata.name}': {e}"
            )
            return None

    def register_all(self, app: typer.Typer) -> None:
        """Register all enabled modules with the app.

        Args:
            app: Typer app to register modules with
        """
        for metadata in self.list_enabled():
            self._register_module(app, metadata)

    def _register_module(
        self,
        app: typer.Typer,
        metadata: ModuleMetadata,
    ) -> bool:
        """Register a single module with the app.

        Args:
            app: Typer app
            metadata: Module metadata

        Returns:
            True if registered successfully
        """
        module = self._load_module(metadata)

        if module is None:
            return False

        # Check for register_commands function
        if hasattr(module, "register_commands"):
            try:
                module.register_commands(app)
                self.logger.debug(f"Registered commands from: {metadata.name}")
                return True
            except Exception as e:
                self.logger.error(
                    f"Error registering commands from '{metadata.name}': {e}"
                )
                return False

        # Check for app attribute (sub-typer)
        if hasattr(module, "app"):
            try:
                # For subcommand groups, use the module name
                app.add_typer(module.app, name=metadata.name)
                self.logger.debug(f"Added typer from: {metadata.name}")
                return True
            except Exception as e:
                self.logger.error(f"Error adding typer from '{metadata.name}': {e}")
                return False

        self.logger.warning(
            f"Module '{metadata.name}' has no register_commands or app"
        )
        return False

    def enable(self, name: str) -> bool:
        """Enable a module.

        Args:
            name: Module name

        Returns:
            True if enabled, False if not found
        """
        metadata = self.get(name)
        if metadata:
            metadata.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a module.

        Args:
            name: Module name

        Returns:
            True if disabled, False if not found
        """
        metadata = self.get(name)
        if metadata:
            metadata.enabled = False
            return True
        return False

    def get_load_status(self) -> dict[str, dict[str, Any]]:
        """Get load status of all modules.

        Returns:
            Dictionary of module name -> status info
        """
        status = {}
        for name, metadata in self._modules.items():
            status[name] = {
                "enabled": metadata.enabled,
                "loaded": name in self._loaded,
                "failed": self._failed.get(name),
                "priority": metadata.priority,
                "module_path": metadata.module_path,
            }
        return status

    def __contains__(self, name: str) -> bool:
        """Check if a module is registered."""
        return name.lower() in self._modules

    def __len__(self) -> int:
        """Get number of registered modules."""
        return len(self._modules)


def get_module_registry() -> CLIModuleRegistry:
    """Get the global module registry.

    Returns:
        The singleton registry instance
    """
    return CLIModuleRegistry.get_instance()


# =============================================================================
# Discovery Utilities
# =============================================================================


def discover_plugins() -> list[ModuleMetadata]:
    """Discover CLI modules from entry points.

    This function scans for entry points in the 'truthound.cli' group
    to find third-party CLI extensions.

    Returns:
        List of discovered module metadata
    """
    discovered = []

    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="truthound.cli")

        for ep in eps:
            try:
                metadata = ModuleMetadata(
                    name=ep.name,
                    description=f"Plugin: {ep.name}",
                    module_path=ep.value,
                    priority=200,  # Plugins load after built-ins
                )
                discovered.append(metadata)
                logger.debug(f"Discovered CLI plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load CLI plugin '{ep.name}': {e}")

    except Exception as e:
        logger.debug(f"Entry point discovery not available: {e}")

    return discovered


def auto_register(app: typer.Typer) -> None:
    """Automatically register all CLI modules with the app.

    This is a convenience function that:
    1. Registers built-in modules
    2. Discovers plugin modules
    3. Registers all with the app

    Args:
        app: Typer app to register with
    """
    registry = get_module_registry()

    # Register built-in modules
    registry.register_builtin_modules()

    # Discover and register plugins
    for metadata in discover_plugins():
        try:
            registry.register(metadata)
        except ValueError:
            pass  # Already registered

    # Register all with app
    registry.register_all(app)
