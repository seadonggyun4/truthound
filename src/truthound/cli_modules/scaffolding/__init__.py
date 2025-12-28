"""Scaffolding system for generating project templates.

This module provides an extensible scaffolding framework for generating
validators, reporters, plugins, and other components.

Architecture:
    - ScaffoldProtocol: Interface for scaffold generators
    - ScaffoldRegistry: Central registry for scaffold types
    - ScaffoldConfig: Configuration for scaffold generation
    - ScaffoldResult: Result of scaffold generation

Example:
    from truthound.cli_modules.scaffolding import (
        ScaffoldRegistry,
        ScaffoldConfig,
        get_registry,
    )

    # Get the singleton registry
    registry = get_registry()

    # Generate a validator
    config = ScaffoldConfig(
        name="my_validator",
        output_dir=Path("."),
        author="John Doe",
    )
    result = registry.generate("validator", config)

Extensibility:
    To add a new scaffold type, implement the ScaffoldProtocol and register it:

    @register_scaffold("my_type")
    class MyScaffold:
        name = "my_type"
        description = "My custom scaffold"

        def generate(self, config: ScaffoldConfig) -> ScaffoldResult:
            ...

        def get_templates(self) -> list[str]:
            ...
"""

from truthound.cli_modules.scaffolding.base import (
    # Protocols
    ScaffoldProtocol,
    # Data classes
    ScaffoldConfig,
    ScaffoldResult,
    ScaffoldFile,
    ScaffoldMetadata,
    ScaffoldType,
    # Registry
    ScaffoldRegistry,
    get_registry,
    register_scaffold,
    # Utilities
    snake_to_pascal,
    snake_to_kebab,
    pascal_to_snake,
)

from truthound.cli_modules.scaffolding.validators import ValidatorScaffold
from truthound.cli_modules.scaffolding.reporters import ReporterScaffold
from truthound.cli_modules.scaffolding.plugins import PluginScaffold

__all__ = [
    # Protocols
    "ScaffoldProtocol",
    # Data classes
    "ScaffoldConfig",
    "ScaffoldResult",
    "ScaffoldFile",
    "ScaffoldMetadata",
    "ScaffoldType",
    # Registry
    "ScaffoldRegistry",
    "get_registry",
    "register_scaffold",
    # Utilities
    "snake_to_pascal",
    "snake_to_kebab",
    "pascal_to_snake",
    # Scaffolds
    "ValidatorScaffold",
    "ReporterScaffold",
    "PluginScaffold",
]
