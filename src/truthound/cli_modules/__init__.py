"""CLI modules for Truthound.

This package provides a modular CLI architecture with:
    - common: Shared infrastructure (protocol, options, output, errors)
    - core: Core commands (learn, check, scan, mask, profile, compare)
    - checkpoint: CI/CD integration commands
    - profiler: Auto-profiling and rule generation commands
    - advanced: Advanced feature commands (docs, ml, lineage, realtime)
    - scaffolding: Code generation for validators, reporters, plugins
    - registry: Module registration and auto-discovery

Architecture:
    The CLI is built on a modular, extensible architecture where each
    command group is a separate module that can be independently
    developed, tested, and enabled/disabled.

    Key design principles:
    1. Convention over Configuration - Modules follow naming conventions
    2. Protocol-based - Commands implement CommandProtocol
    3. Auto-discovery - Modules can be discovered from entry points
    4. Lazy loading - Modules are loaded on demand

Usage:
    # In main cli.py
    from truthound.cli_modules.registry import auto_register

    app = typer.Typer()
    auto_register(app)  # Registers all CLI modules

    # Or manual registration
    from truthound.cli_modules import core, checkpoint, profiler, advanced
    core.register_commands(app)
    checkpoint.register_commands(app)
    profiler.register_commands(app)
    advanced.register_commands(app)
"""

# Re-export common utilities
from truthound.cli_modules.common import (
    CommandProtocol,
    CommandResult,
    BaseCommand,
    command,
    CommandRegistry,
    get_command_registry,
    FileArg,
    OutputOpt,
    FormatOpt,
    StrictOpt,
    VerboseOpt,
    file_exists_callback,
    OutputFormatter,
    ConsoleOutput,
    JsonOutput,
    CLIError,
    handle_cli_error,
)

# Re-export registry utilities
from truthound.cli_modules.registry import (
    CLIModuleRegistry,
    ModuleMetadata,
    get_module_registry,
    auto_register,
    discover_plugins,
)

__all__ = [
    # Common - Protocol
    "CommandProtocol",
    "CommandResult",
    "BaseCommand",
    "command",
    "CommandRegistry",
    "get_command_registry",
    # Common - Options
    "FileArg",
    "OutputOpt",
    "FormatOpt",
    "StrictOpt",
    "VerboseOpt",
    "file_exists_callback",
    # Common - Output
    "OutputFormatter",
    "ConsoleOutput",
    "JsonOutput",
    # Common - Errors
    "CLIError",
    "handle_cli_error",
    # Registry
    "CLIModuleRegistry",
    "ModuleMetadata",
    "get_module_registry",
    "auto_register",
    "discover_plugins",
    # Submodules (lazy import)
    "common",
    "core",
    "checkpoint",
    "profiler",
    "advanced",
    "scaffolding",
]


def __getattr__(name: str):
    """Lazy import submodules."""
    import importlib

    _submodules = {
        "common": "truthound.cli_modules.common",
        "core": "truthound.cli_modules.core",
        "checkpoint": "truthound.cli_modules.checkpoint",
        "profiler": "truthound.cli_modules.profiler",
        "advanced": "truthound.cli_modules.advanced",
        "scaffolding": "truthound.cli_modules.scaffolding",
    }

    if name in _submodules:
        module = importlib.import_module(_submodules[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
