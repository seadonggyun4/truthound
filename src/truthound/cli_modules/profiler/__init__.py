"""Profiler CLI commands for Truthound.

This package contains auto-profiling and rule generation commands:
    - auto_profile: Profile data with auto-detection
    - generate_suite: Generate validation rules from profile
    - quick_suite: Profile and generate rules in one step
    - list_formats: List available output formats
    - list_presets: List available configuration presets
    - list_categories: List available rule categories
"""

import typer

from truthound.cli_modules.profiler.auto_profile import auto_profile_cmd
from truthound.cli_modules.profiler.suite import (
    generate_suite_cmd,
    quick_suite_cmd,
)
from truthound.cli_modules.profiler.metadata import (
    list_formats_cmd,
    list_presets_cmd,
    list_categories_cmd,
)

# Profiler app isn't a subcommand - these are top-level commands
# We just organize them here for modularity


def register_commands(parent_app: typer.Typer) -> None:
    """Register profiler commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to
    """
    parent_app.command(name="auto-profile")(auto_profile_cmd)
    parent_app.command(name="generate-suite")(generate_suite_cmd)
    parent_app.command(name="quick-suite")(quick_suite_cmd)
    parent_app.command(name="list-formats")(list_formats_cmd)
    parent_app.command(name="list-presets")(list_presets_cmd)
    parent_app.command(name="list-categories")(list_categories_cmd)


__all__ = [
    "register_commands",
    "auto_profile_cmd",
    "generate_suite_cmd",
    "quick_suite_cmd",
    "list_formats_cmd",
    "list_presets_cmd",
    "list_categories_cmd",
]
