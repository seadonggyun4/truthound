"""CLI commands for plugin management.

This module provides Typer commands for managing plugins from the command line.

Commands:
    - plugin list: List all discovered/loaded plugins
    - plugin install: Install a plugin from PyPI or path
    - plugin uninstall: Uninstall a plugin
    - plugin enable: Enable a plugin
    - plugin disable: Disable a plugin
    - plugin info: Show plugin details
    - plugin create: Create a new plugin template
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional
import json

import typer

from truthound.plugins import (
    PluginManager,
    PluginType,
    PluginState,
    get_plugin_manager,
)


# Create Typer app for plugin commands
app = typer.Typer(
    name="plugin",
    help="Plugin management commands",
    no_args_is_help=True,
)


def _format_state(state: PluginState) -> str:
    """Format plugin state with color."""
    colors = {
        PluginState.DISCOVERED: "yellow",
        PluginState.LOADING: "cyan",
        PluginState.LOADED: "blue",
        PluginState.ACTIVE: "green",
        PluginState.INACTIVE: "dim",
        PluginState.ERROR: "red",
        PluginState.UNLOADING: "yellow",
    }
    color = colors.get(state, "white")
    return f"[{color}]{state.value}[/{color}]"


def _format_type(plugin_type: PluginType) -> str:
    """Format plugin type."""
    return plugin_type.value


@app.command("list")
def list_plugins(
    plugin_type: Annotated[
        Optional[str],
        typer.Option("--type", "-t", help="Filter by plugin type"),
    ] = None,
    state: Annotated[
        Optional[str],
        typer.Option("--state", "-s", help="Filter by state"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """List all discovered and loaded plugins."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        console = None

    manager = get_plugin_manager()
    manager.discover_plugins()

    # Collect plugin data
    plugins_data = []

    # Add discovered but not loaded
    for name, cls in manager.discovery.discovered_plugins.items():
        if name not in manager.registry:
            try:
                temp = cls()
                plugins_data.append({
                    "name": name,
                    "version": temp.version,
                    "type": temp.plugin_type.value,
                    "state": "discovered",
                    "description": temp.info.description[:50] if temp.info.description else "",
                    "author": temp.info.author,
                })
            except Exception:
                plugins_data.append({
                    "name": name,
                    "version": "?",
                    "type": "?",
                    "state": "discovered",
                    "description": "",
                    "author": "",
                })

    # Add loaded plugins
    for plugin in manager.registry:
        plugins_data.append({
            "name": plugin.name,
            "version": plugin.version,
            "type": plugin.plugin_type.value,
            "state": plugin.state.value,
            "description": plugin.info.description[:50] if plugin.info.description else "",
            "author": plugin.info.author,
        })

    # Apply filters
    if plugin_type:
        plugins_data = [p for p in plugins_data if p["type"] == plugin_type]
    if state:
        plugins_data = [p for p in plugins_data if p["state"] == state]

    if json_output:
        typer.echo(json.dumps(plugins_data, indent=2))
        return

    if not plugins_data:
        typer.echo("No plugins found.")
        return

    if console:
        table = Table(title="Truthound Plugins")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Version", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("State")
        if verbose:
            table.add_column("Description")
            table.add_column("Author")

        for p in plugins_data:
            state_style = {
                "discovered": "yellow",
                "loaded": "blue",
                "active": "green",
                "inactive": "dim",
                "error": "red",
            }.get(p["state"], "white")

            row = [
                p["name"],
                p["version"],
                p["type"],
                f"[{state_style}]{p['state']}[/{state_style}]",
            ]
            if verbose:
                row.extend([p["description"], p["author"]])

            table.add_row(*row)

        console.print(table)
    else:
        # Fallback to plain text
        for p in plugins_data:
            if verbose:
                typer.echo(
                    f"{p['name']} v{p['version']} ({p['type']}) - {p['state']}\n"
                    f"  {p['description']}"
                )
            else:
                typer.echo(f"{p['name']} v{p['version']} ({p['type']}) - {p['state']}")


@app.command("info")
def plugin_info(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Show detailed information about a plugin."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
    except ImportError:
        console = None

    manager = get_plugin_manager()
    manager.discover_plugins()

    # Try to find plugin
    plugin = manager.registry.get_or_none(name)
    if not plugin:
        # Try discovered
        cls = manager.discovery.discovered_plugins.get(name)
        if cls:
            try:
                plugin = cls()
            except Exception as e:
                typer.echo(f"Error instantiating plugin: {e}", err=True)
                raise typer.Exit(1)
        else:
            typer.echo(f"Plugin '{name}' not found.", err=True)
            raise typer.Exit(1)

    info = plugin.info

    if json_output:
        data = {
            "name": info.name,
            "version": info.version,
            "type": info.plugin_type.value,
            "description": info.description,
            "author": info.author,
            "homepage": info.homepage,
            "license": info.license,
            "min_truthound_version": info.min_truthound_version,
            "max_truthound_version": info.max_truthound_version,
            "dependencies": list(info.dependencies),
            "python_dependencies": list(info.python_dependencies),
            "tags": list(info.tags),
            "state": plugin.state.value if hasattr(plugin, "_state") else "discovered",
        }
        typer.echo(json.dumps(data, indent=2))
        return

    if console:
        # Build info panel
        content = f"""[bold cyan]Name:[/] {info.name}
[bold cyan]Version:[/] {info.version}
[bold cyan]Type:[/] {info.plugin_type.value}
[bold cyan]Description:[/] {info.description or 'N/A'}
[bold cyan]Author:[/] {info.author or 'N/A'}
[bold cyan]Homepage:[/] {info.homepage or 'N/A'}
[bold cyan]License:[/] {info.license or 'N/A'}

[bold yellow]Compatibility:[/]
  Min Truthound: {info.min_truthound_version or 'Any'}
  Max Truthound: {info.max_truthound_version or 'Any'}

[bold yellow]Dependencies:[/]
  Plugins: {', '.join(info.dependencies) or 'None'}
  Python: {', '.join(info.python_dependencies) or 'None'}

[bold yellow]Tags:[/] {', '.join(info.tags) or 'None'}"""

        console.print(Panel(content, title=f"Plugin: {info.name}"))
    else:
        typer.echo(f"Name: {info.name}")
        typer.echo(f"Version: {info.version}")
        typer.echo(f"Type: {info.plugin_type.value}")
        typer.echo(f"Description: {info.description}")
        typer.echo(f"Author: {info.author}")


@app.command("load")
def load_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to load")],
    activate: Annotated[
        bool,
        typer.Option("--activate/--no-activate", help="Activate after loading"),
    ] = True,
) -> None:
    """Load a discovered plugin."""
    manager = get_plugin_manager()
    manager.discover_plugins()

    try:
        plugin = manager.load_plugin(name, activate=activate)
        typer.echo(f"Loaded plugin: {plugin.name} v{plugin.version}")
        if plugin.state == PluginState.ACTIVE:
            typer.echo("Plugin is now active.")
    except Exception as e:
        typer.echo(f"Error loading plugin: {e}", err=True)
        raise typer.Exit(1)


@app.command("unload")
def unload_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to unload")],
) -> None:
    """Unload a loaded plugin."""
    manager = get_plugin_manager()
    manager.discover_plugins()

    # Load the plugin first if only discovered (not loaded)
    if not manager.is_plugin_loaded(name):
        typer.echo(f"Plugin '{name}' is not loaded.", err=True)
        raise typer.Exit(1)

    try:
        manager.unload_plugin(name)
        typer.echo(f"Unloaded plugin: {name}")
    except Exception as e:
        typer.echo(f"Error unloading plugin: {e}", err=True)
        raise typer.Exit(1)


@app.command("enable")
def enable_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to enable")],
) -> None:
    """Enable a plugin."""
    manager = get_plugin_manager()
    manager.discover_plugins()

    try:
        # Load if not loaded
        if not manager.is_plugin_loaded(name):
            manager.load_plugin(name)

        manager.enable_plugin(name)
        typer.echo(f"Enabled plugin: {name}")
    except Exception as e:
        typer.echo(f"Error enabling plugin: {e}", err=True)
        raise typer.Exit(1)


@app.command("disable")
def disable_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to disable")],
) -> None:
    """Disable a plugin."""
    manager = get_plugin_manager()
    manager.discover_plugins()

    # Load if not loaded
    if not manager.is_plugin_loaded(name):
        manager.load_plugin(name)

    try:
        manager.disable_plugin(name)
        typer.echo(f"Disabled plugin: {name}")
    except Exception as e:
        typer.echo(f"Error disabling plugin: {e}", err=True)
        raise typer.Exit(1)


@app.command("create")
def create_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("."),
    plugin_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Plugin type"),
    ] = "validator",
    author: Annotated[
        Optional[str],
        typer.Option("--author", help="Author name"),
    ] = None,
) -> None:
    """Create a new plugin from template."""
    # Validate plugin name
    import re

    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        typer.echo(
            "Invalid plugin name. Must start with lowercase letter and "
            "contain only lowercase letters, numbers, hyphens, and underscores.",
            err=True,
        )
        raise typer.Exit(1)

    # Create plugin directory
    plugin_dir = output_dir / f"truthound-plugin-{name}"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Create package directory
    pkg_name = name.replace("-", "_")
    pkg_dir = plugin_dir / pkg_name
    pkg_dir.mkdir(exist_ok=True)

    # Create __init__.py
    init_content = f'''"""Truthound plugin: {name}"""

from {pkg_name}.plugin import {_to_class_name(name)}Plugin

__all__ = ["{_to_class_name(name)}Plugin"]
'''
    (pkg_dir / "__init__.py").write_text(init_content)

    # Create plugin.py
    plugin_content = _generate_plugin_template(name, plugin_type, author)
    (pkg_dir / "plugin.py").write_text(plugin_content)

    # Create pyproject.toml
    pyproject_content = _generate_pyproject(name, plugin_type, author)
    (plugin_dir / "pyproject.toml").write_text(pyproject_content)

    # Create README.md
    readme_content = f"""# Truthound Plugin: {name}

A {plugin_type} plugin for Truthound.

## Installation

```bash
pip install truthound-plugin-{name}
```

## Usage

The plugin will be automatically discovered by Truthound.

```python
from truthound.plugins import get_plugin_manager

manager = get_plugin_manager()
manager.discover_plugins()
manager.load_plugin("{name}")
```
"""
    (plugin_dir / "README.md").write_text(readme_content)

    typer.echo(f"Created plugin template at: {plugin_dir}")
    typer.echo("\nNext steps:")
    typer.echo(f"  1. cd {plugin_dir}")
    typer.echo(f"  2. Edit {pkg_name}/plugin.py to implement your plugin")
    typer.echo("  3. pip install -e .")
    typer.echo("  4. truthound plugin list")


def _to_class_name(name: str) -> str:
    """Convert plugin name to class name."""
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)


def _generate_plugin_template(name: str, plugin_type: str, author: str | None) -> str:
    """Generate plugin.py template."""
    class_name = _to_class_name(name)
    pkg_name = name.replace("-", "_")

    if plugin_type == "validator":
        return f'''"""Plugin implementation for {name}."""

from truthound.plugins import (
    ValidatorPlugin,
    PluginInfo,
    PluginType,
)
from truthound.validators.base import Validator, ValidatorConfig, ValidationIssue
from truthound.types import Severity

import polars as pl


class {class_name}Validator(Validator):
    """Custom validator implementation."""

    name = "{name}"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Perform validation.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues found.
        """
        issues: list[ValidationIssue] = []

        # TODO: Implement your validation logic here
        # Example:
        # for col in self._get_target_columns(lf):
        #     # Check something
        #     issues.append(ValidationIssue(
        #         column=col,
        #         issue_type=self.name,
        #         count=1,
        #         severity=Severity.MEDIUM,
        #         details="Description of the issue",
        #     ))

        return issues


class {class_name}Plugin(ValidatorPlugin):
    """Plugin that provides custom validators."""

    def _get_plugin_name(self) -> str:
        return "{name}"

    def _get_plugin_version(self) -> str:
        return "0.1.0"

    def _get_description(self) -> str:
        return "Custom validators for Truthound"

    def get_validators(self) -> list[type]:
        """Return validator classes to register."""
        return [{class_name}Validator]
'''

    elif plugin_type == "reporter":
        return f'''"""Plugin implementation for {name}."""

from truthound.plugins import (
    ReporterPlugin,
    PluginInfo,
    PluginType,
)
from truthound.reporters.base import ValidationReporter, ReporterConfig
from truthound.core import ValidationResult


class {class_name}Reporter(ValidationReporter[ReporterConfig]):
    """Custom reporter implementation."""

    name = "{name}"
    file_extension = ".txt"
    content_type = "text/plain"

    def render(self, data: ValidationResult) -> str:
        """Render validation result to string.

        Args:
            data: Validation result to render.

        Returns:
            Rendered string.
        """
        # TODO: Implement your rendering logic here
        lines = [
            f"Validation Report: {{data.source}}",
            f"Total Issues: {{len(data.issues)}}",
            "",
        ]

        for issue in data.issues:
            lines.append(f"- {{issue.column}}: {{issue.issue_type}} ({{issue.severity.value}})")

        return "\\n".join(lines)


class {class_name}Plugin(ReporterPlugin):
    """Plugin that provides custom reporters."""

    def _get_plugin_name(self) -> str:
        return "{name}"

    def _get_plugin_version(self) -> str:
        return "0.1.0"

    def _get_description(self) -> str:
        return "Custom reporter for Truthound"

    def get_reporters(self) -> dict[str, type]:
        """Return reporter classes to register."""
        return {{"{name}": {class_name}Reporter}}
'''

    elif plugin_type == "hook":
        return f'''"""Plugin implementation for {name}."""

from typing import Any, Callable

from truthound.plugins import (
    HookPlugin,
    PluginInfo,
    PluginType,
    HookType,
)


def on_validation_start(datasource: Any, validators: list, **kwargs: Any) -> None:
    """Called before validation starts."""
    print(f"Starting validation on {{datasource}}")


def on_validation_complete(datasource: Any, result: Any, issues: list, **kwargs: Any) -> None:
    """Called after validation completes."""
    print(f"Validation complete: {{len(issues)}} issues found")


class {class_name}Plugin(HookPlugin):
    """Plugin that provides event hooks."""

    def _get_plugin_name(self) -> str:
        return "{name}"

    def _get_plugin_version(self) -> str:
        return "0.1.0"

    def _get_description(self) -> str:
        return "Custom hooks for Truthound"

    def get_hooks(self) -> dict[str, Callable]:
        """Return hooks to register."""
        return {{
            HookType.BEFORE_VALIDATION.value: on_validation_start,
            HookType.AFTER_VALIDATION.value: on_validation_complete,
        }}
'''

    else:
        return f'''"""Plugin implementation for {name}."""

from truthound.plugins import (
    Plugin,
    PluginInfo,
    PluginType,
    PluginConfig,
)


class {class_name}Plugin(Plugin[PluginConfig]):
    """Custom plugin implementation."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="{name}",
            version="0.1.0",
            plugin_type=PluginType.CUSTOM,
            description="Custom plugin for Truthound",
            author="{author or ''}",
        )

    def setup(self) -> None:
        """Initialize the plugin."""
        pass

    def teardown(self) -> None:
        """Cleanup plugin resources."""
        pass

    def register(self, manager: "PluginManager") -> None:
        """Register plugin components."""
        # TODO: Register your components
        pass
'''


def _generate_pyproject(name: str, plugin_type: str, author: str | None) -> str:
    """Generate pyproject.toml template."""
    pkg_name = name.replace("-", "_")
    class_name = _to_class_name(name)

    return f'''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "truthound-plugin-{name}"
version = "0.1.0"
description = "A {plugin_type} plugin for Truthound"
readme = "README.md"
license = {{text = "MIT"}}
authors = [
    {{name = "{author or 'Your Name'}", email = "your@email.com"}}
]
requires-python = ">=3.10"
dependencies = [
    "truthound>=0.1.0",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.entry-points."truthound.plugins"]
{name} = "{pkg_name}:{class_name}Plugin"

[tool.setuptools.packages.find]
where = ["."]
'''
