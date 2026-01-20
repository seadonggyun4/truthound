"""CLI commands for scaffolding.

This module provides the 'new' subcommand group for generating
validators, reporters, plugins, and other components.

Commands:
    - truthound new validator: Create a new validator
    - truthound new reporter: Create a new reporter
    - truthound new plugin: Create a new plugin
    - truthound new list: List available scaffolds
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.scaffolding import (
    ScaffoldConfig,
    ScaffoldResult,
    ScaffoldType,
    get_registry,
)

logger = logging.getLogger(__name__)

# Create Typer app for scaffolding commands
app = typer.Typer(
    name="new",
    help="Generate new components (validators, reporters, plugins)",
    no_args_is_help=True,
)


def register_commands(parent_app: typer.Typer) -> None:
    """Register scaffolding commands with the parent app.

    Registers the scaffolding app under the 'new' command name
    for backward compatibility (e.g., 'truthound new validator').

    Args:
        parent_app: Parent Typer app to register commands to
    """
    parent_app.add_typer(app, name="new")


def _print_result(result: ScaffoldResult, output_dir: Path) -> None:
    """Print generation result to console.

    Args:
        result: Scaffold generation result
        output_dir: Output directory
    """
    if result.success:
        typer.echo(f"\nSuccessfully generated {result.file_count} files:")
        for file in result.files:
            full_path = output_dir / file.path
            typer.echo(f"  {full_path}")

        if result.warnings:
            typer.echo("\nWarnings:")
            for warning in result.warnings:
                typer.echo(f"  - {warning}")
    else:
        typer.echo("Generation failed:", err=True)
        for error in result.errors:
            typer.echo(f"  - {error}", err=True)


def _validate_name(name: str) -> str:
    """Validate component name.

    Args:
        name: Name to validate

    Returns:
        Validated name

    Raises:
        typer.BadParameter: If name is invalid
    """
    import re

    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise typer.BadParameter(
            "Name must start with a lowercase letter and contain only "
            "lowercase letters, numbers, and underscores."
        )
    return name


@app.command("validator")
def new_validator(
    name: Annotated[str, typer.Argument(help="Validator name (snake_case)")],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("."),
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template variant (basic, column, pattern, range, comparison, composite, full)",
        ),
    ] = "basic",
    author: Annotated[
        Optional[str],
        typer.Option("--author", "-a", help="Author name"),
    ] = None,
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Validator description"),
    ] = None,
    category: Annotated[
        str,
        typer.Option("--category", "-c", help="Validator category"),
    ] = "custom",
    include_tests: Annotated[
        bool,
        typer.Option("--tests/--no-tests", help="Generate test files"),
    ] = True,
    include_docs: Annotated[
        bool,
        typer.Option("--docs/--no-docs", help="Generate documentation"),
    ] = False,
    severity: Annotated[
        str,
        typer.Option("--severity", "-s", help="Default severity level"),
    ] = "MEDIUM",
    pattern: Annotated[
        Optional[str],
        typer.Option("--pattern", help="Regex pattern for pattern validators"),
    ] = None,
    min_value: Annotated[
        Optional[float],
        typer.Option("--min", help="Minimum value for range validators"),
    ] = None,
    max_value: Annotated[
        Optional[float],
        typer.Option("--max", help="Maximum value for range validators"),
    ] = None,
) -> None:
    """Create a new custom validator.

    Template variants:
        - basic: Minimal validator with core structure
        - column: Column-level validator with target column support
        - pattern: Pattern matching validator with regex
        - range: Numeric range validator
        - comparison: Cross-column comparison validator
        - composite: Multi-validator composite
        - full: Full-featured with tests and documentation

    Examples:
        # Create a basic validator
        truthound new validator my_validator

        # Create a column validator with tests
        truthound new validator null_check --template column --tests

        # Create a pattern validator
        truthound new validator email_format --template pattern --pattern "^[a-z@.]+$"

        # Create a range validator
        truthound new validator percentage --template range --min 0 --max 100

        # Create a full-featured validator
        truthound new validator customer_data --template full --docs --author "John Doe"
    """
    try:
        name = _validate_name(name)
    except typer.BadParameter as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    registry = get_registry()
    scaffold = registry.get("validator")

    if scaffold is None:
        typer.echo("Error: Validator scaffold not found", err=True)
        raise typer.Exit(1)

    # Validate template
    available_templates = scaffold.get_template_variants()
    if template not in available_templates:
        typer.echo(
            f"Error: Invalid template '{template}'. "
            f"Available: {', '.join(available_templates)}",
            err=True,
        )
        raise typer.Exit(1)

    # Build extra options
    extra = {
        "severity": severity.upper(),
        "category": category,
    }
    if pattern:
        extra["pattern"] = pattern
    if min_value is not None:
        extra["min_value"] = min_value
    if max_value is not None:
        extra["max_value"] = max_value

    # Create configuration
    config = ScaffoldConfig(
        name=name,
        output_dir=output_dir,
        scaffold_type=ScaffoldType.VALIDATOR,
        template_variant=template,
        author=author or "",
        description=description or "",
        category=category,
        include_tests=include_tests,
        include_docs=include_docs,
        extra=extra,
    )

    typer.echo(f"Creating validator '{name}'...")

    # Generate scaffold
    result = scaffold.generate(config)

    if result.success:
        # Write files
        written = result.write_files(output_dir)
        result.files = [f for f in result.files if (output_dir / f.path) in written]
        _print_result(result, output_dir)

        typer.echo("\nNext steps:")
        typer.echo(f"  1. cd {output_dir / name}")
        typer.echo(f"  2. Edit {name}/validator.py to implement your logic")
        if include_tests:
            typer.echo(f"  3. Run tests: pytest {name}/tests/")
    else:
        _print_result(result, output_dir)
        raise typer.Exit(1)


@app.command("reporter")
def new_reporter(
    name: Annotated[str, typer.Argument(help="Reporter name (snake_case)")],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("."),
    template: Annotated[
        str,
        typer.Option("--template", "-t", help="Template variant (basic, full)"),
    ] = "basic",
    author: Annotated[
        Optional[str],
        typer.Option("--author", "-a", help="Author name"),
    ] = None,
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Reporter description"),
    ] = None,
    include_tests: Annotated[
        bool,
        typer.Option("--tests/--no-tests", help="Generate test files"),
    ] = True,
    include_docs: Annotated[
        bool,
        typer.Option("--docs/--no-docs", help="Generate documentation"),
    ] = False,
    file_extension: Annotated[
        str,
        typer.Option("--extension", "-e", help="Output file extension"),
    ] = ".txt",
    content_type: Annotated[
        str,
        typer.Option("--content-type", help="MIME content type"),
    ] = "text/plain",
) -> None:
    """Create a new custom reporter.

    Template variants:
        - basic: Minimal reporter with core structure
        - full: Full-featured with filtering, sorting, and tests

    Examples:
        # Create a basic reporter
        truthound new reporter my_reporter

        # Create a JSON reporter
        truthound new reporter json_export --extension .json --content-type application/json

        # Create a full-featured reporter
        truthound new reporter detailed_report --template full --docs
    """
    try:
        name = _validate_name(name)
    except typer.BadParameter as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    registry = get_registry()
    scaffold = registry.get("reporter")

    if scaffold is None:
        typer.echo("Error: Reporter scaffold not found", err=True)
        raise typer.Exit(1)

    # Build extra options
    extra = {
        "file_extension": file_extension,
        "content_type": content_type,
    }

    # Create configuration
    config = ScaffoldConfig(
        name=name,
        output_dir=output_dir,
        scaffold_type=ScaffoldType.REPORTER,
        template_variant=template,
        author=author or "",
        description=description or "",
        include_tests=include_tests,
        include_docs=include_docs,
        extra=extra,
    )

    typer.echo(f"Creating reporter '{name}'...")

    # Generate scaffold
    result = scaffold.generate(config)

    if result.success:
        written = result.write_files(output_dir)
        result.files = [f for f in result.files if (output_dir / f.path) in written]
        _print_result(result, output_dir)

        typer.echo("\nNext steps:")
        typer.echo(f"  1. cd {output_dir / name}")
        typer.echo(f"  2. Edit {name}/reporter.py to implement your format")
        if include_tests:
            typer.echo(f"  3. Run tests: pytest {name}/tests/")
    else:
        _print_result(result, output_dir)
        raise typer.Exit(1)


def _install_plugin(plugin_dir: Path) -> bool:
    """Install a plugin in editable mode.

    Args:
        plugin_dir: Path to the plugin directory.

    Returns:
        True if installation succeeded, False otherwise.
    """
    import subprocess
    import sys

    typer.echo(f"\nInstalling plugin from {plugin_dir}...")

    try:
        # Run pip install -e . in the plugin directory
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=plugin_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            typer.echo("✓ Plugin installed successfully!")
            return True
        else:
            typer.echo("✗ Plugin installation failed:", err=True)
            if result.stderr:
                # Show only the last few lines of error
                error_lines = result.stderr.strip().split("\n")[-5:]
                for line in error_lines:
                    typer.echo(f"  {line}", err=True)
            return False

    except Exception as e:
        typer.echo(f"✗ Installation error: {e}", err=True)
        return False


@app.command("plugin")
def new_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name (snake_case)")],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("."),
    plugin_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Plugin type (validator, reporter, hook, datasource, action, full)",
        ),
    ] = "validator",
    author: Annotated[
        Optional[str],
        typer.Option("--author", "-a", help="Author name"),
    ] = None,
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Plugin description"),
    ] = None,
    include_tests: Annotated[
        bool,
        typer.Option("--tests/--no-tests", help="Generate test files"),
    ] = True,
    install: Annotated[
        bool,
        typer.Option(
            "--install/--no-install",
            "-i",
            help="Install plugin in editable mode after generation",
        ),
    ] = False,
    min_truthound_version: Annotated[
        str,
        typer.Option("--min-version", help="Minimum Truthound version"),
    ] = "0.1.0",
    python_version: Annotated[
        str,
        typer.Option("--python", help="Minimum Python version"),
    ] = "3.10",
) -> None:
    """Create a new Truthound plugin.

    Plugin types:
        - validator: Plugin that provides custom validators
        - reporter: Plugin that provides custom reporters
        - hook: Plugin that provides event hooks
        - datasource: Plugin that provides data source connectors
        - action: Plugin that provides checkpoint actions
        - full: Full-featured plugin with all components

    Examples:
        # Create a validator plugin
        truthound new plugin my_validators

        # Create and install immediately (recommended)
        truthound new plugin my_validators --install

        # Create a reporter plugin
        truthound new plugin custom_reports --type reporter

        # Create a hook plugin
        truthound new plugin my_hooks --type hook

        # Create a full-featured plugin with auto-install
        truthound new plugin enterprise --type full --author "Company Inc." --install
    """
    try:
        name = _validate_name(name)
    except typer.BadParameter as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    registry = get_registry()
    scaffold = registry.get("plugin")

    if scaffold is None:
        typer.echo("Error: Plugin scaffold not found", err=True)
        raise typer.Exit(1)

    # Validate plugin type
    available_types = scaffold.get_template_variants()
    if plugin_type not in available_types:
        typer.echo(
            f"Error: Invalid plugin type '{plugin_type}'. "
            f"Available: {', '.join(available_types)}",
            err=True,
        )
        raise typer.Exit(1)

    # Build extra options
    extra = {
        "plugin_type": plugin_type,
        "min_truthound_version": min_truthound_version,
        "python_version": python_version,
    }

    # Create configuration
    config = ScaffoldConfig(
        name=name,
        output_dir=output_dir,
        scaffold_type=ScaffoldType.PLUGIN,
        template_variant=plugin_type,
        author=author or "",
        description=description or "",
        include_tests=include_tests,
        extra=extra,
    )

    # Plugin output directory
    plugin_dir = output_dir / f"truthound-plugin-{name}"

    typer.echo(f"Creating plugin '{name}'...")

    # Generate scaffold
    result = scaffold.generate(config)

    if result.success:
        written = result.write_files(plugin_dir)
        result.files = [f for f in result.files if (plugin_dir / f.path) in written]
        _print_result(result, plugin_dir)

        pkg_name = name.replace("-", "_")

        # Auto-install if requested
        if install:
            install_success = _install_plugin(plugin_dir)
            if install_success:
                typer.echo("\nNext steps:")
                typer.echo(f"  1. Edit {plugin_dir}/{pkg_name}/plugin.py to implement your plugin")
                typer.echo("  2. truthound plugin list")
            else:
                typer.echo("\nPlugin created but installation failed.")
                typer.echo("You can install manually:")
                typer.echo(f"  cd {plugin_dir} && pip install -e .")
                raise typer.Exit(1)
        else:
            typer.echo("\nNext steps:")
            typer.echo(f"  1. cd {plugin_dir}")
            typer.echo(f"  2. Edit {pkg_name}/plugin.py to implement your plugin")
            typer.echo("  3. pip install -e .")
            typer.echo("  4. truthound plugin list")
            typer.echo("\nTip: Use --install flag to auto-install the plugin")
    else:
        _print_result(result, plugin_dir)
        raise typer.Exit(1)


@app.command("list")
def list_scaffolds(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
) -> None:
    """List available scaffold types and templates.

    Examples:
        # List all scaffolds
        truthound new list

        # Show detailed information
        truthound new list --verbose
    """
    registry = get_registry()
    scaffolds = registry.list_scaffolds()

    if not scaffolds:
        typer.echo("No scaffolds available.")
        return

    typer.echo("Available scaffolds:\n")

    for name, description in scaffolds:
        scaffold = registry.get(name)
        if scaffold is None:
            continue

        typer.echo(f"  {name}")
        typer.echo(f"    {description}")

        if verbose:
            # Show aliases
            if scaffold.aliases:
                typer.echo(f"    Aliases: {', '.join(scaffold.aliases)}")

            # Show template variants
            variants = scaffold.get_template_variants()
            if variants:
                typer.echo(f"    Templates: {', '.join(variants)}")

            # Show options
            options = scaffold.get_options()
            if options:
                typer.echo("    Options:")
                for opt_name, opt_info in options.items():
                    default = opt_info.get("default", "None")
                    typer.echo(f"      --{opt_name}: {opt_info.get('description', '')} (default: {default})")

        typer.echo()


@app.command("templates")
def list_templates(
    scaffold_type: Annotated[
        str,
        typer.Argument(help="Scaffold type (validator, reporter, plugin)"),
    ],
) -> None:
    """List available templates for a scaffold type.

    Examples:
        truthound new templates validator
        truthound new templates plugin
    """
    registry = get_registry()
    scaffold = registry.get(scaffold_type)

    if scaffold is None:
        typer.echo(
            f"Error: Unknown scaffold type '{scaffold_type}'. "
            f"Available: {', '.join(registry.list_names())}",
            err=True,
        )
        raise typer.Exit(1)

    variants = scaffold.get_template_variants()

    typer.echo(f"Templates for '{scaffold_type}':\n")
    for variant in variants:
        typer.echo(f"  - {variant}")

    typer.echo(f"\nUsage: truthound new {scaffold_type} <name> --template <template>")
