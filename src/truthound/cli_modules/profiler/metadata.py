"""Metadata listing commands for profiler.

This module implements commands for listing available formats, presets,
and categories for suite generation.
"""

from __future__ import annotations

import typer

from truthound.cli_modules.common.errors import error_boundary


@error_boundary
def list_formats_cmd() -> None:
    """List available output formats for suite generation.

    Displays all supported output formats with descriptions.
    """
    try:
        from truthound.profiler import get_available_formats

        typer.echo("Available output formats:")
        typer.echo("")

        formats_info = {
            "yaml": "Human-readable YAML format (default)",
            "json": "Machine-readable JSON format",
            "python": "Executable Python code with validators",
            "toml": "TOML configuration format",
            "checkpoint": "Truthound checkpoint format for CI/CD",
        }

        for fmt in get_available_formats():
            desc = formats_info.get(fmt, "")
            typer.echo(f"  {fmt:12} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@error_boundary
def list_presets_cmd() -> None:
    """List available configuration presets for suite generation.

    Displays all supported presets with descriptions.
    """
    try:
        from truthound.profiler import get_available_presets

        typer.echo("Available configuration presets:")
        typer.echo("")

        presets_info = {
            "default": "Balanced settings (medium strictness, all categories)",
            "strict": "Strict validation rules with high confidence",
            "loose": "Relaxed validation for flexible data",
            "minimal": "Only high-confidence schema rules",
            "comprehensive": "All generators with detailed output",
            "schema_only": "Schema and completeness rules only",
            "format_only": "Format and pattern rules only",
            "ci_cd": "Optimized for CI/CD pipelines (checkpoint format)",
            "development": "Development-friendly (Python code output)",
            "production": "Production-ready (strict, high confidence)",
        }

        for preset in get_available_presets():
            desc = presets_info.get(preset, "")
            typer.echo(f"  {preset:16} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@error_boundary
def list_categories_cmd() -> None:
    """List available rule categories for suite generation.

    Displays all supported rule categories with descriptions.
    """
    try:
        from truthound.profiler import get_available_categories

        typer.echo("Available rule categories:")
        typer.echo("")

        categories_info = {
            "schema": "Column existence, types, and structure",
            "completeness": "Null values and data completeness",
            "uniqueness": "Unique constraints and cardinality",
            "format": "Data format validation (email, phone, etc.)",
            "distribution": "Statistical distribution checks",
            "pattern": "Regex pattern matching",
            "temporal": "Date/time validation",
            "relationship": "Cross-column relationships",
            "anomaly": "Anomaly detection rules",
        }

        for cat in get_available_categories():
            desc = categories_info.get(cat, "")
            typer.echo(f"  {cat:14} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
