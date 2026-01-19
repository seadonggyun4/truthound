"""Reusable CLI options and arguments.

This module provides standardized, reusable CLI options using Typer's
Annotated type pattern for consistency across all commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, TypeVar

import typer

# =============================================================================
# Callback Functions
# =============================================================================


def file_exists_callback(path: Path) -> Path:
    """Validate that a file exists.

    Args:
        path: File path to validate

    Returns:
        The validated path

    Raises:
        typer.BadParameter: If file doesn't exist
    """
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    return path


def dir_exists_callback(path: Path) -> Path:
    """Validate that a directory exists.

    Args:
        path: Directory path to validate

    Returns:
        The validated path

    Raises:
        typer.BadParameter: If directory doesn't exist
    """
    if not path.exists():
        raise typer.BadParameter(f"Directory not found: {path}")
    if not path.is_dir():
        raise typer.BadParameter(f"Not a directory: {path}")
    return path


def parse_list_option(value: str | None, separator: str = ",") -> list[str] | None:
    """Parse a comma-separated string into a list.

    Args:
        value: Comma-separated string or None
        separator: Separator character

    Returns:
        List of strings or None if input is None
    """
    if value is None:
        return None
    return [item.strip() for item in value.split(separator) if item.strip()]


def parse_list_callback(
    values: list[str] | None,
    separator: str = ",",
) -> list[str] | None:
    """Parse multiple list options into a single flat list.

    Args:
        values: List of comma-separated strings
        separator: Separator character

    Returns:
        Flattened list of strings
    """
    if values is None:
        return None
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(separator) if item.strip())
    return result if result else None


# =============================================================================
# Common Arguments
# =============================================================================


# File input argument (required)
FileArg = Annotated[
    Path,
    typer.Argument(
        help="Path to the data file",
        exists=False,  # We handle existence check in command
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]

# Optional file argument
OptionalFileArg = Annotated[
    Path | None,
    typer.Argument(
        default=None,
        help="Path to the data file",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]

# Directory argument
DirArg = Annotated[
    Path,
    typer.Argument(
        help="Path to the directory",
        exists=False,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]


# =============================================================================
# Common Options
# =============================================================================


# Output file path
OutputOpt = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help="Output file path",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
]

# Output format
FormatOpt = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Output format (console, json, html)",
    ),
]

# Strict mode (exit with error code on issues)
StrictOpt = Annotated[
    bool,
    typer.Option(
        "--strict",
        help="Exit with code 1 if issues are found",
    ),
]

# Verbose mode
VerboseOpt = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
]

# Quiet mode
QuietOpt = Annotated[
    bool,
    typer.Option(
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
]

# Debug mode
DebugOpt = Annotated[
    bool,
    typer.Option(
        "--debug",
        help="Enable debug output",
    ),
]

# Column selection
ColumnsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--columns",
        "-c",
        help="Columns to process (comma-separated, can be specified multiple times)",
    ),
]

# Validator selection
ValidatorsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--validators",
        "-v",
        help="Validators to use (comma-separated, can be specified multiple times)",
    ),
]

# Minimum severity filter
SeverityOpt = Annotated[
    str | None,
    typer.Option(
        "--min-severity",
        "-s",
        help="Minimum severity level (low, medium, high, critical)",
    ),
]

# Schema file path
SchemaOpt = Annotated[
    Path | None,
    typer.Option(
        "--schema",
        help="Schema file for validation",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
]

# Auto schema mode
AutoSchemaOpt = Annotated[
    bool,
    typer.Option(
        "--auto-schema",
        help="Auto-learn and cache schema (zero-config mode)",
    ),
]

# Configuration file
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Configuration file (YAML/JSON)",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
]

# Sample size for profiling
SampleSizeOpt = Annotated[
    int | None,
    typer.Option(
        "--sample",
        "-s",
        help="Sample size for processing (default: all rows)",
        min=1,
    ),
]

# Name option
NameOpt = Annotated[
    str | None,
    typer.Option(
        "--name",
        "-n",
        help="Name for the output",
    ),
]

# Title option
TitleOpt = Annotated[
    str,
    typer.Option(
        "--title",
        "-t",
        help="Title for the output",
    ),
]

# Threshold option
ThresholdOpt = Annotated[
    float | None,
    typer.Option(
        "--threshold",
        "-t",
        help="Threshold value",
    ),
]

# Method selection
MethodOpt = Annotated[
    str,
    typer.Option(
        "--method",
        "-m",
        help="Method to use",
    ),
]

# Iterations
IterationsOpt = Annotated[
    int,
    typer.Option(
        "--iterations",
        "-i",
        help="Number of iterations",
        min=1,
    ),
]


# =============================================================================
# Option Groups (for related options)
# =============================================================================


class OutputOptions:
    """Standard output-related options."""

    output: Path | None = None
    format: str = "console"
    verbose: bool = False
    quiet: bool = False


class ValidationOptions:
    """Standard validation-related options."""

    validators: list[str] | None = None
    min_severity: str | None = None
    schema: Path | None = None
    auto_schema: bool = False
    strict: bool = False


class ProfilingOptions:
    """Standard profiling-related options."""

    sample_size: int | None = None
    include_patterns: bool = True
    include_correlations: bool = False
    top_n: int = 10


# =============================================================================
# Factory Functions
# =============================================================================


T = TypeVar("T")


def create_option(
    name: str,
    help_text: str,
    default: T = None,
    short: str | None = None,
    **kwargs: Any,
) -> Annotated[T, Any]:
    """Create a custom option with consistent styling.

    Args:
        name: Option name (without --)
        help_text: Help text
        default: Default value
        short: Short option name (without -)
        **kwargs: Additional typer.Option arguments

    Returns:
        Annotated type for the option
    """
    names = [f"--{name}"]
    if short:
        names.insert(0, f"-{short}")

    return Annotated[
        type(default) if default is not None else Any,
        typer.Option(*names, help=help_text, default=default, **kwargs),
    ]


def create_argument(
    name: str,
    help_text: str,
    **kwargs: Any,
) -> Annotated[Any, Any]:
    """Create a custom argument with consistent styling.

    Args:
        name: Argument name (for documentation)
        help_text: Help text
        **kwargs: Additional typer.Argument arguments

    Returns:
        Annotated type for the argument
    """
    return Annotated[
        Any,
        typer.Argument(help=help_text, **kwargs),
    ]
