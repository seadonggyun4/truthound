"""Output formatting utilities for CLI commands.

This module provides standardized output formatting for consistent
display across all CLI commands.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TextIO

import typer


# =============================================================================
# Output Levels
# =============================================================================


class OutputLevel(Enum):
    """Output importance levels."""

    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4


# =============================================================================
# Color Theme
# =============================================================================


@dataclass(frozen=True)
class ColorTheme:
    """Color theme for terminal output."""

    success: str = "green"
    error: str = "red"
    warning: str = "yellow"
    info: str = "blue"
    debug: str = "dim"
    header: str = "bold"
    key: str = "cyan"
    value: str = "white"
    muted: str = "dim"


DEFAULT_THEME = ColorTheme()


# =============================================================================
# Output Formatter Protocol
# =============================================================================


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format(self, data: Any) -> str:
        """Format data for output.

        Args:
            data: Data to format

        Returns:
            Formatted string
        """
        pass

    @abstractmethod
    def write(self, content: str, level: OutputLevel = OutputLevel.INFO) -> None:
        """Write content to output.

        Args:
            content: Content to write
            level: Output level
        """
        pass


# =============================================================================
# Console Output
# =============================================================================


class ConsoleOutput(OutputFormatter):
    """Console output formatter with color support.

    Provides rich terminal output with colors, progress indicators,
    and structured display.
    """

    def __init__(
        self,
        theme: ColorTheme = DEFAULT_THEME,
        no_color: bool = False,
        quiet: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize console output.

        Args:
            theme: Color theme to use
            no_color: Disable colored output
            quiet: Suppress non-essential output
            verbose: Enable verbose output
        """
        self.theme = theme
        self.no_color = no_color
        self.quiet = quiet
        self.verbose = verbose

    def format(self, data: Any) -> str:
        """Format data for console display.

        Args:
            data: Data to format

        Returns:
            Formatted string
        """
        if isinstance(data, dict):
            return self._format_dict(data)
        elif isinstance(data, list):
            return self._format_list(data)
        else:
            return str(data)

    def _format_dict(self, data: dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for display."""
        lines = []
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    def _format_list(self, data: list[Any], indent: int = 0) -> str:
        """Format list for display."""
        lines = []
        prefix = "  " * indent
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(self._format_dict(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)

    def write(self, content: str, level: OutputLevel = OutputLevel.INFO) -> None:
        """Write content to console.

        Args:
            content: Content to write
            level: Output level
        """
        if self.quiet and level.value < OutputLevel.WARNING.value:
            return

        if not self.verbose and level == OutputLevel.DEBUG:
            return

        # Get color for level
        color = self._get_color_for_level(level)

        if self.no_color or color is None:
            typer.echo(content)
        else:
            typer.echo(typer.style(content, fg=color))

    def _get_color_for_level(self, level: OutputLevel) -> str | None:
        """Get color for output level."""
        colors = {
            OutputLevel.DEBUG: self.theme.debug,
            OutputLevel.INFO: None,  # Default color
            OutputLevel.SUCCESS: self.theme.success,
            OutputLevel.WARNING: self.theme.warning,
            OutputLevel.ERROR: self.theme.error,
        }
        return colors.get(level)

    # Convenience methods

    def info(self, message: str) -> None:
        """Write info message."""
        self.write(message, OutputLevel.INFO)

    def success(self, message: str) -> None:
        """Write success message."""
        self.write(message, OutputLevel.SUCCESS)

    def warning(self, message: str) -> None:
        """Write warning message."""
        self.write(f"Warning: {message}", OutputLevel.WARNING)

    def error(self, message: str) -> None:
        """Write error message."""
        self.write(f"Error: {message}", OutputLevel.ERROR)

    def debug(self, message: str) -> None:
        """Write debug message."""
        self.write(f"[DEBUG] {message}", OutputLevel.DEBUG)

    def header(self, title: str, width: int = 60) -> None:
        """Write a section header.

        Args:
            title: Header title
            width: Total width including decoration
        """
        line = "=" * width
        self.write(line)
        self.write(title.center(width))
        self.write(line)

    def subheader(self, title: str, width: int = 60) -> None:
        """Write a subsection header.

        Args:
            title: Subheader title
            width: Total width
        """
        self.write(f"\n{title}")
        self.write("-" * min(len(title), width))

    def key_value(self, key: str, value: Any, indent: int = 0) -> None:
        """Write a key-value pair.

        Args:
            key: Key name
            value: Value
            indent: Indentation level
        """
        prefix = "  " * indent
        self.write(f"{prefix}{key}: {value}")

    def bullet(self, item: str, indent: int = 0) -> None:
        """Write a bullet point.

        Args:
            item: Item text
            indent: Indentation level
        """
        prefix = "  " * indent
        self.write(f"{prefix}- {item}")

    def table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        column_widths: list[int] | None = None,
    ) -> None:
        """Write a simple table.

        Args:
            headers: Column headers
            rows: Table rows
            column_widths: Optional column widths
        """
        if not column_widths:
            column_widths = [
                max(len(str(h)), max(len(str(row[i])) for row in rows) if rows else 0)
                for i, h in enumerate(headers)
            ]

        # Header row
        header_str = " | ".join(
            str(h).ljust(w) for h, w in zip(headers, column_widths)
        )
        self.write(header_str)

        # Separator
        sep = "-+-".join("-" * w for w in column_widths)
        self.write(sep)

        # Data rows
        for row in rows:
            row_str = " | ".join(
                str(cell).ljust(w) for cell, w in zip(row, column_widths)
            )
            self.write(row_str)


# =============================================================================
# JSON Output
# =============================================================================


class JsonOutput(OutputFormatter):
    """JSON output formatter.

    Formats data as JSON for machine consumption.
    """

    def __init__(
        self,
        pretty: bool = True,
        indent: int = 2,
    ) -> None:
        """Initialize JSON output.

        Args:
            pretty: Enable pretty printing
            indent: Indentation level for pretty printing
        """
        self.pretty = pretty
        self.indent = indent if pretty else None

    def format(self, data: Any) -> str:
        """Format data as JSON.

        Args:
            data: Data to format

        Returns:
            JSON string
        """
        return json.dumps(data, indent=self.indent, default=str)

    def write(self, content: str, level: OutputLevel = OutputLevel.INFO) -> None:
        """Write JSON content.

        Args:
            content: Content to write
            level: Output level (ignored for JSON)
        """
        typer.echo(content)

    def write_data(self, data: Any) -> None:
        """Write data as JSON.

        Args:
            data: Data to write
        """
        self.write(self.format(data))

    def write_to_file(self, data: Any, path: Path) -> None:
        """Write data to JSON file.

        Args:
            data: Data to write
            path: Output file path
        """
        path.write_text(self.format(data), encoding="utf-8")


# =============================================================================
# Table Output
# =============================================================================


class TableOutput(OutputFormatter):
    """Table output formatter.

    Formats data as ASCII tables.
    """

    def __init__(
        self,
        max_width: int = 120,
        truncate: bool = True,
    ) -> None:
        """Initialize table output.

        Args:
            max_width: Maximum table width
            truncate: Truncate long content
        """
        self.max_width = max_width
        self.truncate = truncate

    def format(self, data: Any) -> str:
        """Format data as table.

        Args:
            data: Data to format (list of dicts expected)

        Returns:
            Table string
        """
        if not data:
            return "(empty)"

        if isinstance(data, list) and data and isinstance(data[0], dict):
            return self._format_dict_list(data)
        else:
            return str(data)

    def _format_dict_list(self, data: list[dict[str, Any]]) -> str:
        """Format list of dictionaries as table."""
        if not data:
            return "(empty)"

        # Get all keys
        all_keys = []
        for item in data:
            for key in item:
                if key not in all_keys:
                    all_keys.append(key)

        # Calculate column widths
        widths = {}
        for key in all_keys:
            max_value_len = max(
                len(str(item.get(key, ""))) for item in data
            )
            widths[key] = max(len(key), max_value_len)

        # Build table
        lines = []

        # Header
        header = " | ".join(key.ljust(widths[key]) for key in all_keys)
        lines.append(header)

        # Separator
        sep = "-+-".join("-" * widths[key] for key in all_keys)
        lines.append(sep)

        # Rows
        for item in data:
            row = " | ".join(
                str(item.get(key, "")).ljust(widths[key]) for key in all_keys
            )
            lines.append(row)

        return "\n".join(lines)

    def write(self, content: str, level: OutputLevel = OutputLevel.INFO) -> None:
        """Write table content.

        Args:
            content: Content to write
            level: Output level (ignored for tables)
        """
        typer.echo(content)


# =============================================================================
# Utility Functions
# =============================================================================


def get_formatter(
    format_type: str,
    no_color: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> OutputFormatter:
    """Get an output formatter by type.

    Args:
        format_type: Format type (console, json, table)
        no_color: Disable colors for console
        quiet: Quiet mode for console
        verbose: Verbose mode for console

    Returns:
        OutputFormatter instance
    """
    formatters = {
        "console": lambda: ConsoleOutput(
            no_color=no_color, quiet=quiet, verbose=verbose
        ),
        "json": lambda: JsonOutput(pretty=True),
        "table": lambda: TableOutput(),
    }

    factory = formatters.get(format_type.lower(), formatters["console"])
    return factory()


def write_output(
    data: Any,
    output_path: Path | None = None,
    format_type: str = "console",
    **formatter_kwargs: Any,
) -> None:
    """Write output in specified format.

    Args:
        data: Data to output
        output_path: Optional output file path
        format_type: Output format
        **formatter_kwargs: Additional formatter arguments
    """
    formatter = get_formatter(format_type, **formatter_kwargs)
    content = formatter.format(data)

    if output_path:
        output_path.write_text(content, encoding="utf-8")
        typer.echo(f"Output written to: {output_path}")
    else:
        formatter.write(content)
