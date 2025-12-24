"""Console/Terminal format reporter.

This module provides a reporter that outputs validation results to the console
using Rich for formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from truthound.reporters.base import (
    ReporterConfig,
    RenderError,
    ValidationReporter,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class ConsoleReporterConfig(ReporterConfig):
    """Configuration for console reporter.

    Attributes:
        color: Whether to use colors in output.
        width: Maximum width of output (None for auto).
        show_header: Whether to show the report header.
        show_summary: Whether to show the summary section.
        show_issues_table: Whether to show the issues table.
        compact: Whether to use compact output format.
        severity_colors: Color mapping for severity levels.
    """

    color: bool = True
    width: int | None = None
    show_header: bool = True
    show_summary: bool = True
    show_issues_table: bool = True
    compact: bool = False
    severity_colors: dict[str, str] | None = None

    def get_severity_color(self, severity: str) -> str:
        """Get the color for a severity level."""
        default_colors = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
        }

        colors = self.severity_colors or default_colors
        return colors.get(severity.lower(), "white")


class ConsoleReporter(ValidationReporter[ConsoleReporterConfig]):
    """Console format reporter for validation results.

    Outputs validation results to the terminal with Rich formatting,
    including colors, tables, and panels.

    Example:
        >>> reporter = ConsoleReporter(color=True)
        >>> output = reporter.render(validation_result)
        >>> print(output)
        >>>
        >>> # Or print directly
        >>> reporter.print(validation_result)
    """

    name = "console"
    file_extension = ".txt"
    content_type = "text/plain"

    @classmethod
    def _default_config(cls) -> ConsoleReporterConfig:
        """Create default configuration."""
        return ConsoleReporterConfig()

    def _create_console(self, capture: bool = True) -> Console:
        """Create a Rich console instance.

        Args:
            capture: Whether to capture output (for render()).

        Returns:
            Configured Console instance.
        """
        return Console(
            force_terminal=self._config.color,
            width=self._config.width,
            record=capture,
            file=StringIO() if capture else None,
        )

    def _render_header(self, console: Console, result: "ValidationResult") -> None:
        """Render the report header.

        Args:
            console: The Rich console.
            result: The validation result.
        """
        status_style = "green" if result.success else "red"
        status_text = "PASSED" if result.success else "FAILED"

        header_text = Text()
        header_text.append(f"{self._config.title}\n", style="bold")
        header_text.append(f"Data Asset: ", style="dim")
        header_text.append(f"{result.data_asset}\n", style="cyan")
        header_text.append(f"Run ID: ", style="dim")
        header_text.append(f"{result.run_id}\n", style="cyan")
        header_text.append(f"Status: ", style="dim")
        header_text.append(status_text, style=f"bold {status_style}")

        console.print(Panel(header_text, title="Truthound Report", border_style="blue"))

    def _render_summary(self, console: Console, result: "ValidationResult") -> None:
        """Render the summary section.

        Args:
            console: The Rich console.
            result: The validation result.
        """
        stats = result.statistics

        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Total Rows", f"{stats.total_rows:,}")
        table.add_row("Total Columns", f"{stats.total_columns:,}")
        table.add_row("Total Issues", f"{stats.total_issues:,}")
        table.add_row(
            "Pass Rate",
            f"{stats.pass_rate:.1%}" if stats.total_validators > 0 else "N/A",
        )

        if stats.execution_time_ms > 0:
            if stats.execution_time_ms < 1000:
                time_str = f"{stats.execution_time_ms:.1f}ms"
            else:
                time_str = f"{stats.execution_time_ms / 1000:.2f}s"
            table.add_row("Execution Time", time_str)

        console.print()
        console.print(table)

        # Severity breakdown
        if stats.total_issues > 0:
            severity_text = Text()
            severity_text.append("Issues by Severity: ", style="dim")

            parts = []
            if stats.critical_issues:
                parts.append(f"[bold red]{stats.critical_issues} critical[/]")
            if stats.high_issues:
                parts.append(f"[red]{stats.high_issues} high[/]")
            if stats.medium_issues:
                parts.append(f"[yellow]{stats.medium_issues} medium[/]")
            if stats.low_issues:
                parts.append(f"[dim]{stats.low_issues} low[/]")

            console.print()
            console.print(f"Issues by Severity: {', '.join(parts)}")

    def _render_issues_table(self, console: Console, result: "ValidationResult") -> None:
        """Render the issues table.

        Args:
            console: The Rich console.
            result: The validation result.
        """
        issues = [r for r in result.results if not r.success]

        if not issues:
            console.print()
            console.print("[green]✓ No issues found[/green]")
            return

        table = Table(title="Issues", show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Issue Type")
        table.add_column("Count", justify="right")
        table.add_column("Severity", justify="center")
        table.add_column("Message", max_width=40)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            issues,
            key=lambda x: severity_order.get(x.severity.lower() if x.severity else "low", 4),
        )

        for issue in sorted_issues:
            severity = issue.severity or "low"
            severity_style = self._config.get_severity_color(severity)

            table.add_row(
                issue.column or "-",
                issue.issue_type or issue.validator_name,
                f"{issue.count:,}",
                f"[{severity_style}]{severity}[/{severity_style}]",
                (issue.message or "")[:40] + "..." if issue.message and len(issue.message) > 40 else (issue.message or "-"),
            )

        console.print()
        console.print(table)

    def _render_compact(self, console: Console, result: "ValidationResult") -> None:
        """Render a compact version of the report.

        Args:
            console: The Rich console.
            result: The validation result.
        """
        status_icon = "✓" if result.success else "✗"
        status_style = "green" if result.success else "red"

        console.print(
            f"[{status_style}]{status_icon}[/{status_style}] "
            f"[bold]{result.data_asset}[/bold]: "
            f"{result.status.value} "
            f"({result.statistics.total_issues} issues)"
        )

        if not result.success:
            for issue in result.results:
                if not issue.success:
                    severity_style = self._config.get_severity_color(issue.severity or "low")
                    console.print(
                        f"  [{severity_style}]•[/{severity_style}] "
                        f"{issue.column or '-'}: {issue.issue_type}"
                    )

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as console output.

        Args:
            data: The validation result to render.

        Returns:
            Formatted console output as a string.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            console = self._create_console(capture=True)

            if self._config.compact:
                self._render_compact(console, data)
            else:
                if self._config.show_header:
                    self._render_header(console, data)

                if self._config.show_summary:
                    self._render_summary(console, data)

                if self._config.show_issues_table:
                    self._render_issues_table(console, data)

                console.print()

            return console.export_text(clear=True)

        except Exception as e:
            raise RenderError(f"Failed to render console output: {e}")

    def print(self, data: "ValidationResult") -> None:
        """Print validation result directly to the console.

        Args:
            data: The validation result to print.
        """
        console = Console(
            force_terminal=self._config.color,
            width=self._config.width,
        )

        if self._config.compact:
            self._render_compact(console, data)
        else:
            if self._config.show_header:
                self._render_header(console, data)

            if self._config.show_summary:
                self._render_summary(console, data)

            if self._config.show_issues_table:
                self._render_issues_table(console, data)

            console.print()
