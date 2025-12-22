"""Report generation for validation results."""

import json
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from truthound.types import Severity
from truthound.validators.base import ValidationIssue


@dataclass
class Report:
    """Validation report containing all issues found."""

    issues: list[ValidationIssue] = field(default_factory=list)
    source: str = "unknown"
    row_count: int = 0
    column_count: int = 0

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the report to a Rich console."""
        console.print()
        console.print("[bold]Truthound Report[/bold]")
        console.print("━" * 52)

        if not self.issues:
            console.print("[green]✓ No issues found[/green]")
            console.print()
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Issue", style="white")
        table.add_column("Count", justify="right")
        table.add_column("Severity", justify="center")

        # Sort issues by severity (highest first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        sorted_issues = sorted(self.issues, key=lambda x: severity_order[x.severity])

        for issue in sorted_issues:
            severity_style = self._get_severity_style(issue.severity)
            table.add_row(
                issue.column,
                issue.issue_type,
                f"{issue.count:,}",
                f"[{severity_style}]{issue.severity.value}[/{severity_style}]",
            )

        console.print(table)
        console.print()

        # Summary
        unique_columns = len({i.column for i in self.issues if i.column != "*"})
        console.print(f"Summary: {len(self.issues)} issues found in {unique_columns} columns")
        console.print()

    def _get_severity_style(self, severity: Severity) -> str:
        """Get Rich style for severity level."""
        return {
            Severity.CRITICAL: "bold red",
            Severity.HIGH: "red",
            Severity.MEDIUM: "yellow",
            Severity.LOW: "dim",
        }[severity]

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def filter_by_severity(self, min_severity: Severity) -> "Report":
        """Return a new report with only issues at or above the given severity.

        Args:
            min_severity: Minimum severity level to include.

        Returns:
            New Report with filtered issues.
        """
        filtered_issues = [i for i in self.issues if i.severity >= min_severity]
        return Report(
            issues=filtered_issues,
            source=self.source,
            row_count=self.row_count,
            column_count=self.column_count,
        )

    @property
    def has_issues(self) -> bool:
        """Check if the report contains any issues."""
        return len(self.issues) > 0

    @property
    def has_critical(self) -> bool:
        """Check if the report contains critical issues."""
        return any(i.severity == Severity.CRITICAL for i in self.issues)

    @property
    def has_high(self) -> bool:
        """Check if the report contains high severity issues."""
        return any(i.severity >= Severity.HIGH for i in self.issues)


@dataclass
class PIIReport:
    """Report for PII scan results."""

    findings: list[dict] = field(default_factory=list)
    source: str = "unknown"
    row_count: int = 0

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the PII report to a Rich console."""
        console.print()
        console.print("[bold]Truthound PII Scan[/bold]")
        console.print("━" * 52)

        if not self.findings:
            console.print("[green]✓ No PII detected[/green]")
            console.print()
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("PII Type", style="white")
        table.add_column("Count", justify="right")
        table.add_column("Confidence", justify="center")

        for finding in self.findings:
            table.add_row(
                finding["column"],
                finding["pii_type"],
                f"{finding['count']:,}",
                f"{finding['confidence']}%",
            )

        console.print(table)
        console.print()

        console.print(f"[yellow]Warning: Found {len(self.findings)} columns with potential PII[/yellow]")
        console.print()

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "finding_count": len(self.findings),
            "findings": self.findings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def has_pii(self) -> bool:
        """Check if PII was detected."""
        return len(self.findings) > 0


@dataclass
class ProfileReport:
    """Report for dataset profiling."""

    source: str = "unknown"
    row_count: int = 0
    column_count: int = 0
    size_bytes: int = 0
    columns: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=80)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the profile report to a Rich console."""
        console.print()
        console.print("[bold]Truthound Profile[/bold]")
        console.print("━" * 52)

        size_str = self._format_size(self.size_bytes)
        console.print(f"Dataset: {self.source}")
        console.print(f"Rows: {self.row_count:,} | Columns: {self.column_count} | Size: {size_str}")
        console.print()

        if not self.columns:
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Nulls", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")

        for col in self.columns:
            table.add_row(
                col["name"],
                col["dtype"],
                col["null_pct"],
                col["unique_pct"],
                col.get("min", "-"),
                col.get("max", "-"),
            )

        console.print(table)
        console.print()

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "size_bytes": self.size_bytes,
            "columns": self.columns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
