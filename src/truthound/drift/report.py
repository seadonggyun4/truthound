"""Drift detection report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from truthound.drift.detectors import DriftLevel, DriftResult


@dataclass
class ColumnDrift:
    """Drift information for a single column."""

    column: str
    dtype: str
    result: DriftResult
    baseline_stats: dict
    current_stats: dict

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "drift": self.result.to_dict(),
            "baseline_stats": self.baseline_stats,
            "current_stats": self.current_stats,
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""

    baseline_source: str
    current_source: str
    baseline_rows: int
    current_rows: int
    columns: list[ColumnDrift] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a formatted string representation using Rich."""
        console = Console(force_terminal=True, width=100)
        with console.capture() as capture:
            self._print_to_console(console)
        return capture.get()

    def _print_to_console(self, console: Console) -> None:
        """Print the drift report to a Rich console."""
        console.print()
        console.print("[bold]Truthound Drift Report[/bold]")
        console.print("━" * 70)

        # Summary info
        console.print(f"Baseline: {self.baseline_source} ({self.baseline_rows:,} rows)")
        console.print(f"Current:  {self.current_source} ({self.current_rows:,} rows)")
        console.print()

        if not self.columns:
            console.print("[green]✓ No columns analyzed[/green]")
            console.print()
            return

        # Drift summary
        drifted = [c for c in self.columns if c.result.drifted]
        if not drifted:
            console.print("[green]✓ No drift detected in any column[/green]")
            console.print()
            return

        # Drift table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Method", style="white")
        table.add_column("Statistic", justify="right")
        table.add_column("P-value", justify="right")
        table.add_column("Drift", justify="center")

        # Sort by drift level (highest first)
        level_order = {DriftLevel.HIGH: 0, DriftLevel.MEDIUM: 1, DriftLevel.LOW: 2, DriftLevel.NONE: 3}
        sorted_cols = sorted(self.columns, key=lambda x: level_order[x.result.level])

        for col in sorted_cols:
            r = col.result
            drift_style = self._get_drift_style(r.level)
            p_val = f"{r.p_value:.4f}" if r.p_value is not None else "-"

            table.add_row(
                col.column,
                col.dtype,
                r.method,
                f"{r.statistic:.4f}",
                p_val,
                f"[{drift_style}]{r.level.value}[/{drift_style}]",
            )

        console.print(table)
        console.print()

        # Summary
        high = sum(1 for c in self.columns if c.result.level == DriftLevel.HIGH)
        medium = sum(1 for c in self.columns if c.result.level == DriftLevel.MEDIUM)
        low = sum(1 for c in self.columns if c.result.level == DriftLevel.LOW)

        if high > 0:
            console.print(f"[bold red]⚠ {high} column(s) with HIGH drift[/bold red]")
        if medium > 0:
            console.print(f"[yellow]{medium} column(s) with MEDIUM drift[/yellow]")
        if low > 0:
            console.print(f"[dim]{low} column(s) with LOW drift[/dim]")
        console.print()

    def _get_drift_style(self, level: DriftLevel) -> str:
        """Get Rich style for drift level."""
        return {
            DriftLevel.HIGH: "bold red",
            DriftLevel.MEDIUM: "yellow",
            DriftLevel.LOW: "dim",
            DriftLevel.NONE: "green",
        }[level]

    def print(self) -> None:
        """Print the report to stdout."""
        console = Console()
        self._print_to_console(console)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        drifted_count = sum(1 for c in self.columns if c.result.drifted)
        return {
            "baseline_source": self.baseline_source,
            "current_source": self.current_source,
            "baseline_rows": self.baseline_rows,
            "current_rows": self.current_rows,
            "total_columns": len(self.columns),
            "drifted_columns": drifted_count,
            "columns": [c.to_dict() for c in self.columns],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def has_drift(self) -> bool:
        """Check if any column has drift."""
        return any(c.result.drifted for c in self.columns)

    @property
    def has_high_drift(self) -> bool:
        """Check if any column has high drift."""
        return any(c.result.level == DriftLevel.HIGH for c in self.columns)

    def get_drifted_columns(self) -> list[str]:
        """Get list of column names with drift."""
        return [c.column for c in self.columns if c.result.drifted]
