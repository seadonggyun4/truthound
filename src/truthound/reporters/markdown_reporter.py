"""Markdown format reporter.

This module provides a reporter that outputs validation results in Markdown format.
No external dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from truthound.reporters.base import (
    ReporterConfig,
    RenderError,
    ValidationReporter,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class MarkdownReporterConfig(ReporterConfig):
    """Configuration for Markdown reporter.

    Attributes:
        include_toc: Whether to include a table of contents.
        heading_level: Starting heading level (1-6).
        include_badges: Whether to include status badges.
        table_style: Table style ("github" or "simple").
    """

    include_toc: bool = True
    heading_level: int = 1
    include_badges: bool = True
    table_style: str = "github"


class MarkdownReporter(ValidationReporter[MarkdownReporterConfig]):
    """Markdown format reporter for validation results.

    Outputs validation results as Markdown that can be rendered in
    GitHub, GitLab, or other Markdown-compatible platforms.

    Example:
        >>> reporter = MarkdownReporter(include_toc=True)
        >>> markdown = reporter.render(validation_result)
        >>> reporter.write(validation_result, "report.md")
    """

    name = "markdown"
    file_extension = ".md"
    content_type = "text/markdown"

    @classmethod
    def _default_config(cls) -> MarkdownReporterConfig:
        """Create default configuration."""
        return MarkdownReporterConfig()

    def _heading(self, text: str, level: int = 1) -> str:
        """Create a Markdown heading.

        Args:
            text: Heading text.
            level: Heading level (1-6).

        Returns:
            Markdown heading string.
        """
        adjusted_level = min(6, max(1, level + self._config.heading_level - 1))
        return f"{'#' * adjusted_level} {text}"

    def _badge(self, label: str, value: str, color: str = "blue") -> str:
        """Create a shields.io badge.

        Args:
            label: Badge label.
            value: Badge value.
            color: Badge color.

        Returns:
            Markdown image for the badge.
        """
        encoded_label = label.replace(" ", "%20").replace("-", "--")
        encoded_value = value.replace(" ", "%20").replace("-", "--")
        return f"![{label}](https://img.shields.io/badge/{encoded_label}-{encoded_value}-{color})"

    def _table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Create a Markdown table.

        Args:
            headers: Table headers.
            rows: Table rows.

        Returns:
            Markdown table string.
        """
        lines: list[str] = []

        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        for row in rows:
            # Ensure row has correct number of columns
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[: len(headers)]) + " |")

        return "\n".join(lines)

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as Markdown.

        Args:
            data: The validation result to render.

        Returns:
            Markdown string representation.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            sections: list[str] = []

            # Title
            sections.append(self._heading(self._config.title, 1))
            sections.append("")

            # Badges
            if self._config.include_badges:
                status_color = "success" if data.success else "critical"
                status_text = "PASSED" if data.success else "FAILED"

                badges = [
                    self._badge("Status", status_text, status_color),
                    self._badge("Issues", str(data.statistics.total_issues), "blue"),
                    self._badge("Rows", f"{data.statistics.total_rows:,}", "blue"),
                ]
                sections.append(" ".join(badges))
                sections.append("")

            # Table of Contents
            if self._config.include_toc:
                sections.append(self._heading("Table of Contents", 2))
                sections.append("")
                sections.append("- [Overview](#overview)")
                sections.append("- [Statistics](#statistics)")
                if data.statistics.total_issues > 0:
                    sections.append("- [Issues](#issues)")
                if self._config.include_metadata:
                    sections.append("- [Metadata](#metadata)")
                sections.append("")

            # Overview
            sections.append(self._heading("Overview", 2))
            sections.append("")
            sections.append(f"- **Data Asset**: `{data.data_asset}`")
            sections.append(f"- **Run ID**: `{data.run_id}`")
            sections.append(f"- **Run Time**: {data.run_time.strftime(self._config.timestamp_format)}")
            sections.append(f"- **Status**: {'✅ Passed' if data.success else '❌ Failed'}")
            sections.append("")

            # Statistics
            sections.append(self._heading("Statistics", 2))
            sections.append("")

            stats = data.statistics
            stats_rows = [
                ["Total Rows", f"{stats.total_rows:,}"],
                ["Total Columns", f"{stats.total_columns:,}"],
                ["Total Issues", f"{stats.total_issues:,}"],
                ["Pass Rate", f"{stats.pass_rate:.1%}" if stats.total_validators > 0 else "N/A"],
            ]

            if stats.execution_time_ms > 0:
                if stats.execution_time_ms < 1000:
                    time_str = f"{stats.execution_time_ms:.1f}ms"
                else:
                    time_str = f"{stats.execution_time_ms / 1000:.2f}s"
                stats_rows.append(["Execution Time", time_str])

            sections.append(self._table(["Metric", "Value"], stats_rows))
            sections.append("")

            # Severity breakdown
            if stats.total_issues > 0:
                sections.append(self._heading("Issues by Severity", 3))
                sections.append("")

                severity_rows = []
                if stats.critical_issues:
                    severity_rows.append(["🔴 Critical", str(stats.critical_issues)])
                if stats.high_issues:
                    severity_rows.append(["🟠 High", str(stats.high_issues)])
                if stats.medium_issues:
                    severity_rows.append(["🟡 Medium", str(stats.medium_issues)])
                if stats.low_issues:
                    severity_rows.append(["⚪ Low", str(stats.low_issues)])

                if severity_rows:
                    sections.append(self._table(["Severity", "Count"], severity_rows))
                    sections.append("")

            # Issues
            if data.statistics.total_issues > 0:
                sections.append(self._heading("Issues", 2))
                sections.append("")

                issues = [r for r in data.results if not r.success]

                # Sort by severity
                severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                sorted_issues = sorted(
                    issues,
                    key=lambda x: severity_order.get(
                        x.severity.lower() if x.severity else "low", 4
                    ),
                )

                issue_rows: list[list[str]] = []
                for issue in sorted_issues:
                    severity = issue.severity or "low"
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "⚪",
                    }.get(severity.lower(), "⚪")

                    message = issue.message or ""
                    if len(message) > 50:
                        message = message[:50] + "..."

                    issue_rows.append([
                        f"`{issue.column or '-'}`",
                        issue.issue_type or issue.validator_name,
                        f"{issue.count:,}",
                        f"{severity_emoji} {severity}",
                        message,
                    ])

                sections.append(
                    self._table(
                        ["Column", "Issue Type", "Count", "Severity", "Message"],
                        issue_rows,
                    )
                )
                sections.append("")

            # Metadata
            if self._config.include_metadata:
                sections.append(self._heading("Metadata", 2))
                sections.append("")

                if data.tags:
                    sections.append(self._heading("Tags", 3))
                    sections.append("")
                    for key, value in data.tags.items():
                        sections.append(f"- **{key}**: {value}")
                    sections.append("")

                if data.suite_name:
                    sections.append(f"- **Suite Name**: {data.suite_name}")
                    sections.append("")

                sections.append(f"*Report generated at {datetime.now().strftime(self._config.timestamp_format)}*")
                sections.append("")

            # Footer
            sections.append("---")
            sections.append("")
            sections.append("*Generated by [Truthound](https://github.com/seadonggyun4/Truthound)*")

            return "\n".join(sections)

        except Exception as e:
            raise RenderError(f"Failed to render Markdown: {e}")
