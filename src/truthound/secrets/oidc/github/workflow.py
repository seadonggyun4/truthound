"""GitHub Actions Workflow Integration Utilities.

This module provides utilities for integrating with GitHub Actions workflows:
- Setting outputs and environment variables
- Creating job summaries
- Logging and masking
- Step outputs

Example:
    >>> from truthound.secrets.oidc.github import (
    ...     set_output,
    ...     set_env,
    ...     add_mask,
    ...     create_summary,
    ... )
    >>>
    >>> # Set workflow output
    >>> set_output("validation_status", "success")
    >>> set_output("issues_found", 5)
    >>>
    >>> # Mask sensitive value
    >>> add_mask(api_key)
    >>>
    >>> # Create job summary
    >>> summary = create_summary()
    >>> summary.add_heading("Validation Results", level=2)
    >>> summary.add_table([
    ...     ["Column", "Status", "Issues"],
    ...     ["email", "Failed", "5"],
    ...     ["phone", "Passed", "0"],
    ... ])
    >>> summary.write()
"""

from __future__ import annotations

import os
import sys
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Sequence


# =============================================================================
# Output Utilities
# =============================================================================


@dataclass
class GitHubActionsOutput:
    """GitHub Actions output manager.

    Handles writing outputs to GITHUB_OUTPUT file for modern GitHub Actions.

    Example:
        >>> output = GitHubActionsOutput()
        >>> output.set("status", "success")
        >>> output.set("data", {"key": "value"})
    """

    _output_file: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._output_file = os.environ.get("GITHUB_OUTPUT")

    def set(self, name: str, value: Any) -> None:
        """Set an output value.

        Args:
            name: Output name.
            value: Output value (will be converted to string/JSON).
        """
        # Convert value to string
        if isinstance(value, (dict, list)):
            str_value = json.dumps(value)
        elif isinstance(value, bool):
            str_value = "true" if value else "false"
        else:
            str_value = str(value)

        if self._output_file:
            # Modern output format (multiline safe)
            with open(self._output_file, "a") as f:
                if "\n" in str_value:
                    # Multiline values use heredoc syntax
                    delimiter = f"EOF_{hash(str_value) % 10000}"
                    f.write(f"{name}<<{delimiter}\n{str_value}\n{delimiter}\n")
                else:
                    f.write(f"{name}={str_value}\n")
        else:
            # Fall back to deprecated ::set-output command
            print(f"::set-output name={name}::{str_value}")

    def set_many(self, outputs: dict[str, Any]) -> None:
        """Set multiple outputs at once.

        Args:
            outputs: Dictionary of output names and values.
        """
        for name, value in outputs.items():
            self.set(name, value)


def set_output(name: str, value: Any) -> None:
    """Set a GitHub Actions output.

    Args:
        name: Output name.
        value: Output value.
    """
    GitHubActionsOutput().set(name, value)


# =============================================================================
# Environment Variable Utilities
# =============================================================================


def set_env(name: str, value: Any) -> None:
    """Set a GitHub Actions environment variable.

    The variable will be available in subsequent steps.

    Args:
        name: Variable name.
        value: Variable value.
    """
    # Convert value to string
    if isinstance(value, (dict, list)):
        str_value = json.dumps(value)
    elif isinstance(value, bool):
        str_value = "true" if value else "false"
    else:
        str_value = str(value)

    env_file = os.environ.get("GITHUB_ENV")

    if env_file:
        with open(env_file, "a") as f:
            if "\n" in str_value:
                delimiter = f"EOF_{hash(str_value) % 10000}"
                f.write(f"{name}<<{delimiter}\n{str_value}\n{delimiter}\n")
            else:
                f.write(f"{name}={str_value}\n")
    else:
        # Fall back to deprecated ::set-env command
        print(f"::set-env name={name}::{str_value}")


def add_path(path: str | Path) -> None:
    """Add a path to GitHub Actions PATH.

    Args:
        path: Directory path to add.
    """
    path_str = str(path)
    path_file = os.environ.get("GITHUB_PATH")

    if path_file:
        with open(path_file, "a") as f:
            f.write(f"{path_str}\n")
    else:
        # Fall back to deprecated ::add-path command
        print(f"::add-path::{path_str}")


# =============================================================================
# Logging Utilities
# =============================================================================


def add_mask(value: str) -> None:
    """Mask a value in GitHub Actions logs.

    The value will be replaced with *** in all subsequent logs.

    Args:
        value: Value to mask.
    """
    print(f"::add-mask::{value}")


def set_failed(message: str) -> None:
    """Set the action as failed with a message.

    Args:
        message: Error message.
    """
    print(f"::error::{message}")
    # Set exit code
    sys.exit(1)


def set_warning(message: str, *, file: str | None = None, line: int | None = None) -> None:
    """Log a warning message.

    Args:
        message: Warning message.
        file: Optional file path.
        line: Optional line number.
    """
    params = []
    if file:
        params.append(f"file={file}")
    if line:
        params.append(f"line={line}")

    if params:
        print(f"::warning {','.join(params)}::{message}")
    else:
        print(f"::warning::{message}")


def set_error(
    message: str,
    *,
    file: str | None = None,
    line: int | None = None,
    col: int | None = None,
    title: str | None = None,
) -> None:
    """Log an error message.

    Args:
        message: Error message.
        file: Optional file path.
        line: Optional line number.
        col: Optional column number.
        title: Optional error title.
    """
    params = []
    if file:
        params.append(f"file={file}")
    if line:
        params.append(f"line={line}")
    if col:
        params.append(f"col={col}")
    if title:
        params.append(f"title={title}")

    if params:
        print(f"::error {','.join(params)}::{message}")
    else:
        print(f"::error::{message}")


def set_notice(
    message: str,
    *,
    file: str | None = None,
    line: int | None = None,
    title: str | None = None,
) -> None:
    """Log a notice message.

    Args:
        message: Notice message.
        file: Optional file path.
        line: Optional line number.
        title: Optional notice title.
    """
    params = []
    if file:
        params.append(f"file={file}")
    if line:
        params.append(f"line={line}")
    if title:
        params.append(f"title={title}")

    if params:
        print(f"::notice {','.join(params)}::{message}")
    else:
        print(f"::notice::{message}")


def debug(message: str) -> None:
    """Log a debug message.

    Only visible when ACTIONS_STEP_DEBUG is set.

    Args:
        message: Debug message.
    """
    print(f"::debug::{message}")


@contextmanager
def log_group(name: str) -> Generator[None, None, None]:
    """Create a collapsible log group.

    Args:
        name: Group name.

    Yields:
        None.

    Example:
        >>> with log_group("Validation Details"):
        ...     print("Checking email column...")
        ...     print("Found 5 issues")
    """
    print(f"::group::{name}")
    try:
        yield
    finally:
        print("::endgroup::")


# =============================================================================
# Job Summary
# =============================================================================


class WorkflowSummary:
    """GitHub Actions job summary builder.

    Creates markdown content for the job summary.

    Example:
        >>> summary = WorkflowSummary()
        >>> summary.add_heading("Validation Results", level=2)
        >>> summary.add_paragraph("All checks passed!")
        >>> summary.add_table([
        ...     ["Check", "Status"],
        ...     ["Email", ":white_check_mark:"],
        ...     ["Phone", ":white_check_mark:"],
        ... ])
        >>> summary.write()
    """

    def __init__(self) -> None:
        self._content: list[str] = []
        self._summary_file = os.environ.get("GITHUB_STEP_SUMMARY")

    def add_raw(self, text: str) -> "WorkflowSummary":
        """Add raw markdown content.

        Args:
            text: Raw markdown text.

        Returns:
            Self for chaining.
        """
        self._content.append(text)
        return self

    def add_heading(self, text: str, level: int = 1) -> "WorkflowSummary":
        """Add a heading.

        Args:
            text: Heading text.
            level: Heading level (1-6).

        Returns:
            Self for chaining.
        """
        level = max(1, min(6, level))
        self._content.append(f"{'#' * level} {text}")
        return self

    def add_paragraph(self, text: str) -> "WorkflowSummary":
        """Add a paragraph.

        Args:
            text: Paragraph text.

        Returns:
            Self for chaining.
        """
        self._content.append(f"\n{text}\n")
        return self

    def add_list(self, items: Sequence[str], ordered: bool = False) -> "WorkflowSummary":
        """Add a list.

        Args:
            items: List items.
            ordered: If True, create numbered list.

        Returns:
            Self for chaining.
        """
        lines = []
        for i, item in enumerate(items, 1):
            if ordered:
                lines.append(f"{i}. {item}")
            else:
                lines.append(f"- {item}")
        self._content.append("\n".join(lines))
        return self

    def add_table(
        self,
        rows: Sequence[Sequence[str]],
        align: str | list[str] | None = None,
    ) -> "WorkflowSummary":
        """Add a markdown table.

        Args:
            rows: Table rows (first row is header).
            align: Column alignment ('left', 'center', 'right') or list.

        Returns:
            Self for chaining.
        """
        if not rows:
            return self

        # Build header
        header = rows[0]
        col_count = len(header)

        # Handle alignment
        alignments: list[str] = []
        if align is None:
            alignments = ["---"] * col_count
        elif isinstance(align, str):
            alignments = [self._get_alignment(align)] * col_count
        else:
            alignments = [self._get_alignment(a) for a in align]
            while len(alignments) < col_count:
                alignments.append("---")

        lines = [
            "| " + " | ".join(str(cell) for cell in header) + " |",
            "| " + " | ".join(alignments[:col_count]) + " |",
        ]

        for row in rows[1:]:
            cells = list(row) + [""] * (col_count - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in cells[:col_count]) + " |")

        self._content.append("\n".join(lines))
        return self

    def _get_alignment(self, align: str) -> str:
        """Convert alignment to markdown format."""
        align = align.lower()
        if align == "center":
            return ":---:"
        elif align == "right":
            return "---:"
        else:
            return "---"

    def add_code_block(self, code: str, language: str = "") -> "WorkflowSummary":
        """Add a code block.

        Args:
            code: Code content.
            language: Language for syntax highlighting.

        Returns:
            Self for chaining.
        """
        self._content.append(f"```{language}\n{code}\n```")
        return self

    def add_quote(self, text: str) -> "WorkflowSummary":
        """Add a blockquote.

        Args:
            text: Quote text.

        Returns:
            Self for chaining.
        """
        lines = text.split("\n")
        quoted = "\n".join(f"> {line}" for line in lines)
        self._content.append(quoted)
        return self

    def add_separator(self) -> "WorkflowSummary":
        """Add a horizontal separator.

        Returns:
            Self for chaining.
        """
        self._content.append("---")
        return self

    def add_link(self, text: str, url: str) -> "WorkflowSummary":
        """Add a link.

        Args:
            text: Link text.
            url: Link URL.

        Returns:
            Self for chaining.
        """
        self._content.append(f"[{text}]({url})")
        return self

    def add_image(self, alt: str, url: str) -> "WorkflowSummary":
        """Add an image.

        Args:
            alt: Alt text.
            url: Image URL.

        Returns:
            Self for chaining.
        """
        self._content.append(f"![{alt}]({url})")
        return self

    def add_collapsible(
        self,
        summary: str,
        content: str,
    ) -> "WorkflowSummary":
        """Add a collapsible section.

        Args:
            summary: Section summary (visible when collapsed).
            content: Section content.

        Returns:
            Self for chaining.
        """
        self._content.append(f"<details>\n<summary>{summary}</summary>\n\n{content}\n\n</details>")
        return self

    def add_badge(
        self,
        label: str,
        message: str,
        color: str = "blue",
    ) -> "WorkflowSummary":
        """Add a shields.io badge.

        Args:
            label: Badge label.
            message: Badge message.
            color: Badge color.

        Returns:
            Self for chaining.
        """
        url = f"https://img.shields.io/badge/{label}-{message}-{color}"
        self._content.append(f"![{label}]({url})")
        return self

    def add_validation_result(
        self,
        name: str,
        passed: bool,
        details: str | None = None,
    ) -> "WorkflowSummary":
        """Add a validation result entry.

        Args:
            name: Validation name.
            passed: Whether validation passed.
            details: Optional details.

        Returns:
            Self for chaining.
        """
        icon = ":white_check_mark:" if passed else ":x:"
        text = f"{icon} **{name}**"
        if details:
            text += f": {details}"
        self._content.append(text)
        return self

    def to_markdown(self) -> str:
        """Convert to markdown string.

        Returns:
            Markdown content.
        """
        return "\n\n".join(self._content)

    def write(self) -> None:
        """Write summary to GITHUB_STEP_SUMMARY file."""
        content = self.to_markdown()

        if self._summary_file:
            with open(self._summary_file, "a") as f:
                f.write(content)
                f.write("\n")
        else:
            # Print to stdout as fallback
            print("\n--- Job Summary ---")
            print(content)
            print("--- End Summary ---\n")

    def clear(self) -> None:
        """Clear the summary content."""
        self._content = []


def create_summary() -> WorkflowSummary:
    """Create a new workflow summary builder.

    Returns:
        WorkflowSummary instance.
    """
    return WorkflowSummary()


# =============================================================================
# Checkpoint Integration
# =============================================================================


@dataclass
class WorkflowOutputConfig:
    """Configuration for workflow output integration.

    Attributes:
        set_status_output: Set 'status' output.
        set_issues_output: Set 'issues' output.
        create_summary: Create job summary.
        fail_on_error: Fail workflow on validation error.
        annotations: Create file annotations.
    """

    set_status_output: bool = True
    set_issues_output: bool = True
    create_summary: bool = True
    fail_on_error: bool = True
    annotations: bool = True


def report_checkpoint_result(
    result: Any,  # CheckpointResult
    config: WorkflowOutputConfig | None = None,
) -> None:
    """Report checkpoint result to GitHub Actions.

    Args:
        result: Checkpoint result.
        config: Output configuration.
    """
    if config is None:
        config = WorkflowOutputConfig()

    output = GitHubActionsOutput()

    # Set outputs
    if config.set_status_output:
        status = getattr(result, "status", "unknown")
        output.set("status", str(status.value) if hasattr(status, "value") else str(status))

    if config.set_issues_output:
        issues = getattr(result, "total_issues", 0)
        output.set("issues", issues)
        output.set("has_issues", issues > 0)

    # Additional outputs
    output.set("checkpoint_name", getattr(result, "checkpoint_name", ""))
    output.set("run_id", getattr(result, "run_id", ""))

    # Create summary
    if config.create_summary:
        summary = create_summary()
        summary.add_heading("Truthound Validation Report", level=2)

        checkpoint_name = getattr(result, "checkpoint_name", "Unknown")
        status = getattr(result, "status", "unknown")
        total_issues = getattr(result, "total_issues", 0)

        # Status badge
        if str(status) in ("success", "passed", "CheckpointStatus.SUCCESS"):
            summary.add_badge("Status", "Passed", "success")
        else:
            summary.add_badge("Status", "Failed", "critical")

        # Summary table
        summary.add_table([
            ["Metric", "Value"],
            ["Checkpoint", str(checkpoint_name)],
            ["Status", str(status)],
            ["Issues Found", str(total_issues)],
            ["Timestamp", datetime.now().isoformat()],
        ])

        summary.write()

    # Fail on error
    if config.fail_on_error:
        status = getattr(result, "status", None)
        if status and str(status) not in ("success", "passed", "CheckpointStatus.SUCCESS"):
            total_issues = getattr(result, "total_issues", 0)
            set_failed(f"Validation failed with {total_issues} issues")
