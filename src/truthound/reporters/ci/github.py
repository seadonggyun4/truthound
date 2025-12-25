"""GitHub Actions Reporter.

This module provides a reporter that outputs validation results using
GitHub Actions workflow commands for annotations and job summaries.

Reference:
    https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from truthound.reporters.ci.base import (
    BaseCIReporter,
    CIAnnotation,
    CIPlatform,
    CIReporterConfig,
    AnnotationLevel,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class GitHubActionsConfig(CIReporterConfig):
    """Configuration for GitHub Actions reporter.

    Attributes:
        step_summary: Write summary to GITHUB_STEP_SUMMARY file.
        use_groups: Use ::group:: commands for collapsible sections.
        emoji_enabled: Include emojis in output.
        set_output: Set workflow output variables.
        output_name: Name for the output variable.
    """

    step_summary: bool = True
    use_groups: bool = True
    emoji_enabled: bool = True
    set_output: bool = False
    output_name: str = "validation_result"


class GitHubActionsReporter(BaseCIReporter):
    """GitHub Actions reporter with annotations and job summaries.

    Outputs validation results using GitHub Actions workflow commands:
    - ::error:: / ::warning:: / ::notice:: for code annotations
    - ::group:: / ::endgroup:: for collapsible sections
    - Job summary via GITHUB_STEP_SUMMARY

    Example:
        >>> reporter = GitHubActionsReporter()
        >>> exit_code = reporter.report_to_ci(validation_result)
        >>> sys.exit(exit_code)

    Environment Variables:
        GITHUB_ACTIONS: Set to 'true' when running in GitHub Actions.
        GITHUB_STEP_SUMMARY: Path to file for job summary.
        GITHUB_OUTPUT: Path to file for output variables.
    """

    platform = CIPlatform.GITHUB_ACTIONS
    name = "github"
    file_extension = ".md"
    content_type = "text/markdown"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 50  # GitHub limits to 50 annotations per step

    @classmethod
    def _default_config(cls) -> GitHubActionsConfig:
        """Create default GitHub Actions configuration."""
        return GitHubActionsConfig()

    @property
    def github_config(self) -> GitHubActionsConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # Annotation Formatting
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format an annotation as a GitHub Actions workflow command.

        Args:
            annotation: The annotation to format.

        Returns:
            GitHub workflow command string.
        """
        # Map level to GitHub command
        command_map = {
            AnnotationLevel.ERROR: "error",
            AnnotationLevel.WARNING: "warning",
            AnnotationLevel.NOTICE: "notice",
            AnnotationLevel.INFO: "notice",
        }
        command = command_map.get(annotation.level, "notice")

        # Build parameters
        params: list[str] = []

        if annotation.file:
            params.append(f"file={annotation.file}")
        if annotation.line:
            params.append(f"line={annotation.line}")
        if annotation.end_line and annotation.end_line != annotation.line:
            params.append(f"endLine={annotation.end_line}")
        if annotation.column:
            params.append(f"col={annotation.column}")
        if annotation.end_column:
            params.append(f"endColumn={annotation.end_column}")
        if annotation.title:
            params.append(f"title={self._escape_property(annotation.title)}")

        # Build command
        param_str = ",".join(params)
        message = self._escape_message(annotation.message)

        if param_str:
            return f"::{command} {param_str}::{message}"
        return f"::{command}::{message}"

    def format_group_start(self, name: str) -> str:
        """Format a collapsible group start.

        Args:
            name: Group name.

        Returns:
            GitHub group command.
        """
        if self.github_config.use_groups:
            return f"::group::{name}"
        return ""

    def format_group_end(self) -> str:
        """Format a collapsible group end.

        Returns:
            GitHub endgroup command.
        """
        if self.github_config.use_groups:
            return "::endgroup::"
        return ""

    # =========================================================================
    # Summary Formatting
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a job summary in Markdown.

        Args:
            result: The validation result.

        Returns:
            Markdown summary string.
        """
        lines: list[str] = []
        stats = result.statistics
        config = self.github_config

        # Header with status
        status_emoji = "✅" if result.success else "❌"
        if config.emoji_enabled:
            lines.append(f"## {status_emoji} Truthound Validation Report")
        else:
            lines.append(f"## Truthound Validation Report - {result.status.value.upper()}")

        lines.append("")

        # Quick stats
        lines.append("### Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| **Status** | {result.status.value.upper()} |")
        lines.append(f"| **Data Asset** | `{result.data_asset}` |")
        lines.append(f"| **Total Validators** | {stats.total_validators} |")
        lines.append(f"| **Passed** | {stats.passed_validators} |")
        lines.append(f"| **Failed** | {stats.failed_validators} |")
        lines.append(f"| **Pass Rate** | {stats.pass_rate:.1%} |")
        lines.append(f"| **Execution Time** | {self.format_duration(stats.execution_time_ms)} |")
        lines.append("")

        # Issues by severity
        if stats.total_issues > 0:
            lines.append("### Issues by Severity")
            lines.append("")

            if config.emoji_enabled:
                lines.append(f"| Severity | Count |")
                lines.append(f"|----------|-------|")
                lines.append(f"| 🔴 Critical | {stats.critical_issues} |")
                lines.append(f"| 🟠 High | {stats.high_issues} |")
                lines.append(f"| 🟡 Medium | {stats.medium_issues} |")
                lines.append(f"| 🟢 Low | {stats.low_issues} |")
            else:
                lines.append(f"| Severity | Count |")
                lines.append(f"|----------|-------|")
                lines.append(f"| Critical | {stats.critical_issues} |")
                lines.append(f"| High | {stats.high_issues} |")
                lines.append(f"| Medium | {stats.medium_issues} |")
                lines.append(f"| Low | {stats.low_issues} |")

            lines.append("")

            # Detailed issues list (collapsible)
            lines.append("<details>")
            lines.append("<summary>View All Issues</summary>")
            lines.append("")

            for validator_result in result.results:
                if not validator_result.success:
                    severity_emoji = self.severity_emoji(validator_result.severity) if config.emoji_enabled else ""
                    lines.append(
                        f"- {severity_emoji} **{validator_result.validator_name}**: "
                        f"{validator_result.message or 'Validation failed'}"
                    )

            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Metadata
        lines.append("---")
        lines.append(
            f"*Run ID: `{result.run_id}` | "
            f"Generated: {result.run_time.strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        return "\n".join(lines)

    # =========================================================================
    # Output Methods
    # =========================================================================

    def render(self, data: "ValidationResult") -> str:
        """Render the complete GitHub Actions output.

        Args:
            data: The validation result.

        Returns:
            Complete output string.
        """
        parts: list[str] = []

        # Annotations (printed to stdout)
        if self._config.annotations_enabled:
            annotations = self.render_annotations(data)
            if annotations:
                parts.append(annotations)

        # Summary is handled separately via file
        if self._config.summary_enabled:
            summary = self.format_summary(data)
            parts.append(summary)

        return "\n\n".join(parts)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to GitHub Actions and return exit code.

        Handles:
        - Printing annotations to stdout
        - Writing job summary to GITHUB_STEP_SUMMARY
        - Setting output variables via GITHUB_OUTPUT

        Args:
            result: The validation result.

        Returns:
            Exit code.
        """
        # Print annotations
        if self._config.annotations_enabled:
            annotations = self.render_annotations(result)
            if annotations:
                print(annotations)

        # Write job summary
        if self._config.summary_enabled and self.github_config.step_summary:
            summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_path:
                summary = self.format_summary(result)
                with open(summary_path, "a") as f:
                    f.write(summary)
                    f.write("\n")

        # Set output variables
        if self.github_config.set_output:
            self._set_output(result)

        return self.get_exit_code(result)

    def _set_output(self, result: "ValidationResult") -> None:
        """Set GitHub Actions output variables.

        Args:
            result: The validation result.
        """
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            return

        outputs = {
            f"{self.github_config.output_name}_success": str(result.success).lower(),
            f"{self.github_config.output_name}_status": result.status.value,
            f"{self.github_config.output_name}_issues": str(result.statistics.total_issues),
        }

        with open(output_path, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _escape_message(self, message: str) -> str:
        """Escape message for GitHub Actions command.

        Args:
            message: The message to escape.

        Returns:
            Escaped message.
        """
        # GitHub Actions requires escaping certain characters
        return (
            message.replace("%", "%25")
            .replace("\r", "%0D")
            .replace("\n", "%0A")
        )

    def _escape_property(self, value: str) -> str:
        """Escape property value for GitHub Actions command.

        Args:
            value: The value to escape.

        Returns:
            Escaped value.
        """
        return (
            value.replace("%", "%25")
            .replace("\r", "%0D")
            .replace("\n", "%0A")
            .replace(":", "%3A")
            .replace(",", "%2C")
        )

    @staticmethod
    def is_github_actions() -> bool:
        """Check if running in GitHub Actions environment.

        Returns:
            True if running in GitHub Actions.
        """
        return os.environ.get("GITHUB_ACTIONS") == "true"

    @staticmethod
    def get_workflow_info() -> dict[str, str | None]:
        """Get GitHub Actions workflow information.

        Returns:
            Dictionary of workflow context values.
        """
        return {
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
            "job": os.environ.get("GITHUB_JOB"),
            "action": os.environ.get("GITHUB_ACTION"),
            "actor": os.environ.get("GITHUB_ACTOR"),
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "ref": os.environ.get("GITHUB_REF"),
            "sha": os.environ.get("GITHUB_SHA"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        }
