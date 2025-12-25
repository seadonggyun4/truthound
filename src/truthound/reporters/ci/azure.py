"""Azure DevOps Reporter.

This module provides a reporter that outputs validation results using
Azure DevOps (VSO) logging commands for annotations, task results, and variables.

Reference:
    https://docs.microsoft.com/en-us/azure/devops/pipelines/scripts/logging-commands
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
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
class AzureDevOpsConfig(CIReporterConfig):
    """Configuration for Azure DevOps reporter.

    Attributes:
        set_variable: Set pipeline variables with results.
        variable_prefix: Prefix for output variables.
        upload_summary: Upload markdown summary as attachment.
        summary_path: Path for markdown summary file.
        use_task_commands: Use task result commands.
        timeline_records: Create timeline records for issues.
    """

    set_variable: bool = True
    variable_prefix: str = "TRUTHOUND"
    upload_summary: bool = True
    summary_path: str = "truthound-summary.md"
    use_task_commands: bool = True
    timeline_records: bool = False


class AzureDevOpsReporter(BaseCIReporter):
    """Azure DevOps reporter with VSO logging commands.

    Outputs validation results using Azure Pipelines logging commands:
    - ##vso[task.logissue] for warnings and errors
    - ##vso[task.setvariable] for output variables
    - ##vso[task.complete] for task status
    - ##vso[task.uploadsummary] for markdown summaries

    Example:
        >>> reporter = AzureDevOpsReporter()
        >>> exit_code = reporter.report_to_ci(validation_result)

    Environment Variables:
        TF_BUILD: Set to 'True' when running in Azure Pipelines.
        SYSTEM_TEAMFOUNDATIONCOLLECTIONURI: Azure DevOps organization URL.
        BUILD_BUILDID: Current build ID.
    """

    platform = CIPlatform.AZURE_DEVOPS
    name = "azure"
    file_extension = ".md"
    content_type = "text/markdown"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 100

    @classmethod
    def _default_config(cls) -> AzureDevOpsConfig:
        """Create default Azure DevOps configuration."""
        return AzureDevOpsConfig()

    @property
    def azure_config(self) -> AzureDevOpsConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # VSO Logging Commands
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format annotation as VSO logging command.

        Args:
            annotation: The annotation to format.

        Returns:
            VSO task.logissue command.
        """
        # Map level to VSO type
        type_map = {
            AnnotationLevel.ERROR: "error",
            AnnotationLevel.WARNING: "warning",
            AnnotationLevel.NOTICE: "warning",
            AnnotationLevel.INFO: "warning",
        }
        issue_type = type_map.get(annotation.level, "warning")

        # Build properties
        properties: list[str] = [f"type={issue_type}"]

        if annotation.file:
            properties.append(f"sourcepath={annotation.file}")
        if annotation.line:
            properties.append(f"linenumber={annotation.line}")
        if annotation.column:
            properties.append(f"columnnumber={annotation.column}")
        if annotation.title:
            properties.append(f"code={self._escape_property(annotation.title)}")

        props_str = ";".join(properties)
        message = self._escape_message(annotation.message)

        return f"##vso[task.logissue {props_str}]{message}"

    def format_group_start(self, name: str) -> str:
        """Format a collapsible group start.

        Args:
            name: Group name.

        Returns:
            VSO group command.
        """
        return f"##[group]{name}"

    def format_group_end(self) -> str:
        """Format a collapsible group end.

        Returns:
            VSO endgroup command.
        """
        return "##[endgroup]"

    # =========================================================================
    # Variable Commands
    # =========================================================================

    def set_variable(self, name: str, value: str, is_output: bool = True) -> str:
        """Generate command to set a pipeline variable.

        Args:
            name: Variable name.
            value: Variable value.
            is_output: Whether this is an output variable.

        Returns:
            VSO setvariable command.
        """
        output_flag = "isOutput=true;" if is_output else ""
        full_name = f"{self.azure_config.variable_prefix}_{name}"
        return f"##vso[task.setvariable variable={full_name};{output_flag}]{value}"

    def generate_variable_commands(self, result: "ValidationResult") -> list[str]:
        """Generate variable setting commands for the result.

        Args:
            result: The validation result.

        Returns:
            List of VSO setvariable commands.
        """
        if not self.azure_config.set_variable:
            return []

        commands = [
            self.set_variable("SUCCESS", str(result.success).lower()),
            self.set_variable("STATUS", result.status.value),
            self.set_variable("TOTAL_ISSUES", str(result.statistics.total_issues)),
            self.set_variable("CRITICAL_ISSUES", str(result.statistics.critical_issues)),
            self.set_variable("HIGH_ISSUES", str(result.statistics.high_issues)),
            self.set_variable("PASS_RATE", f"{result.statistics.pass_rate:.2f}"),
        ]

        return commands

    # =========================================================================
    # Task Commands
    # =========================================================================

    def generate_task_result(self, result: "ValidationResult") -> str:
        """Generate task completion command.

        Args:
            result: The validation result.

        Returns:
            VSO task.complete command.
        """
        if not self.azure_config.use_task_commands:
            return ""

        if result.success:
            return "##vso[task.complete result=Succeeded;]Validation passed"
        elif result.statistics.critical_issues > 0:
            return "##vso[task.complete result=Failed;]Validation failed with critical issues"
        else:
            return "##vso[task.complete result=SucceededWithIssues;]Validation completed with warnings"

    def generate_upload_summary(self, path: str) -> str:
        """Generate command to upload summary as build attachment.

        Args:
            path: Path to the summary file.

        Returns:
            VSO task.uploadsummary command.
        """
        return f"##vso[task.uploadsummary]{path}"

    # =========================================================================
    # Summary Format
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a markdown summary for Azure DevOps.

        Args:
            result: The validation result.

        Returns:
            Markdown summary string.
        """
        lines: list[str] = []
        stats = result.statistics

        # Header
        status_icon = ":white_check_mark:" if result.success else ":x:"
        lines.append(f"# {status_icon} Truthound Validation Report")
        lines.append("")

        # Quick summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| **Status** | {result.status.value.upper()} |")
        lines.append(f"| **Data Asset** | `{result.data_asset}` |")
        lines.append(f"| **Run ID** | `{result.run_id}` |")
        lines.append(f"| **Timestamp** | {result.run_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        lines.append("")

        # Statistics
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Validators | {stats.total_validators} |")
        lines.append(f"| Passed | {stats.passed_validators} |")
        lines.append(f"| Failed | {stats.failed_validators} |")
        lines.append(f"| Pass Rate | {stats.pass_rate:.1%} |")
        lines.append(f"| Execution Time | {self.format_duration(stats.execution_time_ms)} |")
        lines.append("")

        # Issues
        if stats.total_issues > 0:
            lines.append("## Issues by Severity")
            lines.append("")
            lines.append(f"| Severity | Count |")
            lines.append(f"|----------|-------|")
            lines.append(f"| :red_circle: Critical | {stats.critical_issues} |")
            lines.append(f"| :orange_circle: High | {stats.high_issues} |")
            lines.append(f"| :yellow_circle: Medium | {stats.medium_issues} |")
            lines.append(f"| :green_circle: Low | {stats.low_issues} |")
            lines.append("")

            # Issue details
            lines.append("## Issue Details")
            lines.append("")

            for validator_result in result.results:
                if not validator_result.success:
                    severity = validator_result.severity or "unknown"
                    lines.append(
                        f"- **[{severity.upper()}]** {validator_result.validator_name}: "
                        f"{validator_result.message or 'Validation failed'}"
                    )

            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by Truthound*")

        return "\n".join(lines)

    # =========================================================================
    # Output Methods
    # =========================================================================

    def render(self, data: "ValidationResult") -> str:
        """Render the complete Azure DevOps output.

        Args:
            data: The validation result.

        Returns:
            Complete output with VSO commands.
        """
        parts: list[str] = []

        # Annotations
        if self._config.annotations_enabled:
            annotations = self.render_annotations(data)
            if annotations:
                parts.append(annotations)

        # Variables
        variable_commands = self.generate_variable_commands(data)
        if variable_commands:
            parts.append("\n".join(variable_commands))

        # Task result
        task_result = self.generate_task_result(data)
        if task_result:
            parts.append(task_result)

        # Summary (as markdown for uploading)
        if self._config.summary_enabled:
            parts.append(self.format_summary(data))

        return "\n\n".join(parts)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to Azure DevOps and return exit code.

        Args:
            result: The validation result.

        Returns:
            Exit code.
        """
        # Print annotations as VSO commands
        if self._config.annotations_enabled:
            annotations = self.render_annotations(result)
            if annotations:
                print(annotations)

        # Set variables
        variable_commands = self.generate_variable_commands(result)
        for cmd in variable_commands:
            print(cmd)

        # Write and upload summary
        if self._config.summary_enabled and self.azure_config.upload_summary:
            summary_path = self.azure_config.summary_path
            summary = self.format_summary(result)

            with open(summary_path, "w") as f:
                f.write(summary)

            print(self.generate_upload_summary(summary_path))

        # Task completion
        if self.azure_config.use_task_commands:
            print(self.generate_task_result(result))

        return self.get_exit_code(result)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _escape_message(self, message: str) -> str:
        """Escape message for VSO command.

        Args:
            message: The message to escape.

        Returns:
            Escaped message.
        """
        # VSO uses different escape sequences
        return (
            message.replace("\r", "%0D")
            .replace("\n", "%0A")
            .replace("]", "%5D")
            .replace(";", "%3B")
        )

    def _escape_property(self, value: str) -> str:
        """Escape property value for VSO command.

        Args:
            value: The value to escape.

        Returns:
            Escaped value.
        """
        return (
            value.replace("\r", "")
            .replace("\n", " ")
            .replace("]", "")
            .replace(";", "")
        )

    @staticmethod
    def is_azure_devops() -> bool:
        """Check if running in Azure DevOps environment.

        Returns:
            True if running in Azure DevOps.
        """
        return os.environ.get("TF_BUILD") == "True"

    @staticmethod
    def get_pipeline_info() -> dict[str, str | None]:
        """Get Azure DevOps pipeline information.

        Returns:
            Dictionary of pipeline context values.
        """
        return {
            "collection_uri": os.environ.get("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI"),
            "project": os.environ.get("SYSTEM_TEAMPROJECT"),
            "build_id": os.environ.get("BUILD_BUILDID"),
            "build_number": os.environ.get("BUILD_BUILDNUMBER"),
            "definition_name": os.environ.get("BUILD_DEFINITIONNAME"),
            "source_branch": os.environ.get("BUILD_SOURCEBRANCH"),
            "source_version": os.environ.get("BUILD_SOURCEVERSION"),
            "repository_name": os.environ.get("BUILD_REPOSITORY_NAME"),
            "agent_name": os.environ.get("AGENT_NAME"),
            "stage_name": os.environ.get("SYSTEM_STAGENAME"),
            "job_name": os.environ.get("SYSTEM_JOBDISPLAYNAME"),
            "pull_request_id": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID"),
        }
