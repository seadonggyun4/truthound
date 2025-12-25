"""Bitbucket Pipelines Reporter.

This module provides a reporter that outputs validation results in formats
compatible with Bitbucket Pipelines:
- Build status API reports
- Code annotations (Reports API)
- Pipe-compatible output

Reference:
    https://support.atlassian.com/bitbucket-cloud/docs/code-insights/
    https://support.atlassian.com/bitbucket-cloud/docs/test-reporting-in-pipelines/
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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
class BitbucketConfig(CIReporterConfig):
    """Configuration for Bitbucket Pipelines reporter.

    Attributes:
        report_id: Unique identifier for the report.
        report_title: Title for the Code Insights report.
        reporter_name: Name of the reporter (shown in UI).
        logo_url: URL for reporter logo (optional).
        report_type: Type of report ('BUG', 'VULNERABILITY', 'CODE_SMELL', 'SECURITY').
        json_path: Path for JSON report artifact.
        use_pipes_format: Output in Bitbucket Pipes format.
        create_report_file: Create report file for Reports API.
        report_file_path: Path for report file.
    """

    report_id: str = "truthound-validation"
    report_title: str = "Truthound Data Validation"
    reporter_name: str = "Truthound"
    logo_url: str | None = None
    report_type: str = "CODE_SMELL"  # BUG, VULNERABILITY, CODE_SMELL, SECURITY
    json_path: str = "validation-report.json"
    use_pipes_format: bool = True
    create_report_file: bool = True
    report_file_path: str = "bitbucket-report.json"


class BitbucketPipelinesReporter(BaseCIReporter):
    """Bitbucket Pipelines reporter with Code Insights support.

    Outputs validation results for Bitbucket:
    - Code Insights report format (annotations on PRs)
    - Console output with status indicators
    - JSON artifacts for detailed analysis

    Example:
        >>> reporter = BitbucketPipelinesReporter(
        ...     report_title="Data Quality Check"
        ... )
        >>> reporter.report(validation_result)

    Environment Variables:
        BITBUCKET_BUILD_NUMBER: Current build number.
        BITBUCKET_COMMIT: Current commit SHA.
        BITBUCKET_REPO_SLUG: Repository slug.
    """

    platform = CIPlatform.BITBUCKET
    name = "bitbucket"
    file_extension = ".json"
    content_type = "application/json"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 1000  # Bitbucket supports up to 1000 annotations

    @classmethod
    def _default_config(cls) -> BitbucketConfig:
        """Create default Bitbucket configuration."""
        return BitbucketConfig()

    @property
    def bitbucket_config(self) -> BitbucketConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # Annotation Formatting
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format annotation for console output.

        Bitbucket Pipelines uses standard terminal output with
        basic formatting.

        Args:
            annotation: The annotation to format.

        Returns:
            Formatted string.
        """
        # Status indicators for Bitbucket pipes
        indicator_map = {
            AnnotationLevel.ERROR: "✖",
            AnnotationLevel.WARNING: "⚠",
            AnnotationLevel.NOTICE: "ℹ",
            AnnotationLevel.INFO: "·",
        }
        indicator = indicator_map.get(annotation.level, "·")

        parts = [f"{indicator} [{annotation.level.value.upper()}]"]

        if annotation.title:
            parts.append(f" {annotation.title}:")

        parts.append(f" {annotation.message}")

        if annotation.file:
            location = annotation.file
            if annotation.line:
                location += f":{annotation.line}"
            parts.append(f" ({location})")

        return "".join(parts)

    def format_group_start(self, name: str) -> str:
        """Format section header.

        Args:
            name: Section name.

        Returns:
            Section header string.
        """
        return f"\n{'━' * 50}\n{name}\n{'━' * 50}"

    def format_group_end(self) -> str:
        """Format section end.

        Returns:
            Empty string.
        """
        return ""

    # =========================================================================
    # Code Insights Report Format
    # =========================================================================

    def generate_report(self, result: "ValidationResult") -> dict[str, Any]:
        """Generate Bitbucket Code Insights report format.

        This format can be used with the Bitbucket Reports API
        to display annotations on pull requests.

        Args:
            result: The validation result.

        Returns:
            Report dictionary.
        """
        config = self.bitbucket_config

        # Map status to Bitbucket result
        if result.success:
            report_result = "PASSED"
        elif result.statistics.critical_issues > 0:
            report_result = "FAILED"
        else:
            report_result = "PENDING"  # Warnings only

        report: dict[str, Any] = {
            "title": config.report_title,
            "details": self._generate_report_details(result),
            "report_type": config.report_type,
            "reporter": config.reporter_name,
            "result": report_result,
            "data": self._generate_report_data(result),
        }

        if config.logo_url:
            report["logo_url"] = config.logo_url

        return report

    def _generate_report_details(self, result: "ValidationResult") -> str:
        """Generate report details summary.

        Args:
            result: The validation result.

        Returns:
            Details string.
        """
        stats = result.statistics
        return (
            f"Data validation completed with {stats.total_issues} issues. "
            f"Pass rate: {stats.pass_rate:.1%}. "
            f"Critical: {stats.critical_issues}, High: {stats.high_issues}, "
            f"Medium: {stats.medium_issues}, Low: {stats.low_issues}"
        )

    def _generate_report_data(self, result: "ValidationResult") -> list[dict[str, Any]]:
        """Generate report data items for summary display.

        Args:
            result: The validation result.

        Returns:
            List of data items.
        """
        stats = result.statistics

        return [
            {
                "title": "Pass Rate",
                "type": "PERCENTAGE",
                "value": int(stats.pass_rate * 100),
            },
            {
                "title": "Total Issues",
                "type": "NUMBER",
                "value": stats.total_issues,
            },
            {
                "title": "Critical Issues",
                "type": "NUMBER",
                "value": stats.critical_issues,
            },
            {
                "title": "Execution Time",
                "type": "DURATION",
                "value": int(stats.execution_time_ms),
            },
        ]

    def generate_annotations(self, result: "ValidationResult") -> list[dict[str, Any]]:
        """Generate Bitbucket Code Insights annotations.

        Args:
            result: The validation result.

        Returns:
            List of annotation dictionaries.
        """
        annotations: list[dict[str, Any]] = []

        for validator_result in result.results:
            if validator_result.success:
                continue

            annotation = self.create_annotation(validator_result)

            # Map level to Bitbucket severity
            severity_map = {
                AnnotationLevel.ERROR: "CRITICAL",
                AnnotationLevel.WARNING: "HIGH",
                AnnotationLevel.NOTICE: "MEDIUM",
                AnnotationLevel.INFO: "LOW",
            }

            # Map level to Bitbucket annotation type
            type_map = {
                AnnotationLevel.ERROR: "BUG",
                AnnotationLevel.WARNING: "CODE_SMELL",
                AnnotationLevel.NOTICE: "CODE_SMELL",
                AnnotationLevel.INFO: "CODE_SMELL",
            }

            bitbucket_annotation: dict[str, Any] = {
                "external_id": self._generate_external_id(validator_result),
                "annotation_type": type_map.get(annotation.level, "CODE_SMELL"),
                "summary": annotation.message[:450],  # Bitbucket limit
                "severity": severity_map.get(annotation.level, "MEDIUM"),
            }

            if annotation.file:
                bitbucket_annotation["path"] = annotation.file
                if annotation.line:
                    bitbucket_annotation["line"] = annotation.line

            if annotation.title:
                bitbucket_annotation["details"] = annotation.title

            annotations.append(bitbucket_annotation)

        return annotations

    def _generate_external_id(self, validator_result: Any) -> str:
        """Generate unique external ID for annotation.

        Args:
            validator_result: The validator result.

        Returns:
            Unique ID string.
        """
        content = (
            f"{validator_result.validator_name}:"
            f"{validator_result.column or 'table'}:"
            f"{validator_result.message or ''}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # =========================================================================
    # Pipes Format
    # =========================================================================

    def format_pipes_output(self, result: "ValidationResult") -> str:
        """Format output compatible with Bitbucket Pipes.

        Uses special markers that Bitbucket Pipes recognize.

        Args:
            result: The validation result.

        Returns:
            Pipes-formatted output string.
        """
        lines: list[str] = []

        # Pipe status indicator
        if result.success:
            lines.append("✔ Validation PASSED")
        else:
            lines.append("✖ Validation FAILED")

        lines.append("")

        # Key metrics in pipe format
        stats = result.statistics
        lines.append(f"TRUTHOUND_STATUS={result.status.value}")
        lines.append(f"TRUTHOUND_PASS_RATE={stats.pass_rate:.2f}")
        lines.append(f"TRUTHOUND_TOTAL_ISSUES={stats.total_issues}")
        lines.append(f"TRUTHOUND_CRITICAL_ISSUES={stats.critical_issues}")

        return "\n".join(lines)

    # =========================================================================
    # Summary Format
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a summary for Bitbucket Pipelines console.

        Args:
            result: The validation result.

        Returns:
            Formatted summary string.
        """
        lines: list[str] = []
        stats = result.statistics

        # Header
        lines.append("")
        lines.append("╔" + "═" * 48 + "╗")
        if result.success:
            lines.append("║" + " ✓ TRUTHOUND VALIDATION PASSED ".center(48) + "║")
        else:
            lines.append("║" + " ✗ TRUTHOUND VALIDATION FAILED ".center(48) + "║")
        lines.append("╚" + "═" * 48 + "╝")
        lines.append("")

        # Info
        lines.append(f"Data Asset: {result.data_asset}")
        lines.append(f"Run ID: {result.run_id}")
        lines.append(f"Timestamp: {result.run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Statistics table
        lines.append("┌" + "─" * 30 + "┬" + "─" * 15 + "┐")
        lines.append("│" + " Metric".ljust(30) + "│" + " Value".ljust(15) + "│")
        lines.append("├" + "─" * 30 + "┼" + "─" * 15 + "┤")
        lines.append("│" + " Total Validators".ljust(30) + "│" + f" {stats.total_validators}".ljust(15) + "│")
        lines.append("│" + " Passed".ljust(30) + "│" + f" {stats.passed_validators}".ljust(15) + "│")
        lines.append("│" + " Failed".ljust(30) + "│" + f" {stats.failed_validators}".ljust(15) + "│")
        lines.append("│" + " Pass Rate".ljust(30) + "│" + f" {stats.pass_rate:.1%}".ljust(15) + "│")
        lines.append("│" + " Execution Time".ljust(30) + "│" + f" {self.format_duration(stats.execution_time_ms)}".ljust(15) + "│")
        lines.append("└" + "─" * 30 + "┴" + "─" * 15 + "┘")

        # Issues breakdown
        if stats.total_issues > 0:
            lines.append("")
            lines.append("Issues by Severity:")
            lines.append(f"  ● Critical: {stats.critical_issues}")
            lines.append(f"  ● High:     {stats.high_issues}")
            lines.append(f"  ● Medium:   {stats.medium_issues}")
            lines.append(f"  ● Low:      {stats.low_issues}")

        # Build info
        build_num = os.environ.get("BITBUCKET_BUILD_NUMBER")
        if build_num:
            lines.append("")
            lines.append(f"Build #{build_num}")

        return "\n".join(lines)

    # =========================================================================
    # Output Methods
    # =========================================================================

    def render(self, data: "ValidationResult") -> str:
        """Render the complete Bitbucket output.

        Args:
            data: The validation result.

        Returns:
            Complete output string.
        """
        parts: list[str] = []

        # Summary
        if self._config.summary_enabled:
            parts.append(self.format_summary(data))

        # Annotations
        if self._config.annotations_enabled:
            annotations = self.render_annotations(data)
            if annotations:
                parts.append(annotations)

        # Pipes format
        if self.bitbucket_config.use_pipes_format:
            parts.append(self.format_pipes_output(data))

        return "\n\n".join(parts)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to Bitbucket Pipelines and return exit code.

        Args:
            result: The validation result.

        Returns:
            Exit code.
        """
        # Print summary
        if self._config.summary_enabled:
            print(self.format_summary(result))
            print()

        # Print annotations
        if self._config.annotations_enabled:
            annotations = self.render_annotations(result)
            if annotations:
                print(annotations)

        # Write report file for Reports API
        if self.bitbucket_config.create_report_file:
            self._write_report_file(result)

        # Write JSON artifact
        self._write_json_artifact(result)

        # Print pipes format
        if self.bitbucket_config.use_pipes_format:
            print()
            print(self.format_pipes_output(result))

        return self.get_exit_code(result)

    def _write_report_file(self, result: "ValidationResult") -> None:
        """Write Code Insights report file.

        This file can be used with the Bitbucket Reports API
        or the reports pipe.

        Args:
            result: The validation result.
        """
        path = self.bitbucket_config.report_file_path

        report_data = {
            "report": self.generate_report(result),
            "annotations": self.generate_annotations(result),
        }

        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\nCode Insights report written to: {path}")

    def _write_json_artifact(self, result: "ValidationResult") -> None:
        """Write JSON artifact.

        Args:
            result: The validation result.
        """
        path = self.bitbucket_config.json_path

        report = {
            "version": "1.0",
            "platform": "bitbucket",
            "generated_at": datetime.now().isoformat(),
            "validation": {
                "run_id": result.run_id,
                "data_asset": result.data_asset,
                "status": result.status.value,
                "success": result.success,
            },
            "statistics": result.statistics.to_dict(),
            "issues": [r.to_dict() for r in result.results if not r.success],
            "bitbucket": {
                "build_number": os.environ.get("BITBUCKET_BUILD_NUMBER"),
                "commit": os.environ.get("BITBUCKET_COMMIT"),
                "repo_slug": os.environ.get("BITBUCKET_REPO_SLUG"),
                "workspace": os.environ.get("BITBUCKET_WORKSPACE"),
                "branch": os.environ.get("BITBUCKET_BRANCH"),
                "pr_id": os.environ.get("BITBUCKET_PR_ID"),
            },
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"JSON report written to: {path}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def is_bitbucket() -> bool:
        """Check if running in Bitbucket Pipelines environment.

        Returns:
            True if running in Bitbucket Pipelines.
        """
        return bool(os.environ.get("BITBUCKET_BUILD_NUMBER"))

    @staticmethod
    def get_pipeline_info() -> dict[str, str | None]:
        """Get Bitbucket Pipelines information.

        Returns:
            Dictionary of pipeline context values.
        """
        return {
            "build_number": os.environ.get("BITBUCKET_BUILD_NUMBER"),
            "commit": os.environ.get("BITBUCKET_COMMIT"),
            "repo_slug": os.environ.get("BITBUCKET_REPO_SLUG"),
            "repo_owner": os.environ.get("BITBUCKET_REPO_OWNER"),
            "workspace": os.environ.get("BITBUCKET_WORKSPACE"),
            "branch": os.environ.get("BITBUCKET_BRANCH"),
            "tag": os.environ.get("BITBUCKET_TAG"),
            "pr_id": os.environ.get("BITBUCKET_PR_ID"),
            "pr_destination_branch": os.environ.get("BITBUCKET_PR_DESTINATION_BRANCH"),
            "deployment_environment": os.environ.get("BITBUCKET_DEPLOYMENT_ENVIRONMENT"),
            "pipeline_uuid": os.environ.get("BITBUCKET_PIPELINE_UUID"),
            "step_uuid": os.environ.get("BITBUCKET_STEP_UUID"),
        }
