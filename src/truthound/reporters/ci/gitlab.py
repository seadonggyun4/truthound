"""GitLab CI Reporter.

This module provides a reporter that outputs validation results in formats
compatible with GitLab CI/CD features:
- Code Quality report (codequality.json)
- JUnit report for test results
- Merge request annotations

Reference:
    https://docs.gitlab.com/ee/ci/testing/code_quality.html
    https://docs.gitlab.com/ee/ci/testing/unit_test_reports.html
"""

from __future__ import annotations

import json
import hashlib
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
class GitLabCIConfig(CIReporterConfig):
    """Configuration for GitLab CI reporter.

    Attributes:
        code_quality_path: Path for Code Quality report artifact.
        junit_path: Path for JUnit XML report artifact.
        output_format: Primary output format ('code_quality', 'junit', 'both').
        include_fingerprint: Include fingerprint for deduplication.
        collapse_sections: Use collapsible sections in logs.
    """

    code_quality_path: str = "gl-code-quality-report.json"
    junit_path: str = "gl-junit-report.xml"
    output_format: str = "code_quality"  # 'code_quality', 'junit', 'both'
    include_fingerprint: bool = True
    collapse_sections: bool = True


class GitLabCIReporter(BaseCIReporter):
    """GitLab CI reporter with Code Quality and JUnit support.

    Outputs validation results in GitLab-compatible formats:
    - Code Quality JSON for merge request diffs
    - JUnit XML for test result visualization
    - ANSI colored console output with sections

    Example:
        >>> reporter = GitLabCIReporter(
        ...     code_quality_path="quality.json",
        ...     junit_path="results.xml",
        ...     output_format="both"
        ... )
        >>> reporter.report(validation_result)

    Environment Variables:
        GITLAB_CI: Set to 'true' when running in GitLab CI.
        CI_PROJECT_DIR: Project directory path.
        CI_COMMIT_SHA: Current commit SHA.
    """

    platform = CIPlatform.GITLAB_CI
    name = "gitlab"
    file_extension = ".json"
    content_type = "application/json"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 100

    @classmethod
    def _default_config(cls) -> GitLabCIConfig:
        """Create default GitLab CI configuration."""
        return GitLabCIConfig()

    @property
    def gitlab_config(self) -> GitLabCIConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # Code Quality Report Format
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format annotation for console output.

        Args:
            annotation: The annotation to format.

        Returns:
            ANSI-formatted string.
        """
        # GitLab supports ANSI colors in job logs
        color_map = {
            AnnotationLevel.ERROR: "\033[31m",    # Red
            AnnotationLevel.WARNING: "\033[33m",  # Yellow
            AnnotationLevel.NOTICE: "\033[36m",   # Cyan
            AnnotationLevel.INFO: "\033[37m",     # White
        }
        reset = "\033[0m"
        color = color_map.get(annotation.level, "")

        parts = [f"{color}[{annotation.level.value.upper()}]{reset}"]

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
        """Format a collapsible section start.

        Args:
            name: Section name.

        Returns:
            GitLab section start command.
        """
        if self.gitlab_config.collapse_sections:
            # GitLab collapsible section syntax
            section_id = name.lower().replace(" ", "_")
            return f"\033[0Ksection_start:{self._get_timestamp()}:{section_id}\r\033[0K{name}"
        return f"=== {name} ==="

    def format_group_end(self) -> str:
        """Format a collapsible section end.

        Returns:
            GitLab section end command.
        """
        if self.gitlab_config.collapse_sections:
            return f"\033[0Ksection_end:{self._get_timestamp()}:section\r\033[0K"
        return ""

    def _get_timestamp(self) -> int:
        """Get current Unix timestamp.

        Returns:
            Unix timestamp.
        """
        import time
        return int(time.time())

    # =========================================================================
    # Code Quality JSON Format
    # =========================================================================

    def generate_code_quality_report(self, result: "ValidationResult") -> list[dict[str, Any]]:
        """Generate GitLab Code Quality report format.

        The Code Quality report is a JSON array of issues that GitLab
        displays in merge request diffs.

        Args:
            result: The validation result.

        Returns:
            List of Code Quality issue dictionaries.
        """
        issues: list[dict[str, Any]] = []

        for validator_result in result.results:
            if validator_result.success:
                continue

            annotation = self.create_annotation(validator_result)

            # Map severity to Code Quality severity
            severity_map = {
                AnnotationLevel.ERROR: "critical",
                AnnotationLevel.WARNING: "major",
                AnnotationLevel.NOTICE: "minor",
                AnnotationLevel.INFO: "info",
            }
            severity = severity_map.get(annotation.level, "minor")

            issue: dict[str, Any] = {
                "description": annotation.message,
                "check_name": annotation.validator_name or "truthound",
                "severity": severity,
                "categories": ["Data Quality"],
            }

            # Location
            if annotation.file:
                issue["location"] = {
                    "path": annotation.file,
                    "lines": {
                        "begin": annotation.line or 1,
                    },
                }
                if annotation.end_line:
                    issue["location"]["lines"]["end"] = annotation.end_line
            else:
                # GitLab requires a location, use data asset as fallback
                issue["location"] = {
                    "path": result.data_asset,
                    "lines": {"begin": 1},
                }

            # Fingerprint for deduplication
            if self.gitlab_config.include_fingerprint:
                fingerprint_data = (
                    f"{issue['check_name']}:"
                    f"{issue['location']['path']}:"
                    f"{issue['location']['lines']['begin']}:"
                    f"{annotation.message}"
                )
                issue["fingerprint"] = hashlib.md5(
                    fingerprint_data.encode()
                ).hexdigest()

            issues.append(issue)

        return issues

    # =========================================================================
    # JUnit XML Format
    # =========================================================================

    def generate_junit_report(self, result: "ValidationResult") -> str:
        """Generate JUnit XML report format.

        Args:
            result: The validation result.

        Returns:
            JUnit XML string.
        """
        from xml.etree import ElementTree as ET

        # Create root testsuite element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", f"Truthound: {result.data_asset}")
        testsuite.set("tests", str(result.statistics.total_validators))
        testsuite.set("failures", str(result.statistics.failed_validators))
        testsuite.set("errors", str(result.statistics.error_validators))
        testsuite.set("time", str(result.statistics.execution_time_ms / 1000))
        testsuite.set("timestamp", result.run_time.isoformat())

        # Add test cases
        for validator_result in result.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", validator_result.validator_name)
            testcase.set("classname", f"truthound.{validator_result.column or 'table'}")
            testcase.set("time", str(validator_result.execution_time_ms / 1000))

            if not validator_result.success:
                if validator_result.severity in ("critical", "high"):
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("type", validator_result.issue_type or "ValidationError")
                    failure.set("message", validator_result.message or "Validation failed")
                    failure.text = self._format_failure_details(validator_result)
                else:
                    # Treat medium/low as warnings (system-out)
                    system_out = ET.SubElement(testcase, "system-out")
                    system_out.text = (
                        f"Warning: {validator_result.message}\n"
                        f"Count: {validator_result.count}"
                    )

        # Convert to string
        return ET.tostring(testsuite, encoding="unicode", xml_declaration=True)

    def _format_failure_details(self, validator_result: Any) -> str:
        """Format detailed failure information.

        Args:
            validator_result: The validator result.

        Returns:
            Formatted details string.
        """
        lines = [
            f"Validator: {validator_result.validator_name}",
            f"Column: {validator_result.column or 'N/A'}",
            f"Issue Type: {validator_result.issue_type or 'N/A'}",
            f"Count: {validator_result.count}",
            f"Severity: {validator_result.severity or 'N/A'}",
        ]

        if validator_result.details:
            lines.append("Details:")
            for key, value in validator_result.details.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    # =========================================================================
    # Summary Format
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a summary for GitLab CI logs.

        Args:
            result: The validation result.

        Returns:
            Formatted summary string.
        """
        lines: list[str] = []
        stats = result.statistics

        # Header with color
        if result.success:
            lines.append("\033[32m✓ Validation Passed\033[0m")
        else:
            lines.append("\033[31m✗ Validation Failed\033[0m")

        lines.append("")
        lines.append(f"Data Asset: {result.data_asset}")
        lines.append(f"Run ID: {result.run_id}")
        lines.append("")

        # Statistics
        lines.append("Statistics:")
        lines.append(f"  Total Validators: {stats.total_validators}")
        lines.append(f"  Passed: \033[32m{stats.passed_validators}\033[0m")
        lines.append(f"  Failed: \033[31m{stats.failed_validators}\033[0m")
        lines.append(f"  Pass Rate: {stats.pass_rate:.1%}")
        lines.append(f"  Execution Time: {self.format_duration(stats.execution_time_ms)}")

        if stats.total_issues > 0:
            lines.append("")
            lines.append("Issues by Severity:")
            lines.append(f"  🔴 Critical: {stats.critical_issues}")
            lines.append(f"  🟠 High: {stats.high_issues}")
            lines.append(f"  🟡 Medium: {stats.medium_issues}")
            lines.append(f"  🟢 Low: {stats.low_issues}")

        return "\n".join(lines)

    # =========================================================================
    # Output Methods
    # =========================================================================

    def render(self, data: "ValidationResult") -> str:
        """Render the primary output format.

        Args:
            data: The validation result.

        Returns:
            Rendered output string.
        """
        format_type = self.gitlab_config.output_format

        if format_type == "junit":
            return self.generate_junit_report(data)
        else:
            # Default to Code Quality JSON
            issues = self.generate_code_quality_report(data)
            return json.dumps(issues, indent=2)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to GitLab CI and return exit code.

        Handles:
        - Printing summary to stdout
        - Writing Code Quality artifact
        - Writing JUnit artifact

        Args:
            result: The validation result.

        Returns:
            Exit code.
        """
        # Print summary and annotations to console
        if self._config.summary_enabled:
            print(self.format_summary(result))
            print()

        if self._config.annotations_enabled:
            annotations = self.render_annotations(result)
            if annotations:
                print(annotations)

        # Write artifact files
        format_type = self.gitlab_config.output_format

        if format_type in ("code_quality", "both"):
            self._write_code_quality_artifact(result)

        if format_type in ("junit", "both"):
            self._write_junit_artifact(result)

        return self.get_exit_code(result)

    def _write_code_quality_artifact(self, result: "ValidationResult") -> None:
        """Write Code Quality report artifact.

        Args:
            result: The validation result.
        """
        path = self.gitlab_config.code_quality_path
        issues = self.generate_code_quality_report(result)

        with open(path, "w") as f:
            json.dump(issues, f, indent=2)

        print(f"Code Quality report written to: {path}")

    def _write_junit_artifact(self, result: "ValidationResult") -> None:
        """Write JUnit report artifact.

        Args:
            result: The validation result.
        """
        path = self.gitlab_config.junit_path
        report = self.generate_junit_report(result)

        with open(path, "w") as f:
            f.write(report)

        print(f"JUnit report written to: {path}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def is_gitlab_ci() -> bool:
        """Check if running in GitLab CI environment.

        Returns:
            True if running in GitLab CI.
        """
        return os.environ.get("GITLAB_CI") == "true"

    @staticmethod
    def get_pipeline_info() -> dict[str, str | None]:
        """Get GitLab CI pipeline information.

        Returns:
            Dictionary of pipeline context values.
        """
        return {
            "project_id": os.environ.get("CI_PROJECT_ID"),
            "project_name": os.environ.get("CI_PROJECT_NAME"),
            "project_path": os.environ.get("CI_PROJECT_PATH"),
            "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
            "pipeline_source": os.environ.get("CI_PIPELINE_SOURCE"),
            "job_id": os.environ.get("CI_JOB_ID"),
            "job_name": os.environ.get("CI_JOB_NAME"),
            "commit_sha": os.environ.get("CI_COMMIT_SHA"),
            "commit_branch": os.environ.get("CI_COMMIT_BRANCH"),
            "merge_request_iid": os.environ.get("CI_MERGE_REQUEST_IID"),
        }
