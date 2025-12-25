"""Jenkins Reporter.

This module provides a reporter that outputs validation results in formats
compatible with Jenkins CI/CD:
- JUnit XML for test results
- Jenkins Pipeline warnings-ng format
- Console output with Jenkins annotations

Reference:
    https://plugins.jenkins.io/junit/
    https://plugins.jenkins.io/warnings-ng/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

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
class JenkinsConfig(CIReporterConfig):
    """Configuration for Jenkins reporter.

    Attributes:
        junit_path: Path for JUnit XML report.
        warnings_path: Path for warnings-ng compatible report.
        output_format: Primary output format ('junit', 'warnings', 'both').
        testsuite_name: Name for the JUnit test suite.
        include_stdout: Include stdout/stderr in test cases.
        use_pipeline_steps: Use Jenkins Pipeline step annotations.
    """

    junit_path: str = "junit-report.xml"
    warnings_path: str = "warnings-report.json"
    output_format: str = "junit"  # 'junit', 'warnings', 'both'
    testsuite_name: str = "Truthound Validation"
    include_stdout: bool = True
    use_pipeline_steps: bool = True


class JenkinsReporter(BaseCIReporter):
    """Jenkins reporter with JUnit and warnings-ng support.

    Outputs validation results in Jenkins-compatible formats:
    - JUnit XML for test result visualization
    - warnings-ng JSON for issue tracking
    - Pipeline step annotations for Blue Ocean

    Example:
        >>> reporter = JenkinsReporter(
        ...     junit_path="test-results.xml",
        ...     output_format="junit"
        ... )
        >>> reporter.report(validation_result)

    Environment Variables:
        JENKINS_URL: Jenkins server URL.
        BUILD_NUMBER: Current build number.
        JOB_NAME: Name of the Jenkins job.
    """

    platform = CIPlatform.JENKINS
    name = "jenkins"
    file_extension = ".xml"
    content_type = "application/xml"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 100

    @classmethod
    def _default_config(cls) -> JenkinsConfig:
        """Create default Jenkins configuration."""
        return JenkinsConfig()

    @property
    def jenkins_config(self) -> JenkinsConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # Annotation Formatting
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format annotation for Jenkins console output.

        Uses Jenkins Pipeline annotation format for visibility in logs.

        Args:
            annotation: The annotation to format.

        Returns:
            Formatted annotation string.
        """
        # Jenkins Pipeline step annotations
        if self.jenkins_config.use_pipeline_steps:
            level_map = {
                AnnotationLevel.ERROR: "ERROR",
                AnnotationLevel.WARNING: "WARNING",
                AnnotationLevel.NOTICE: "INFO",
                AnnotationLevel.INFO: "INFO",
            }
            level = level_map.get(annotation.level, "INFO")

            parts = [f"[{level}]"]
            if annotation.title:
                parts.append(f" {annotation.title}:")
            parts.append(f" {annotation.message}")

            if annotation.file:
                location = annotation.file
                if annotation.line:
                    location += f":{annotation.line}"
                parts.append(f" at {location}")

            return "".join(parts)

        # Simple format fallback
        return f"[{annotation.level.value}] {annotation.message}"

    def format_group_start(self, name: str) -> str:
        """Format a pipeline stage start.

        Args:
            name: Stage name.

        Returns:
            Stage start marker.
        """
        return f"\n{'='*60}\n{name}\n{'='*60}"

    def format_group_end(self) -> str:
        """Format a pipeline stage end.

        Returns:
            Stage end marker.
        """
        return ""

    # =========================================================================
    # JUnit XML Format
    # =========================================================================

    def generate_junit_report(self, result: "ValidationResult") -> str:
        """Generate JUnit XML report.

        Creates a detailed JUnit report that Jenkins can parse for
        test result visualization.

        Args:
            result: The validation result.

        Returns:
            JUnit XML string.
        """
        # Create testsuites root
        testsuites = ET.Element("testsuites")
        testsuites.set("name", "Truthound Validation Results")
        testsuites.set("tests", str(result.statistics.total_validators))
        testsuites.set("failures", str(result.statistics.failed_validators))
        testsuites.set("errors", str(result.statistics.error_validators))
        testsuites.set("time", str(result.statistics.execution_time_ms / 1000))

        # Create testsuite
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", self.jenkins_config.testsuite_name)
        testsuite.set("tests", str(result.statistics.total_validators))
        testsuite.set("failures", str(result.statistics.failed_validators))
        testsuite.set("errors", str(result.statistics.error_validators))
        testsuite.set("time", str(result.statistics.execution_time_ms / 1000))
        testsuite.set("timestamp", result.run_time.isoformat())
        testsuite.set("hostname", os.environ.get("HOSTNAME", "localhost"))

        # Properties
        properties = ET.SubElement(testsuite, "properties")
        self._add_property(properties, "data_asset", result.data_asset)
        self._add_property(properties, "run_id", result.run_id)
        self._add_property(properties, "status", result.status.value)

        # Add Jenkins build info if available
        if os.environ.get("BUILD_NUMBER"):
            self._add_property(properties, "build_number", os.environ.get("BUILD_NUMBER", ""))
        if os.environ.get("JOB_NAME"):
            self._add_property(properties, "job_name", os.environ.get("JOB_NAME", ""))

        # Test cases - group by column
        columns: dict[str | None, list[Any]] = {}
        for validator_result in result.results:
            col = validator_result.column
            if col not in columns:
                columns[col] = []
            columns[col].append(validator_result)

        for column, validators in columns.items():
            classname = f"truthound.{column}" if column else "truthound.table"

            for validator_result in validators:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", validator_result.validator_name)
                testcase.set("classname", classname)
                testcase.set("time", str(validator_result.execution_time_ms / 1000))

                if not validator_result.success:
                    self._add_failure_element(testcase, validator_result)

                if self.jenkins_config.include_stdout and validator_result.details:
                    system_out = ET.SubElement(testcase, "system-out")
                    system_out.text = self._format_details(validator_result.details)

        # System output for the suite
        system_out = ET.SubElement(testsuite, "system-out")
        system_out.text = result.summary()

        return self._to_xml_string(testsuites)

    def _add_property(self, properties: ET.Element, name: str, value: str) -> None:
        """Add a property element.

        Args:
            properties: Parent properties element.
            name: Property name.
            value: Property value.
        """
        prop = ET.SubElement(properties, "property")
        prop.set("name", name)
        prop.set("value", value)

    def _add_failure_element(self, testcase: ET.Element, validator_result: Any) -> None:
        """Add failure or error element to test case.

        Args:
            testcase: Parent testcase element.
            validator_result: The validator result.
        """
        severity = (validator_result.severity or "").lower()

        if severity in ("critical",):
            element = ET.SubElement(testcase, "error")
        else:
            element = ET.SubElement(testcase, "failure")

        element.set("type", validator_result.issue_type or "ValidationFailure")
        element.set("message", validator_result.message or "Validation failed")

        # Detailed text content
        details_lines = [
            f"Validator: {validator_result.validator_name}",
            f"Severity: {validator_result.severity or 'unknown'}",
            f"Issue Count: {validator_result.count}",
        ]

        if validator_result.column:
            details_lines.append(f"Column: {validator_result.column}")

        if validator_result.details:
            details_lines.append("\nDetails:")
            for key, value in validator_result.details.items():
                details_lines.append(f"  {key}: {value}")

        element.text = "\n".join(details_lines)

    def _format_details(self, details: dict[str, Any]) -> str:
        """Format details dictionary as string.

        Args:
            details: Details dictionary.

        Returns:
            Formatted string.
        """
        lines = []
        for key, value in details.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value[:5]:  # Limit items
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _to_xml_string(self, element: ET.Element) -> str:
        """Convert element to XML string with declaration.

        Args:
            element: Root element.

        Returns:
            XML string.
        """
        # Add XML declaration
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + ET.tostring(element, encoding="unicode")

    # =========================================================================
    # Warnings-NG Format
    # =========================================================================

    def generate_warnings_report(self, result: "ValidationResult") -> str:
        """Generate warnings-ng compatible JSON report.

        Args:
            result: The validation result.

        Returns:
            JSON string.
        """
        import json

        issues: list[dict[str, Any]] = []

        for validator_result in result.results:
            if validator_result.success:
                continue

            annotation = self.create_annotation(validator_result)

            # Map to warnings-ng severity
            severity_map = {
                AnnotationLevel.ERROR: "ERROR",
                AnnotationLevel.WARNING: "WARNING_HIGH",
                AnnotationLevel.NOTICE: "WARNING_NORMAL",
                AnnotationLevel.INFO: "WARNING_LOW",
            }

            issue: dict[str, Any] = {
                "fileName": annotation.file or result.data_asset,
                "lineStart": annotation.line or 0,
                "lineEnd": annotation.end_line or annotation.line or 0,
                "columnStart": annotation.column or 0,
                "columnEnd": annotation.end_column or 0,
                "message": annotation.message,
                "category": "Data Quality",
                "type": annotation.validator_name or "truthound",
                "severity": severity_map.get(annotation.level, "WARNING_NORMAL"),
            }

            issues.append(issue)

        report = {
            "_class": "io.jenkins.plugins.analysis.core.restapi.ReportApi",
            "issues": issues,
            "size": len(issues),
        }

        return json.dumps(report, indent=2)

    # =========================================================================
    # Summary Format
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a summary for Jenkins console output.

        Args:
            result: The validation result.

        Returns:
            Formatted summary string.
        """
        lines: list[str] = []
        stats = result.statistics

        # Header
        lines.append("")
        lines.append("=" * 60)
        lines.append("TRUTHOUND VALIDATION REPORT")
        lines.append("=" * 60)

        # Status
        status_indicator = "PASSED" if result.success else "FAILED"
        lines.append(f"Status: {status_indicator}")
        lines.append(f"Data Asset: {result.data_asset}")
        lines.append(f"Run ID: {result.run_id}")
        lines.append(f"Timestamp: {result.run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Statistics table
        lines.append("-" * 40)
        lines.append(f"{'Metric':<25} {'Value':>10}")
        lines.append("-" * 40)
        lines.append(f"{'Total Validators':<25} {stats.total_validators:>10}")
        lines.append(f"{'Passed':<25} {stats.passed_validators:>10}")
        lines.append(f"{'Failed':<25} {stats.failed_validators:>10}")
        lines.append(f"{'Pass Rate':<25} {stats.pass_rate:>9.1%}")
        lines.append(f"{'Execution Time':<25} {self.format_duration(stats.execution_time_ms):>10}")
        lines.append("-" * 40)

        # Issues breakdown
        if stats.total_issues > 0:
            lines.append("")
            lines.append("Issues by Severity:")
            lines.append(f"  Critical: {stats.critical_issues}")
            lines.append(f"  High:     {stats.high_issues}")
            lines.append(f"  Medium:   {stats.medium_issues}")
            lines.append(f"  Low:      {stats.low_issues}")

        lines.append("")
        lines.append("=" * 60)

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
        format_type = self.jenkins_config.output_format

        if format_type == "warnings":
            return self.generate_warnings_report(data)
        else:
            return self.generate_junit_report(data)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to Jenkins and return exit code.

        Args:
            result: The validation result.

        Returns:
            Exit code.
        """
        # Print summary to console
        if self._config.summary_enabled:
            print(self.format_summary(result))

        # Print annotations
        if self._config.annotations_enabled:
            annotations = self.render_annotations(result)
            if annotations:
                print(annotations)

        # Write artifact files
        format_type = self.jenkins_config.output_format

        if format_type in ("junit", "both"):
            self._write_junit_artifact(result)

        if format_type in ("warnings", "both"):
            self._write_warnings_artifact(result)

        return self.get_exit_code(result)

    def _write_junit_artifact(self, result: "ValidationResult") -> None:
        """Write JUnit XML artifact.

        Args:
            result: The validation result.
        """
        path = self.jenkins_config.junit_path
        report = self.generate_junit_report(result)

        with open(path, "w") as f:
            f.write(report)

        print(f"JUnit report written to: {path}")

    def _write_warnings_artifact(self, result: "ValidationResult") -> None:
        """Write warnings-ng artifact.

        Args:
            result: The validation result.
        """
        path = self.jenkins_config.warnings_path
        report = self.generate_warnings_report(result)

        with open(path, "w") as f:
            f.write(report)

        print(f"Warnings report written to: {path}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def is_jenkins() -> bool:
        """Check if running in Jenkins environment.

        Returns:
            True if running in Jenkins.
        """
        return bool(os.environ.get("JENKINS_URL") or os.environ.get("BUILD_NUMBER"))

    @staticmethod
    def get_build_info() -> dict[str, str | None]:
        """Get Jenkins build information.

        Returns:
            Dictionary of build context values.
        """
        return {
            "jenkins_url": os.environ.get("JENKINS_URL"),
            "build_number": os.environ.get("BUILD_NUMBER"),
            "build_id": os.environ.get("BUILD_ID"),
            "build_url": os.environ.get("BUILD_URL"),
            "job_name": os.environ.get("JOB_NAME"),
            "job_base_name": os.environ.get("JOB_BASE_NAME"),
            "workspace": os.environ.get("WORKSPACE"),
            "node_name": os.environ.get("NODE_NAME"),
            "branch_name": os.environ.get("BRANCH_NAME"),
            "change_id": os.environ.get("CHANGE_ID"),
        }
