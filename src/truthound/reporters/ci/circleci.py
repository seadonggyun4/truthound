"""CircleCI Reporter.

This module provides a reporter that outputs validation results in formats
compatible with CircleCI:
- JUnit XML for test metadata
- Store test results API
- Build artifacts

Reference:
    https://circleci.com/docs/collect-test-data/
    https://circleci.com/docs/artifacts/
"""

from __future__ import annotations

import json
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
class CircleCIConfig(CIReporterConfig):
    """Configuration for CircleCI reporter.

    Attributes:
        test_results_path: Path for JUnit test results (store_test_results).
        artifacts_path: Path for artifacts output.
        output_format: Output format ('junit', 'json', 'both').
        json_path: Path for JSON report artifact.
        use_timing: Include timing data for test splitting.
        create_insights: Generate insights-compatible data.
    """

    test_results_path: str = "test-results"
    artifacts_path: str = "artifacts"
    output_format: str = "junit"  # 'junit', 'json', 'both'
    json_path: str = "validation-report.json"
    use_timing: bool = True
    create_insights: bool = True


class CircleCIReporter(BaseCIReporter):
    """CircleCI reporter with test metadata and artifacts.

    Outputs validation results for CircleCI:
    - JUnit XML for test results tab and test splitting
    - JSON artifacts for detailed analysis
    - Timing data for parallel test optimization

    Example:
        >>> reporter = CircleCIReporter(
        ...     test_results_path="test-results",
        ...     output_format="junit"
        ... )
        >>> reporter.report(validation_result)

    Environment Variables:
        CIRCLECI: Set to 'true' when running in CircleCI.
        CIRCLE_BUILD_NUM: Current build number.
        CIRCLE_NODE_INDEX: Node index for parallelism.
    """

    platform = CIPlatform.CIRCLECI
    name = "circleci"
    file_extension = ".xml"
    content_type = "application/xml"
    supports_annotations = True
    supports_summary = True
    max_annotations_limit = 100

    @classmethod
    def _default_config(cls) -> CircleCIConfig:
        """Create default CircleCI configuration."""
        return CircleCIConfig()

    @property
    def circleci_config(self) -> CircleCIConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    # =========================================================================
    # Annotation Formatting
    # =========================================================================

    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format annotation for CircleCI console output.

        CircleCI uses standard terminal output, so we use ANSI colors.

        Args:
            annotation: The annotation to format.

        Returns:
            ANSI-formatted string.
        """
        color_map = {
            AnnotationLevel.ERROR: "\033[1;31m",    # Bold red
            AnnotationLevel.WARNING: "\033[1;33m",  # Bold yellow
            AnnotationLevel.NOTICE: "\033[1;36m",   # Bold cyan
            AnnotationLevel.INFO: "\033[0;37m",     # White
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
            parts.append(f" \033[0;90m({location}){reset}")

        return "".join(parts)

    def format_group_start(self, name: str) -> str:
        """Format a section header.

        Args:
            name: Section name.

        Returns:
            Section header string.
        """
        # CircleCI doesn't have native folding, use visual separators
        return f"\n\033[1;34m{'─' * 50}\n{name}\n{'─' * 50}\033[0m"

    def format_group_end(self) -> str:
        """Format section end.

        Returns:
            Empty string (CircleCI doesn't support collapsible sections).
        """
        return ""

    # =========================================================================
    # JUnit XML Format
    # =========================================================================

    def generate_junit_report(self, result: "ValidationResult") -> str:
        """Generate JUnit XML for CircleCI test results.

        Creates JUnit XML optimized for CircleCI's test metadata features
        including timing data for test splitting.

        Args:
            result: The validation result.

        Returns:
            JUnit XML string.
        """
        testsuites = ET.Element("testsuites")

        # Main testsuite
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", f"Truthound: {result.data_asset}")
        testsuite.set("tests", str(result.statistics.total_validators))
        testsuite.set("failures", str(result.statistics.failed_validators))
        testsuite.set("errors", str(result.statistics.error_validators))
        testsuite.set("time", str(result.statistics.execution_time_ms / 1000))
        testsuite.set("timestamp", result.run_time.isoformat())

        # Add CircleCI-specific properties
        properties = ET.SubElement(testsuite, "properties")
        self._add_property(properties, "data_asset", result.data_asset)
        self._add_property(properties, "run_id", result.run_id)

        # Add parallelism info if available
        node_index = os.environ.get("CIRCLE_NODE_INDEX")
        node_total = os.environ.get("CIRCLE_NODE_TOTAL")
        if node_index and node_total:
            self._add_property(properties, "node_index", node_index)
            self._add_property(properties, "node_total", node_total)

        # Group test cases by column/category for better organization
        validators_by_category = self._group_validators(result)

        for category, validators in validators_by_category.items():
            for validator_result in validators:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", validator_result.validator_name)
                testcase.set("classname", f"truthound.{category}")

                # Timing data is important for CircleCI test splitting
                if self.circleci_config.use_timing:
                    testcase.set("time", str(validator_result.execution_time_ms / 1000))

                # File attribute helps with test location
                if validator_result.details.get("file"):
                    testcase.set("file", validator_result.details["file"])

                if not validator_result.success:
                    self._add_failure(testcase, validator_result)

        return self._to_xml_string(testsuites)

    def _group_validators(self, result: "ValidationResult") -> dict[str, list[Any]]:
        """Group validators by category/column.

        Args:
            result: The validation result.

        Returns:
            Dictionary mapping category to validators.
        """
        groups: dict[str, list[Any]] = {}

        for validator_result in result.results:
            category = validator_result.column or "table"
            if category not in groups:
                groups[category] = []
            groups[category].append(validator_result)

        return groups

    def _add_property(self, properties: ET.Element, name: str, value: str) -> None:
        """Add a property element.

        Args:
            properties: Parent element.
            name: Property name.
            value: Property value.
        """
        prop = ET.SubElement(properties, "property")
        prop.set("name", name)
        prop.set("value", value)

    def _add_failure(self, testcase: ET.Element, validator_result: Any) -> None:
        """Add failure element to test case.

        Args:
            testcase: Parent testcase element.
            validator_result: The validator result.
        """
        failure = ET.SubElement(testcase, "failure")
        failure.set("type", validator_result.issue_type or "ValidationFailure")
        failure.set("message", validator_result.message or "Validation failed")

        # Detailed content
        details = [
            f"Severity: {validator_result.severity or 'unknown'}",
            f"Count: {validator_result.count}",
        ]

        if validator_result.details:
            details.append("")
            details.append("Details:")
            for key, value in validator_result.details.items():
                if isinstance(value, list):
                    details.append(f"  {key}: {', '.join(str(v) for v in value[:5])}")
                else:
                    details.append(f"  {key}: {value}")

        failure.text = "\n".join(details)

    def _to_xml_string(self, element: ET.Element) -> str:
        """Convert element to XML string.

        Args:
            element: Root element.

        Returns:
            XML string with declaration.
        """
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
            element, encoding="unicode"
        )

    # =========================================================================
    # JSON Report Format
    # =========================================================================

    def generate_json_report(self, result: "ValidationResult") -> str:
        """Generate JSON report for artifacts.

        Args:
            result: The validation result.

        Returns:
            JSON string.
        """
        report = {
            "version": "1.0",
            "platform": "circleci",
            "generated_at": datetime.now().isoformat(),
            "validation": {
                "run_id": result.run_id,
                "data_asset": result.data_asset,
                "status": result.status.value,
                "success": result.success,
                "run_time": result.run_time.isoformat(),
            },
            "statistics": result.statistics.to_dict(),
            "issues": [],
            "circleci": {
                "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
                "job": os.environ.get("CIRCLE_JOB"),
                "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
                "node_index": os.environ.get("CIRCLE_NODE_INDEX"),
                "node_total": os.environ.get("CIRCLE_NODE_TOTAL"),
            },
        }

        # Add issues
        for validator_result in result.results:
            if not validator_result.success:
                report["issues"].append(validator_result.to_dict())

        return json.dumps(report, indent=2)

    # =========================================================================
    # Insights Format
    # =========================================================================

    def generate_insights_data(self, result: "ValidationResult") -> dict[str, Any]:
        """Generate data for CircleCI Insights.

        This format is useful for tracking validation metrics over time.

        Args:
            result: The validation result.

        Returns:
            Insights-compatible dictionary.
        """
        return {
            "metrics": {
                "truthound.pass_rate": result.statistics.pass_rate,
                "truthound.total_issues": result.statistics.total_issues,
                "truthound.critical_issues": result.statistics.critical_issues,
                "truthound.high_issues": result.statistics.high_issues,
                "truthound.execution_time_ms": result.statistics.execution_time_ms,
            },
            "dimensions": {
                "data_asset": result.data_asset,
                "status": result.status.value,
            },
        }

    # =========================================================================
    # Summary Format
    # =========================================================================

    def format_summary(self, result: "ValidationResult") -> str:
        """Format a summary for CircleCI console.

        Args:
            result: The validation result.

        Returns:
            Formatted summary string.
        """
        lines: list[str] = []
        stats = result.statistics

        # Header with status
        if result.success:
            lines.append("\033[1;32m✓ Validation Passed\033[0m")
        else:
            lines.append("\033[1;31m✗ Validation Failed\033[0m")

        lines.append("")

        # Build info
        build_num = os.environ.get("CIRCLE_BUILD_NUM", "local")
        job = os.environ.get("CIRCLE_JOB", "validation")
        lines.append(f"\033[0;90mBuild #{build_num} • Job: {job}\033[0m")
        lines.append("")

        # Summary table
        lines.append(f"Data Asset: \033[1m{result.data_asset}\033[0m")
        lines.append(f"Run ID:     {result.run_id}")
        lines.append("")

        # Statistics
        lines.append("\033[1mStatistics:\033[0m")
        lines.append(f"  Validators: {stats.passed_validators}/{stats.total_validators} passed")
        lines.append(f"  Pass Rate:  {stats.pass_rate:.1%}")
        lines.append(f"  Duration:   {self.format_duration(stats.execution_time_ms)}")

        # Issues
        if stats.total_issues > 0:
            lines.append("")
            lines.append("\033[1mIssues:\033[0m")
            lines.append(f"  \033[31mCritical: {stats.critical_issues}\033[0m")
            lines.append(f"  \033[33mHigh:     {stats.high_issues}\033[0m")
            lines.append(f"  \033[33mMedium:   {stats.medium_issues}\033[0m")
            lines.append(f"  \033[32mLow:      {stats.low_issues}\033[0m")

        # Parallelism info
        node_index = os.environ.get("CIRCLE_NODE_INDEX")
        node_total = os.environ.get("CIRCLE_NODE_TOTAL")
        if node_index and node_total:
            lines.append("")
            lines.append(f"\033[0;90mParallel execution: Node {int(node_index) + 1} of {node_total}\033[0m")

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
        format_type = self.circleci_config.output_format

        if format_type == "json":
            return self.generate_json_report(data)
        else:
            return self.generate_junit_report(data)

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output report to CircleCI and return exit code.

        Handles:
        - Console output with summary
        - Writing JUnit XML to test results directory
        - Writing JSON artifacts

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

        # Write artifacts
        format_type = self.circleci_config.output_format

        if format_type in ("junit", "both"):
            self._write_junit_artifact(result)

        if format_type in ("json", "both"):
            self._write_json_artifact(result)

        return self.get_exit_code(result)

    def _write_junit_artifact(self, result: "ValidationResult") -> None:
        """Write JUnit XML artifact.

        Args:
            result: The validation result.
        """
        # Create directory if needed
        test_results_dir = self.circleci_config.test_results_path
        os.makedirs(test_results_dir, exist_ok=True)

        # Write JUnit XML
        path = os.path.join(test_results_dir, "results.xml")
        report = self.generate_junit_report(result)

        with open(path, "w") as f:
            f.write(report)

        print(f"\n\033[0;90mTest results written to: {path}\033[0m")
        print(f"\033[0;90mAdd 'store_test_results' step with path: {test_results_dir}\033[0m")

    def _write_json_artifact(self, result: "ValidationResult") -> None:
        """Write JSON artifact.

        Args:
            result: The validation result.
        """
        # Create artifacts directory if needed
        artifacts_dir = self.circleci_config.artifacts_path
        os.makedirs(artifacts_dir, exist_ok=True)

        # Write JSON
        path = os.path.join(artifacts_dir, self.circleci_config.json_path)
        report = self.generate_json_report(result)

        with open(path, "w") as f:
            f.write(report)

        print(f"\033[0;90mJSON artifact written to: {path}\033[0m")
        print(f"\033[0;90mAdd 'store_artifacts' step with path: {artifacts_dir}\033[0m")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def is_circleci() -> bool:
        """Check if running in CircleCI environment.

        Returns:
            True if running in CircleCI.
        """
        return os.environ.get("CIRCLECI") == "true"

    @staticmethod
    def get_build_info() -> dict[str, str | None]:
        """Get CircleCI build information.

        Returns:
            Dictionary of build context values.
        """
        return {
            "build_num": os.environ.get("CIRCLE_BUILD_NUM"),
            "build_url": os.environ.get("CIRCLE_BUILD_URL"),
            "job": os.environ.get("CIRCLE_JOB"),
            "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
            "workflow_name": os.environ.get("CIRCLE_WORKFLOW_JOB_ID"),
            "project_reponame": os.environ.get("CIRCLE_PROJECT_REPONAME"),
            "project_username": os.environ.get("CIRCLE_PROJECT_USERNAME"),
            "branch": os.environ.get("CIRCLE_BRANCH"),
            "sha1": os.environ.get("CIRCLE_SHA1"),
            "tag": os.environ.get("CIRCLE_TAG"),
            "pr_number": os.environ.get("CIRCLE_PR_NUMBER"),
            "node_index": os.environ.get("CIRCLE_NODE_INDEX"),
            "node_total": os.environ.get("CIRCLE_NODE_TOTAL"),
        }
