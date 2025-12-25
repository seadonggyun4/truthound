"""CI/CD platform-specific reporters.

This module provides reporters that output results in formats
suitable for various CI/CD platforms.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.ci.detector import CIPlatform, detect_ci_platform

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


class CIReporter(ABC):
    """Abstract base class for CI/CD reporters."""

    platform: CIPlatform = CIPlatform.UNKNOWN

    @abstractmethod
    def report_status(self, result: "CheckpointResult") -> None:
        """Report checkpoint status to the CI platform.

        Args:
            result: Checkpoint result to report.
        """
        pass

    @abstractmethod
    def set_output(self, name: str, value: Any) -> None:
        """Set an output variable for the CI platform.

        Args:
            name: Output variable name.
            value: Output value.
        """
        pass

    def fail_build(self, message: str) -> None:
        """Mark the build as failed.

        Args:
            message: Failure message.
        """
        print(f"::error::{message}")

    def warn(self, message: str) -> None:
        """Emit a warning.

        Args:
            message: Warning message.
        """
        print(f"::warning::{message}")

    def log_group(self, name: str) -> "LogGroup":
        """Create a collapsible log group.

        Args:
            name: Group name.

        Returns:
            Context manager for the group.
        """
        return LogGroup(name, self)


class LogGroup:
    """Context manager for CI log groups."""

    def __init__(self, name: str, reporter: CIReporter) -> None:
        self.name = name
        self.reporter = reporter

    def __enter__(self) -> "LogGroup":
        print(f"::group::{self.name}")
        return self

    def __exit__(self, *args: Any) -> None:
        print("::endgroup::")


class GitHubActionsReporter(CIReporter):
    """Reporter for GitHub Actions."""

    platform = CIPlatform.GITHUB_ACTIONS

    def report_status(self, result: "CheckpointResult") -> None:
        """Report status to GitHub Actions."""
        # Write job summary
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            self._write_summary(result, Path(summary_path))

        # Set outputs
        self._set_outputs(result)

        # Log status
        status = result.status.value
        if status in ("failure", "error"):
            self.fail_build(f"Checkpoint {result.checkpoint_name} failed: {result.error or 'Issues found'}")
        elif status == "warning":
            self.warn(f"Checkpoint {result.checkpoint_name} completed with warnings")

    def _write_summary(self, result: "CheckpointResult", path: Path) -> None:
        """Write GitHub Actions job summary."""
        validation = result.validation_result
        stats = validation.statistics if validation else None
        status = result.status.value

        status_emoji = {
            "success": ":white_check_mark:",
            "failure": ":x:",
            "error": ":exclamation:",
            "warning": ":warning:",
        }.get(status, ":question:")

        summary = f"""## {status_emoji} Truthound Checkpoint: {result.checkpoint_name}

| Metric | Value |
|--------|-------|
| Status | **{status.upper()}** |
| Data Asset | `{result.data_asset}` |
| Run ID | `{result.run_id}` |
| Duration | {result.duration_ms:.2f}ms |

### Validation Results

| Severity | Count |
|----------|-------|
| :red_circle: Critical | {stats.critical_issues if stats else 0} |
| :orange_circle: High | {stats.high_issues if stats else 0} |
| :yellow_circle: Medium | {stats.medium_issues if stats else 0} |
| :blue_circle: Low | {stats.low_issues if stats else 0} |
| **Total** | **{stats.total_issues if stats else 0}** |

"""
        if result.action_results:
            summary += "### Actions\n\n"
            for action in result.action_results:
                action_emoji = ":white_check_mark:" if action.success else ":x:"
                summary += f"- {action_emoji} **{action.action_name}**: {action.message}\n"

        with open(path, "a") as f:
            f.write(summary)

    def _set_outputs(self, result: "CheckpointResult") -> None:
        """Set GitHub Actions outputs."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        outputs = {
            "status": result.status.value,
            "checkpoint": result.checkpoint_name,
            "run_id": result.run_id,
            "total_issues": str(stats.total_issues if stats else 0),
            "critical_issues": str(stats.critical_issues if stats else 0),
            "high_issues": str(stats.high_issues if stats else 0),
            "has_failures": str(result.status.value in ("failure", "error")).lower(),
        }

        for name, value in outputs.items():
            self.set_output(name, value)

    def set_output(self, name: str, value: Any) -> None:
        """Set GitHub Actions output."""
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a") as f:
                f.write(f"{name}={value}\n")
        else:
            # Legacy format
            print(f"::set-output name={name}::{value}")

    def fail_build(self, message: str) -> None:
        """Emit error annotation."""
        print(f"::error::{message}")

    def warn(self, message: str) -> None:
        """Emit warning annotation."""
        print(f"::warning::{message}")


class GitLabCIReporter(CIReporter):
    """Reporter for GitLab CI."""

    platform = CIPlatform.GITLAB_CI

    def report_status(self, result: "CheckpointResult") -> None:
        """Report status to GitLab CI."""
        # GitLab uses dotenv artifacts for passing variables
        self._write_dotenv(result)

        # Log status
        status = result.status.value
        if status in ("failure", "error"):
            print(f"\033[31mCheckpoint {result.checkpoint_name} failed\033[0m")
        elif status == "warning":
            print(f"\033[33mCheckpoint {result.checkpoint_name} completed with warnings\033[0m")
        else:
            print(f"\033[32mCheckpoint {result.checkpoint_name} passed\033[0m")

        # Print summary
        self._print_summary(result)

    def _write_dotenv(self, result: "CheckpointResult") -> None:
        """Write variables to dotenv file for artifacts."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        dotenv_content = f"""TRUTHOUND_STATUS={result.status.value}
TRUTHOUND_CHECKPOINT={result.checkpoint_name}
TRUTHOUND_RUN_ID={result.run_id}
TRUTHOUND_TOTAL_ISSUES={stats.total_issues if stats else 0}
TRUTHOUND_CRITICAL_ISSUES={stats.critical_issues if stats else 0}
TRUTHOUND_HAS_FAILURES={str(result.status.value in ('failure', 'error')).lower()}
"""
        Path("truthound.env").write_text(dotenv_content)

    def _print_summary(self, result: "CheckpointResult") -> None:
        """Print summary to console."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        print("\n" + "=" * 60)
        print(f"Checkpoint: {result.checkpoint_name}")
        print(f"Status: {result.status.value.upper()}")
        print(f"Data Asset: {result.data_asset}")
        print("-" * 60)
        print(f"Total Issues: {stats.total_issues if stats else 0}")
        print(f"  Critical: {stats.critical_issues if stats else 0}")
        print(f"  High: {stats.high_issues if stats else 0}")
        print(f"  Medium: {stats.medium_issues if stats else 0}")
        print(f"  Low: {stats.low_issues if stats else 0}")
        print("=" * 60 + "\n")

    def set_output(self, name: str, value: Any) -> None:
        """Set output (writes to dotenv)."""
        with open("truthound.env", "a") as f:
            f.write(f"TRUTHOUND_{name.upper()}={value}\n")


class JenkinsCIReporter(CIReporter):
    """Reporter for Jenkins CI."""

    platform = CIPlatform.JENKINS

    def report_status(self, result: "CheckpointResult") -> None:
        """Report status to Jenkins."""
        # Write properties file
        self._write_properties(result)

        # Write JUnit-style XML for test results
        self._write_junit_xml(result)

        # Console output
        self._print_summary(result)

    def _write_properties(self, result: "CheckpointResult") -> None:
        """Write properties file for Jenkins."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        properties = f"""truthound.status={result.status.value}
truthound.checkpoint={result.checkpoint_name}
truthound.run_id={result.run_id}
truthound.total_issues={stats.total_issues if stats else 0}
truthound.critical_issues={stats.critical_issues if stats else 0}
truthound.has_failures={str(result.status.value in ('failure', 'error')).lower()}
"""
        Path("truthound.properties").write_text(properties)

    def _write_junit_xml(self, result: "CheckpointResult") -> None:
        """Write JUnit XML for Jenkins test reporting."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        failures = stats.total_issues if stats else 0
        status = result.status.value

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Truthound" tests="1" failures="{1 if status in ('failure', 'error') else 0}" errors="0" time="{result.duration_ms / 1000:.3f}">
    <testcase name="{result.checkpoint_name}" classname="truthound.checkpoint" time="{result.duration_ms / 1000:.3f}">
"""
        if status in ("failure", "error"):
            xml += f"""        <failure message="Validation failed with {failures} issues">
            Status: {status}
            Total Issues: {failures}
            Critical: {stats.critical_issues if stats else 0}
            High: {stats.high_issues if stats else 0}
        </failure>
"""
        xml += """    </testcase>
</testsuite>
"""
        Path("truthound-junit.xml").write_text(xml)

    def _print_summary(self, result: "CheckpointResult") -> None:
        """Print summary for Jenkins console."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        print("\n" + "=" * 60)
        print("TRUTHOUND VALIDATION REPORT")
        print("=" * 60)
        print(f"Checkpoint: {result.checkpoint_name}")
        print(f"Status: {result.status.value.upper()}")
        print(f"Data Asset: {result.data_asset}")
        print(f"Duration: {result.duration_ms:.2f}ms")
        print("-" * 60)
        print(f"Total Issues: {stats.total_issues if stats else 0}")
        print(f"  Critical: {stats.critical_issues if stats else 0}")
        print(f"  High: {stats.high_issues if stats else 0}")
        print("=" * 60 + "\n")

    def set_output(self, name: str, value: Any) -> None:
        """Set output (writes to properties)."""
        with open("truthound.properties", "a") as f:
            f.write(f"truthound.{name}={value}\n")


class CircleCIReporter(CIReporter):
    """Reporter for CircleCI."""

    platform = CIPlatform.CIRCLECI

    def report_status(self, result: "CheckpointResult") -> None:
        """Report status to CircleCI."""
        # Write test results in JUnit format
        self._write_junit_xml(result)

        # Write JSON for artifacts
        self._write_json(result)

        # Console output
        self._print_summary(result)

    def _write_junit_xml(self, result: "CheckpointResult") -> None:
        """Write JUnit XML for CircleCI test reporting."""
        # Create test-results directory
        results_dir = Path("test-results/truthound")
        results_dir.mkdir(parents=True, exist_ok=True)

        validation = result.validation_result
        stats = validation.statistics if validation else None
        status = result.status.value

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Truthound.{result.checkpoint_name}" tests="1" failures="{1 if status in ('failure', 'error') else 0}" time="{result.duration_ms / 1000:.3f}">
    <testcase name="validation" classname="Truthound.{result.checkpoint_name}" time="{result.duration_ms / 1000:.3f}">
"""
        if status in ("failure", "error"):
            xml += f"""        <failure message="Validation failed">
Issues found: {stats.total_issues if stats else 0}
Critical: {stats.critical_issues if stats else 0}
High: {stats.high_issues if stats else 0}
        </failure>
"""
        xml += """    </testcase>
</testsuite>
"""
        (results_dir / "results.xml").write_text(xml)

    def _write_json(self, result: "CheckpointResult") -> None:
        """Write JSON results for artifacts."""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "truthound-result.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )

    def _print_summary(self, result: "CheckpointResult") -> None:
        """Print summary for CircleCI console."""
        validation = result.validation_result
        stats = validation.statistics if validation else None

        print("\n╔" + "═" * 58 + "╗")
        print("║" + " TRUTHOUND VALIDATION REPORT".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║ Checkpoint: {result.checkpoint_name:<45}║")
        print(f"║ Status: {result.status.value.upper():<49}║")
        print(f"║ Issues: {stats.total_issues if stats else 0:<49}║")
        print("╚" + "═" * 58 + "╝\n")

    def set_output(self, name: str, value: Any) -> None:
        """Set output (writes to file)."""
        outputs_file = Path("truthound-outputs.json")
        outputs: dict[str, Any] = {}
        if outputs_file.exists():
            outputs = json.loads(outputs_file.read_text())
        outputs[name] = value
        outputs_file.write_text(json.dumps(outputs, indent=2))


class GenericCIReporter(CIReporter):
    """Generic reporter for unknown CI platforms."""

    platform = CIPlatform.UNKNOWN

    def report_status(self, result: "CheckpointResult") -> None:
        """Report status generically."""
        # Write JSON result
        Path("truthound-result.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )

        # Console output
        print(result.summary())

    def set_output(self, name: str, value: Any) -> None:
        """Set output to environment."""
        os.environ[f"TRUTHOUND_{name.upper()}"] = str(value)


def get_ci_reporter() -> CIReporter:
    """Get the appropriate CI reporter for the current environment.

    Returns:
        CI reporter instance.
    """
    platform = detect_ci_platform()

    reporters: dict[CIPlatform, type[CIReporter]] = {
        CIPlatform.GITHUB_ACTIONS: GitHubActionsReporter,
        CIPlatform.GITLAB_CI: GitLabCIReporter,
        CIPlatform.JENKINS: JenkinsCIReporter,
        CIPlatform.CIRCLECI: CircleCIReporter,
    }

    reporter_class = reporters.get(platform, GenericCIReporter)
    return reporter_class()
