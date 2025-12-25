"""Base classes for CI/CD reporters.

This module defines abstract base classes and common types for CI reporters.
All CI platform-specific reporters inherit from these base classes.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from truthound.reporters.base import (
    ReporterConfig,
    ValidationReporter,
    RenderError,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


class CIPlatform(str, Enum):
    """Supported CI/CD platforms."""

    GITHUB_ACTIONS = "github"
    GITLAB_CI = "gitlab"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure"
    CIRCLECI = "circleci"
    BITBUCKET = "bitbucket"
    GENERIC = "generic"  # Fallback for unknown platforms

    def __str__(self) -> str:
        return self.value


class AnnotationLevel(str, Enum):
    """Annotation severity levels (normalized across platforms)."""

    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFO = "info"

    @classmethod
    def from_severity(cls, severity: str | None) -> "AnnotationLevel":
        """Convert validation severity to annotation level.

        Args:
            severity: Validation severity (critical, high, medium, low).

        Returns:
            Corresponding annotation level.
        """
        if severity is None:
            return cls.INFO

        severity = severity.lower()
        if severity in ("critical", "high"):
            return cls.ERROR
        elif severity == "medium":
            return cls.WARNING
        else:
            return cls.NOTICE


@dataclass
class CIAnnotation:
    """Represents a code annotation for CI platforms.

    This is a platform-agnostic representation that gets converted
    to platform-specific formats by each reporter.

    Attributes:
        message: The annotation message.
        level: Severity level of the annotation.
        file: Path to the file (optional).
        line: Line number (optional).
        end_line: End line number for multi-line annotations (optional).
        column: Column number (optional).
        end_column: End column number (optional).
        title: Short title for the annotation (optional).
        validator_name: Name of the validator that created this annotation.
        raw_severity: Original severity from the validator.
    """

    message: str
    level: AnnotationLevel = AnnotationLevel.WARNING
    file: str | None = None
    line: int | None = None
    end_line: int | None = None
    column: int | None = None
    end_column: int | None = None
    title: str | None = None
    validator_name: str | None = None
    raw_severity: str | None = None

    def with_file_context(
        self,
        file: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> "CIAnnotation":
        """Create a new annotation with updated file context.

        Args:
            file: File path.
            line: Line number.
            column: Column number.

        Returns:
            New annotation with updated context.
        """
        return CIAnnotation(
            message=self.message,
            level=self.level,
            file=file or self.file,
            line=line or self.line,
            end_line=self.end_line,
            column=column or self.column,
            end_column=self.end_column,
            title=self.title,
            validator_name=self.validator_name,
            raw_severity=self.raw_severity,
        )


@dataclass
class CIReporterConfig(ReporterConfig):
    """Configuration for CI reporters.

    Attributes:
        fail_on_error: Whether to exit with non-zero code on errors.
        fail_on_warning: Whether to exit with non-zero code on warnings.
        annotations_enabled: Whether to emit inline code annotations.
        summary_enabled: Whether to emit summary reports.
        max_annotations: Maximum number of annotations to emit (platform limits).
        group_by_file: Group annotations by file path.
        include_passed: Include passed validations in output.
        artifact_path: Path for CI artifact output.
        custom_properties: Platform-specific custom properties.
    """

    fail_on_error: bool = True
    fail_on_warning: bool = False
    annotations_enabled: bool = True
    summary_enabled: bool = True
    max_annotations: int = 50  # GitHub limits to 50 annotations per step
    group_by_file: bool = True
    include_passed: bool = False
    artifact_path: str | None = None
    custom_properties: dict[str, Any] = field(default_factory=dict)


class BaseCIReporter(ValidationReporter[CIReporterConfig]):
    """Abstract base class for all CI/CD reporters.

    Provides common functionality for generating CI platform-specific output
    from validation results. Subclasses implement platform-specific formatting.

    Class Attributes:
        platform: The CI platform this reporter targets.
        supports_annotations: Whether the platform supports inline code annotations.
        supports_summary: Whether the platform supports job/build summaries.
        max_annotations_limit: Platform-specific limit on annotations.

    Example:
        >>> class MyPlatformReporter(BaseCIReporter):
        ...     platform = CIPlatform.GENERIC
        ...     supports_annotations = True
        ...
        ...     def format_annotation(self, annotation: CIAnnotation) -> str:
        ...         return f"[{annotation.level}] {annotation.message}"
        ...
        ...     def format_summary(self, result: ValidationResult) -> str:
        ...         return f"Validation: {result.status}"
    """

    # Class-level platform configuration
    platform: CIPlatform = CIPlatform.GENERIC
    supports_annotations: bool = True
    supports_summary: bool = True
    max_annotations_limit: int = 50

    @classmethod
    def _default_config(cls) -> CIReporterConfig:
        """Create default CI reporter configuration."""
        return CIReporterConfig(
            max_annotations=cls.max_annotations_limit,
        )

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Format a single annotation for the target platform.

        Args:
            annotation: The annotation to format.

        Returns:
            Platform-specific formatted string.
        """
        pass

    @abstractmethod
    def format_summary(self, result: "ValidationResult") -> str:
        """Format a summary report for the target platform.

        Args:
            result: The validation result.

        Returns:
            Platform-specific summary string.
        """
        pass

    # =========================================================================
    # Optional Override Methods
    # =========================================================================

    def format_group_start(self, name: str) -> str:
        """Format the start of a collapsible group.

        Override in subclasses if the platform supports grouping.

        Args:
            name: Name of the group.

        Returns:
            Platform-specific group start string (empty by default).
        """
        return ""

    def format_group_end(self) -> str:
        """Format the end of a collapsible group.

        Override in subclasses if the platform supports grouping.

        Returns:
            Platform-specific group end string (empty by default).
        """
        return ""

    def get_exit_code(self, result: "ValidationResult") -> int:
        """Determine the exit code based on validation results.

        Args:
            result: The validation result.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        if not result.success and self._config.fail_on_error:
            return 1

        if result.statistics.medium_issues > 0 and self._config.fail_on_warning:
            return 1

        return 0

    def should_emit_annotation(self, validator_result: "ValidatorResult") -> bool:
        """Determine if an annotation should be emitted for this result.

        Args:
            validator_result: The validator result to check.

        Returns:
            True if an annotation should be emitted.
        """
        if validator_result.success:
            return self._config.include_passed
        return True

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def create_annotation(
        self,
        validator_result: "ValidatorResult",
    ) -> CIAnnotation:
        """Create a CIAnnotation from a validator result.

        Args:
            validator_result: The validator result.

        Returns:
            A CIAnnotation object.
        """
        level = AnnotationLevel.from_severity(validator_result.severity)

        # Extract file context from details if available
        file_path = validator_result.details.get("file")
        line_number = validator_result.details.get("line")
        column_number = validator_result.details.get("column")

        # Build message
        message = validator_result.message or f"Validation failed: {validator_result.validator_name}"
        if validator_result.count > 0:
            message = f"{message} ({validator_result.count} occurrences)"

        return CIAnnotation(
            message=message,
            level=level,
            file=file_path,
            line=line_number,
            column=column_number,
            title=validator_result.issue_type or validator_result.validator_name,
            validator_name=validator_result.validator_name,
            raw_severity=validator_result.severity,
        )

    def iter_annotations(
        self,
        result: "ValidationResult",
    ) -> Iterator[CIAnnotation]:
        """Iterate over annotations for a validation result.

        Respects the max_annotations limit from config.

        Args:
            result: The validation result.

        Yields:
            CIAnnotation objects.
        """
        count = 0
        max_count = min(self._config.max_annotations, self.max_annotations_limit)

        for validator_result in result.results:
            if count >= max_count:
                break

            if self.should_emit_annotation(validator_result):
                yield self.create_annotation(validator_result)
                count += 1

    def render_annotations(self, result: "ValidationResult") -> str:
        """Render all annotations as a string.

        Args:
            result: The validation result.

        Returns:
            Formatted annotations string.
        """
        if not self._config.annotations_enabled or not self.supports_annotations:
            return ""

        lines: list[str] = []

        if self._config.group_by_file:
            # Group annotations by file
            by_file: dict[str | None, list[CIAnnotation]] = {}
            for annotation in self.iter_annotations(result):
                file_key = annotation.file
                if file_key not in by_file:
                    by_file[file_key] = []
                by_file[file_key].append(annotation)

            for file_path, annotations in by_file.items():
                group_name = file_path or "General"
                group_start = self.format_group_start(group_name)
                if group_start:
                    lines.append(group_start)

                for annotation in annotations:
                    lines.append(self.format_annotation(annotation))

                group_end = self.format_group_end()
                if group_end:
                    lines.append(group_end)
        else:
            for annotation in self.iter_annotations(result):
                lines.append(self.format_annotation(annotation))

        return "\n".join(lines)

    def render(self, data: "ValidationResult") -> str:
        """Render the complete CI output.

        Args:
            data: The validation result.

        Returns:
            Complete formatted output for the CI platform.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            parts: list[str] = []

            # Annotations
            if self._config.annotations_enabled and self.supports_annotations:
                annotations = self.render_annotations(data)
                if annotations:
                    parts.append(annotations)

            # Summary
            if self._config.summary_enabled and self.supports_summary:
                summary = self.format_summary(data)
                if summary:
                    parts.append(summary)

            return "\n\n".join(parts)

        except Exception as e:
            raise RenderError(f"Failed to render CI output: {e}")

    def report_to_ci(self, result: "ValidationResult") -> int:
        """Output the report directly to CI platform and return exit code.

        This method prints directly to stdout (for CI to capture) and
        returns an appropriate exit code.

        Args:
            result: The validation result.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        output = self.render(result)
        if output:
            print(output)

        return self.get_exit_code(result)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_environment_info(self) -> dict[str, str | None]:
        """Get CI environment information.

        Returns:
            Dictionary of environment variable values.
        """
        # Common CI environment variables
        env_vars = [
            "CI",
            "BUILD_NUMBER",
            "BUILD_ID",
            "BUILD_URL",
            "JOB_NAME",
            "BRANCH_NAME",
            "COMMIT_SHA",
            "PR_NUMBER",
        ]

        return {var: os.environ.get(var) for var in env_vars}

    def escape_message(self, message: str) -> str:
        """Escape special characters in messages.

        Override in subclasses for platform-specific escaping.

        Args:
            message: The message to escape.

        Returns:
            Escaped message string.
        """
        return message

    @staticmethod
    def format_duration(ms: float) -> str:
        """Format duration in milliseconds to human-readable string.

        Args:
            ms: Duration in milliseconds.

        Returns:
            Human-readable duration string.
        """
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms / 1000:.1f}s"
        else:
            minutes = int(ms // 60000)
            seconds = (ms % 60000) / 1000
            return f"{minutes}m {seconds:.0f}s"

    @staticmethod
    def severity_emoji(severity: str | None) -> str:
        """Get an emoji for the severity level.

        Args:
            severity: The severity level.

        Returns:
            Emoji character.
        """
        emojis = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }
        return emojis.get((severity or "").lower(), "⚪")
