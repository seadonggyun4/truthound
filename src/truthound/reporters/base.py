"""Base classes and interfaces for reporters.

This module defines the abstract base classes and protocols that all reporter
implementations must follow. Reporters are responsible for transforming
validation results into human-readable or machine-parseable formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult
    from truthound.report import Report


# =============================================================================
# Exceptions
# =============================================================================


class ReporterError(Exception):
    """Base exception for all reporter-related errors."""

    pass


class RenderError(ReporterError):
    """Raised when rendering a report fails."""

    pass


class WriteError(ReporterError):
    """Raised when writing a report to file fails."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ReporterConfig:
    """Base configuration for all reporters.

    Subclasses can extend this with format-specific options.

    Attributes:
        output_path: Optional path to write the report to.
        title: Title for the report.
        include_metadata: Whether to include metadata in the report.
        include_statistics: Whether to include statistics in the report.
        include_details: Whether to include detailed issue information.
        timestamp_format: Format string for timestamps.
        max_sample_values: Maximum number of sample values to include.
    """

    output_path: str | Path | None = None
    title: str = "Truthound Validation Report"
    include_metadata: bool = True
    include_statistics: bool = True
    include_details: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    max_sample_values: int = 5

    def get_output_path(self) -> Path | None:
        """Get the output path as a Path object."""
        if self.output_path is None:
            return None
        return Path(self.output_path)


# =============================================================================
# Type Variables
# =============================================================================

ConfigT = TypeVar("ConfigT", bound=ReporterConfig)
InputT = TypeVar("InputT")


# =============================================================================
# Abstract Base Reporter
# =============================================================================


class BaseReporter(ABC, Generic[ConfigT, InputT]):
    """Abstract base class for all reporters.

    Reporters transform validation results into various output formats.
    They can render to strings or write directly to files.

    Type Parameters:
        ConfigT: The configuration type for this reporter.
        InputT: The input type this reporter accepts.

    Example:
        >>> class MyReporter(BaseReporter[MyConfig, ValidationResult]):
        ...     def render(self, result: ValidationResult) -> str:
        ...         return f"Status: {result.status}"
    """

    # Class-level attributes
    name: str = "base"
    file_extension: str = ".txt"
    content_type: str = "text/plain"

    def __init__(self, config: ConfigT | None = None, **kwargs: Any) -> None:
        """Initialize the reporter with optional configuration.

        Args:
            config: Reporter configuration. If None, uses default configuration.
            **kwargs: Additional configuration options to override.
        """
        self._config = config or self._default_config()

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration for this reporter type."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get the reporter configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def render(self, data: InputT) -> str:
        """Render the input data to a string.

        Args:
            data: The data to render.

        Returns:
            The rendered report as a string.

        Raises:
            RenderError: If rendering fails.
        """
        pass

    # -------------------------------------------------------------------------
    # Concrete Methods
    # -------------------------------------------------------------------------

    def render_to_bytes(self, data: InputT, encoding: str = "utf-8") -> bytes:
        """Render the input data to bytes.

        Args:
            data: The data to render.
            encoding: The encoding to use.

        Returns:
            The rendered report as bytes.
        """
        return self.render(data).encode(encoding)

    def write(self, data: InputT, path: str | Path | None = None) -> Path:
        """Write the rendered report to a file.

        Args:
            data: The data to render.
            path: Optional path to write to. Uses config.output_path if not specified.

        Returns:
            The path where the report was written.

        Raises:
            WriteError: If no path is specified and config.output_path is None.
            WriteError: If writing fails.
        """
        output_path = Path(path) if path else self._config.get_output_path()

        if output_path is None:
            raise WriteError(
                "No output path specified. Either pass a path argument "
                "or set output_path in the reporter configuration."
            )

        try:
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Render and write
            content = self.render(data)
            output_path.write_text(content, encoding="utf-8")

            return output_path

        except (OSError, IOError) as e:
            raise WriteError(f"Failed to write report to {output_path}: {e}")

    def report(self, data: InputT, path: str | Path | None = None) -> str:
        """Generate a report, optionally writing to file.

        This is a convenience method that renders the report and optionally
        writes it to a file if a path is specified.

        Args:
            data: The data to render.
            path: Optional path to write to.

        Returns:
            The rendered report as a string.
        """
        content = self.render(data)

        output_path = Path(path) if path else self._config.get_output_path()
        if output_path:
            self.write(data, output_path)

        return content

    def generate_filename(
        self,
        data: InputT,
        timestamp: bool = True,
    ) -> str:
        """Generate a filename for the report.

        Args:
            data: The data being reported (used for naming).
            timestamp: Whether to include a timestamp in the filename.

        Returns:
            A generated filename with the appropriate extension.
        """
        base_name = self.name

        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{ts}"

        return f"{base_name}{self.file_extension}"


# =============================================================================
# Specialized Reporter Types
# =============================================================================


class ValidationReporter(BaseReporter[ConfigT, "ValidationResult"], Generic[ConfigT]):
    """Reporter specialized for ValidationResult objects.

    Provides additional helper methods for working with validation results.
    """

    def get_severity_counts(self, result: "ValidationResult") -> dict[str, int]:
        """Get counts of issues by severity.

        Args:
            result: The validation result.

        Returns:
            Dictionary mapping severity to count.
        """
        counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for validator_result in result.results:
            if not validator_result.success and validator_result.severity:
                severity = validator_result.severity.lower()
                if severity in counts:
                    counts[severity] += 1

        return counts

    def get_column_issues(
        self,
        result: "ValidationResult",
    ) -> dict[str, list[dict[str, Any]]]:
        """Get issues grouped by column.

        Args:
            result: The validation result.

        Returns:
            Dictionary mapping column names to lists of issues.
        """
        column_issues: dict[str, list[dict[str, Any]]] = {}

        for validator_result in result.results:
            if not validator_result.success:
                column = validator_result.column or "_table_"
                if column not in column_issues:
                    column_issues[column] = []
                column_issues[column].append({
                    "validator": validator_result.validator_name,
                    "issue_type": validator_result.issue_type,
                    "count": validator_result.count,
                    "severity": validator_result.severity,
                    "message": validator_result.message,
                })

        return column_issues


class ReportReporter(BaseReporter[ConfigT, "Report"], Generic[ConfigT]):
    """Reporter specialized for Report objects (legacy format).

    Provides compatibility with the existing Report class.
    """

    pass
