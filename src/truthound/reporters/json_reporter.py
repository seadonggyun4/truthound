"""JSON format reporter.

This module provides a reporter that outputs validation results in JSON format.
No external dependencies required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from truthound.reporters.base import (
    ReporterConfig,
    RenderError,
    ValidationReporter,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


@dataclass
class JSONReporterConfig(ReporterConfig):
    """Configuration for JSON reporter.

    Attributes:
        indent: Number of spaces for indentation (None for compact).
        sort_keys: Whether to sort dictionary keys.
        ensure_ascii: Whether to escape non-ASCII characters.
        include_null_values: Whether to include null/None values.
        date_format: Format for date serialization ("iso" or "timestamp").
    """

    indent: int | None = 2
    sort_keys: bool = False
    ensure_ascii: bool = False
    include_null_values: bool = True
    date_format: str = "iso"  # "iso" or "timestamp"


class JSONReporter(ValidationReporter[JSONReporterConfig]):
    """JSON format reporter for validation results.

    Outputs validation results as well-formatted JSON that can be
    easily parsed by other tools or stored for later analysis.

    Example:
        >>> reporter = JSONReporter(indent=2)
        >>> json_str = reporter.render(validation_result)
        >>> reporter.write(validation_result, "report.json")
    """

    name = "json"
    file_extension = ".json"
    content_type = "application/json"

    @classmethod
    def _default_config(cls) -> JSONReporterConfig:
        """Create default configuration."""
        return JSONReporterConfig()

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation.
        """
        if isinstance(obj, datetime):
            if self._config.date_format == "timestamp":
                return obj.timestamp()
            return obj.isoformat()

        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        if hasattr(obj, "__dict__"):
            return obj.__dict__

        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _prepare_data(self, result: "ValidationResult") -> dict[str, Any]:
        """Prepare validation result data for JSON serialization.

        Args:
            result: The validation result.

        Returns:
            Dictionary ready for JSON serialization.
        """
        data: dict[str, Any] = {
            "report_type": "validation",
            "generated_at": datetime.now().strftime(self._config.timestamp_format),
            "title": self._config.title,
        }

        # Core result data
        data["result"] = {
            "run_id": result.run_id,
            "run_time": result.run_time.isoformat(),
            "data_asset": result.data_asset,
            "status": result.status.value,
            "success": result.success,
        }

        # Metadata
        if self._config.include_metadata:
            data["metadata"] = {
                "tags": result.tags,
                "suite_name": result.suite_name,
                "runtime_environment": result.runtime_environment,
                **result.metadata,
            }

        # Statistics
        if self._config.include_statistics:
            data["statistics"] = result.statistics.to_dict()

        # Issues/Results
        if self._config.include_details:
            issues: list[dict[str, Any]] = []

            for validator_result in result.results:
                if not validator_result.success:
                    issue_data = validator_result.to_dict()

                    # Limit sample values
                    if "details" in issue_data and "sample_values" in issue_data["details"]:
                        samples = issue_data["details"]["sample_values"]
                        if samples and len(samples) > self._config.max_sample_values:
                            issue_data["details"]["sample_values"] = samples[
                                : self._config.max_sample_values
                            ]

                    issues.append(issue_data)

            data["issues"] = issues
            data["issue_count"] = len(issues)

            # Group by severity
            data["issues_by_severity"] = self.get_severity_counts(result)

            # Group by column
            data["issues_by_column"] = self.get_column_issues(result)

        # Remove null values if configured
        if not self._config.include_null_values:
            data = self._remove_null_values(data)

        return data

    def _remove_null_values(self, obj: Any) -> Any:
        """Recursively remove null values from a data structure.

        Args:
            obj: The object to process.

        Returns:
            Object with null values removed.
        """
        if isinstance(obj, dict):
            return {
                k: self._remove_null_values(v)
                for k, v in obj.items()
                if v is not None
            }
        elif isinstance(obj, list):
            return [self._remove_null_values(item) for item in obj if item is not None]
        return obj

    def render(self, data: "ValidationResult") -> str:
        """Render validation result as JSON.

        Args:
            data: The validation result to render.

        Returns:
            JSON string representation.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            prepared_data = self._prepare_data(data)

            return json.dumps(
                prepared_data,
                indent=self._config.indent,
                sort_keys=self._config.sort_keys,
                ensure_ascii=self._config.ensure_ascii,
                default=self._json_serializer,
            )

        except (TypeError, ValueError) as e:
            raise RenderError(f"Failed to render JSON: {e}")

    def render_compact(self, data: "ValidationResult") -> str:
        """Render validation result as compact JSON (no whitespace).

        Args:
            data: The validation result to render.

        Returns:
            Compact JSON string.
        """
        original_indent = self._config.indent
        try:
            self._config.indent = None
            return self.render(data)
        finally:
            self._config.indent = original_indent

    def render_lines(self, data: "ValidationResult") -> str:
        """Render validation result as JSON Lines (NDJSON) format.

        Each issue is output as a separate JSON object on its own line.

        Args:
            data: The validation result to render.

        Returns:
            JSON Lines string.
        """
        lines: list[str] = []

        # Header line with metadata
        header = {
            "type": "header",
            "run_id": data.run_id,
            "data_asset": data.data_asset,
            "run_time": data.run_time.isoformat(),
            "status": data.status.value,
        }
        lines.append(json.dumps(header, default=self._json_serializer))

        # Issue lines
        for validator_result in data.results:
            if not validator_result.success:
                issue = {
                    "type": "issue",
                    **validator_result.to_dict(),
                }
                lines.append(json.dumps(issue, default=self._json_serializer))

        # Summary line
        summary = {
            "type": "summary",
            **data.statistics.to_dict(),
        }
        lines.append(json.dumps(summary, default=self._json_serializer))

        return "\n".join(lines)
