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
    from truthound.core.results import ValidationRunResult
    from truthound.reporters.presentation import RunPresentation


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

    def _prepare_data(self, presentation: "RunPresentation") -> dict[str, Any]:
        """Prepare validation result data for JSON serialization.

        Args:
            presentation: Shared presentation model.

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
            "run_id": presentation.run_id,
            "run_time": presentation.run_time.isoformat(),
            "data_asset": presentation.source,
            "status": presentation.status,
            "success": presentation.success,
        }

        # Metadata
        if self._config.include_metadata:
            data["metadata"] = {
                "tags": presentation.metadata.get("tags", {}),
                "suite_name": presentation.suite_name,
                "runtime_environment": presentation.metadata.get("runtime_environment", {}),
                **presentation.metadata,
            }

        # Statistics
        if self._config.include_statistics:
            data["statistics"] = presentation.summary.to_legacy_statistics_dict()

        # Issues/Results
        if self._config.include_details:
            issues = [issue.to_legacy_issue_dict() for issue in presentation.issues]
            data["issues"] = issues
            data["issue_count"] = len(issues)

            # Group by severity
            data["issues_by_severity"] = dict(presentation.issue_counts_by_severity)

            # Group by column
            issues_by_column: dict[str, list[dict[str, Any]]] = {}
            for issue in presentation.issues:
                column = issue.column or "_table_"
                issues_by_column.setdefault(column, []).append({
                    "validator": issue.validator_name,
                    "issue_type": issue.issue_type,
                    "count": issue.count,
                    "severity": issue.severity,
                    "message": issue.message,
                })
            data["issues_by_column"] = issues_by_column

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

    def render(self, data: "ValidationRunResult") -> str:
        """Render validation result as JSON.

        Args:
            data: The validation result to render.

        Returns:
            JSON string representation.

        Raises:
            RenderError: If rendering fails.
        """
        try:
            presentation = self.present(data)
            prepared_data = self._prepare_data(presentation)

            return json.dumps(
                prepared_data,
                indent=self._config.indent,
                sort_keys=self._config.sort_keys,
                ensure_ascii=self._config.ensure_ascii,
                default=self._json_serializer,
            )

        except (TypeError, ValueError) as e:
            raise RenderError(f"Failed to render JSON: {e}")

    def render_compact(self, data: "ValidationRunResult") -> str:
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

    def render_lines(self, data: "ValidationRunResult") -> str:
        """Render validation result as JSON Lines (NDJSON) format.

        Each issue is output as a separate JSON object on its own line.

        Args:
            data: The validation result to render.

        Returns:
            JSON Lines string.
        """
        presentation = self.present(data)
        lines: list[str] = []

        # Header line with metadata
        header = {
            "type": "header",
            "run_id": presentation.run_id,
            "data_asset": presentation.source,
            "run_time": presentation.run_time.isoformat(),
            "status": presentation.status,
        }
        lines.append(json.dumps(header, default=self._json_serializer))

        # Issue lines
        for issue in presentation.issues:
            lines.append(
                json.dumps(
                    {
                        "type": "issue",
                        **issue.to_legacy_issue_dict(),
                    },
                    default=self._json_serializer,
                )
            )

        # Summary line
        summary = {
            "type": "summary",
            **presentation.summary.to_legacy_statistics_dict(),
        }
        lines.append(json.dumps(summary, default=self._json_serializer))

        return "\n".join(lines)
