"""JSON exporter for Data Docs.

This module provides JSON export functionality for structured data output.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, TYPE_CHECKING

from truthound.datadocs.exporters.base import BaseExporter, ExportOptions

if TYPE_CHECKING:
    from truthound.datadocs.engine.context import ReportContext


class JsonExporter(BaseExporter):
    """JSON output exporter.

    Exports report data as structured JSON.

    Options:
        indent: Indentation level (None for compact).
        include_raw_data: Include raw profile data.
        include_html: Include rendered HTML in output.
        sort_keys: Sort dictionary keys.
    """

    def __init__(
        self,
        indent: int | None = 2,
        include_raw_data: bool = False,
        include_html: bool = False,
        sort_keys: bool = False,
        options: ExportOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the JSON exporter.

        Args:
            indent: Indentation level.
            include_raw_data: Include raw profile data.
            include_html: Include rendered HTML.
            sort_keys: Sort dictionary keys.
            options: Export options.
            name: Exporter name.
        """
        super().__init__(options=options, name=name or "JsonExporter")
        self._indent = indent
        self._include_raw_data = include_raw_data
        self._include_html = include_html
        self._sort_keys = sort_keys

    @property
    def format(self) -> str:
        return "json"

    def _do_export(
        self,
        content: str,
        ctx: "ReportContext",
    ) -> str:
        """Export to JSON.

        Args:
            content: Rendered HTML content.
            ctx: Report context.

        Returns:
            JSON string.
        """
        data = ctx.data
        metadata = data.metadata

        output = {
            "metadata": {
                "title": metadata.get("title", "Data Quality Report"),
                "subtitle": metadata.get("subtitle", ""),
                "generated_at": metadata.get("generated_at", datetime.now().isoformat()),
                "locale": ctx.locale,
                "theme": ctx.theme,
                "format_version": "1.0",
            },
            "summary": self._build_summary(data, metadata),
            "sections": self._build_sections(data),
            "alerts": self._build_alerts(data.alerts),
            "recommendations": data.recommendations,
        }

        # Add quality score if present
        if metadata.get("quality_score"):
            output["quality_score"] = metadata["quality_score"]

        # Optionally include raw data
        if self._include_raw_data:
            output["raw_data"] = data.raw

        # Optionally include rendered HTML
        if self._include_html:
            output["html"] = content

        return json.dumps(
            output,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=self._json_serializer,
            ensure_ascii=False,
        )

    def _build_summary(self, data: Any, metadata: dict) -> dict:
        """Build summary section.

        Args:
            data: Report data.
            metadata: Report metadata.

        Returns:
            Summary dictionary.
        """
        summary = metadata.get("summary", {})
        quality = metadata.get("quality_score", {})

        return {
            "row_count": summary.get("row_count", data.raw.get("row_count", 0)),
            "column_count": summary.get("column_count", data.raw.get("column_count", 0)),
            "quality_score": quality.get("overall") if isinstance(quality, dict) else None,
            "quality_grade": quality.get("grade") if isinstance(quality, dict) else None,
            "alert_count": len(data.alerts),
            "recommendation_count": len(data.recommendations),
        }

    def _build_sections(self, data: Any) -> dict:
        """Build sections for JSON output.

        Args:
            data: Report data.

        Returns:
            Sections dictionary.
        """
        return {
            name: self._serialize_section(section_data)
            for name, section_data in data.sections.items()
        }

    def _serialize_section(self, section: Any) -> Any:
        """Serialize a section for JSON.

        Args:
            section: Section data.

        Returns:
            Serializable section data.
        """
        if isinstance(section, dict):
            return {
                k: self._serialize_value(v)
                for k, v in section.items()
            }
        elif isinstance(section, list):
            return [self._serialize_value(v) for v in section]
        else:
            return self._serialize_value(section)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON.

        Args:
            value: Value to serialize.

        Returns:
            Serializable value.
        """
        if isinstance(value, (datetime,)):
            return value.isoformat()
        elif isinstance(value, (set, frozenset)):
            return list(value)
        elif hasattr(value, "to_dict"):
            return value.to_dict()
        elif hasattr(value, "__dict__"):
            return {
                k: self._serialize_value(v)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }
        return value

    def _build_alerts(self, alerts: list) -> list:
        """Build alerts for JSON output.

        Args:
            alerts: List of alerts.

        Returns:
            Serializable alerts list.
        """
        result = []
        for alert in alerts:
            result.append({
                "title": alert.get("title", ""),
                "message": alert.get("message", ""),
                "severity": alert.get("severity", "info"),
                "column": alert.get("column"),
                "metric": alert.get("metric"),
                "value": alert.get("value"),
                "threshold": alert.get("threshold"),
                "suggestion": alert.get("suggestion"),
            })
        return result

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types.

        Args:
            obj: Object to serialize.

        Returns:
            Serializable value.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return {
                k: v for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CompactJsonExporter(JsonExporter):
    """Compact JSON exporter without indentation.

    Produces minimal JSON output for reduced file size.
    """

    def __init__(
        self,
        options: ExportOptions | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            indent=None,
            include_raw_data=False,
            include_html=False,
            sort_keys=False,
            options=options,
            name=name or "CompactJsonExporter",
        )
