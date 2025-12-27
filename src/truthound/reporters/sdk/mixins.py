"""Mixins for common reporter functionality.

This module provides reusable mixins that can be combined with reporter
base classes to add common functionality:

- FormattingMixin: Text formatting, tables, colors
- AggregationMixin: Grouping, counting, statistics
- FilteringMixin: Filtering by severity, column, validator
- SerializationMixin: JSON, YAML, XML serialization helpers
- TemplatingMixin: Template rendering with Jinja2
- StreamingMixin: Streaming output for large datasets

Example:
    >>> class MyReporter(FormattingMixin, FilteringMixin, ValidationReporter[Config]):
    ...     def render(self, data):
    ...         issues = self.filter_by_severity(data, min_severity="medium")
    ...         return self.format_as_table(issues)
"""

from __future__ import annotations

import csv
import io
import json
from abc import ABC
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


T = TypeVar("T")


# =============================================================================
# Severity Ordering
# =============================================================================

SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "none": 5,
}


def _get_severity_level(severity: str | None) -> int:
    """Get numeric severity level for comparison."""
    if severity is None:
        return SEVERITY_ORDER.get("none", 5)
    return SEVERITY_ORDER.get(severity.lower(), 5)


# =============================================================================
# Formatting Mixin
# =============================================================================


class FormattingMixin(ABC):
    """Mixin providing text formatting utilities.

    Provides methods for:
    - Table formatting (ASCII, Markdown, HTML)
    - Text alignment and padding
    - Color/style helpers
    - Number formatting
    - Date/time formatting
    """

    # -------------------------------------------------------------------------
    # Table Formatting
    # -------------------------------------------------------------------------

    def format_as_table(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str] | None = None,
        headers: dict[str, str] | None = None,
        style: str = "ascii",
        max_width: int | None = None,
    ) -> str:
        """Format data as a text table.

        Args:
            rows: Sequence of dictionaries representing rows.
            columns: Column names to include (default: all columns from first row).
            headers: Optional header labels (column name -> display name).
            style: Table style ("ascii", "markdown", "simple", "grid").
            max_width: Maximum column width (truncates content).

        Returns:
            Formatted table as string.

        Example:
            >>> rows = [{"name": "col1", "count": 10}, {"name": "col2", "count": 5}]
            >>> print(self.format_as_table(rows, style="markdown"))
            | name | count |
            |------|-------|
            | col1 | 10    |
            | col2 | 5     |
        """
        if not rows:
            return ""

        # Determine columns
        if columns is None:
            columns = list(rows[0].keys())

        headers = headers or {}

        # Calculate column widths
        widths: dict[str, int] = {}
        for col in columns:
            header_text = headers.get(col, col)
            widths[col] = len(str(header_text))
            for row in rows:
                value = str(row.get(col, ""))
                if max_width:
                    value = value[:max_width]
                widths[col] = max(widths[col], len(value))

        # Build table based on style
        if style == "markdown":
            return self._format_markdown_table(rows, columns, headers, widths, max_width)
        elif style == "grid":
            return self._format_grid_table(rows, columns, headers, widths, max_width)
        elif style == "simple":
            return self._format_simple_table(rows, columns, headers, widths, max_width)
        else:  # ascii
            return self._format_ascii_table(rows, columns, headers, widths, max_width)

    def _format_markdown_table(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str],
        headers: dict[str, str],
        widths: dict[str, int],
        max_width: int | None,
    ) -> str:
        """Format as Markdown table."""
        lines = []

        # Header row
        header_cells = [
            headers.get(col, col).ljust(widths[col]) for col in columns
        ]
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator
        sep_cells = ["-" * widths[col] for col in columns]
        lines.append("| " + " | ".join(sep_cells) + " |")

        # Data rows
        for row in rows:
            cells = []
            for col in columns:
                value = str(row.get(col, ""))
                if max_width:
                    value = value[:max_width]
                cells.append(value.ljust(widths[col]))
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _format_ascii_table(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str],
        headers: dict[str, str],
        widths: dict[str, int],
        max_width: int | None,
    ) -> str:
        """Format as ASCII table with borders."""
        lines = []

        # Top border
        border = "+" + "+".join("-" * (widths[col] + 2) for col in columns) + "+"
        lines.append(border)

        # Header row
        header_cells = [
            " " + headers.get(col, col).ljust(widths[col]) + " " for col in columns
        ]
        lines.append("|" + "|".join(header_cells) + "|")
        lines.append(border)

        # Data rows
        for row in rows:
            cells = []
            for col in columns:
                value = str(row.get(col, ""))
                if max_width:
                    value = value[:max_width]
                cells.append(" " + value.ljust(widths[col]) + " ")
            lines.append("|" + "|".join(cells) + "|")

        lines.append(border)
        return "\n".join(lines)

    def _format_grid_table(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str],
        headers: dict[str, str],
        widths: dict[str, int],
        max_width: int | None,
    ) -> str:
        """Format as grid table with double borders."""
        lines = []

        # Top border
        border = "╔" + "╤".join("═" * (widths[col] + 2) for col in columns) + "╗"
        lines.append(border)

        # Header row
        header_cells = [
            " " + headers.get(col, col).ljust(widths[col]) + " " for col in columns
        ]
        lines.append("║" + "│".join(header_cells) + "║")

        # Header separator
        sep = "╠" + "╪".join("═" * (widths[col] + 2) for col in columns) + "╣"
        lines.append(sep)

        # Data rows
        for i, row in enumerate(rows):
            cells = []
            for col in columns:
                value = str(row.get(col, ""))
                if max_width:
                    value = value[:max_width]
                cells.append(" " + value.ljust(widths[col]) + " ")
            lines.append("║" + "│".join(cells) + "║")

        # Bottom border
        border = "╚" + "╧".join("═" * (widths[col] + 2) for col in columns) + "╝"
        lines.append(border)
        return "\n".join(lines)

    def _format_simple_table(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str],
        headers: dict[str, str],
        widths: dict[str, int],
        max_width: int | None,
    ) -> str:
        """Format as simple table with minimal separators."""
        lines = []

        # Header row
        header_cells = [headers.get(col, col).ljust(widths[col]) for col in columns]
        lines.append("  ".join(header_cells))

        # Separator
        sep_cells = ["-" * widths[col] for col in columns]
        lines.append("  ".join(sep_cells))

        # Data rows
        for row in rows:
            cells = []
            for col in columns:
                value = str(row.get(col, ""))
                if max_width:
                    value = value[:max_width]
                cells.append(value.ljust(widths[col]))
            lines.append("  ".join(cells))

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Text Formatting
    # -------------------------------------------------------------------------

    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length.

        Args:
            text: Text to truncate.
            max_length: Maximum length including suffix.
            suffix: Suffix to add when truncating.

        Returns:
            Truncated text.
        """
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    def indent(self, text: str, prefix: str = "  ", first_line: bool = True) -> str:
        """Indent text with a prefix.

        Args:
            text: Text to indent.
            prefix: Prefix to add to each line.
            first_line: Whether to indent the first line.

        Returns:
            Indented text.
        """
        lines = text.split("\n")
        if first_line:
            return "\n".join(prefix + line for line in lines)
        return lines[0] + "\n" + "\n".join(prefix + line for line in lines[1:])

    def wrap(self, text: str, width: int = 80) -> str:
        """Wrap text to specified width.

        Args:
            text: Text to wrap.
            width: Maximum line width.

        Returns:
            Wrapped text.
        """
        import textwrap

        return textwrap.fill(text, width=width)

    # -------------------------------------------------------------------------
    # Number Formatting
    # -------------------------------------------------------------------------

    def format_number(
        self,
        value: int | float,
        precision: int = 2,
        thousands_sep: str = ",",
    ) -> str:
        """Format a number with thousands separator.

        Args:
            value: Number to format.
            precision: Decimal precision for floats.
            thousands_sep: Thousands separator character.

        Returns:
            Formatted number string.
        """
        if isinstance(value, float):
            formatted = f"{value:,.{precision}f}"
        else:
            formatted = f"{value:,}"
        return formatted.replace(",", thousands_sep)

    def format_percentage(
        self,
        value: float,
        precision: int = 1,
        include_sign: bool = False,
    ) -> str:
        """Format a value as percentage.

        Args:
            value: Value to format (0.0 to 1.0 or 0 to 100).
            precision: Decimal precision.
            include_sign: Whether to include + sign for positive values.

        Returns:
            Formatted percentage string.
        """
        # Assume value is already a percentage if > 1
        if abs(value) <= 1:
            value = value * 100

        if include_sign and value > 0:
            return f"+{value:.{precision}f}%"
        return f"{value:.{precision}f}%"

    def format_bytes(self, size: int) -> str:
        """Format byte size as human readable.

        Args:
            size: Size in bytes.

        Returns:
            Human readable size (e.g., "1.5 MB").
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size) < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024  # type: ignore
        return f"{size:.1f} PB"

    # -------------------------------------------------------------------------
    # Date/Time Formatting
    # -------------------------------------------------------------------------

    def format_datetime(
        self,
        dt: datetime,
        format: str = "%Y-%m-%d %H:%M:%S",
    ) -> str:
        """Format datetime object.

        Args:
            dt: Datetime to format.
            format: strftime format string.

        Returns:
            Formatted datetime string.
        """
        return dt.strftime(format)

    def format_duration(self, seconds: float) -> str:
        """Format duration as human readable.

        Args:
            seconds: Duration in seconds.

        Returns:
            Human readable duration (e.g., "2h 30m 15s").
        """
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def format_relative_time(self, dt: datetime) -> str:
        """Format datetime as relative time.

        Args:
            dt: Datetime to format.

        Returns:
            Relative time string (e.g., "5 minutes ago").
        """
        now = datetime.now(dt.tzinfo)
        delta = now - dt

        seconds = delta.total_seconds()

        if seconds < 60:
            return "just now"
        if seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        if seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

        return self.format_datetime(dt, "%Y-%m-%d")


# =============================================================================
# Aggregation Mixin
# =============================================================================


class AggregationMixin(ABC):
    """Mixin providing data aggregation utilities.

    Provides methods for:
    - Grouping by various keys
    - Counting and statistics
    - Summarization
    """

    def group_by_column(
        self,
        results: list["ValidatorResult"],
    ) -> dict[str, list["ValidatorResult"]]:
        """Group validator results by column.

        Args:
            results: List of validator results.

        Returns:
            Dictionary mapping column name to results.
        """
        grouped: dict[str, list[ValidatorResult]] = defaultdict(list)
        for result in results:
            column = result.column or "_table_"
            grouped[column].append(result)
        return dict(grouped)

    def group_by_severity(
        self,
        results: list["ValidatorResult"],
    ) -> dict[str, list["ValidatorResult"]]:
        """Group validator results by severity.

        Args:
            results: List of validator results.

        Returns:
            Dictionary mapping severity to results.
        """
        grouped: dict[str, list[ValidatorResult]] = defaultdict(list)
        for result in results:
            severity = result.severity or "unknown"
            grouped[severity.lower()].append(result)
        return dict(grouped)

    def group_by_validator(
        self,
        results: list["ValidatorResult"],
    ) -> dict[str, list["ValidatorResult"]]:
        """Group validator results by validator name.

        Args:
            results: List of validator results.

        Returns:
            Dictionary mapping validator name to results.
        """
        grouped: dict[str, list[ValidatorResult]] = defaultdict(list)
        for result in results:
            grouped[result.validator_name].append(result)
        return dict(grouped)

    def group_by(
        self,
        results: list["ValidatorResult"],
        key: Callable[["ValidatorResult"], str],
    ) -> dict[str, list["ValidatorResult"]]:
        """Group validator results by custom key function.

        Args:
            results: List of validator results.
            key: Function to extract grouping key from result.

        Returns:
            Dictionary mapping key to results.
        """
        grouped: dict[str, list[ValidatorResult]] = defaultdict(list)
        for result in results:
            grouped[key(result)].append(result)
        return dict(grouped)

    def count_by_severity(
        self,
        results: list["ValidatorResult"],
    ) -> dict[str, int]:
        """Count results by severity.

        Args:
            results: List of validator results.

        Returns:
            Dictionary mapping severity to count.
        """
        counts: dict[str, int] = defaultdict(int)
        for result in results:
            severity = result.severity or "unknown"
            counts[severity.lower()] += 1
        return dict(counts)

    def count_by_column(
        self,
        results: list["ValidatorResult"],
    ) -> dict[str, int]:
        """Count results by column.

        Args:
            results: List of validator results.

        Returns:
            Dictionary mapping column to count.
        """
        counts: dict[str, int] = defaultdict(int)
        for result in results:
            column = result.column or "_table_"
            counts[column] += 1
        return dict(counts)

    def get_summary_stats(
        self,
        result: "ValidationResult",
    ) -> dict[str, Any]:
        """Get summary statistics from validation result.

        Args:
            result: Validation result.

        Returns:
            Dictionary with summary statistics.
        """
        failed = [r for r in result.results if not r.success]

        return {
            "total_validators": len(result.results),
            "passed": len(result.results) - len(failed),
            "failed": len(failed),
            "pass_rate": (
                (len(result.results) - len(failed)) / len(result.results) * 100
                if result.results
                else 0
            ),
            "by_severity": self.count_by_severity(failed),
            "by_column": self.count_by_column(failed),
            "most_affected_column": (
                max(self.count_by_column(failed).items(), key=lambda x: x[1])[0]
                if failed
                else None
            ),
        }


# =============================================================================
# Filtering Mixin
# =============================================================================


class FilteringMixin(ABC):
    """Mixin providing data filtering utilities.

    Provides methods for:
    - Filtering by severity, column, validator
    - Sorting results
    - Limiting/pagination
    """

    def filter_by_severity(
        self,
        results: list["ValidatorResult"],
        min_severity: str | None = None,
        max_severity: str | None = None,
        include_severities: list[str] | None = None,
        exclude_severities: list[str] | None = None,
    ) -> list["ValidatorResult"]:
        """Filter results by severity.

        Args:
            results: List of validator results.
            min_severity: Minimum severity to include.
            max_severity: Maximum severity to include.
            include_severities: List of severities to include (exclusive with min/max).
            exclude_severities: List of severities to exclude.

        Returns:
            Filtered list of results.
        """
        filtered = []

        for result in results:
            severity = (result.severity or "none").lower()
            level = _get_severity_level(severity)

            # Check include list
            if include_severities:
                if severity not in [s.lower() for s in include_severities]:
                    continue
            else:
                # Check min severity
                if min_severity:
                    min_level = _get_severity_level(min_severity)
                    if level > min_level:
                        continue

                # Check max severity
                if max_severity:
                    max_level = _get_severity_level(max_severity)
                    if level < max_level:
                        continue

            # Check exclude list
            if exclude_severities:
                if severity in [s.lower() for s in exclude_severities]:
                    continue

            filtered.append(result)

        return filtered

    def filter_by_column(
        self,
        results: list["ValidatorResult"],
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> list["ValidatorResult"]:
        """Filter results by column.

        Args:
            results: List of validator results.
            include_columns: Columns to include.
            exclude_columns: Columns to exclude.

        Returns:
            Filtered list of results.
        """
        filtered = []

        for result in results:
            column = result.column or "_table_"

            if include_columns and column not in include_columns:
                continue
            if exclude_columns and column in exclude_columns:
                continue

            filtered.append(result)

        return filtered

    def filter_by_validator(
        self,
        results: list["ValidatorResult"],
        include_validators: list[str] | None = None,
        exclude_validators: list[str] | None = None,
    ) -> list["ValidatorResult"]:
        """Filter results by validator name.

        Args:
            results: List of validator results.
            include_validators: Validator names to include.
            exclude_validators: Validator names to exclude.

        Returns:
            Filtered list of results.
        """
        filtered = []

        for result in results:
            if include_validators and result.validator_name not in include_validators:
                continue
            if exclude_validators and result.validator_name in exclude_validators:
                continue

            filtered.append(result)

        return filtered

    def filter_failed(
        self,
        results: list["ValidatorResult"],
    ) -> list["ValidatorResult"]:
        """Filter to only failed results.

        Args:
            results: List of validator results.

        Returns:
            List of failed results.
        """
        return [r for r in results if not r.success]

    def filter_passed(
        self,
        results: list["ValidatorResult"],
    ) -> list["ValidatorResult"]:
        """Filter to only passed results.

        Args:
            results: List of validator results.

        Returns:
            List of passed results.
        """
        return [r for r in results if r.success]

    def sort_by_severity(
        self,
        results: list["ValidatorResult"],
        ascending: bool = False,
    ) -> list["ValidatorResult"]:
        """Sort results by severity.

        Args:
            results: List of validator results.
            ascending: If True, sort from low to critical.

        Returns:
            Sorted list of results.
        """
        return sorted(
            results,
            key=lambda r: _get_severity_level(r.severity),
            reverse=not ascending,
        )

    def sort_by_column(
        self,
        results: list["ValidatorResult"],
        ascending: bool = True,
    ) -> list["ValidatorResult"]:
        """Sort results by column name.

        Args:
            results: List of validator results.
            ascending: Sort order.

        Returns:
            Sorted list of results.
        """
        return sorted(
            results,
            key=lambda r: r.column or "",
            reverse=not ascending,
        )

    def limit(
        self,
        results: list["ValidatorResult"],
        count: int,
        offset: int = 0,
    ) -> list["ValidatorResult"]:
        """Limit results with optional offset.

        Args:
            results: List of validator results.
            count: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Limited list of results.
        """
        return results[offset : offset + count]


# =============================================================================
# Serialization Mixin
# =============================================================================


class SerializationMixin(ABC):
    """Mixin providing serialization utilities.

    Provides methods for:
    - JSON serialization with custom encoders
    - XML element building
    - YAML formatting
    - CSV generation
    """

    def to_json(
        self,
        data: Any,
        indent: int | None = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> str:
        """Serialize data to JSON.

        Args:
            data: Data to serialize.
            indent: Indentation level (None for compact).
            sort_keys: Whether to sort dictionary keys.
            ensure_ascii: Whether to escape non-ASCII characters.

        Returns:
            JSON string.
        """
        return json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=self._json_encoder,
        )

    def _json_encoder(self, obj: Any) -> Any:
        """Custom JSON encoder for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def to_csv(
        self,
        rows: Sequence[dict[str, Any]],
        columns: list[str] | None = None,
        delimiter: str = ",",
        include_header: bool = True,
    ) -> str:
        """Serialize data to CSV.

        Args:
            rows: Sequence of dictionaries.
            columns: Column order (default: keys from first row).
            delimiter: Field delimiter.
            include_header: Whether to include header row.

        Returns:
            CSV string.
        """
        if not rows:
            return ""

        if columns is None:
            columns = list(rows[0].keys())

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            delimiter=delimiter,
            extrasaction="ignore",
        )

        if include_header:
            writer.writeheader()

        for row in rows:
            # Convert non-string values
            converted = {}
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                elif isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, Enum):
                    value = value.value
                converted[col] = value
            writer.writerow(converted)

        return output.getvalue()

    def to_xml_element(
        self,
        tag: str,
        value: Any = None,
        attributes: dict[str, str] | None = None,
        children: list[str] | None = None,
    ) -> str:
        """Create an XML element string.

        Args:
            tag: Element tag name.
            value: Text content.
            attributes: Element attributes.
            children: Child element strings.

        Returns:
            XML element string.
        """
        attrs = ""
        if attributes:
            attrs = " " + " ".join(f'{k}="{self._escape_xml(str(v))}"' for k, v in attributes.items())

        if children:
            content = "\n".join(children)
            return f"<{tag}{attrs}>\n{content}\n</{tag}>"
        elif value is not None:
            return f"<{tag}{attrs}>{self._escape_xml(str(value))}</{tag}>"
        else:
            return f"<{tag}{attrs}/>"

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


# =============================================================================
# Templating Mixin
# =============================================================================


class TemplatingMixin(ABC):
    """Mixin providing template rendering utilities.

    Provides methods for:
    - Jinja2 template rendering
    - Template string interpolation
    - Conditional content
    """

    _jinja_env: Any = None

    def render_template(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """Render a Jinja2 template string.

        Args:
            template: Jinja2 template string.
            context: Template context variables.

        Returns:
            Rendered template.

        Raises:
            ImportError: If jinja2 is not installed.
        """
        try:
            from jinja2 import Environment, BaseLoader

            if self._jinja_env is None:
                self._jinja_env = Environment(loader=BaseLoader())

            tmpl = self._jinja_env.from_string(template)
            return tmpl.render(**context)

        except ImportError:
            raise ImportError(
                "jinja2 is required for template rendering. "
                "Install with: pip install jinja2"
            )

    def render_template_file(
        self,
        template_path: str,
        context: dict[str, Any],
    ) -> str:
        """Render a Jinja2 template file.

        Args:
            template_path: Path to template file.
            context: Template context variables.

        Returns:
            Rendered template.
        """
        try:
            from jinja2 import Environment, FileSystemLoader
            import os

            template_dir = os.path.dirname(template_path)
            template_name = os.path.basename(template_path)

            env = Environment(loader=FileSystemLoader(template_dir))
            tmpl = env.get_template(template_name)
            return tmpl.render(**context)

        except ImportError:
            raise ImportError(
                "jinja2 is required for template rendering. "
                "Install with: pip install jinja2"
            )

    def interpolate(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """Simple string interpolation without Jinja2.

        Args:
            template: Template string with {key} placeholders.
            context: Values to interpolate.

        Returns:
            Interpolated string.
        """
        return template.format(**context)


# =============================================================================
# Streaming Mixin
# =============================================================================


class StreamingMixin(ABC):
    """Mixin providing streaming output utilities.

    Provides methods for:
    - Streaming output for large datasets
    - Progress tracking
    - Chunked processing
    """

    def stream_results(
        self,
        results: list["ValidatorResult"],
        chunk_size: int = 100,
    ) -> Iterator[list["ValidatorResult"]]:
        """Stream results in chunks.

        Args:
            results: List of validator results.
            chunk_size: Number of results per chunk.

        Yields:
            Chunks of validator results.
        """
        for i in range(0, len(results), chunk_size):
            yield results[i : i + chunk_size]

    def stream_lines(
        self,
        results: list["ValidatorResult"],
        formatter: Callable[["ValidatorResult"], str],
    ) -> Iterator[str]:
        """Stream formatted lines for each result.

        Args:
            results: List of validator results.
            formatter: Function to format each result.

        Yields:
            Formatted lines.
        """
        for result in results:
            yield formatter(result)

    def render_streaming(
        self,
        results: list["ValidatorResult"],
        formatter: Callable[["ValidatorResult"], str],
        separator: str = "\n",
    ) -> str:
        """Render all results using streaming formatter.

        Args:
            results: List of validator results.
            formatter: Function to format each result.
            separator: Separator between formatted results.

        Returns:
            Complete rendered output.
        """
        return separator.join(self.stream_lines(results, formatter))
