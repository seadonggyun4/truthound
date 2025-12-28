"""Filter transformers for the report pipeline.

These transformers filter and select which data to include in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Set

from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.transformers.base import BaseTransformer


class FilterTransformer(BaseTransformer):
    """General-purpose filter transformer.

    Filters sections, columns, alerts, and other data based on configuration.

    Example:
        # Include only specific sections
        transformer = FilterTransformer(
            include_sections=["overview", "quality", "alerts"],
        )

        # Exclude certain columns
        transformer = FilterTransformer(
            exclude_columns=["internal_id", "temp_*"],
        )
    """

    def __init__(
        self,
        include_sections: list[str] | None = None,
        exclude_sections: list[str] | None = None,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        min_alert_severity: str | None = None,
        max_recommendations: int | None = None,
        custom_filter: Callable[[ReportData], ReportData] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the filter transformer.

        Args:
            include_sections: Sections to include (whitelist).
            exclude_sections: Sections to exclude (blacklist).
            include_columns: Columns to include.
            exclude_columns: Columns to exclude.
            min_alert_severity: Minimum alert severity to include.
            max_recommendations: Maximum number of recommendations.
            custom_filter: Custom filter function.
            name: Transformer name.
        """
        super().__init__(name=name or "FilterTransformer")
        self._include_sections = set(include_sections) if include_sections else None
        self._exclude_sections = set(exclude_sections) if exclude_sections else set()
        self._include_columns = set(include_columns) if include_columns else None
        self._exclude_columns = set(exclude_columns) if exclude_columns else set()
        self._min_alert_severity = min_alert_severity
        self._max_recommendations = max_recommendations
        self._custom_filter = custom_filter

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        """Apply filters to the context.

        Args:
            ctx: Input context.

        Returns:
            Filtered context.
        """
        data = ctx.data

        # Filter sections
        data = self._filter_sections(data)

        # Filter columns in raw data
        data = self._filter_columns(data)

        # Filter alerts
        data = self._filter_alerts(data)

        # Filter recommendations
        data = self._filter_recommendations(data)

        # Apply custom filter
        if self._custom_filter:
            data = self._custom_filter(data)

        return ctx.with_data(data)

    def _filter_sections(self, data: ReportData) -> ReportData:
        """Filter sections based on include/exclude lists.

        Args:
            data: Report data.

        Returns:
            Data with filtered sections.
        """
        if not data.sections:
            return data

        filtered = {}
        for name, section in data.sections.items():
            # Check exclude first
            if name in self._exclude_sections:
                continue

            # Check include if specified
            if self._include_sections is not None:
                if name not in self._include_sections:
                    continue

            filtered[name] = section

        return replace(data, sections=filtered)

    def _filter_columns(self, data: ReportData) -> ReportData:
        """Filter columns in raw data.

        Args:
            data: Report data.

        Returns:
            Data with filtered columns.
        """
        if not data.raw.get("columns"):
            return data

        columns = data.raw.get("columns", [])
        filtered_columns = []

        for col in columns:
            col_name = col.get("name", "")

            # Check exclude patterns
            if self._matches_patterns(col_name, self._exclude_columns):
                continue

            # Check include if specified
            if self._include_columns is not None:
                if not self._matches_patterns(col_name, self._include_columns):
                    continue

            filtered_columns.append(col)

        new_raw = dict(data.raw)
        new_raw["columns"] = filtered_columns
        return replace(data, raw=new_raw)

    def _filter_alerts(self, data: ReportData) -> ReportData:
        """Filter alerts based on severity.

        Args:
            data: Report data.

        Returns:
            Data with filtered alerts.
        """
        if not data.alerts or not self._min_alert_severity:
            return data

        severity_levels = {
            "info": 0,
            "warning": 1,
            "error": 2,
            "critical": 3,
        }

        min_level = severity_levels.get(self._min_alert_severity.lower(), 0)

        filtered_alerts = [
            alert for alert in data.alerts
            if severity_levels.get(
                alert.get("severity", "info").lower(), 0
            ) >= min_level
        ]

        return replace(data, alerts=filtered_alerts)

    def _filter_recommendations(self, data: ReportData) -> ReportData:
        """Limit the number of recommendations.

        Args:
            data: Report data.

        Returns:
            Data with filtered recommendations.
        """
        if not data.recommendations or self._max_recommendations is None:
            return data

        return replace(
            data,
            recommendations=data.recommendations[:self._max_recommendations]
        )

    def _matches_patterns(self, name: str, patterns: Set[str]) -> bool:
        """Check if a name matches any pattern.

        Supports simple wildcard matching with '*'.

        Args:
            name: Name to check.
            patterns: Set of patterns.

        Returns:
            True if any pattern matches.
        """
        import fnmatch

        for pattern in patterns:
            if "*" in pattern:
                if fnmatch.fnmatch(name, pattern):
                    return True
            elif name == pattern:
                return True
        return False


class SectionFilter(BaseTransformer):
    """Simple section filter transformer.

    Example:
        filter = SectionFilter(["overview", "quality"])
        ctx = filter.transform(ctx)
    """

    def __init__(
        self,
        sections: list[str],
        mode: str = "include",
        name: str | None = None,
    ) -> None:
        """Initialize section filter.

        Args:
            sections: List of section names.
            mode: "include" or "exclude".
            name: Transformer name.
        """
        super().__init__(name=name or "SectionFilter")
        self._sections = set(sections)
        self._mode = mode

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data

        if not data.sections:
            return ctx

        if self._mode == "include":
            filtered = {
                k: v for k, v in data.sections.items()
                if k in self._sections
            }
        else:
            filtered = {
                k: v for k, v in data.sections.items()
                if k not in self._sections
            }

        return ctx.with_data(replace(data, sections=filtered))


class ColumnFilter(BaseTransformer):
    """Filter columns based on various criteria.

    Example:
        # Filter by name pattern
        filter = ColumnFilter(pattern="user_*")

        # Filter by type
        filter = ColumnFilter(types=["integer", "float"])

        # Filter by null ratio
        filter = ColumnFilter(max_null_ratio=0.5)
    """

    def __init__(
        self,
        pattern: str | None = None,
        types: list[str] | None = None,
        max_null_ratio: float | None = None,
        min_unique_ratio: float | None = None,
        names: list[str] | None = None,
        exclude_names: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize column filter.

        Args:
            pattern: Name pattern to match (supports wildcards).
            types: List of column types to include.
            max_null_ratio: Maximum null ratio to include.
            min_unique_ratio: Minimum unique ratio to include.
            names: Specific column names to include.
            exclude_names: Column names to exclude.
            name: Transformer name.
        """
        super().__init__(name=name or "ColumnFilter")
        self._pattern = pattern
        self._types = set(types) if types else None
        self._max_null_ratio = max_null_ratio
        self._min_unique_ratio = min_unique_ratio
        self._names = set(names) if names else None
        self._exclude_names = set(exclude_names) if exclude_names else set()

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        import fnmatch

        data = ctx.data
        columns = data.raw.get("columns", [])

        if not columns:
            return ctx

        filtered = []
        for col in columns:
            col_name = col.get("name", "")

            # Exclude check
            if col_name in self._exclude_names:
                continue

            # Name pattern check
            if self._pattern:
                if not fnmatch.fnmatch(col_name, self._pattern):
                    continue

            # Specific names check
            if self._names is not None:
                if col_name not in self._names:
                    continue

            # Type check
            if self._types is not None:
                col_type = col.get("inferred_type", col.get("physical_type", ""))
                if col_type not in self._types:
                    continue

            # Null ratio check
            if self._max_null_ratio is not None:
                null_ratio = col.get("null_ratio", 0)
                if null_ratio > self._max_null_ratio:
                    continue

            # Unique ratio check
            if self._min_unique_ratio is not None:
                unique_ratio = col.get("unique_ratio", 0)
                if unique_ratio < self._min_unique_ratio:
                    continue

            filtered.append(col)

        new_raw = dict(data.raw)
        new_raw["columns"] = filtered
        return ctx.with_data(replace(data, raw=new_raw))


class AlertFilter(BaseTransformer):
    """Filter alerts based on criteria.

    Example:
        # Only show errors and critical
        filter = AlertFilter(min_severity="error")

        # Only alerts for specific columns
        filter = AlertFilter(columns=["user_id", "email"])
    """

    def __init__(
        self,
        min_severity: str | None = None,
        max_severity: str | None = None,
        columns: list[str] | None = None,
        types: list[str] | None = None,
        limit: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize alert filter.

        Args:
            min_severity: Minimum severity level.
            max_severity: Maximum severity level.
            columns: Filter to alerts for these columns.
            types: Filter to these alert types.
            limit: Maximum number of alerts.
            name: Transformer name.
        """
        super().__init__(name=name or "AlertFilter")
        self._min_severity = min_severity
        self._max_severity = max_severity
        self._columns = set(columns) if columns else None
        self._types = set(types) if types else None
        self._limit = limit

        self._severity_levels = {
            "info": 0,
            "warning": 1,
            "error": 2,
            "critical": 3,
        }

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        data = ctx.data

        if not data.alerts:
            return ctx

        filtered = []
        for alert in data.alerts:
            # Severity check
            severity = alert.get("severity", "info").lower()
            level = self._severity_levels.get(severity, 0)

            if self._min_severity:
                min_level = self._severity_levels.get(self._min_severity.lower(), 0)
                if level < min_level:
                    continue

            if self._max_severity:
                max_level = self._severity_levels.get(self._max_severity.lower(), 3)
                if level > max_level:
                    continue

            # Column check
            if self._columns is not None:
                col = alert.get("column")
                if col and col not in self._columns:
                    continue

            # Type check
            if self._types is not None:
                alert_type = alert.get("type", "")
                if alert_type not in self._types:
                    continue

            filtered.append(alert)

        # Apply limit
        if self._limit is not None:
            filtered = filtered[:self._limit]

        return ctx.with_data(replace(data, alerts=filtered))
