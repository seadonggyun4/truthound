"""State management for the Reflex dashboard.

This module defines the application state and data management
for the interactive dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


@dataclass
class FilterState:
    """State for filtering and search."""

    search_query: str = ""
    selected_types: list[str] = field(default_factory=list)
    show_only_issues: bool = False
    min_null_ratio: float = 0.0
    max_null_ratio: float = 1.0
    sort_by: str = "name"
    sort_order: str = "asc"


@dataclass
class ChartState:
    """State for chart configuration."""

    selected_chart_type: str = "bar"
    selected_columns: list[str] = field(default_factory=list)
    show_grid: bool = True
    animation_enabled: bool = True


@dataclass
class ProfileState:
    """State for profile data."""

    # Raw data
    raw_data: dict[str, Any] = field(default_factory=dict)

    # Computed values
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    duplicate_count: int = 0
    quality_score: float = 100.0

    # Column data
    columns: list[dict[str, Any]] = field(default_factory=list)
    correlations: list[tuple[str, str, float]] = field(default_factory=list)

    # Alerts
    alerts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "ProfileState":
        """Create state from profile data."""
        state = cls()
        state.raw_data = profile
        state.row_count = profile.get("row_count", 0)
        state.column_count = profile.get("column_count", 0)
        state.memory_bytes = profile.get("estimated_memory_bytes", 0)
        state.duplicate_count = profile.get("duplicate_row_count", 0)
        state.columns = profile.get("columns", [])

        # Parse correlations
        corrs = profile.get("correlations", [])
        if corrs:
            if isinstance(corrs[0], dict):
                state.correlations = [
                    (c.get("column1", ""), c.get("column2", ""), c.get("correlation", 0))
                    for c in corrs
                ]
            else:
                state.correlations = corrs

        # Calculate quality score
        state.quality_score = cls._calculate_quality(profile)

        # Generate alerts
        state.alerts = cls._generate_alerts(profile)

        return state

    @staticmethod
    def _calculate_quality(profile: dict[str, Any]) -> float:
        """Calculate overall quality score."""
        columns = profile.get("columns", [])
        if not columns:
            return 100.0

        avg_null = sum(c.get("null_ratio", 0) for c in columns) / len(columns)
        return round((1 - avg_null) * 100, 1)

    @staticmethod
    def _generate_alerts(profile: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate alerts from profile."""
        alerts = []

        for col in profile.get("columns", []):
            name = col.get("name", "")
            null_ratio = col.get("null_ratio", 0)

            if null_ratio > 0.5:
                alerts.append({
                    "type": "warning" if null_ratio < 0.8 else "error",
                    "title": f"High missing values in '{name}'",
                    "message": f"{null_ratio:.1%} of values are null",
                    "column": name,
                })

            if col.get("is_constant", False):
                alerts.append({
                    "type": "info",
                    "title": f"Constant column: '{name}'",
                    "message": "Contains only one unique value",
                    "column": name,
                })

        dup_ratio = profile.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.1:
            alerts.append({
                "type": "warning",
                "title": "Duplicate rows detected",
                "message": f"{dup_ratio:.1%} of rows are duplicates",
            })

        return alerts


@dataclass
class DashboardState:
    """Main dashboard state."""

    # Profile data
    profile: ProfileState = field(default_factory=ProfileState)

    # Filter state
    filters: FilterState = field(default_factory=FilterState)

    # Chart state
    chart: ChartState = field(default_factory=ChartState)

    # UI state
    sidebar_open: bool = True
    active_tab: str = "overview"
    selected_column: str | None = None
    theme: str = "light"

    # Loading states
    is_loading: bool = False
    error_message: str | None = None

    def load_profile(self, path: str | Path) -> None:
        """Load profile from file."""
        self.is_loading = True
        self.error_message = None

        try:
            path = Path(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profile = ProfileState.from_profile(data)
        except Exception as e:
            self.error_message = str(e)
        finally:
            self.is_loading = False

    def load_profile_data(self, data: dict[str, Any]) -> None:
        """Load profile from dict."""
        self.is_loading = True
        self.error_message = None

        try:
            self.profile = ProfileState.from_profile(data)
        except Exception as e:
            self.error_message = str(e)
        finally:
            self.is_loading = False

    @property
    def filtered_columns(self) -> list[dict[str, Any]]:
        """Get filtered column list."""
        columns = self.profile.columns

        # Apply search filter
        if self.filters.search_query:
            query = self.filters.search_query.lower()
            columns = [c for c in columns if query in c.get("name", "").lower()]

        # Apply type filter
        if self.filters.selected_types:
            columns = [
                c for c in columns
                if c.get("inferred_type", c.get("physical_type", "")) in self.filters.selected_types
            ]

        # Apply null ratio filter
        columns = [
            c for c in columns
            if self.filters.min_null_ratio <= c.get("null_ratio", 0) <= self.filters.max_null_ratio
        ]

        # Apply issues filter
        if self.filters.show_only_issues:
            columns = [
                c for c in columns
                if c.get("null_ratio", 0) > 0.2 or c.get("is_constant", False)
            ]

        # Sort
        reverse = self.filters.sort_order == "desc"
        if self.filters.sort_by == "name":
            columns = sorted(columns, key=lambda c: c.get("name", ""), reverse=reverse)
        elif self.filters.sort_by == "null_ratio":
            columns = sorted(columns, key=lambda c: c.get("null_ratio", 0), reverse=reverse)
        elif self.filters.sort_by == "unique_ratio":
            columns = sorted(columns, key=lambda c: c.get("unique_ratio", 0), reverse=reverse)

        return columns

    def toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        self.sidebar_open = not self.sidebar_open

    def set_tab(self, tab: str) -> None:
        """Set active tab."""
        self.active_tab = tab

    def select_column(self, column: str) -> None:
        """Select a column for detail view."""
        self.selected_column = column

    def toggle_theme(self) -> None:
        """Toggle between light and dark theme."""
        self.theme = "dark" if self.theme == "light" else "light"
