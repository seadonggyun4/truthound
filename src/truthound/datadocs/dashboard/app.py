"""Main Reflex dashboard application.

This module provides the main dashboard application that brings together
all components and state management for an interactive profile exploration
experience.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DashboardConfig:
    """Configuration for the dashboard application."""

    # Server settings
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    # Theme
    theme: str = "light"  # light, dark
    primary_color: str = "blue"

    # Features
    show_raw_data: bool = True
    show_correlations: bool = True
    show_patterns: bool = True
    enable_export: bool = True

    # Data
    profile_path: str | None = None
    profile_data: dict[str, Any] | None = None

    # Custom branding
    title: str = "Truthound Dashboard"
    logo_url: str | None = None


class DashboardApp:
    """Interactive dashboard application using Reflex.

    This class provides a full-featured dashboard for exploring
    data profiles with interactive visualizations.
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        """Initialize the dashboard application.

        Args:
            config: Dashboard configuration
        """
        self._check_dependencies()
        self.config = config or DashboardConfig()
        self._app = None
        self._state = None

    def _check_dependencies(self) -> None:
        """Check if Reflex is installed."""
        try:
            import reflex as rx  # noqa: F401
        except ImportError:
            raise ImportError(
                "Dashboard requires Reflex. "
                "Install with: pip install truthound[dashboard]"
            )

    def _create_app(self) -> Any:
        """Create the Reflex application."""
        import reflex as rx

        from truthound.datadocs.dashboard.state import DashboardState
        from truthound.datadocs.dashboard import components as c

        # Create state class with Reflex decorators
        class State(rx.State):
            """Application state."""

            # Profile data
            profile_data: dict = {}
            row_count: int = 0
            column_count: int = 0
            memory_bytes: int = 0
            quality_score: float = 100.0
            columns: list = []
            correlations: list = []
            alerts: list = []

            # UI state
            sidebar_open: bool = True
            active_tab: str = "overview"
            selected_column: str = ""
            search_query: str = ""
            theme: str = "light"
            is_loading: bool = True

            def load_profile(self, data: dict) -> None:
                """Load profile data into state."""
                self.profile_data = data
                self.row_count = data.get("row_count", 0)
                self.column_count = data.get("column_count", 0)
                self.memory_bytes = data.get("estimated_memory_bytes", 0)
                self.columns = data.get("columns", [])

                # Parse correlations
                corrs = data.get("correlations", [])
                if corrs and isinstance(corrs[0], dict):
                    self.correlations = [
                        (c.get("column1", ""), c.get("column2", ""), c.get("correlation", 0))
                        for c in corrs
                    ]
                else:
                    self.correlations = corrs

                # Calculate quality
                if self.columns:
                    avg_null = sum(c.get("null_ratio", 0) for c in self.columns) / len(self.columns)
                    self.quality_score = round((1 - avg_null) * 100, 1)

                # Generate alerts
                self.alerts = self._generate_alerts(data)
                self.is_loading = False

            def _generate_alerts(self, data: dict) -> list:
                """Generate alerts from profile."""
                alerts = []
                for col in data.get("columns", []):
                    null_ratio = col.get("null_ratio", 0)
                    if null_ratio > 0.5:
                        alerts.append({
                            "type": "warning" if null_ratio < 0.8 else "error",
                            "title": f"High missing values in '{col.get('name', '')}'",
                            "message": f"{null_ratio:.1%} of values are null",
                        })
                return alerts

            def toggle_sidebar(self) -> None:
                """Toggle sidebar."""
                self.sidebar_open = not self.sidebar_open

            def set_tab(self, tab: str) -> None:
                """Set active tab."""
                self.active_tab = tab

            def select_column(self, column: str) -> None:
                """Select a column."""
                self.selected_column = column

            def set_search(self, query: str) -> None:
                """Set search query."""
                self.search_query = query

            def toggle_theme(self) -> None:
                """Toggle theme."""
                self.theme = "dark" if self.theme == "light" else "light"

            @rx.var
            def filtered_columns(self) -> list:
                """Get filtered columns."""
                cols = self.columns
                if self.search_query:
                    q = self.search_query.lower()
                    cols = [c for c in cols if q in c.get("name", "").lower()]
                return cols

            @rx.var
            def format_memory(self) -> str:
                """Format memory size."""
                bytes_val = self.memory_bytes
                for unit in ["B", "KB", "MB", "GB"]:
                    if abs(bytes_val) < 1024:
                        return f"{bytes_val:.1f} {unit}"
                    bytes_val /= 1024
                return f"{bytes_val:.1f} TB"

        # Define pages
        def overview_page() -> rx.Component:
            """Overview tab content."""
            return rx.vstack(
                # Metrics grid
                rx.grid(
                    rx.box(
                        rx.vstack(
                            rx.text("Rows", color="gray.500", size="2"),
                            rx.heading(State.row_count.to_string(), size="6"),
                            align_items="start",
                        ),
                        padding="4",
                        border_radius="lg",
                        border="1px solid",
                        border_color="gray.200",
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Columns", color="gray.500", size="2"),
                            rx.heading(State.column_count.to_string(), size="6"),
                            align_items="start",
                        ),
                        padding="4",
                        border_radius="lg",
                        border="1px solid",
                        border_color="gray.200",
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Memory", color="gray.500", size="2"),
                            rx.heading(State.format_memory, size="6"),
                            align_items="start",
                        ),
                        padding="4",
                        border_radius="lg",
                        border="1px solid",
                        border_color="gray.200",
                    ),
                    rx.box(
                        rx.vstack(
                            rx.text("Quality Score", color="gray.500", size="2"),
                            rx.heading(
                                rx.text(State.quality_score.to_string() + "%"),
                                size="6",
                            ),
                            align_items="start",
                        ),
                        padding="4",
                        border_radius="lg",
                        border="1px solid",
                        border_color="gray.200",
                    ),
                    columns="4",
                    spacing="4",
                    width="100%",
                ),
                # Alerts
                rx.cond(
                    State.alerts.length() > 0,
                    rx.vstack(
                        rx.heading("Alerts", size="4"),
                        rx.foreach(
                            State.alerts,
                            lambda alert: rx.box(
                                rx.hstack(
                                    rx.icon("alert-triangle", color="yellow.500"),
                                    rx.vstack(
                                        rx.text(alert["title"], font_weight="bold"),
                                        rx.text(alert["message"], size="2", color="gray.500"),
                                        align_items="start",
                                        spacing="0",
                                    ),
                                ),
                                padding="3",
                                border_radius="md",
                                bg="yellow.50",
                                border="1px solid",
                                border_color="yellow.200",
                                width="100%",
                            ),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                ),
                spacing="6",
                width="100%",
            )

        def columns_page() -> rx.Component:
            """Columns tab content."""
            return rx.vstack(
                # Search
                rx.input(
                    placeholder="Search columns...",
                    value=State.search_query,
                    on_change=State.set_search,
                    width="100%",
                ),
                # Column cards grid
                rx.grid(
                    rx.foreach(
                        State.filtered_columns,
                        lambda col: rx.box(
                            rx.vstack(
                                rx.hstack(
                                    rx.heading(col["name"], size="3"),
                                    rx.badge(
                                        col.get("inferred_type", col.get("physical_type", "unknown")),
                                    ),
                                    width="100%",
                                    justify_content="space-between",
                                ),
                                rx.hstack(
                                    rx.vstack(
                                        rx.text("Null", size="1", color="gray.500"),
                                        rx.text(
                                            (col.get("null_ratio", 0) * 100).to_string() + "%",
                                            font_weight="bold",
                                        ),
                                        spacing="0",
                                    ),
                                    rx.vstack(
                                        rx.text("Unique", size="1", color="gray.500"),
                                        rx.text(
                                            (col.get("unique_ratio", 0) * 100).to_string() + "%",
                                            font_weight="bold",
                                        ),
                                        spacing="0",
                                    ),
                                    rx.vstack(
                                        rx.text("Distinct", size="1", color="gray.500"),
                                        rx.text(
                                            col.get("distinct_count", 0).to_string(),
                                            font_weight="bold",
                                        ),
                                        spacing="0",
                                    ),
                                    width="100%",
                                    justify_content="space-around",
                                ),
                                align_items="start",
                                spacing="3",
                            ),
                            padding="4",
                            border_radius="lg",
                            border="1px solid",
                            border_color="gray.200",
                            _hover={"shadow": "md"},
                        ),
                    ),
                    columns="3",
                    spacing="4",
                    width="100%",
                ),
                spacing="4",
                width="100%",
            )

        def quality_page() -> rx.Component:
            """Quality tab content."""
            return rx.vstack(
                rx.heading("Data Quality Metrics", size="4"),
                rx.text(
                    "Quality analysis based on completeness, uniqueness, and validity.",
                    color="gray.500",
                ),
                # Quality scores would go here
                rx.center(
                    rx.vstack(
                        rx.text("Overall Quality", size="2", color="gray.500"),
                        rx.heading(State.quality_score.to_string() + "%", size="9"),
                    ),
                    padding="8",
                ),
                spacing="4",
                width="100%",
            )

        def index() -> rx.Component:
            """Main page layout."""
            return rx.box(
                rx.hstack(
                    # Sidebar
                    rx.cond(
                        State.sidebar_open,
                        rx.box(
                            rx.vstack(
                                rx.heading(self.config.title, size="4", padding="4"),
                                rx.divider(),
                                rx.vstack(
                                    rx.button(
                                        rx.icon("layout-dashboard"),
                                        rx.text("Overview"),
                                        width="100%",
                                        justify_content="start",
                                        variant="ghost" if State.active_tab != "overview" else "solid",
                                        on_click=lambda: State.set_tab("overview"),
                                    ),
                                    rx.button(
                                        rx.icon("columns"),
                                        rx.text("Columns"),
                                        width="100%",
                                        justify_content="start",
                                        variant="ghost" if State.active_tab != "columns" else "solid",
                                        on_click=lambda: State.set_tab("columns"),
                                    ),
                                    rx.button(
                                        rx.icon("bar-chart-2"),
                                        rx.text("Quality"),
                                        width="100%",
                                        justify_content="start",
                                        variant="ghost" if State.active_tab != "quality" else "solid",
                                        on_click=lambda: State.set_tab("quality"),
                                    ),
                                    padding="4",
                                    spacing="2",
                                ),
                                height="100vh",
                            ),
                            width="250px",
                            border_right="1px solid",
                            border_color="gray.200",
                            bg="white",
                        ),
                    ),
                    # Main content
                    rx.box(
                        rx.vstack(
                            # Header
                            rx.hstack(
                                rx.button(
                                    rx.icon("menu"),
                                    on_click=State.toggle_sidebar,
                                    variant="ghost",
                                ),
                                rx.spacer(),
                                rx.button(
                                    rx.cond(
                                        State.theme == "light",
                                        rx.icon("moon"),
                                        rx.icon("sun"),
                                    ),
                                    on_click=State.toggle_theme,
                                    variant="ghost",
                                ),
                                width="100%",
                                padding="4",
                                border_bottom="1px solid",
                                border_color="gray.200",
                            ),
                            # Content
                            rx.box(
                                rx.cond(
                                    State.is_loading,
                                    rx.center(
                                        rx.vstack(
                                            rx.spinner(size="3"),
                                            rx.text("Loading profile..."),
                                        ),
                                        height="400px",
                                    ),
                                    rx.match(
                                        State.active_tab,
                                        ("overview", overview_page()),
                                        ("columns", columns_page()),
                                        ("quality", quality_page()),
                                        overview_page(),
                                    ),
                                ),
                                padding="6",
                                width="100%",
                            ),
                        ),
                        flex="1",
                    ),
                    width="100%",
                    height="100vh",
                ),
                bg=rx.cond(State.theme == "dark", "gray.900", "gray.50"),
            )

        # Create app
        app = rx.App()

        @app.add_page
        def main():
            return index()

        # Store state class for data loading
        self._state_class = State
        self._app = app

        return app

    def load_profile(self, profile_path: str | Path | None = None, profile_data: dict | None = None) -> None:
        """Load profile data.

        Args:
            profile_path: Path to profile JSON file
            profile_data: Profile data dict
        """
        if profile_path:
            path = Path(profile_path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif profile_data:
            data = profile_data
        else:
            raise ValueError("Must provide either profile_path or profile_data")

        # Store for app initialization
        self._profile_data = data

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
    ) -> None:
        """Run the dashboard server.

        Args:
            host: Server host (overrides config)
            port: Server port (overrides config)
            debug: Debug mode (overrides config)
        """
        if self._app is None:
            self._create_app()

        # Load profile if configured
        if self.config.profile_path:
            self.load_profile(profile_path=self.config.profile_path)
        elif self.config.profile_data:
            self.load_profile(profile_data=self.config.profile_data)

        # Run the app
        self._app.run(
            host=host or self.config.host,
            port=port or self.config.port,
        )


def launch_dashboard(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    port: int = 8080,
    host: str = "localhost",
    title: str = "Truthound Dashboard",
    debug: bool = False,
) -> None:
    """Launch the interactive dashboard.

    Args:
        profile_path: Path to profile JSON file
        profile_data: Profile data dict
        port: Server port
        host: Server host
        title: Dashboard title
        debug: Debug mode
    """
    config = DashboardConfig(
        host=host,
        port=port,
        title=title,
        debug=debug,
        profile_path=str(profile_path) if profile_path else None,
        profile_data=profile_data,
    )

    app = DashboardApp(config)
    app.run()


def create_app(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    config: DashboardConfig | None = None,
) -> DashboardApp:
    """Create a dashboard application instance.

    Args:
        profile_path: Path to profile JSON file
        profile_data: Profile data dict
        config: Full configuration

    Returns:
        DashboardApp instance
    """
    if config is None:
        config = DashboardConfig(
            profile_path=str(profile_path) if profile_path else None,
            profile_data=profile_data,
        )

    return DashboardApp(config)
