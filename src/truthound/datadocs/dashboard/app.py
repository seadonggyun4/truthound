"""Main Reflex dashboard application.

This module provides the main dashboard application that brings together
all components and state management for an interactive profile exploration
experience.
"""

import json
from dataclasses import dataclass
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
        self._profile_data = None

    def _check_dependencies(self) -> None:
        """Check if Reflex is installed."""
        try:
            import reflex as rx  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Dashboard requires Reflex. Install with: pip install truthound[dashboard]"
            ) from error

    def _create_app(self) -> Any:
        """Create the Reflex application."""
        import reflex as rx

        from truthound.datadocs.dashboard.state import ProfileState

        if self._profile_data is None:
            if self.config.profile_path:
                self.load_profile(profile_path=self.config.profile_path)
            elif self.config.profile_data is not None:
                self.load_profile(profile_data=self.config.profile_data)
        initial = ProfileState.from_profile(self._profile_data or {})

        # Create state class with Reflex decorators
        class State(rx.State):
            """Application state."""

            # Profile data
            profile_data: dict[str, Any] = initial.raw_data
            row_count: int = initial.row_count
            column_count: int = initial.column_count
            memory_bytes: int = initial.memory_bytes
            quality_score: float = initial.quality_score
            columns: list[dict[str, Any]] = initial.columns
            correlations: list[tuple[str, str, float]] = initial.correlations
            alerts: list[dict[str, str]] = initial.alerts

            # UI state
            sidebar_open: bool = True
            active_tab: str = "overview"
            selected_column: str = ""
            search_query: str = ""
            is_loading: bool = False

            def load_profile(self, data: dict) -> None:
                """Load profile data into state."""
                profile = ProfileState.from_profile(data)
                self.profile_data = profile.raw_data
                self.row_count = profile.row_count
                self.column_count = profile.column_count
                self.memory_bytes = profile.memory_bytes
                self.columns = profile.columns
                self.correlations = profile.correlations
                self.quality_score = profile.quality_score
                self.alerts = profile.alerts
                self.is_loading = False

            @rx.var
            def profile_json(self) -> str:
                return json.dumps(self.profile_data, ensure_ascii=False, allow_nan=False, indent=2)

            @rx.var
            def pattern_json(self) -> str:
                return json.dumps(
                    [
                        {"column": column.get("name", ""), "patterns": column.get("patterns", [])}
                        for column in self.columns
                    ],
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )

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

            @rx.var
            def filtered_columns(self) -> list[dict[str, Any]]:
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
                                    rx.icon("triangle-alert", color="yellow.500"),
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
                                        col.get(
                                            "inferred_type", col.get("physical_type", "unknown")
                                        ),
                                    ),
                                    width="100%",
                                    justify_content="space-between",
                                ),
                                rx.hstack(
                                    rx.vstack(
                                        rx.text("Null", size="1", color="gray.500"),
                                        rx.text(
                                            (col.get("null_ratio", 0).to(float) * 100).to_string()
                                            + "%",
                                            font_weight="bold",
                                        ),
                                        spacing="0",
                                    ),
                                    rx.vstack(
                                        rx.text("Unique", size="1", color="gray.500"),
                                        rx.text(
                                            (col.get("unique_ratio", 0).to(float) * 100).to_string()
                                            + "%",
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
                                *(
                                    [rx.image(src=self.config.logo_url, alt=self.config.title)]
                                    if self.config.logo_url
                                    else []
                                ),
                                rx.heading(self.config.title, size="4", padding="4"),
                                rx.divider(),
                                rx.vstack(
                                    rx.button(
                                        rx.icon("layout-dashboard"),
                                        rx.text("Overview"),
                                        width="100%",
                                        justify_content="start",
                                        variant=rx.cond(
                                            State.active_tab != "overview", "ghost", "solid"
                                        ),
                                        on_click=lambda: State.set_tab("overview"),
                                    ),
                                    rx.button(
                                        rx.icon("columns-2"),
                                        rx.text("Columns"),
                                        width="100%",
                                        justify_content="start",
                                        variant=rx.cond(
                                            State.active_tab != "columns", "ghost", "solid"
                                        ),
                                        on_click=lambda: State.set_tab("columns"),
                                    ),
                                    rx.button(
                                        rx.icon("bar-chart-2"),
                                        rx.text("Quality"),
                                        width="100%",
                                        justify_content="start",
                                        variant=rx.cond(
                                            State.active_tab != "quality", "ghost", "solid"
                                        ),
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
                                    rx.color_mode_cond(
                                        rx.icon("moon"),
                                        rx.icon("sun"),
                                    ),
                                    on_click=rx.toggle_color_mode,
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
                            *(
                                [
                                    rx.box(
                                        rx.heading("Correlations", size="4"),
                                        rx.data_table(
                                            data=State.correlations,
                                            columns=["Column 1", "Column 2", "Correlation"],
                                        ),
                                        padding="6",
                                        width="100%",
                                    )
                                ]
                                if self.config.show_correlations
                                else []
                            ),
                            *(
                                [
                                    rx.box(
                                        rx.heading("Patterns", size="4"),
                                        rx.code_block(State.pattern_json, language="json"),
                                        padding="6",
                                        width="100%",
                                    )
                                ]
                                if self.config.show_patterns
                                else []
                            ),
                            *(
                                [
                                    rx.box(
                                        rx.heading("Profile data", size="4"),
                                        rx.code_block(State.profile_json, language="json"),
                                        padding="6",
                                        width="100%",
                                    )
                                ]
                                if self.config.show_raw_data
                                else []
                            ),
                            *(
                                [
                                    rx.button(
                                        "Export profile",
                                        on_click=rx.download(
                                            data=State.profile_json,
                                            filename="truthound-profile.json",
                                        ),
                                    )
                                ]
                                if self.config.enable_export
                                else []
                            ),
                        ),
                        flex="1",
                    ),
                    width="100%",
                    height="100vh",
                ),
                bg=rx.color_mode_cond("gray.50", "gray.900"),
            )

        # Create app
        app = rx.App(
            theme=rx.theme(appearance=self.config.theme, accent_color=self.config.primary_color)
        )

        app.add_page(index(), route="/", title=self.config.title)

        # Store state class for data loading
        self._state_class = State
        self._app = app

        return app

    def load_profile(
        self, profile_path: str | Path | None = None, profile_data: dict | None = None
    ) -> None:
        """Load profile data.

        Args:
            profile_path: Path to profile JSON file
            profile_data: Profile data dict
        """
        if profile_path:
            path = Path(profile_path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif profile_data is not None:
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
        import tempfile
        from dataclasses import replace

        from truthound.datadocs.dashboard.server import ReflexDashboardServer

        config = replace(
            self.config,
            host=self.config.host if host is None else host,
            port=self.config.port if port is None else port,
            debug=self.config.debug if debug is None else debug,
        )
        if self._profile_data is not None:
            config = replace(config, profile_path=None, profile_data=self._profile_data)
        with (
            tempfile.TemporaryDirectory(prefix="truthound-reflex-dashboard-") as directory,
            ReflexDashboardServer(Path(directory) / "project", config) as server,
        ):
            server.prepare()
            server.start()
            server.wait()


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
