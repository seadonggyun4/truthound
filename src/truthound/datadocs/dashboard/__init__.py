"""Interactive Dashboard module for Data Docs.

This module provides a Reflex-based interactive dashboard for exploring
data profiles. It requires the 'dashboard' extra to be installed.

Installation:
    pip install truthound[dashboard]

Usage:
    from truthound.datadocs.dashboard import launch_dashboard, create_app

    # Launch directly
    launch_dashboard(profile_path="profile.json", port=8080)

    # Or create app for customization
    app = create_app(profile_path="profile.json")
    app.run(port=8080)
"""

from truthound.datadocs.dashboard.app import (
    DashboardApp,
    DashboardConfig,
    launch_dashboard,
    create_app,
)

from truthound.datadocs.dashboard.state import (
    DashboardState,
    ProfileState,
    FilterState,
    ChartState,
)

from truthound.datadocs.dashboard.components import (
    header,
    sidebar,
    metric_card,
    chart_container,
    column_card,
    data_table,
    alert_banner,
)

__all__ = [
    # App
    "DashboardApp",
    "DashboardConfig",
    "launch_dashboard",
    "create_app",
    # State
    "DashboardState",
    "ProfileState",
    "FilterState",
    "ChartState",
    # Components
    "header",
    "sidebar",
    "metric_card",
    "chart_container",
    "column_card",
    "data_table",
    "alert_banner",
]
