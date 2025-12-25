"""Profile Visualization Dashboard.

This package provides a Streamlit-based dashboard for interactive
data profiling and exploration.

Features:
- Real-time data profiling
- Interactive visualizations
- Column drill-down
- Export capabilities
- Comparison views

Example:
    # Run dashboard from command line:
    # truthound dashboard --port 8501

    # Or programmatically:
    from truthound.profiler.dashboard import run_dashboard, DashboardConfig

    config = DashboardConfig(port=8501, theme="dark")
    run_dashboard(config)
"""

from truthound.profiler.dashboard.config import (
    DashboardConfig,
    DashboardTheme,
)
from truthound.profiler.dashboard.app import (
    create_app,
    run_dashboard,
    DashboardApp,
)
from truthound.profiler.dashboard.components import (
    render_overview,
    render_column_details,
    render_quality_metrics,
    render_patterns,
)

__all__ = [
    "DashboardConfig",
    "DashboardTheme",
    "create_app",
    "run_dashboard",
    "DashboardApp",
    "render_overview",
    "render_column_details",
    "render_quality_metrics",
    "render_patterns",
]
