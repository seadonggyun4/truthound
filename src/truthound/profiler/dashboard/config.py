"""Dashboard configuration.

Provides configuration options for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DashboardTheme(str, Enum):
    """Dashboard theme options."""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


@dataclass
class DashboardConfig:
    """Configuration for the dashboard.

    Attributes:
        host: Host address to bind
        port: Port number
        theme: Visual theme
        title: Dashboard title
        page_icon: Emoji or path to favicon
        enable_uploads: Allow file uploads
        max_upload_size: Maximum upload size in MB
        cache_ttl: Cache time-to-live in seconds
        enable_export: Enable data export
        show_raw_data: Show raw data tables
        default_sample_size: Default sample size for large files
    """

    host: str = "localhost"
    port: int = 8501
    theme: DashboardTheme = DashboardTheme.LIGHT
    title: str = "Truthound Data Profiler"
    page_icon: str = "📊"
    enable_uploads: bool = True
    max_upload_size: int = 200  # MB
    cache_ttl: int = 3600  # seconds
    enable_export: bool = True
    show_raw_data: bool = True
    default_sample_size: int = 100000
    # Layout options
    wide_layout: bool = True
    sidebar_state: str = "expanded"
    # Feature flags
    enable_comparison: bool = True
    enable_recommendations: bool = True
    enable_ml_inference: bool = True
    # Security
    require_auth: bool = False
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".csv", ".parquet", ".json", ".xlsx", ".feather"
    ])
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_streamlit_config(self) -> Dict[str, Any]:
        """Convert to Streamlit page config."""
        return {
            "page_title": self.title,
            "page_icon": self.page_icon,
            "layout": "wide" if self.wide_layout else "centered",
            "initial_sidebar_state": self.sidebar_state,
        }
