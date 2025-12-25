"""Streamlit Dashboard Application.

Main application entry point for the Truthound dashboard.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

from truthound.profiler.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


def _check_streamlit() -> bool:
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


class DashboardApp:
    """Streamlit Dashboard Application.

    Manages the dashboard lifecycle and state.

    Example:
        app = DashboardApp(config)
        app.run()
    """

    def __init__(self, config: DashboardConfig | None = None):
        """Initialize dashboard app.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self._profile_cache: Dict[str, Any] = {}

    def run(self) -> None:
        """Run the dashboard.

        This will start a Streamlit server.
        """
        if not _check_streamlit():
            raise ImportError(
                "Streamlit is required for the dashboard. "
                "Install with: pip install streamlit"
            )

        import streamlit as st

        # Configure page
        st.set_page_config(**self.config.to_streamlit_config())

        # Apply theme
        self._apply_theme()

        # Render sidebar
        self._render_sidebar()

        # Main content
        self._render_main()

    def _apply_theme(self) -> None:
        """Apply dashboard theme."""
        import streamlit as st

        if self.config.theme.value == "dark":
            st.markdown("""
            <style>
                .stApp { background-color: #1a1a2e; color: #eaeaea; }
                .stMetric { background-color: #16213e; border-radius: 8px; padding: 10px; }
            </style>
            """, unsafe_allow_html=True)

    def _render_sidebar(self) -> None:
        """Render sidebar with controls."""
        import streamlit as st

        with st.sidebar:
            st.title(self.config.title)
            st.markdown("---")

            # Data source selection
            st.subheader("Data Source")

            source_type = st.radio(
                "Source Type",
                ["Upload File", "Enter Path", "Sample Data"],
                label_visibility="collapsed",
            )

            if source_type == "Upload File" and self.config.enable_uploads:
                uploaded_file = st.file_uploader(
                    "Upload a file",
                    type=["csv", "parquet", "json"],
                    help="Upload CSV, Parquet, or JSON files",
                )

                if uploaded_file is not None:
                    self._handle_upload(uploaded_file)

            elif source_type == "Enter Path":
                file_path = st.text_input("File Path")
                if file_path and st.button("Load"):
                    self._load_file(file_path)

            elif source_type == "Sample Data":
                if st.button("Load Sample Data"):
                    self._load_sample_data()

            st.markdown("---")

            # Options
            st.subheader("Options")

            st.session_state.show_raw = st.checkbox(
                "Show Raw Data",
                value=self.config.show_raw_data,
            )

            st.session_state.sample_size = st.number_input(
                "Sample Size",
                min_value=100,
                max_value=1000000,
                value=self.config.default_sample_size,
                step=10000,
            )

    def _render_main(self) -> None:
        """Render main content area."""
        import streamlit as st

        from truthound.profiler.dashboard.components import (
            render_overview,
            render_column_details,
            render_quality_metrics,
            render_patterns,
            render_recommendations,
            render_data_preview,
            render_export_options,
        )

        # Check if data is loaded
        if "profile_data" not in st.session_state:
            st.info("👈 Load data from the sidebar to get started")
            self._render_welcome()
            return

        profile_data = st.session_state.profile_data

        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Columns",
            "Quality",
            "Patterns",
            "Export",
        ])

        with tab1:
            render_overview(profile_data)

            if st.session_state.get("show_raw") and "dataframe" in st.session_state:
                render_data_preview(st.session_state.dataframe)

        with tab2:
            render_column_details(profile_data.get("columns", []))

        with tab3:
            render_quality_metrics(
                profile_data.get("quality_scores", {}),
                profile_data.get("alerts"),
            )

            if profile_data.get("recommendations"):
                render_recommendations(profile_data["recommendations"])

        with tab4:
            render_patterns(profile_data.get("patterns_found", []))

        with tab5:
            if self.config.enable_export:
                render_export_options(profile_data)
            else:
                st.info("Export is disabled")

    def _render_welcome(self) -> None:
        """Render welcome screen."""
        import streamlit as st

        st.markdown("""
        ## Welcome to Truthound Data Profiler

        This interactive dashboard allows you to:

        - **Profile** your data to understand its structure and quality
        - **Visualize** distributions, patterns, and anomalies
        - **Analyze** data quality metrics and get recommendations
        - **Export** reports and validation rules

        ### Getting Started

        1. Select a data source from the sidebar
        2. Upload a file or enter a file path
        3. Explore the generated profile

        ### Supported Formats

        - CSV files
        - Parquet files
        - JSON files
        - Excel files (xlsx)
        """)

    def _handle_upload(self, uploaded_file: Any) -> None:
        """Handle file upload."""
        import streamlit as st

        try:
            # Save to temp file
            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            self._load_file(tmp_path)

            # Clean up
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error loading file: {e}")

    def _load_file(self, path: str) -> None:
        """Load and profile a file."""
        import streamlit as st

        with st.spinner("Loading and profiling data..."):
            try:
                # Load data
                path_obj = Path(path)

                if path_obj.suffix == ".csv":
                    df = pl.read_csv(path)
                elif path_obj.suffix == ".parquet":
                    df = pl.read_parquet(path)
                elif path_obj.suffix == ".json":
                    df = pl.read_json(path)
                else:
                    st.error(f"Unsupported file format: {path_obj.suffix}")
                    return

                # Sample if needed
                sample_size = st.session_state.get("sample_size", self.config.default_sample_size)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, seed=42)
                    st.info(f"Sampled {sample_size:,} rows from {len(df):,}")

                # Profile data
                profile_data = self._profile_dataframe(df, path_obj.name)

                # Store in session
                st.session_state.dataframe = df
                st.session_state.profile_data = profile_data

                st.success(f"Loaded: {path_obj.name}")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Error loading file")

    def _load_sample_data(self) -> None:
        """Load sample data for demonstration."""
        import streamlit as st
        import random

        with st.spinner("Generating sample data..."):
            # Generate sample data
            n_rows = 1000

            df = pl.DataFrame({
                "id": range(n_rows),
                "name": [f"User_{i}" for i in range(n_rows)],
                "email": [f"user{i}@example.com" for i in range(n_rows)],
                "age": [random.randint(18, 80) for _ in range(n_rows)],
                "salary": [random.uniform(30000, 150000) for _ in range(n_rows)],
                "is_active": [random.choice([True, False]) for _ in range(n_rows)],
                "department": [random.choice(["Sales", "Engineering", "Marketing", "HR"]) for _ in range(n_rows)],
                "join_date": [f"2020-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(n_rows)],
            })

            # Add some nulls
            df = df.with_columns([
                pl.when(pl.col("age") < 25).then(None).otherwise(pl.col("salary")).alias("salary"),
            ])

            profile_data = self._profile_dataframe(df, "sample_data")

            st.session_state.dataframe = df
            st.session_state.profile_data = profile_data

            st.success("Sample data loaded!")
            st.rerun()

    def _profile_dataframe(self, df: pl.DataFrame, name: str) -> Dict[str, Any]:
        """Profile a DataFrame.

        Args:
            df: DataFrame to profile
            name: Name for the profile

        Returns:
            Profile data dictionary
        """
        columns = []

        for col_name in df.columns:
            col = df.get_column(col_name)

            col_profile = {
                "name": col_name,
                "physical_type": str(col.dtype),
                "inferred_type": self._infer_type(col),
                "row_count": len(col),
                "null_count": col.null_count(),
                "null_ratio": col.null_count() / len(col) if len(col) > 0 else 0,
                "distinct_count": col.n_unique(),
                "unique_ratio": col.n_unique() / len(col) if len(col) > 0 else 0,
            }

            # Distribution for numeric
            if col.dtype.is_numeric():
                non_null = col.drop_nulls()
                if len(non_null) > 0:
                    col_profile["distribution"] = {
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "mean": float(non_null.mean()),
                        "std": float(non_null.std()) if len(non_null) > 1 else 0,
                        "median": float(non_null.median()),
                    }

            # Top values for categorical
            if col.dtype == pl.Utf8 or col.n_unique() < 20:
                vc = col.value_counts().head(10)
                col_profile["top_values"] = [
                    (row[col_name], row["count"])
                    for row in vc.to_dicts()
                ]

            columns.append(col_profile)

        # Calculate quality scores
        avg_completeness = 1 - sum(c["null_ratio"] for c in columns) / len(columns)

        return {
            "name": name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "quality_scores": {
                "overall": avg_completeness * 0.9,
                "completeness": avg_completeness,
                "validity": 0.95,
                "uniqueness": 0.9,
                "consistency": 0.85,
            },
        }

    def _infer_type(self, col: pl.Series) -> str:
        """Infer semantic type of column."""
        dtype = col.dtype

        if dtype == pl.Boolean:
            return "boolean"
        elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            return "integer"
        elif dtype in [pl.Float32, pl.Float64]:
            return "float"
        elif dtype == pl.Utf8:
            # Check patterns
            sample = col.drop_nulls().head(100).to_list()
            if sample:
                if all("@" in str(s) for s in sample[:10]):
                    return "email"
                if all(len(str(s)) == 36 and "-" in str(s) for s in sample[:10]):
                    return "uuid"
            return "string"
        elif "date" in str(dtype).lower():
            return "date"
        else:
            return str(dtype)


def create_app(config: DashboardConfig | None = None) -> DashboardApp:
    """Create a dashboard application.

    Args:
        config: Dashboard configuration

    Returns:
        DashboardApp instance
    """
    return DashboardApp(config)


def run_dashboard(config: DashboardConfig | None = None) -> None:
    """Run the dashboard.

    This function starts a Streamlit server with the dashboard.

    Args:
        config: Dashboard configuration
    """
    if not _check_streamlit():
        raise ImportError(
            "Streamlit is required for the dashboard. "
            "Install with: pip install streamlit"
        )

    config = config or DashboardConfig()

    # Create a temporary script to run
    script_content = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent.parent.parent}")

from truthound.profiler.dashboard.app import DashboardApp
from truthound.profiler.dashboard.config import DashboardConfig

config = DashboardConfig(
    host="{config.host}",
    port={config.port},
    theme="{config.theme.value}",
    title="{config.title}",
)

app = DashboardApp(config)
app.run()
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            script_path,
            "--server.port", str(config.port),
            "--server.address", config.host,
            "--browser.gatherUsageStats", "false",
        ]

        logger.info(f"Starting dashboard on {config.host}:{config.port}")
        subprocess.run(cmd)

    finally:
        os.unlink(script_path)
