"""Dashboard UI components.

Provides reusable Streamlit components for the dashboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import polars as pl


def _check_streamlit() -> bool:
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def render_overview(profile_data: Dict[str, Any]) -> None:
    """Render the overview section.

    Args:
        profile_data: Profile data dictionary
    """
    if not _check_streamlit():
        return

    import streamlit as st

    st.header("Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{profile_data.get('row_count', 0):,}")

    with col2:
        st.metric("Columns", profile_data.get("column_count", 0))

    with col3:
        quality = profile_data.get("quality_scores", {}).get("overall", 0)
        st.metric("Quality Score", f"{quality:.1%}")

    with col4:
        nulls = sum(
            col.get("null_ratio", 0)
            for col in profile_data.get("columns", [])
        ) / max(1, len(profile_data.get("columns", [])))
        st.metric("Avg Null Rate", f"{nulls:.1%}")

    # Data type distribution
    st.subheader("Data Type Distribution")

    columns = profile_data.get("columns", [])
    if columns:
        type_counts: Dict[str, int] = {}
        for col in columns:
            dtype = col.get("inferred_type", col.get("physical_type", "unknown"))
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        # Create chart data
        import pandas as pd

        chart_df = pd.DataFrame({
            "Type": list(type_counts.keys()),
            "Count": list(type_counts.values()),
        })

        st.bar_chart(chart_df.set_index("Type"))


def render_column_details(
    columns: List[Dict[str, Any]],
    selected_column: Optional[str] = None,
) -> Optional[str]:
    """Render column details section.

    Args:
        columns: List of column profiles
        selected_column: Pre-selected column name

    Returns:
        Selected column name
    """
    if not _check_streamlit():
        return None

    import streamlit as st
    import pandas as pd

    st.header("Column Details")

    if not columns:
        st.warning("No column data available")
        return None

    # Column selector
    column_names = [col.get("name", "") for col in columns]
    selected = st.selectbox(
        "Select Column",
        column_names,
        index=column_names.index(selected_column) if selected_column in column_names else 0,
    )

    # Find selected column data
    col_data = next((c for c in columns if c.get("name") == selected), None)

    if col_data:
        # Column info cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Type**")
            st.code(col_data.get("inferred_type", "unknown"))

        with col2:
            st.markdown("**Physical Type**")
            st.code(col_data.get("physical_type", "unknown"))

        with col3:
            st.markdown("**Unique Ratio**")
            st.write(f"{col_data.get('unique_ratio', 0):.1%}")

        # Statistics
        st.subheader("Statistics")

        dist = col_data.get("distribution", {})
        if dist:
            stats_df = pd.DataFrame({
                "Statistic": ["Min", "Max", "Mean", "Std Dev", "Median"],
                "Value": [
                    dist.get("min", "N/A"),
                    dist.get("max", "N/A"),
                    dist.get("mean", "N/A"),
                    dist.get("std", "N/A"),
                    dist.get("median", "N/A"),
                ],
            })
            st.table(stats_df)

        # Top values
        top_values = col_data.get("top_values", [])
        if top_values:
            st.subheader("Top Values")
            top_df = pd.DataFrame(top_values, columns=["Value", "Count"])
            st.bar_chart(top_df.set_index("Value"))

        # Null analysis
        st.subheader("Null Analysis")
        null_ratio = col_data.get("null_ratio", 0)

        null_chart = pd.DataFrame({
            "Status": ["Non-Null", "Null"],
            "Percentage": [1 - null_ratio, null_ratio],
        })
        st.bar_chart(null_chart.set_index("Status"))

    return selected


def render_quality_metrics(
    quality_scores: Dict[str, float],
    alerts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render data quality metrics.

    Args:
        quality_scores: Quality score dictionary
        alerts: Optional list of alerts
    """
    if not _check_streamlit():
        return

    import streamlit as st
    import pandas as pd

    st.header("Data Quality")

    if not quality_scores:
        st.info("No quality scores available")
        return

    # Overall score with gauge-like display
    overall = quality_scores.get("overall", 0)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Overall Quality",
            f"{overall:.1%}",
            delta=f"{(overall - 0.5):.1%}" if overall != 0.5 else None,
        )

        # Color indicator
        if overall >= 0.8:
            st.success("Good quality")
        elif overall >= 0.6:
            st.warning("Moderate quality")
        else:
            st.error("Poor quality")

    with col2:
        # Dimension breakdown
        dimensions = {
            "Completeness": quality_scores.get("completeness", 0),
            "Validity": quality_scores.get("validity", 0),
            "Uniqueness": quality_scores.get("uniqueness", 0),
            "Consistency": quality_scores.get("consistency", 0),
        }

        dim_df = pd.DataFrame({
            "Dimension": list(dimensions.keys()),
            "Score": list(dimensions.values()),
        })

        st.bar_chart(dim_df.set_index("Dimension"))

    # Alerts
    if alerts:
        st.subheader("Quality Alerts")

        for alert in alerts[:10]:
            severity = alert.get("severity", "info")
            message = alert.get("message", "")
            column = alert.get("column", "")

            if severity == "error":
                st.error(f"**{column}**: {message}")
            elif severity == "warning":
                st.warning(f"**{column}**: {message}")
            else:
                st.info(f"**{column}**: {message}")


def render_patterns(
    patterns: List[Dict[str, Any]],
) -> None:
    """Render detected patterns.

    Args:
        patterns: List of detected patterns
    """
    if not _check_streamlit():
        return

    import streamlit as st
    import pandas as pd

    st.header("Detected Patterns")

    if not patterns:
        st.info("No patterns detected")
        return

    # Group by column
    by_column: Dict[str, List[Dict[str, Any]]] = {}
    for pattern in patterns:
        col = pattern.get("column", "unknown")
        by_column.setdefault(col, []).append(pattern)

    # Display per column
    for col_name, col_patterns in by_column.items():
        with st.expander(f"{col_name} ({len(col_patterns)} patterns)"):
            for p in col_patterns:
                col1, col2, col3 = st.columns([2, 1, 2])

                with col1:
                    st.write(f"**{p.get('name', 'Unknown')}**")

                with col2:
                    ratio = p.get("match_ratio", 0)
                    if ratio >= 0.9:
                        st.success(f"{ratio:.1%}")
                    elif ratio >= 0.7:
                        st.warning(f"{ratio:.1%}")
                    else:
                        st.error(f"{ratio:.1%}")

                with col3:
                    st.code(p.get("regex", "")[:50])


def render_recommendations(
    recommendations: List[str],
) -> None:
    """Render recommendations.

    Args:
        recommendations: List of recommendation strings
    """
    if not _check_streamlit():
        return

    import streamlit as st

    st.header("Recommendations")

    if not recommendations:
        st.success("No recommendations - data looks good!")
        return

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")


def render_comparison(
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
) -> None:
    """Render profile comparison view.

    Args:
        profile_a: First profile
        profile_b: Second profile
    """
    if not _check_streamlit():
        return

    import streamlit as st
    import pandas as pd

    st.header("Profile Comparison")

    # Overview comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(profile_a.get("name", "Profile A"))
        st.metric("Rows", f"{profile_a.get('row_count', 0):,}")
        st.metric("Columns", profile_a.get("column_count", 0))

    with col2:
        st.subheader(profile_b.get("name", "Profile B"))
        st.metric("Rows", f"{profile_b.get('row_count', 0):,}")
        st.metric("Columns", profile_b.get("column_count", 0))

    # Column comparison
    st.subheader("Column Changes")

    cols_a = {c.get("name"): c for c in profile_a.get("columns", [])}
    cols_b = {c.get("name"): c for c in profile_b.get("columns", [])}

    all_cols = set(cols_a.keys()) | set(cols_b.keys())

    comparison_data = []
    for col_name in sorted(all_cols):
        in_a = col_name in cols_a
        in_b = col_name in cols_b

        if in_a and in_b:
            status = "Modified" if cols_a[col_name] != cols_b[col_name] else "Unchanged"
        elif in_a:
            status = "Removed"
        else:
            status = "Added"

        null_a = cols_a.get(col_name, {}).get("null_ratio", 0)
        null_b = cols_b.get(col_name, {}).get("null_ratio", 0)

        comparison_data.append({
            "Column": col_name,
            "Status": status,
            "Null % (A)": f"{null_a:.1%}",
            "Null % (B)": f"{null_b:.1%}",
            "Delta": f"{(null_b - null_a):.1%}",
        })

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)


def render_data_preview(
    df: pl.DataFrame,
    max_rows: int = 100,
) -> None:
    """Render data preview table.

    Args:
        df: DataFrame to preview
        max_rows: Maximum rows to show
    """
    if not _check_streamlit():
        return

    import streamlit as st

    st.header("Data Preview")

    # Show shape
    st.caption(f"Showing {min(len(df), max_rows)} of {len(df):,} rows, {len(df.columns)} columns")

    # Convert to pandas for display
    preview_df = df.head(max_rows).to_pandas()
    st.dataframe(preview_df, use_container_width=True)


def render_export_options(
    profile_data: Dict[str, Any],
) -> None:
    """Render export options.

    Args:
        profile_data: Profile data to export
    """
    if not _check_streamlit():
        return

    import streamlit as st
    import json

    st.header("Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        # JSON export
        json_str = json.dumps(profile_data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="profile.json",
            mime="application/json",
        )

    with col2:
        # Generate HTML report
        try:
            from truthound.profiler.visualization import generate_report, ProfileData

            # Convert to ProfileData
            pd = ProfileData(
                table_name=profile_data.get("name", "data"),
                row_count=profile_data.get("row_count", 0),
                column_count=profile_data.get("column_count", 0),
                columns=profile_data.get("columns", []),
                quality_scores=profile_data.get("quality_scores"),
            )

            html = generate_report(pd)

            st.download_button(
                label="Download HTML Report",
                data=html,
                file_name="profile_report.html",
                mime="text/html",
            )
        except Exception as e:
            st.warning(f"HTML export not available: {e}")

    with col3:
        st.info("More export formats coming soon")
