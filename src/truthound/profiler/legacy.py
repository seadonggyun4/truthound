"""Legacy compatibility functions for the profiler module.

This module contains legacy functions that are maintained for backward
compatibility with older code (e.g., api.py). For new code, use the
modern profiler classes like DataProfiler or profile_dataframe instead.
"""

from __future__ import annotations

import polars as pl


def profile_data(lf: pl.LazyFrame, source: str = "unknown") -> dict:
    """Generate a statistical profile of the dataset.

    This is the legacy profile function maintained for backward compatibility
    with api.py. For new code, use DataProfiler or profile_dataframe instead.

    Args:
        lf: Polars LazyFrame to profile.
        source: Source identifier for the dataset.

    Returns:
        Dictionary containing profile information with keys:
            - source: Source identifier
            - row_count: Number of rows
            - column_count: Number of columns
            - size_bytes: Estimated size in bytes
            - columns: List of column info dicts with name, dtype, null_pct,
                      unique_pct, min, max
    """
    df = lf.collect()
    schema = lf.collect_schema()

    row_count = len(df)
    column_count = len(schema)

    # Estimate size in bytes
    size_bytes = df.estimated_size()

    columns_info: list[dict] = []

    numeric_types = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

    for col in schema.names():
        dtype = schema[col]
        col_data = df.get_column(col)

        null_count = col_data.null_count()
        null_pct = f"{(null_count / row_count * 100):.1f}%" if row_count > 0 else "0%"

        unique_count = col_data.n_unique()
        unique_pct = f"{(unique_count / row_count * 100):.0f}%" if row_count > 0 else "0%"

        col_info: dict = {
            "name": col,
            "dtype": str(dtype),
            "null_pct": null_pct,
            "unique_pct": unique_pct,
        }

        # Add min/max for numeric columns
        if dtype in numeric_types:
            non_null = col_data.drop_nulls()
            if len(non_null) > 0:
                min_val = non_null.min()
                max_val = non_null.max()
                col_info["min"] = str(min_val)
                col_info["max"] = str(max_val)
            else:
                col_info["min"] = "-"
                col_info["max"] = "-"
        else:
            col_info["min"] = "-"
            col_info["max"] = "-"

        columns_info.append(col_info)

    return {
        "source": source,
        "row_count": row_count,
        "column_count": column_count,
        "size_bytes": size_bytes,
        "columns": columns_info,
    }


__all__ = ["profile_data"]
