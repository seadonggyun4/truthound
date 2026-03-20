"""Compatibility helpers for optional Pandas/Polars conversions.

These helpers preserve the fast native conversion path when optional
dependencies such as ``pyarrow`` are available, while providing a small
pure-Python fallback for CI environments that intentionally omit them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def _is_missing_pyarrow(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "pyarrow" in message or "no module named 'pyarrow'" in message


def pandas_to_polars_frame(df: "pd.DataFrame") -> "pl.DataFrame":
    """Convert a Pandas DataFrame to Polars without requiring pyarrow."""
    import polars as pl

    try:
        return pl.from_pandas(df)
    except (ImportError, ModuleNotFoundError) as exc:
        if not _is_missing_pyarrow(exc):
            raise

    data: dict[str, list[Any]] = {}
    for column in df.columns:
        series = df[column].astype(object)
        series = series.where(df[column].notna(), None)
        data[str(column)] = series.tolist()

    return pl.DataFrame(data)


def polars_to_pandas_frame(df: "pl.DataFrame") -> "pd.DataFrame":
    """Convert a Polars DataFrame to Pandas without requiring pyarrow."""
    import pandas as pd

    try:
        return df.to_pandas()
    except (ImportError, ModuleNotFoundError) as exc:
        if not _is_missing_pyarrow(exc):
            raise

    return pd.DataFrame(df.to_dict(as_series=False))
