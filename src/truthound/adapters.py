"""Input adapters for converting various data formats to Polars LazyFrame."""

import json
import tempfile
from pathlib import Path
from typing import Any

import polars as pl


def to_lazyframe(data: Any) -> pl.LazyFrame:
    """Convert various input formats to a Polars LazyFrame.

    Supports:
        - str: File path (CSV, JSON, Parquet)
        - pl.DataFrame: Polars DataFrame
        - pl.LazyFrame: Polars LazyFrame (passthrough)
        - dict: Python dictionary
        - pd.DataFrame: pandas DataFrame

    Args:
        data: Input data in any supported format.

    Returns:
        Polars LazyFrame for lazy evaluation.

    Raises:
        ValueError: If the input format is not supported.
        FileNotFoundError: If a file path is provided but doesn't exist.
    """
    # Already a LazyFrame
    if isinstance(data, pl.LazyFrame):
        return data

    # Polars DataFrame
    if isinstance(data, pl.DataFrame):
        return data.lazy()

    # Dictionary
    if isinstance(data, dict):
        return pl.DataFrame(data).lazy()

    # File path
    if isinstance(data, str):
        return _load_file(data)

    # Try pandas DataFrame
    if _is_pandas_dataframe(data):
        return pl.from_pandas(data).lazy()

    raise ValueError(
        f"Unsupported input type: {type(data).__name__}. "
        "Supported types: str (file path), pl.DataFrame, pl.LazyFrame, dict, pd.DataFrame"
    )


def _load_file(path: str) -> pl.LazyFrame:
    """Load a file into a Polars LazyFrame based on extension.

    Args:
        path: Path to the file.

    Returns:
        Polars LazyFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file extension is not supported.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pl.scan_csv(path)
    elif suffix == ".json":
        return _read_json_auto(file_path)
    elif suffix == ".parquet":
        return pl.scan_parquet(path)
    elif suffix == ".ndjson" or suffix == ".jsonl":
        return pl.scan_ndjson(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {suffix}. "
            "Supported extensions: .csv, .json, .parquet, .ndjson, .jsonl"
        )


def _is_pandas_dataframe(obj: Any) -> bool:
    """Check if an object is a pandas DataFrame without importing pandas.

    Args:
        obj: Object to check.

    Returns:
        True if the object is a pandas DataFrame.
    """
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _read_json_auto(path: Path) -> pl.LazyFrame:
    """Read JSON file, auto-detecting format (JSON Array or NDJSON).

    JSON Array format: [{}, {}, {}]
    NDJSON format: {}\n{}\n{}

    Args:
        path: Path to the JSON file.

    Returns:
        Polars LazyFrame.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        # Skip whitespace to find first meaningful character
        while first_char and first_char in " \t\n\r":
            first_char = f.read(1)

    if first_char == "[":
        # JSON Array format - convert to NDJSON for lazy loading
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False, encoding="utf-8"
        ) as tmp:
            for record in data:
                tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        return pl.scan_ndjson(tmp_path)
    else:
        # NDJSON format - read directly
        return pl.scan_ndjson(path)
