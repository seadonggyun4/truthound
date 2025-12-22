"""Schema system for data validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from truthound.adapters import to_lazyframe


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False

    # Constraints
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    pattern: str | None = None  # Regex pattern for strings
    min_length: int | None = None
    max_length: int | None = None

    # Statistics (learned from data)
    null_ratio: float | None = None
    unique_ratio: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] | None = None  # {"25%": val, "50%": val, "75%": val}

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "unique": self.unique,
        }

        # Add optional constraints
        if self.min_value is not None:
            result["min_value"] = self.min_value
        if self.max_value is not None:
            result["max_value"] = self.max_value
        if self.allowed_values is not None:
            result["allowed_values"] = self.allowed_values
        if self.pattern is not None:
            result["pattern"] = self.pattern
        if self.min_length is not None:
            result["min_length"] = self.min_length
        if self.max_length is not None:
            result["max_length"] = self.max_length

        # Add statistics
        if self.null_ratio is not None:
            result["null_ratio"] = round(self.null_ratio, 4)
        if self.unique_ratio is not None:
            result["unique_ratio"] = round(self.unique_ratio, 4)
        if self.mean is not None:
            result["mean"] = round(self.mean, 4)
        if self.std is not None:
            result["std"] = round(self.std, 4)
        if self.quantiles is not None:
            result["quantiles"] = {k: round(v, 4) for k, v in self.quantiles.items()}

        return result

    @classmethod
    def from_dict(cls, data: dict) -> ColumnSchema:
        """Create ColumnSchema from dictionary."""
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            nullable=data.get("nullable", True),
            unique=data.get("unique", False),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            pattern=data.get("pattern"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            null_ratio=data.get("null_ratio"),
            unique_ratio=data.get("unique_ratio"),
            mean=data.get("mean"),
            std=data.get("std"),
            quantiles=data.get("quantiles"),
        )


@dataclass
class Schema:
    """Complete schema for a dataset."""

    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    row_count: int | None = None
    version: str = "1.0"

    def __getitem__(self, key: str) -> ColumnSchema:
        """Get column schema by name."""
        return self.columns[key]

    def __contains__(self, key: str) -> bool:
        """Check if column exists in schema."""
        return key in self.columns

    def __iter__(self):
        """Iterate over column names."""
        return iter(self.columns)

    def save(self, path: str | Path) -> None:
        """Save schema to YAML file.

        Args:
            path: Path to save the schema file.
        """
        path = Path(path)
        data = self.to_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        """Load schema from YAML file.

        Args:
            path: Path to the schema file.

        Returns:
            Loaded Schema object.
        """
        path = Path(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "version": self.version,
            "row_count": self.row_count,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Schema:
        """Create Schema from dictionary."""
        columns = {
            name: ColumnSchema.from_dict(col_data) for name, col_data in data.get("columns", {}).items()
        }
        return cls(
            columns=columns,
            row_count=data.get("row_count"),
            version=data.get("version", "1.0"),
        )

    def get_column_names(self) -> list[str]:
        """Get list of column names."""
        return list(self.columns.keys())


def learn(
    data: Any,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
) -> Schema:
    """Learn schema from data.

    Analyzes the input data and generates a Schema with:
    - Column types
    - Null ratios
    - Unique ratios
    - Min/max values for numeric columns
    - Allowed values for low-cardinality columns
    - Statistical summaries

    Args:
        data: Input data (file path, DataFrame, dict, etc.)
        infer_constraints: Whether to infer constraints from data.
        categorical_threshold: Max unique values to treat as categorical.

    Returns:
        Schema learned from the data.

    Example:
        >>> schema = th.learn("data.csv")
        >>> schema.save("schema.yaml")
    """
    lf = to_lazyframe(data)
    df = lf.collect()

    row_count = len(df)
    polars_schema = df.schema
    columns: dict[str, ColumnSchema] = {}

    numeric_types = {
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
    }

    for col_name in df.columns:
        col_data = df.get_column(col_name)
        dtype = polars_schema[col_name]
        dtype_str = str(dtype)

        # Basic stats
        null_count = col_data.null_count()
        null_ratio = null_count / row_count if row_count > 0 else 0.0
        unique_count = col_data.n_unique()
        unique_ratio = unique_count / row_count if row_count > 0 else 0.0

        col_schema = ColumnSchema(
            name=col_name,
            dtype=dtype_str,
            nullable=null_count > 0,
            unique=unique_count == row_count and row_count > 0,
            null_ratio=null_ratio,
            unique_ratio=unique_ratio,
        )

        if infer_constraints:
            non_null = col_data.drop_nulls()

            # Numeric columns: min, max, mean, std, quantiles
            if dtype in numeric_types and len(non_null) > 0:
                col_schema.min_value = float(non_null.min())  # type: ignore
                col_schema.max_value = float(non_null.max())  # type: ignore
                col_schema.mean = float(non_null.mean())  # type: ignore
                col_schema.std = float(non_null.std()) if len(non_null) > 1 else 0.0  # type: ignore

                # Quantiles
                q25 = non_null.quantile(0.25)
                q50 = non_null.quantile(0.50)
                q75 = non_null.quantile(0.75)
                if q25 is not None and q50 is not None and q75 is not None:
                    col_schema.quantiles = {
                        "25%": float(q25),
                        "50%": float(q50),
                        "75%": float(q75),
                    }

            # Low cardinality columns: allowed values
            if unique_count <= categorical_threshold and unique_count > 0:
                values = non_null.unique().sort().to_list()
                # Only store if not too many and values are simple types
                if all(isinstance(v, (str, int, float, bool)) for v in values):
                    col_schema.allowed_values = values

            # String columns: length constraints
            if dtype in (pl.String, pl.Utf8) and len(non_null) > 0:
                lengths = non_null.str.len_chars()
                col_schema.min_length = int(lengths.min())  # type: ignore
                col_schema.max_length = int(lengths.max())  # type: ignore

        columns[col_name] = col_schema

    return Schema(columns=columns, row_count=row_count)
