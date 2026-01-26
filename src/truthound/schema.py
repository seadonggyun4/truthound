"""Schema system for data validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from truthound.adapters import to_lazyframe

if TYPE_CHECKING:
    from truthound.datasources.base import BaseDataSource


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
    data: Any = None,
    source: "BaseDataSource | None" = None,
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
        source: Optional DataSource instance. If provided, data is ignored.
                This enables schema learning on SQL databases, Spark, etc.
        infer_constraints: Whether to infer constraints from data.
        categorical_threshold: Max unique values to treat as categorical.

    Returns:
        Schema learned from the data.

    Example:
        >>> schema = th.learn("data.csv")
        >>> schema.save("schema.yaml")

        >>> # Using DataSource for SQL database
        >>> from truthound.datasources.sql import SQLiteDataSource
        >>> source = SQLiteDataSource(database="mydb.db", table="users")
        >>> schema = th.learn(source=source)
    """
    # Handle DataSource if provided
    if source is not None:
        from truthound.datasources.base import BaseDataSource

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )
        lf = source.to_polars_lazyframe()
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")
        lf = to_lazyframe(data)
    polars_schema = lf.collect_schema()
    col_names = list(polars_schema.keys())

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

    # Classify columns by type
    numeric_cols = [c for c in col_names if polars_schema[c] in numeric_types]
    string_cols = [c for c in col_names if polars_schema[c] in (pl.String, pl.Utf8)]

    # Build all statistics in a single select (single pass over data)
    stat_exprs: list[pl.Expr] = [
        pl.len().alias("_row_count"),
    ]

    # Basic stats for all columns
    for c in col_names:
        stat_exprs.extend([
            pl.col(c).null_count().alias(f"{c}__nulls"),
            pl.col(c).n_unique().alias(f"{c}__unique"),
        ])

    if infer_constraints:
        # Numeric column stats
        for c in numeric_cols:
            stat_exprs.extend([
                pl.col(c).min().alias(f"{c}__min"),
                pl.col(c).max().alias(f"{c}__max"),
                pl.col(c).mean().alias(f"{c}__mean"),
                pl.col(c).std().alias(f"{c}__std"),
                pl.col(c).quantile(0.25).alias(f"{c}__q25"),
                pl.col(c).quantile(0.50).alias(f"{c}__q50"),
                pl.col(c).quantile(0.75).alias(f"{c}__q75"),
            ])

        # String column length stats
        for c in string_cols:
            stat_exprs.extend([
                pl.col(c).str.len_chars().min().alias(f"{c}__min_len"),
                pl.col(c).str.len_chars().max().alias(f"{c}__max_len"),
            ])

    # Execute single pass to collect all statistics
    # Use streaming for large datasets (>1M rows)
    stats_row = lf.select(stat_exprs).collect(engine="streaming").row(0, named=True)
    row_count = stats_row["_row_count"]

    # Build column schemas from collected statistics
    columns: dict[str, ColumnSchema] = {}

    for col_name in col_names:
        dtype = polars_schema[col_name]
        dtype_str = str(dtype)

        null_count = stats_row[f"{col_name}__nulls"]
        unique_count = stats_row[f"{col_name}__unique"]
        null_ratio = null_count / row_count if row_count > 0 else 0.0
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
            # Numeric columns: min, max, mean, std, quantiles
            if col_name in numeric_cols:
                min_val = stats_row.get(f"{col_name}__min")
                max_val = stats_row.get(f"{col_name}__max")
                mean_val = stats_row.get(f"{col_name}__mean")
                std_val = stats_row.get(f"{col_name}__std")

                if min_val is not None:
                    col_schema.min_value = float(min_val)
                if max_val is not None:
                    col_schema.max_value = float(max_val)
                if mean_val is not None:
                    col_schema.mean = float(mean_val)
                if std_val is not None:
                    col_schema.std = float(std_val)

                # Quantiles
                q25 = stats_row.get(f"{col_name}__q25")
                q50 = stats_row.get(f"{col_name}__q50")
                q75 = stats_row.get(f"{col_name}__q75")
                if q25 is not None and q50 is not None and q75 is not None:
                    col_schema.quantiles = {
                        "25%": float(q25),
                        "50%": float(q50),
                        "75%": float(q75),
                    }

            # String columns: length constraints
            if col_name in string_cols:
                min_len = stats_row.get(f"{col_name}__min_len")
                max_len = stats_row.get(f"{col_name}__max_len")
                if min_len is not None:
                    col_schema.min_length = int(min_len)
                if max_len is not None:
                    col_schema.max_length = int(max_len)

        columns[col_name] = col_schema

    # Collect allowed values for low cardinality columns (separate pass - necessary for unique values)
    if infer_constraints:
        low_cardinality_cols = [
            c for c in col_names
            if stats_row[f"{c}__unique"] <= categorical_threshold
            and stats_row[f"{c}__unique"] > 0
        ]

        if low_cardinality_cols:
            # Collect unique values - each column may have different lengths,
            # so we use implode() to collect as list then extract
            # Use streaming for large datasets
            unique_exprs = [
                pl.col(c).drop_nulls().unique().sort().implode().alias(c)
                for c in low_cardinality_cols
            ]
            unique_row = lf.select(unique_exprs).collect(engine="streaming").row(0, named=True)

            for c in low_cardinality_cols:
                values = unique_row[c]
                # Only store if values are simple types
                if all(isinstance(v, (str, int, float, bool)) for v in values):
                    columns[c].allowed_values = values

    return Schema(columns=columns, row_count=row_count)
