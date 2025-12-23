"""Base classes for ML feature validators.

This module provides extensible base classes for implementing
machine learning feature validation algorithms.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import numpy as np

from truthound.validators.base import (
    Validator,
    ValidationIssue,
)


@dataclass
class FeatureStats:
    """Container for feature statistics.

    Stores computed statistics about a feature column
    for use in validation.
    """

    name: str
    dtype: str
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    mean: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    scale: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""

    feature1: str
    feature2: str
    correlation: float
    p_value: float | None = None
    is_significant: bool = False


@dataclass
class LeakageResult:
    """Container for target leakage detection results."""

    feature: str
    correlation: float
    mutual_information: float | None = None
    leakage_score: float = 0.0
    is_suspicious: bool = False
    reason: str = ""


class MLFeatureValidator(Validator):
    """Base class for ML feature validators.

    ML feature validators analyze feature quality for
    machine learning pipelines, checking for issues like
    null impact, scale consistency, multicollinearity,
    and target leakage.

    Subclasses should implement:
        - validate_features(): Core validation logic
        - validate(): Full validation method
    """

    category = "ml_feature"

    def __init__(
        self,
        columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        numeric_only: bool = True,
        **kwargs: Any,
    ):
        """Initialize ML feature validator.

        Args:
            columns: Specific columns to validate (None = all)
            exclude_columns: Columns to exclude from validation
            numeric_only: Whether to only consider numeric columns
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.columns = columns
        self.exclude_columns = exclude_columns or []
        self.numeric_only = numeric_only

    def _get_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of feature columns to validate.

        Args:
            df: Input DataFrame

        Returns:
            List of column names
        """
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.columns

        # Exclude specified columns
        cols = [c for c in cols if c not in self.exclude_columns]

        # Filter to numeric only if requested
        if self.numeric_only:
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
            cols = [c for c in cols if df[c].dtype in numeric_types]

        return cols

    def _compute_feature_stats(
        self, df: pl.DataFrame, column: str
    ) -> FeatureStats:
        """Compute statistics for a feature column.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            FeatureStats instance
        """
        col = df[column]
        total = len(col)
        null_count = col.null_count()
        null_ratio = null_count / total if total > 0 else 0.0

        stats = FeatureStats(
            name=column,
            dtype=str(col.dtype),
            null_count=null_count,
            null_ratio=null_ratio,
            unique_count=col.n_unique(),
        )

        # Compute numeric stats if applicable
        if col.dtype in [
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
        ]:
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                values = non_null.to_numpy()
                stats.mean = float(np.mean(values))
                stats.std = float(np.std(values))
                stats.min_value = float(np.min(values))
                stats.max_value = float(np.max(values))

                if stats.std > 0:
                    stats.scale = stats.max_value - stats.min_value

                # Compute skewness and kurtosis if enough values
                if len(values) > 2:
                    mean = stats.mean
                    std = stats.std
                    if std > 0:
                        stats.skewness = float(
                            np.mean(((values - mean) / std) ** 3)
                        )
                        stats.kurtosis = float(
                            np.mean(((values - mean) / std) ** 4) - 3
                        )

        return stats

    def _compute_correlation(
        self, df: pl.DataFrame, col1: str, col2: str
    ) -> float | None:
        """Compute Pearson correlation between two columns.

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column

        Returns:
            Correlation coefficient or None if not computable
        """
        try:
            # Get non-null pairs
            valid_mask = df[col1].is_not_null() & df[col2].is_not_null()
            valid_df = df.filter(valid_mask)

            if len(valid_df) < 3:
                return None

            x = valid_df[col1].to_numpy()
            y = valid_df[col2].to_numpy()

            # Compute correlation
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_std = np.std(x)
            y_std = np.std(y)

            if x_std == 0 or y_std == 0:
                return None

            correlation = np.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std)
            return float(correlation)
        except Exception:
            return None

    @abstractmethod
    def validate_features(
        self, df: pl.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Validate features in the DataFrame.

        Args:
            df: Input DataFrame
            columns: List of feature columns

        Returns:
            List of validation issues
        """
        pass
