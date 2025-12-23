"""Base classes for data profiling validators.

This module provides extensible base classes for implementing
data profiling and quality metrics validators.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl
import numpy as np

from truthound.validators.base import (
    Validator,
    ValidationIssue,
)


@dataclass
class ProfileMetrics:
    """Container for column profiling metrics.

    Attributes:
        total_count: Total number of rows
        null_count: Number of null values
        unique_count: Number of unique values
        cardinality: Unique count / total count
        entropy: Shannon entropy
        most_common: Most common value and count
        least_common: Least common value and count
    """

    total_count: int = 0
    null_count: int = 0
    unique_count: int = 0
    cardinality: float = 0.0
    entropy: float = 0.0
    most_common: tuple[Any, int] | None = None
    least_common: tuple[Any, int] | None = None


class ProfilingValidator(Validator):
    """Base class for data profiling validators.

    Data profiling validators analyze column characteristics
    to detect quality issues based on statistical properties
    like cardinality, entropy, and value distributions.

    Subclasses should implement:
        - validate_profile(): Core validation logic
        - validate(): Full validation method
    """

    category = "profiling"

    def __init__(
        self,
        column: str,
        include_nulls: bool = False,
        **kwargs: Any,
    ):
        """Initialize profiling validator.

        Args:
            column: Column to profile
            include_nulls: Whether to include nulls in calculations
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.include_nulls = include_nulls

    def _compute_value_counts(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute value counts for the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with value counts
        """
        col_data = df[self.column]

        if not self.include_nulls:
            col_data = col_data.drop_nulls()

        return (
            pl.DataFrame({self.column: col_data})
            .group_by(self.column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

    def _compute_entropy(self, value_counts: pl.DataFrame) -> float:
        """Compute Shannon entropy.

        Args:
            value_counts: DataFrame with value counts

        Returns:
            Entropy value in bits
        """
        if len(value_counts) == 0:
            return 0.0

        counts = value_counts["count"].to_numpy()
        total = counts.sum()

        if total == 0:
            return 0.0

        probabilities = counts / total
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]

        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    def _compute_metrics(self, df: pl.DataFrame) -> ProfileMetrics:
        """Compute all profiling metrics.

        Args:
            df: Input DataFrame

        Returns:
            ProfileMetrics object
        """
        metrics = ProfileMetrics()

        col_data = df[self.column]
        metrics.total_count = len(col_data)
        metrics.null_count = col_data.null_count()

        if metrics.total_count == 0:
            return metrics

        # Compute value counts
        value_counts = self._compute_value_counts(df)

        if len(value_counts) > 0:
            metrics.unique_count = len(value_counts)

            # Cardinality
            non_null_count = metrics.total_count - metrics.null_count
            if non_null_count > 0:
                metrics.cardinality = metrics.unique_count / non_null_count

            # Entropy
            metrics.entropy = self._compute_entropy(value_counts)

            # Most common
            first_row = value_counts.row(0)
            metrics.most_common = (first_row[0], first_row[1])

            # Least common
            last_row = value_counts.row(-1)
            metrics.least_common = (last_row[0], last_row[1])

        return metrics

    @abstractmethod
    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate based on profile metrics.

        Args:
            df: Input DataFrame
            metrics: Computed profile metrics

        Returns:
            List of validation issues
        """
        pass
