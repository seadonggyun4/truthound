"""Base classes for data drift validators.

This module provides base classes and utilities for detecting data drift
between reference (baseline) and current datasets.
"""

from abc import abstractmethod
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class DriftValidator(Validator):
    """Base class for drift detection validators.

    Drift validators compare a current dataset against a reference (baseline)
    dataset to detect distribution shifts, statistical changes, or data quality
    degradation over time.

    Key Concepts:
        - Reference data: The baseline/expected distribution (e.g., training data)
        - Current data: The data being validated (e.g., production data)
        - Drift score: A metric quantifying the difference between distributions
        - Threshold: The acceptable level of drift before triggering an alert

    Usage Pattern:
        1. Initialize with reference data and thresholds
        2. Call validate() with current data
        3. Check returned issues for drift detection results
    """

    name = "drift_base"
    category = "drift"

    def __init__(
        self,
        reference_data: pl.LazyFrame | pl.DataFrame,
        **kwargs: Any,
    ):
        """Initialize drift validator.

        Args:
            reference_data: Baseline data to compare against
            **kwargs: Additional config passed to base Validator
        """
        super().__init__(**kwargs)

        # Ensure reference data is LazyFrame for consistent handling
        if isinstance(reference_data, pl.DataFrame):
            self._reference_data = reference_data.lazy()
        else:
            self._reference_data = reference_data

    @property
    def reference_data(self) -> pl.LazyFrame:
        """Get the reference data as LazyFrame."""
        return self._reference_data

    @abstractmethod
    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> float:
        """Calculate drift score between reference and current data.

        Args:
            reference: Reference/baseline data
            current: Current data to check for drift

        Returns:
            Drift score (interpretation depends on specific validator)
        """
        pass

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate current data against reference for drift.

        Args:
            lf: Current LazyFrame to validate

        Returns:
            List of validation issues if drift is detected
        """
        pass

    def _calculate_severity(self, drift_score: float, threshold: float) -> Severity:
        """Calculate severity based on drift magnitude.

        Args:
            drift_score: The calculated drift score
            threshold: The threshold for drift detection

        Returns:
            Severity level based on how much drift exceeds threshold
        """
        if drift_score <= threshold:
            return Severity.LOW
        elif drift_score <= threshold * 1.5:
            return Severity.MEDIUM
        elif drift_score <= threshold * 2:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class ColumnDriftValidator(DriftValidator):
    """Base class for single-column drift validators.

    Validates drift for a specific column between reference and current data.
    """

    name = "column_drift_base"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        **kwargs: Any,
    ):
        """Initialize column drift validator.

        Args:
            column: Column name to check for drift
            reference_data: Baseline data to compare against
            **kwargs: Additional config
        """
        super().__init__(reference_data=reference_data, **kwargs)
        self.column = column

    def _get_column_values(
        self, lf: pl.LazyFrame, drop_nulls: bool = True
    ) -> pl.Series:
        """Extract column values as Series.

        Args:
            lf: LazyFrame to extract from
            drop_nulls: Whether to drop null values

        Returns:
            Series of column values
        """
        if drop_nulls:
            return lf.select(pl.col(self.column)).drop_nulls().collect().to_series()
        return lf.select(pl.col(self.column)).collect().to_series()


class NumericDriftMixin:
    """Mixin for numeric column drift detection utilities."""

    @staticmethod
    def compute_histogram(
        values: pl.Series, n_bins: int = 10, range_min: float | None = None, range_max: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Compute histogram for numeric values.

        Args:
            values: Series of numeric values
            n_bins: Number of histogram bins
            range_min: Minimum value for histogram range
            range_max: Maximum value for histogram range

        Returns:
            Tuple of (bin_edges, frequencies)
        """
        import numpy as np

        arr = values.to_numpy()
        arr = arr[~np.isnan(arr)]  # Remove NaN values

        if len(arr) == 0:
            return [], []

        # Determine range
        if range_min is None:
            range_min = float(arr.min())
        if range_max is None:
            range_max = float(arr.max())

        # Compute histogram
        counts, edges = np.histogram(arr, bins=n_bins, range=(range_min, range_max))

        # Normalize to frequencies
        total = counts.sum()
        if total > 0:
            frequencies = (counts / total).tolist()
        else:
            frequencies = [0.0] * n_bins

        return edges.tolist(), frequencies


class CategoricalDriftMixin:
    """Mixin for categorical column drift detection utilities."""

    @staticmethod
    def compute_category_frequencies(
        values: pl.Series,
    ) -> dict[str, float]:
        """Compute normalized frequencies for categorical values.

        Args:
            values: Series of categorical values

        Returns:
            Dictionary of category -> frequency
        """
        counts = values.value_counts()
        total = len(values)

        if total == 0:
            return {}

        result = {}
        for row in counts.iter_rows():
            category, count = row
            result[str(category)] = count / total

        return result

    @staticmethod
    def align_categories(
        ref_freq: dict[str, float], curr_freq: dict[str, float]
    ) -> tuple[list[float], list[float]]:
        """Align category frequencies between reference and current.

        Ensures both distributions have the same categories in the same order.

        Args:
            ref_freq: Reference category frequencies
            curr_freq: Current category frequencies

        Returns:
            Tuple of (aligned_ref_frequencies, aligned_curr_frequencies)
        """
        all_categories = sorted(set(ref_freq.keys()) | set(curr_freq.keys()))

        ref_aligned = [ref_freq.get(cat, 0.0) for cat in all_categories]
        curr_aligned = [curr_freq.get(cat, 0.0) for cat in all_categories]

        return ref_aligned, curr_aligned
