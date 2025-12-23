"""Base classes for anomaly detection validators.

This module provides extensible base classes for implementing various
anomaly detection algorithms.
"""

from abc import abstractmethod
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)


class AnomalyValidator(Validator, NumericValidatorMixin):
    """Base class for table-wide anomaly detection.

    Anomaly validators detect unusual patterns or outliers in data.
    They can work on single columns or multiple columns simultaneously.

    Subclasses should implement:
        - detect_anomalies(): Returns indices or mask of anomalous rows
    """

    category = "anomaly"

    def __init__(
        self,
        columns: list[str] | None = None,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize anomaly validator.

        Args:
            columns: Specific columns to check. If None, uses all numeric columns.
            max_anomaly_ratio: Maximum acceptable ratio of anomalies (0.0-1.0)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.columns = columns
        self.max_anomaly_ratio = max_anomaly_ratio

    def _get_anomaly_columns(self, lf: pl.LazyFrame) -> list[str]:
        """Get columns to analyze for anomaly detection."""
        if self.columns:
            return self.columns
        return self._get_numeric_columns(lf)

    @abstractmethod
    def detect_anomalies(
        self, data: np.ndarray, column_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies in the data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features)
            column_names: Names of the columns

        Returns:
            Tuple of:
                - Boolean mask array where True indicates anomaly
                - Dictionary of additional info (e.g., scores, thresholds)
        """
        pass

    def _calculate_severity(self, anomaly_ratio: float) -> Severity:
        """Calculate severity based on anomaly ratio."""
        if anomaly_ratio < 0.01:
            return Severity.LOW
        elif anomaly_ratio < 0.05:
            return Severity.MEDIUM
        elif anomaly_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class ColumnAnomalyValidator(Validator, NumericValidatorMixin):
    """Base class for single-column anomaly detection.

    Use this for methods that detect anomalies in individual columns
    independently.
    """

    category = "anomaly"

    def __init__(
        self,
        column: str,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize column anomaly validator.

        Args:
            column: Column to check for anomalies
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.column = column
        self.max_anomaly_ratio = max_anomaly_ratio

    @abstractmethod
    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies in a single column.

        Args:
            values: 1D numpy array of column values (nulls removed)

        Returns:
            Tuple of:
                - Boolean mask array where True indicates anomaly
                - Dictionary of additional info
        """
        pass

    def _calculate_severity(self, anomaly_ratio: float) -> Severity:
        """Calculate severity based on anomaly ratio."""
        if anomaly_ratio < 0.01:
            return Severity.LOW
        elif anomaly_ratio < 0.05:
            return Severity.MEDIUM
        elif anomaly_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


class StatisticalAnomalyMixin:
    """Mixin providing statistical anomaly detection utilities."""

    @staticmethod
    def compute_iqr_bounds(
        values: np.ndarray, multiplier: float = 1.5
    ) -> tuple[float, float]:
        """Compute IQR-based bounds.

        Args:
            values: 1D numpy array
            multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return float(lower), float(upper)

    @staticmethod
    def compute_mad(values: np.ndarray) -> tuple[float, float]:
        """Compute Median Absolute Deviation.

        MAD is a robust measure of variability:
        MAD = median(|X - median(X)|)

        Args:
            values: 1D numpy array

        Returns:
            Tuple of (median, MAD)
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return float(median), float(mad)

    @staticmethod
    def compute_modified_zscore(values: np.ndarray) -> np.ndarray:
        """Compute modified Z-scores using MAD.

        Modified Z-score = 0.6745 * (x - median) / MAD

        The constant 0.6745 makes the MAD consistent with std for normal data.

        Args:
            values: 1D numpy array

        Returns:
            Array of modified Z-scores
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            # Fallback to mean absolute deviation
            mad = np.mean(np.abs(values - median))
            if mad == 0:
                return np.zeros_like(values)

        return 0.6745 * (values - median) / mad


class MLAnomalyMixin:
    """Mixin for machine learning based anomaly detection.

    Provides utilities for working with sklearn-based anomaly detectors.
    """

    @staticmethod
    def normalize_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize data using robust scaling.

        Uses median and IQR for robustness to outliers.

        Args:
            data: 2D numpy array (n_samples, n_features)

        Returns:
            Tuple of (normalized_data, medians, iqrs)
        """
        medians = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqrs = q3 - q1

        # Avoid division by zero
        iqrs = np.where(iqrs == 0, 1, iqrs)

        normalized = (data - medians) / iqrs
        return normalized, medians, iqrs

    @staticmethod
    def validate_sklearn_available() -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False
