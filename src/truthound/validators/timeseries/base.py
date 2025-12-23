"""Base classes for time series validators.

This module provides extensible base classes for implementing various
time series validation algorithms.
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    DatetimeValidatorMixin,
)


class TimeFrequency(str, Enum):
    """Common time series frequencies."""

    SECONDLY = "1s"
    MINUTELY = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    YEARLY = "1y"

    @property
    def timedelta(self) -> timedelta | None:
        """Convert to timedelta (approximate for month/year)."""
        mapping = {
            "1s": timedelta(seconds=1),
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1mo": timedelta(days=30),  # Approximate
            "3mo": timedelta(days=90),  # Approximate
            "1y": timedelta(days=365),  # Approximate
        }
        return mapping.get(self.value)

    @property
    def seconds(self) -> float:
        """Get frequency in seconds."""
        td = self.timedelta
        return td.total_seconds() if td else 0.0


@dataclass
class TimeSeriesStats:
    """Statistics for a time series column.

    Attributes:
        min_time: Earliest timestamp
        max_time: Latest timestamp
        count: Number of data points
        gaps: Number of detected gaps
        duplicates: Number of duplicate timestamps
        avg_interval: Average interval between points
        median_interval: Median interval between points
        std_interval: Standard deviation of intervals
    """

    min_time: datetime | None = None
    max_time: datetime | None = None
    count: int = 0
    gaps: int = 0
    duplicates: int = 0
    avg_interval: timedelta | None = None
    median_interval: timedelta | None = None
    std_interval: float | None = None


class TimeSeriesValidator(Validator, DatetimeValidatorMixin):
    """Base class for time series validators.

    Time series validators check temporal patterns, gaps, trends,
    and other characteristics of time-indexed data.

    Subclasses should implement:
        - validate_series(): Core validation logic
    """

    category = "timeseries"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize time series validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Optional column containing values
            frequency: Expected frequency of the time series
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.value_column = value_column

        if isinstance(frequency, str):
            try:
                self.frequency = TimeFrequency(frequency)
            except ValueError:
                self.frequency = None
                self._custom_frequency = frequency
        else:
            self.frequency = frequency
            self._custom_frequency = None

    def _get_sorted_timestamps(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Get sorted timestamps from the data.

        Args:
            lf: Input LazyFrame

        Returns:
            DataFrame with sorted timestamps
        """
        return (
            lf.select(pl.col(self.timestamp_column))
            .sort(self.timestamp_column)
            .collect()
        )

    def _calculate_intervals(self, df: pl.DataFrame) -> np.ndarray:
        """Calculate intervals between consecutive timestamps.

        Args:
            df: DataFrame with sorted timestamps

        Returns:
            Array of intervals in seconds
        """
        timestamps = df[self.timestamp_column].to_numpy()
        if len(timestamps) < 2:
            return np.array([])

        # Convert to numpy datetime64 for calculation
        intervals = np.diff(timestamps).astype("timedelta64[s]").astype(float)
        return intervals

    def _infer_frequency(self, intervals: np.ndarray) -> timedelta | None:
        """Infer the most common frequency from intervals.

        Args:
            intervals: Array of intervals in seconds

        Returns:
            Inferred frequency as timedelta
        """
        if len(intervals) == 0:
            return None

        # Use median as robust estimate
        median_seconds = np.median(intervals)
        return timedelta(seconds=float(median_seconds))

    def _compute_stats(self, lf: pl.LazyFrame) -> TimeSeriesStats:
        """Compute time series statistics.

        Args:
            lf: Input LazyFrame

        Returns:
            TimeSeriesStats object
        """
        df = self._get_sorted_timestamps(lf)
        stats = TimeSeriesStats()

        if len(df) == 0:
            return stats

        timestamps = df[self.timestamp_column]
        stats.count = len(timestamps)
        stats.min_time = timestamps.min()
        stats.max_time = timestamps.max()

        # Check duplicates
        stats.duplicates = len(timestamps) - len(timestamps.unique())

        # Calculate intervals
        intervals = self._calculate_intervals(df)
        if len(intervals) > 0:
            stats.avg_interval = timedelta(seconds=float(np.mean(intervals)))
            stats.median_interval = timedelta(seconds=float(np.median(intervals)))
            stats.std_interval = float(np.std(intervals))

        return stats

    def _calculate_severity(self, issue_ratio: float) -> Severity:
        """Calculate severity based on issue ratio.

        Args:
            issue_ratio: Ratio of problematic data points

        Returns:
            Appropriate severity level
        """
        if issue_ratio < 0.01:
            return Severity.LOW
        elif issue_ratio < 0.05:
            return Severity.MEDIUM
        elif issue_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    @abstractmethod
    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the time series.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        pass


class ValueTimeSeriesValidator(TimeSeriesValidator):
    """Base class for time series validators that also analyze values.

    Extends TimeSeriesValidator with value analysis capabilities
    for trend detection, seasonality, etc.
    """

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize value time series validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing numeric values
            frequency: Expected frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )

    def _get_sorted_series(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Get sorted time series with values.

        Args:
            lf: Input LazyFrame

        Returns:
            DataFrame sorted by timestamp with values
        """
        return (
            lf.select([
                pl.col(self.timestamp_column),
                pl.col(self.value_column),
            ])
            .sort(self.timestamp_column)
            .collect()
        )

    def _detrend_linear(self, values: np.ndarray) -> np.ndarray:
        """Remove linear trend from values.

        Args:
            values: Array of values

        Returns:
            Detrended values
        """
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend = np.polyval(coeffs, x)
        return values - trend

    def _compute_autocorrelation(
        self, values: np.ndarray, max_lag: int | None = None
    ) -> np.ndarray:
        """Compute autocorrelation function.

        Args:
            values: Array of values
            max_lag: Maximum lag to compute

        Returns:
            Array of autocorrelation values
        """
        n = len(values)
        if max_lag is None:
            max_lag = min(n // 2, 100)

        values = values - np.mean(values)
        variance = np.var(values)

        if variance == 0:
            return np.zeros(max_lag)

        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            acf[lag] = np.sum(values[: n - lag] * values[lag:]) / (n * variance)

        return acf
