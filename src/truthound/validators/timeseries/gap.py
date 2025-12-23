"""Time series gap detection validators.

This module provides validators for detecting gaps and irregularities
in time series data.
"""

from datetime import timedelta
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.timeseries.base import (
    TimeFrequency,
    TimeSeriesValidator,
)


@register_validator
class TimeSeriesGapValidator(TimeSeriesValidator):
    """Validates time series for gaps and irregular intervals.

    Detects:
    - Missing data points based on expected frequency
    - Irregular intervals between consecutive points
    - Large gaps that may indicate data collection issues

    Example:
        validator = TimeSeriesGapValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
            max_gap_multiplier=2.0,
        )
    """

    name = "timeseries_gap"

    def __init__(
        self,
        timestamp_column: str,
        frequency: TimeFrequency | str | None = None,
        max_gap_multiplier: float = 2.0,
        max_allowed_gaps: int | None = None,
        max_gap_ratio: float = 0.05,
        report_gap_locations: bool = True,
        max_reported_gaps: int = 10,
        **kwargs: Any,
    ):
        """Initialize gap validator.

        Args:
            timestamp_column: Column containing timestamps
            frequency: Expected frequency (auto-inferred if None)
            max_gap_multiplier: Multiplier of expected interval to consider a gap
            max_allowed_gaps: Maximum number of gaps allowed (None = no limit)
            max_gap_ratio: Maximum ratio of gaps to total intervals
            report_gap_locations: Whether to report gap locations in details
            max_reported_gaps: Maximum number of gaps to report in details
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            frequency=frequency,
            **kwargs,
        )
        self.max_gap_multiplier = max_gap_multiplier
        self.max_allowed_gaps = max_allowed_gaps
        self.max_gap_ratio = max_gap_ratio
        self.report_gap_locations = report_gap_locations
        self.max_reported_gaps = max_reported_gaps

    def _detect_gaps(
        self, df: pl.DataFrame, expected_interval: float
    ) -> list[tuple[int, float, Any, Any]]:
        """Detect gaps in the time series.

        Args:
            df: DataFrame with sorted timestamps
            expected_interval: Expected interval in seconds

        Returns:
            List of (index, gap_size, start_time, end_time) tuples
        """
        timestamps = df[self.timestamp_column].to_numpy()
        if len(timestamps) < 2:
            return []

        intervals = np.diff(timestamps).astype("timedelta64[s]").astype(float)
        threshold = expected_interval * self.max_gap_multiplier

        gaps = []
        for i, interval in enumerate(intervals):
            if interval > threshold:
                gap_multiplier = interval / expected_interval
                gaps.append((
                    i,
                    gap_multiplier,
                    timestamps[i],
                    timestamps[i + 1],
                ))

        return gaps

    def _estimate_missing_points(
        self, gaps: list[tuple[int, float, Any, Any]], expected_interval: float
    ) -> int:
        """Estimate number of missing data points.

        Args:
            gaps: List of detected gaps
            expected_interval: Expected interval in seconds

        Returns:
            Estimated number of missing points
        """
        missing = 0
        for _, gap_multiplier, _, _ in gaps:
            # Subtract 1 because 2x interval means 1 missing point
            missing += int(gap_multiplier) - 1
        return missing

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate time series for gaps.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_timestamps(lf)
        if len(df) < 2:
            return issues

        # Calculate intervals
        intervals = self._calculate_intervals(df)
        if len(intervals) == 0:
            return issues

        # Determine expected interval
        if self.frequency is not None:
            expected_interval = self.frequency.seconds
        else:
            # Auto-infer using median
            expected_interval = float(np.median(intervals))

        # Detect gaps
        gaps = self._detect_gaps(df, expected_interval)
        gap_count = len(gaps)
        gap_ratio = gap_count / len(intervals) if len(intervals) > 0 else 0

        # Check against thresholds
        has_issue = False

        if self.max_allowed_gaps is not None and gap_count > self.max_allowed_gaps:
            has_issue = True

        if gap_ratio > self.max_gap_ratio:
            has_issue = True

        if has_issue and gap_count > 0:
            missing_points = self._estimate_missing_points(gaps, expected_interval)

            # Build details
            details_parts = [
                f"Found {gap_count} gaps ({gap_ratio:.2%} of intervals). ",
                f"Expected interval: {timedelta(seconds=expected_interval)}. ",
                f"Estimated missing points: {missing_points}.",
            ]

            if self.report_gap_locations and gaps:
                sample_gaps = gaps[: self.max_reported_gaps]
                gap_strs = []
                for idx, mult, start, end in sample_gaps:
                    gap_strs.append(
                        f"[{idx}] {mult:.1f}x gap: {start} -> {end}"
                    )
                details_parts.append(f" Sample gaps: {'; '.join(gap_strs)}")

            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="timeseries_gaps_detected",
                    count=gap_count,
                    severity=self._calculate_severity(gap_ratio),
                    details="".join(details_parts),
                    expected=(
                        f"Gap ratio <= {self.max_gap_ratio:.2%}"
                        if self.max_allowed_gaps is None
                        else f"<= {self.max_allowed_gaps} gaps"
                    ),
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        return self.validate_series(lf)


@register_validator
class TimeSeriesIntervalValidator(TimeSeriesValidator):
    """Validates that time series intervals are within acceptable bounds.

    Checks both minimum and maximum interval constraints, detecting:
    - Too frequent data (intervals below minimum)
    - Too sparse data (intervals above maximum)
    - Interval variance outside expected range

    Example:
        validator = TimeSeriesIntervalValidator(
            timestamp_column="timestamp",
            min_interval=timedelta(seconds=1),
            max_interval=timedelta(hours=1),
        )
    """

    name = "timeseries_interval"

    def __init__(
        self,
        timestamp_column: str,
        min_interval: timedelta | None = None,
        max_interval: timedelta | None = None,
        max_interval_std: float | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize interval validator.

        Args:
            timestamp_column: Column containing timestamps
            min_interval: Minimum allowed interval
            max_interval: Maximum allowed interval
            max_interval_std: Maximum allowed standard deviation of intervals
            frequency: Expected frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            frequency=frequency,
            **kwargs,
        )
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_interval_std = max_interval_std

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate time series intervals.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_timestamps(lf)
        if len(df) < 2:
            return issues

        intervals = self._calculate_intervals(df)
        if len(intervals) == 0:
            return issues

        timestamps = df[self.timestamp_column].to_numpy()

        # Check minimum interval
        if self.min_interval is not None:
            min_seconds = self.min_interval.total_seconds()
            too_short = intervals < min_seconds
            too_short_count = int(np.sum(too_short))

            if too_short_count > 0:
                too_short_indices = np.where(too_short)[0]
                sample_indices = too_short_indices[:5]
                samples = [
                    f"{timestamps[i]} -> {timestamps[i+1]} ({intervals[i]:.1f}s)"
                    for i in sample_indices
                ]

                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="interval_too_short",
                        count=too_short_count,
                        severity=self._calculate_severity(
                            too_short_count / len(intervals)
                        ),
                        details=(
                            f"Found {too_short_count} intervals shorter than "
                            f"{self.min_interval}. Samples: {samples}"
                        ),
                        expected=f"All intervals >= {self.min_interval}",
                    )
                )

        # Check maximum interval
        if self.max_interval is not None:
            max_seconds = self.max_interval.total_seconds()
            too_long = intervals > max_seconds
            too_long_count = int(np.sum(too_long))

            if too_long_count > 0:
                too_long_indices = np.where(too_long)[0]
                sample_indices = too_long_indices[:5]
                samples = [
                    f"{timestamps[i]} -> {timestamps[i+1]} ({intervals[i]:.1f}s)"
                    for i in sample_indices
                ]

                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="interval_too_long",
                        count=too_long_count,
                        severity=self._calculate_severity(
                            too_long_count / len(intervals)
                        ),
                        details=(
                            f"Found {too_long_count} intervals longer than "
                            f"{self.max_interval}. Samples: {samples}"
                        ),
                        expected=f"All intervals <= {self.max_interval}",
                    )
                )

        # Check interval standard deviation
        if self.max_interval_std is not None:
            actual_std = float(np.std(intervals))
            if actual_std > self.max_interval_std:
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="interval_variance_too_high",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Interval standard deviation ({actual_std:.2f}s) exceeds "
                            f"maximum ({self.max_interval_std:.2f}s). Mean interval: "
                            f"{np.mean(intervals):.2f}s"
                        ),
                        expected=f"Interval std <= {self.max_interval_std}s",
                    )
                )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        return self.validate_series(lf)


@register_validator
class TimeSeriesDuplicateValidator(TimeSeriesValidator):
    """Validates that time series has no duplicate timestamps.

    Duplicate timestamps can indicate:
    - Data collection errors
    - Incorrect time zone handling
    - Merge/join issues

    Example:
        validator = TimeSeriesDuplicateValidator(
            timestamp_column="timestamp",
            max_duplicates=0,
        )
    """

    name = "timeseries_duplicate"

    def __init__(
        self,
        timestamp_column: str,
        max_duplicates: int = 0,
        max_duplicate_ratio: float = 0.0,
        report_duplicates: bool = True,
        **kwargs: Any,
    ):
        """Initialize duplicate validator.

        Args:
            timestamp_column: Column containing timestamps
            max_duplicates: Maximum allowed duplicate timestamps
            max_duplicate_ratio: Maximum ratio of duplicates
            report_duplicates: Whether to report duplicate timestamps
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            **kwargs,
        )
        self.max_duplicates = max_duplicates
        self.max_duplicate_ratio = max_duplicate_ratio
        self.report_duplicates = report_duplicates

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for duplicate timestamps.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = lf.select(pl.col(self.timestamp_column)).collect()
        if len(df) == 0:
            return issues

        total_count = len(df)
        unique_count = df[self.timestamp_column].n_unique()
        duplicate_count = total_count - unique_count
        duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0

        has_issue = (
            duplicate_count > self.max_duplicates
            or duplicate_ratio > self.max_duplicate_ratio
        )

        if has_issue and duplicate_count > 0:
            details_parts = [
                f"Found {duplicate_count} duplicate timestamps ",
                f"({duplicate_ratio:.2%} of {total_count} records).",
            ]

            if self.report_duplicates:
                # Find duplicate values
                duplicates = (
                    df.group_by(self.timestamp_column)
                    .agg(pl.len().alias("count"))
                    .filter(pl.col("count") > 1)
                    .sort("count", descending=True)
                    .head(5)
                )

                if len(duplicates) > 0:
                    samples = duplicates.to_dicts()
                    details_parts.append(f" Top duplicates: {samples}")

            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="duplicate_timestamps",
                    count=duplicate_count,
                    severity=self._calculate_severity(duplicate_ratio),
                    details="".join(details_parts),
                    expected=f"<= {self.max_duplicates} duplicate timestamps",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        return self.validate_series(lf)
