"""Time series completeness validators.

This module provides validators for checking the completeness
of time series data.
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.timeseries.base import (
    TimeFrequency,
    TimeSeriesValidator,
    ValueTimeSeriesValidator,
)


@register_validator
class TimeSeriesCompletenessValidator(TimeSeriesValidator):
    """Validates completeness of time series data.

    Checks:
    - Coverage ratio (actual vs expected data points)
    - Date range coverage
    - Missing periods

    Example:
        validator = TimeSeriesCompletenessValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
            min_coverage=0.95,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
        )
    """

    name = "timeseries_completeness"

    def __init__(
        self,
        timestamp_column: str,
        frequency: TimeFrequency | str | None = None,
        min_coverage: float = 0.95,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_bounds: bool = True,
        **kwargs: Any,
    ):
        """Initialize completeness validator.

        Args:
            timestamp_column: Column containing timestamps
            frequency: Expected data frequency
            min_coverage: Minimum coverage ratio (0-1)
            start_time: Expected start of the series
            end_time: Expected end of the series
            include_bounds: Whether bounds are inclusive
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            frequency=frequency,
            **kwargs,
        )
        self.min_coverage = min_coverage
        self.start_time = start_time
        self.end_time = end_time
        self.include_bounds = include_bounds

    def _calculate_expected_count(
        self,
        start: datetime,
        end: datetime,
        interval_seconds: float,
    ) -> int:
        """Calculate expected number of data points.

        Args:
            start: Start timestamp
            end: End timestamp
            interval_seconds: Expected interval in seconds

        Returns:
            Expected count of data points
        """
        duration = (end - start).total_seconds()
        return max(1, int(duration / interval_seconds) + 1)

    def _find_missing_periods(
        self,
        timestamps: np.ndarray,
        expected_interval: float,
    ) -> list[tuple[Any, Any, int]]:
        """Find missing periods in the time series.

        Args:
            timestamps: Sorted array of timestamps
            expected_interval: Expected interval in seconds

        Returns:
            List of (start, end, missing_count) tuples
        """
        missing_periods = []

        if len(timestamps) < 2:
            return missing_periods

        for i in range(len(timestamps) - 1):
            interval = (
                np.datetime64(timestamps[i + 1]) - np.datetime64(timestamps[i])
            )
            interval_seconds = float(interval / np.timedelta64(1, "s"))

            if interval_seconds > expected_interval * 1.5:
                missing_count = int(interval_seconds / expected_interval) - 1
                if missing_count > 0:
                    missing_periods.append((
                        timestamps[i],
                        timestamps[i + 1],
                        missing_count,
                    ))

        return missing_periods

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate time series completeness.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_timestamps(lf)
        if len(df) == 0:
            if self.start_time is not None or self.end_time is not None:
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="empty_timeseries",
                        count=0,
                        severity=Severity.CRITICAL,
                        details="Time series is empty but data was expected.",
                        expected="Non-empty time series",
                    )
                )
            return issues

        timestamps = df[self.timestamp_column].to_numpy()
        actual_count = len(timestamps)

        # Determine time range
        actual_start = timestamps[0]
        actual_end = timestamps[-1]

        expected_start = self.start_time or actual_start
        expected_end = self.end_time or actual_end

        # Convert to datetime if needed
        if isinstance(expected_start, np.datetime64):
            expected_start = expected_start.astype("datetime64[us]").astype(datetime)
        if isinstance(expected_end, np.datetime64):
            expected_end = expected_end.astype("datetime64[us]").astype(datetime)
        if isinstance(actual_start, np.datetime64):
            actual_start = actual_start.astype("datetime64[us]").astype(datetime)
        if isinstance(actual_end, np.datetime64):
            actual_end = actual_end.astype("datetime64[us]").astype(datetime)

        # Determine expected interval
        intervals = self._calculate_intervals(df)
        if self.frequency is not None:
            expected_interval = self.frequency.seconds
        elif len(intervals) > 0:
            expected_interval = float(np.median(intervals))
        else:
            expected_interval = 1.0

        # Calculate expected count
        expected_count = self._calculate_expected_count(
            expected_start, expected_end, expected_interval
        )

        # Calculate coverage
        coverage = actual_count / expected_count if expected_count > 0 else 1.0

        if coverage < self.min_coverage:
            missing_count = expected_count - actual_count

            # Find specific missing periods
            missing_periods = self._find_missing_periods(timestamps, expected_interval)
            period_details = ""
            if missing_periods:
                sample_periods = missing_periods[:5]
                period_strs = [
                    f"{start} to {end} ({count} missing)"
                    for start, end, count in sample_periods
                ]
                period_details = f" Missing periods: {'; '.join(period_strs)}"

            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="incomplete_timeseries",
                    count=missing_count,
                    severity=self._calculate_severity(1 - coverage),
                    details=(
                        f"Time series coverage ({coverage:.2%}) below minimum "
                        f"({self.min_coverage:.2%}). Actual: {actual_count}, "
                        f"Expected: {expected_count}.{period_details}"
                    ),
                    expected=f"Coverage >= {self.min_coverage:.2%}",
                )
            )

        # Check start time
        if self.start_time is not None:
            start_diff = (actual_start - self.start_time).total_seconds()
            if start_diff > expected_interval:
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="late_series_start",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Time series starts at {actual_start}, "
                            f"expected {self.start_time}. "
                            f"Missing initial period of {timedelta(seconds=start_diff)}"
                        ),
                        expected=f"Series start at or before {self.start_time}",
                    )
                )

        # Check end time
        if self.end_time is not None:
            end_diff = (self.end_time - actual_end).total_seconds()
            if end_diff > expected_interval:
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="early_series_end",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Time series ends at {actual_end}, "
                            f"expected {self.end_time}. "
                            f"Missing final period of {timedelta(seconds=end_diff)}"
                        ),
                        expected=f"Series end at or after {self.end_time}",
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
class TimeSeriesValueCompletenessValidator(ValueTimeSeriesValidator):
    """Validates completeness of values in time series.

    Checks for:
    - NULL/NaN values
    - Missing value patterns
    - Value coverage ratio

    Example:
        validator = TimeSeriesValueCompletenessValidator(
            timestamp_column="timestamp",
            value_column="temperature",
            max_null_ratio=0.05,
            max_consecutive_nulls=3,
        )
    """

    name = "timeseries_value_completeness"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        max_null_ratio: float = 0.05,
        max_consecutive_nulls: int | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize value completeness validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values
            max_null_ratio: Maximum ratio of null values
            max_consecutive_nulls: Maximum consecutive null values
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )
        self.max_null_ratio = max_null_ratio
        self.max_consecutive_nulls = max_consecutive_nulls

    def _find_consecutive_nulls(
        self, values: np.ndarray
    ) -> list[tuple[int, int]]:
        """Find runs of consecutive null values.

        Args:
            values: Array of values

        Returns:
            List of (start_index, length) tuples
        """
        runs = []
        null_mask = np.isnan(values) if np.issubdtype(values.dtype, np.floating) else (values == None)  # noqa: E711

        if not np.any(null_mask):
            return runs

        # Find runs
        in_run = False
        run_start = 0

        for i, is_null in enumerate(null_mask):
            if is_null and not in_run:
                in_run = True
                run_start = i
            elif not is_null and in_run:
                runs.append((run_start, i - run_start))
                in_run = False

        if in_run:
            runs.append((run_start, len(values) - run_start))

        return runs

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate value completeness.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) == 0:
            return issues

        values = df[self.value_column].to_numpy()
        timestamps = df[self.timestamp_column].to_numpy()
        total_count = len(values)

        # Check null ratio
        null_mask = np.isnan(values) if np.issubdtype(values.dtype, np.floating) else (values == None)  # noqa: E711
        null_count = int(np.sum(null_mask))
        null_ratio = null_count / total_count if total_count > 0 else 0

        if null_ratio > self.max_null_ratio:
            # Sample null locations
            null_indices = np.where(null_mask)[0][:5]
            sample_times = [str(timestamps[i]) for i in null_indices]

            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="excessive_null_values",
                    count=null_count,
                    severity=self._calculate_severity(null_ratio),
                    details=(
                        f"Null ratio ({null_ratio:.2%}) exceeds maximum "
                        f"({self.max_null_ratio:.2%}). "
                        f"Sample null times: {sample_times}"
                    ),
                    expected=f"Null ratio <= {self.max_null_ratio:.2%}",
                )
            )

        # Check consecutive nulls
        if self.max_consecutive_nulls is not None:
            runs = self._find_consecutive_nulls(values)
            long_runs = [
                (start, length)
                for start, length in runs
                if length > self.max_consecutive_nulls
            ]

            if long_runs:
                run_details = [
                    f"[{timestamps[start]}]: {length} consecutive"
                    for start, length in long_runs[:5]
                ]

                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="consecutive_null_values",
                        count=len(long_runs),
                        severity=Severity.HIGH,
                        details=(
                            f"Found {len(long_runs)} run(s) of consecutive nulls "
                            f"exceeding {self.max_consecutive_nulls}. "
                            f"Runs: {'; '.join(run_details)}"
                        ),
                        expected=f"Max {self.max_consecutive_nulls} consecutive nulls",
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
class TimeSeriesDateRangeValidator(TimeSeriesValidator):
    """Validates that time series covers expected date ranges.

    Useful for:
    - Report period validation
    - Data pipeline freshness checks
    - Historical data completeness

    Example:
        validator = TimeSeriesDateRangeValidator(
            timestamp_column="date",
            required_start=datetime(2024, 1, 1),
            required_end=datetime(2024, 12, 31),
        )
    """

    name = "timeseries_date_range"

    def __init__(
        self,
        timestamp_column: str,
        required_start: datetime | None = None,
        required_end: datetime | None = None,
        max_start_delay: timedelta | None = None,
        max_end_gap: timedelta | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize date range validator.

        Args:
            timestamp_column: Column containing timestamps
            required_start: Required start date
            required_end: Required end date
            max_start_delay: Maximum delay from required start
            max_end_gap: Maximum gap before required end
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            frequency=frequency,
            **kwargs,
        )
        self.required_start = required_start
        self.required_end = required_end
        self.max_start_delay = max_start_delay
        self.max_end_gap = max_end_gap

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate date range coverage.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_timestamps(lf)
        if len(df) == 0:
            if self.required_start or self.required_end:
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="missing_date_range",
                        count=0,
                        severity=Severity.CRITICAL,
                        details="No data found but date range is required.",
                        expected="Data within required date range",
                    )
                )
            return issues

        timestamps = df[self.timestamp_column]
        actual_start = timestamps.min()
        actual_end = timestamps.max()

        # Convert polars datetime to python datetime if needed
        if hasattr(actual_start, "to_list"):
            actual_start = actual_start
        if hasattr(actual_end, "to_list"):
            actual_end = actual_end

        # Check start date
        if self.required_start is not None:
            start_delay = actual_start - self.required_start
            delay_seconds = start_delay.total_seconds() if hasattr(start_delay, "total_seconds") else 0

            if delay_seconds > 0:
                if self.max_start_delay is not None:
                    max_seconds = self.max_start_delay.total_seconds()
                    if delay_seconds > max_seconds:
                        issues.append(
                            ValidationIssue(
                                column=self.timestamp_column,
                                issue_type="start_date_too_late",
                                count=1,
                                severity=Severity.HIGH,
                                details=(
                                    f"Data starts at {actual_start}, "
                                    f"required start is {self.required_start}. "
                                    f"Delay: {timedelta(seconds=delay_seconds)}"
                                ),
                                expected=f"Start within {self.max_start_delay} of {self.required_start}",
                            )
                        )
                else:
                    issues.append(
                        ValidationIssue(
                            column=self.timestamp_column,
                            issue_type="start_date_after_required",
                            count=1,
                            severity=Severity.MEDIUM,
                            details=(
                                f"Data starts at {actual_start}, "
                                f"required start is {self.required_start}."
                            ),
                            expected=f"Start at or before {self.required_start}",
                        )
                    )

        # Check end date
        if self.required_end is not None:
            end_gap = self.required_end - actual_end
            gap_seconds = end_gap.total_seconds() if hasattr(end_gap, "total_seconds") else 0

            if gap_seconds > 0:
                if self.max_end_gap is not None:
                    max_seconds = self.max_end_gap.total_seconds()
                    if gap_seconds > max_seconds:
                        issues.append(
                            ValidationIssue(
                                column=self.timestamp_column,
                                issue_type="end_date_too_early",
                                count=1,
                                severity=Severity.HIGH,
                                details=(
                                    f"Data ends at {actual_end}, "
                                    f"required end is {self.required_end}. "
                                    f"Gap: {timedelta(seconds=gap_seconds)}"
                                ),
                                expected=f"End within {self.max_end_gap} of {self.required_end}",
                            )
                        )
                else:
                    issues.append(
                        ValidationIssue(
                            column=self.timestamp_column,
                            issue_type="end_date_before_required",
                            count=1,
                            severity=Severity.MEDIUM,
                            details=(
                                f"Data ends at {actual_end}, "
                                f"required end is {self.required_end}."
                            ),
                            expected=f"End at or after {self.required_end}",
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
