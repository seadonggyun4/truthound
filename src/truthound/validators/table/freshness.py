"""Table freshness validators.

Validators for checking data freshness and recency.
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.table.base import TableValidator
from truthound.validators.registry import register_validator


@register_validator
class TableFreshnessValidator(TableValidator):
    """Validates that table data is fresh (recent).

    Example:
        # Data should be updated within last 24 hours
        validator = TableFreshnessValidator(
            timestamp_column="updated_at",
            max_age_hours=24,
        )

        # Data should be updated within last 7 days
        validator = TableFreshnessValidator(
            timestamp_column="created_at",
            max_age_days=7,
        )
    """

    name = "table_freshness"
    category = "table"

    def __init__(
        self,
        timestamp_column: str,
        max_age_hours: int | None = None,
        max_age_days: int | None = None,
        max_age_minutes: int | None = None,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ):
        """Initialize freshness validator.

        Args:
            timestamp_column: Column containing timestamp to check
            max_age_hours: Maximum age in hours
            max_age_days: Maximum age in days
            max_age_minutes: Maximum age in minutes
            reference_time: Reference time (default: now)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.max_age_hours = max_age_hours
        self.max_age_days = max_age_days
        self.max_age_minutes = max_age_minutes
        self.reference_time = reference_time

        if max_age_hours is None and max_age_days is None and max_age_minutes is None:
            raise ValueError(
                "At least one of 'max_age_hours', 'max_age_days', or 'max_age_minutes' required"
            )

    def _get_max_age_timedelta(self) -> timedelta:
        """Calculate total max age as timedelta."""
        total_minutes = 0
        if self.max_age_minutes:
            total_minutes += self.max_age_minutes
        if self.max_age_hours:
            total_minutes += self.max_age_hours * 60
        if self.max_age_days:
            total_minutes += self.max_age_days * 24 * 60
        return timedelta(minutes=total_minutes)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get max timestamp from data
        result = lf.select(
            pl.col(self.timestamp_column).max().alias("max_ts")
        ).collect()

        max_ts = result["max_ts"][0]

        if max_ts is None:
            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="no_timestamp_data",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"No valid timestamps in column '{self.timestamp_column}'",
                    expected="At least one valid timestamp",
                )
            )
            return issues

        # Calculate reference time and threshold
        ref_time = self.reference_time or datetime.now()
        max_age = self._get_max_age_timedelta()
        threshold = ref_time - max_age

        # Convert max_ts to datetime if needed
        if isinstance(max_ts, datetime):
            data_time = max_ts
        else:
            # Try to convert from various formats
            try:
                if hasattr(max_ts, 'to_pydatetime'):
                    data_time = max_ts.to_pydatetime()
                else:
                    data_time = datetime.fromisoformat(str(max_ts))
            except (ValueError, AttributeError):
                issues.append(
                    ValidationIssue(
                        column=self.timestamp_column,
                        issue_type="invalid_timestamp_format",
                        count=1,
                        severity=Severity.HIGH,
                        details=f"Cannot parse timestamp: {max_ts}",
                        expected="Valid datetime format",
                    )
                )
                return issues

        # Compare timestamps (remove timezone info for comparison if needed)
        if hasattr(data_time, 'replace') and data_time.tzinfo is not None:
            data_time = data_time.replace(tzinfo=None)

        if data_time < threshold:
            age = ref_time - data_time
            age_str = self._format_timedelta(age)
            max_age_str = self._format_timedelta(max_age)

            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="data_not_fresh",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"Most recent data is {age_str} old, exceeds max age of {max_age_str}",
                    expected=f"Data within {max_age_str}",
                )
            )

        return issues

    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta to human-readable string."""
        total_seconds = int(td.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)


@register_validator
class TableDataRecencyValidator(TableValidator):
    """Validates that a minimum percentage of data is recent.

    Example:
        # At least 80% of data should be from last 30 days
        validator = TableDataRecencyValidator(
            timestamp_column="created_at",
            max_age_days=30,
            min_recent_ratio=0.8,
        )
    """

    name = "table_data_recency"
    category = "table"

    def __init__(
        self,
        timestamp_column: str,
        max_age_days: int,
        min_recent_ratio: float = 0.5,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ):
        """Initialize data recency validator.

        Args:
            timestamp_column: Column containing timestamp
            max_age_days: Maximum age for "recent" data
            min_recent_ratio: Minimum ratio of recent data (0.0 to 1.0)
            reference_time: Reference time (default: now)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.max_age_days = max_age_days
        self.min_recent_ratio = min_recent_ratio
        self.reference_time = reference_time

        if not 0 <= min_recent_ratio <= 1:
            raise ValueError("'min_recent_ratio' must be between 0.0 and 1.0")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        ref_time = self.reference_time or datetime.now()
        threshold = ref_time - timedelta(days=self.max_age_days)

        # Count total and recent rows
        result = lf.select([
            pl.len().alias("total"),
            (pl.col(self.timestamp_column) >= threshold).sum().alias("recent"),
        ]).collect()

        total = result["total"][0]
        recent = result["recent"][0]

        if total == 0:
            return issues

        recent_ratio = recent / total

        if recent_ratio < self.min_recent_ratio:
            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="insufficient_recent_data",
                    count=total - recent,
                    severity=Severity.MEDIUM,
                    details=f"Only {recent_ratio:.1%} of data is within {self.max_age_days} days, expected {self.min_recent_ratio:.1%}",
                    expected=f">= {self.min_recent_ratio:.1%} recent data",
                )
            )

        return issues


@register_validator
class TableUpdateFrequencyValidator(TableValidator):
    """Validates that data updates occur at expected frequency.

    Example:
        # Data should have entries for each day
        validator = TableUpdateFrequencyValidator(
            timestamp_column="date",
            expected_frequency="daily",
            max_gaps=2,  # Allow up to 2 missing days
        )
    """

    name = "table_update_frequency"
    category = "table"

    FREQUENCIES = {
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "monthly": timedelta(days=30),
        "hourly": timedelta(hours=1),
    }

    def __init__(
        self,
        timestamp_column: str,
        expected_frequency: str,
        max_gaps: int = 0,
        check_period_days: int | None = None,
        **kwargs: Any,
    ):
        """Initialize update frequency validator.

        Args:
            timestamp_column: Column containing timestamp
            expected_frequency: Expected update frequency ('daily', 'weekly', 'monthly', 'hourly')
            max_gaps: Maximum allowed gaps in the expected frequency
            check_period_days: Period to check (default: all data)
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.expected_frequency = expected_frequency
        self.max_gaps = max_gaps
        self.check_period_days = check_period_days

        if expected_frequency not in self.FREQUENCIES:
            raise ValueError(
                f"Invalid frequency: {expected_frequency}. Use one of {list(self.FREQUENCIES.keys())}"
            )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get min and max timestamps
        result = lf.select([
            pl.col(self.timestamp_column).min().alias("min_ts"),
            pl.col(self.timestamp_column).max().alias("max_ts"),
        ]).collect()

        min_ts = result["min_ts"][0]
        max_ts = result["max_ts"][0]

        if min_ts is None or max_ts is None:
            return issues

        # Get distinct dates/times based on frequency
        frequency = self.FREQUENCIES[self.expected_frequency]

        if self.expected_frequency == "daily":
            distinct_expr = pl.col(self.timestamp_column).cast(pl.Date).n_unique()
        elif self.expected_frequency == "weekly":
            distinct_expr = (
                pl.col(self.timestamp_column).dt.truncate("1w").n_unique()
            )
        elif self.expected_frequency == "monthly":
            distinct_expr = (
                pl.col(self.timestamp_column).dt.truncate("1mo").n_unique()
            )
        else:  # hourly
            distinct_expr = (
                pl.col(self.timestamp_column).dt.truncate("1h").n_unique()
            )

        distinct_count = lf.select(distinct_expr.alias("count")).collect()["count"][0]

        # Calculate expected count
        if isinstance(min_ts, datetime) and isinstance(max_ts, datetime):
            time_range = max_ts - min_ts
            expected_count = max(1, int(time_range / frequency) + 1)
        else:
            # For date types
            try:
                time_range = max_ts - min_ts
                expected_count = max(1, time_range.days // frequency.days + 1)
            except (TypeError, AttributeError):
                return issues

        gaps = expected_count - distinct_count

        if gaps > self.max_gaps:
            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="update_frequency_gaps",
                    count=gaps,
                    severity=Severity.MEDIUM,
                    details=f"Found {gaps} gaps in {self.expected_frequency} data, allowed {self.max_gaps}",
                    expected=f"<= {self.max_gaps} gaps in {self.expected_frequency} updates",
                )
            )

        return issues
