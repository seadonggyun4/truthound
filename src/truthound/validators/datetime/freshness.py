"""Data freshness and temporal coverage validators."""

from datetime import datetime, timedelta, date
from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator, DatetimeValidatorMixin
from truthound.validators.registry import register_validator


@register_validator
class RecentDataValidator(Validator, DatetimeValidatorMixin):
    """Validates that data contains recent records.

    Example:
        # Data should have records from the last 24 hours
        validator = RecentDataValidator(
            column="created_at",
            max_age_hours=24,
        )
    """

    name = "recent_data"
    category = "datetime"

    def __init__(
        self,
        column: str,
        max_age_hours: float | None = None,
        max_age_days: float | None = None,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.reference_time = reference_time or datetime.now()

        if max_age_hours:
            self.max_age = timedelta(hours=max_age_hours)
        elif max_age_days:
            self.max_age = timedelta(days=max_age_days)
        else:
            self.max_age = timedelta(hours=24)  # Default 24 hours

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        result = lf.select([
            pl.col(self.column).max().alias("_max_date"),
        ]).collect()

        max_date = result["_max_date"][0]

        if max_date is None:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="no_recent_data",
                    count=1,
                    severity=Severity.CRITICAL,
                    details="No data found in datetime column",
                )
            )
            return issues

        # Convert to datetime if needed
        if isinstance(max_date, date) and not isinstance(max_date, datetime):
            max_date = datetime.combine(max_date, datetime.min.time())

        cutoff = self.reference_time - self.max_age
        age = self.reference_time - max_date

        if max_date < cutoff:
            hours_old = age.total_seconds() / 3600
            severity = Severity.CRITICAL if hours_old > self.max_age.total_seconds() / 3600 * 2 else Severity.HIGH

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="stale_data",
                    count=1,
                    severity=severity,
                    details=f"Most recent data is {hours_old:.1f} hours old",
                    expected=f"Data within last {self.max_age.total_seconds() / 3600:.1f} hours",
                    actual=f"Latest: {max_date}",
                )
            )

        return issues


@register_validator
class DatePartCoverageValidator(Validator, DatetimeValidatorMixin):
    """Validates that data covers expected date parts (no gaps).

    Example:
        # Should have data for every day in the range
        validator = DatePartCoverageValidator(
            column="date",
            date_part="day",
            min_coverage=0.95,  # Allow 5% missing days
        )
    """

    name = "date_part_coverage"
    category = "datetime"

    def __init__(
        self,
        column: str,
        date_part: Literal["day", "week", "month", "hour"] = "day",
        min_coverage: float = 1.0,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.date_part = date_part
        self.min_coverage = min_coverage
        self.start_date = start_date
        self.end_date = end_date

    def _count_expected_parts(self, start: datetime, end: datetime) -> int:
        """Count expected date parts in range."""
        diff = end - start

        if self.date_part == "hour":
            return int(diff.total_seconds() / 3600) + 1
        elif self.date_part == "day":
            return diff.days + 1
        elif self.date_part == "week":
            return diff.days // 7 + 1
        elif self.date_part == "month":
            return (end.year - start.year) * 12 + (end.month - start.month) + 1

        return diff.days + 1

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get date range from data
        result = lf.select([
            pl.col(self.column).min().alias("_min"),
            pl.col(self.column).max().alias("_max"),
        ]).collect()

        min_date = result["_min"][0]
        max_date = result["_max"][0]

        if min_date is None or max_date is None:
            return issues

        # Use provided dates or data range
        start = self.start_date or min_date
        end = self.end_date or max_date

        # Convert to datetime
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time())
        if isinstance(end, date) and not isinstance(end, datetime):
            end = datetime.combine(end, datetime.max.time())

        # Count distinct date parts
        if self.date_part == "day":
            truncate_expr = pl.col(self.column).dt.truncate("1d")
        elif self.date_part == "hour":
            truncate_expr = pl.col(self.column).dt.truncate("1h")
        elif self.date_part == "week":
            truncate_expr = pl.col(self.column).dt.truncate("1w")
        elif self.date_part == "month":
            truncate_expr = pl.col(self.column).dt.truncate("1mo")
        else:
            truncate_expr = pl.col(self.column).dt.truncate("1d")

        distinct_parts = lf.select(
            truncate_expr.n_unique().alias("_distinct")
        ).collect()["_distinct"][0]

        expected_parts = self._count_expected_parts(start, end)

        if expected_parts > 0:
            coverage = distinct_parts / expected_parts
            missing = expected_parts - distinct_parts

            if coverage < self.min_coverage:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="date_part_coverage_gap",
                        count=missing,
                        severity=Severity.HIGH if coverage < 0.8 else Severity.MEDIUM,
                        details=f"Coverage {coverage:.1%} < required {self.min_coverage:.1%}",
                        expected=f"{expected_parts} {self.date_part}s",
                        actual=f"{distinct_parts} {self.date_part}s ({missing} missing)",
                    )
                )

        return issues


@register_validator
class GroupedRecentDataValidator(Validator, DatetimeValidatorMixin):
    """Validates that each group has recent data.

    Example:
        # Each store should have data from the last 24 hours
        validator = GroupedRecentDataValidator(
            datetime_column="transaction_time",
            group_column="store_id",
            max_age_hours=24,
        )
    """

    name = "grouped_recent_data"
    category = "datetime"

    def __init__(
        self,
        datetime_column: str,
        group_column: str,
        max_age_hours: float = 24,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.datetime_column = datetime_column
        self.group_column = group_column
        self.max_age_hours = max_age_hours
        self.reference_time = reference_time or datetime.now()

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        cutoff = self.reference_time - timedelta(hours=self.max_age_hours)

        # Find groups with stale data
        group_max = (
            lf.group_by(self.group_column)
            .agg(pl.col(self.datetime_column).max().alias("_max_date"))
            .collect()
        )

        stale_groups = []
        for row in group_max.iter_rows(named=True):
            max_date = row["_max_date"]
            if max_date is None:
                stale_groups.append(row[self.group_column])
            elif isinstance(max_date, date) and not isinstance(max_date, datetime):
                if datetime.combine(max_date, datetime.min.time()) < cutoff:
                    stale_groups.append(row[self.group_column])
            elif max_date < cutoff:
                stale_groups.append(row[self.group_column])

        if stale_groups:
            issues.append(
                ValidationIssue(
                    column=f"{self.group_column}, {self.datetime_column}",
                    issue_type="grouped_stale_data",
                    count=len(stale_groups),
                    severity=Severity.HIGH,
                    details=f"{len(stale_groups)} groups have stale data (>{self.max_age_hours}h old)",
                    sample_values=stale_groups[: self.config.sample_size],
                )
            )

        return issues
