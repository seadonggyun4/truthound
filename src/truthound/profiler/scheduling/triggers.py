"""Trigger implementations for scheduled profiling.

This module provides various trigger types for determining when
profiling should run.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.profiler.scheduling.protocols import ProfileTrigger

logger = logging.getLogger(__name__)


class BaseTrigger(ProfileTrigger, ABC):
    """Abstract base class for triggers."""

    def __init__(self, name: str | None = None):
        """Initialize trigger.

        Args:
            name: Optional trigger name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get trigger name."""
        return self._name


@dataclass
class CronTrigger(BaseTrigger):
    """Cron-based trigger for scheduled profiling.

    Uses a simplified cron expression format:
    - minute hour day_of_month month day_of_week

    Example:
        # Daily at 2 AM
        trigger = CronTrigger("0 2 * * *")

        # Every Monday at 9 AM
        trigger = CronTrigger("0 9 * * 1")

        # Every hour
        trigger = CronTrigger("0 * * * *")
    """

    expression: str
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        """Validate cron expression."""
        self._validate_expression()
        super().__init__(f"cron({self.expression})")

    def _validate_expression(self) -> None:
        """Validate the cron expression format."""
        parts = self.expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: {self.expression}. "
                "Expected format: 'minute hour day month weekday'"
            )

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if current time matches cron expression."""
        if last_run is None:
            return True

        now = datetime.now()
        next_run = self._get_next_occurrence(last_run)
        return now >= next_run if next_run else False

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Get next scheduled run time."""
        base = last_run or datetime.now()
        return self._get_next_occurrence(base)

    def _get_next_occurrence(self, after: datetime) -> datetime | None:
        """Calculate next occurrence after given time."""
        parts = self.expression.split()
        minute, hour, day, month, weekday = parts

        # Simple implementation - just increment by interval
        # For full cron support, use croniter library
        next_time = after + timedelta(minutes=1)

        # Align to minute boundary
        next_time = next_time.replace(second=0, microsecond=0)

        if minute != "*":
            target_minute = int(minute)
            if next_time.minute > target_minute:
                next_time += timedelta(hours=1)
            next_time = next_time.replace(minute=target_minute)

        if hour != "*":
            target_hour = int(hour)
            if next_time.hour > target_hour:
                next_time += timedelta(days=1)
            next_time = next_time.replace(hour=target_hour)

        return next_time

    def _matches_cron(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        parts = self.expression.split()
        minute, hour, day, month, weekday = parts

        if minute != "*" and dt.minute != int(minute):
            return False
        if hour != "*" and dt.hour != int(hour):
            return False
        if day != "*" and dt.day != int(day):
            return False
        if month != "*" and dt.month != int(month):
            return False
        if weekday != "*" and dt.weekday() != int(weekday):
            return False

        return True


@dataclass
class IntervalTrigger(BaseTrigger):
    """Interval-based trigger for periodic profiling.

    Supports multiple ways to specify the interval:

    1. Using component parameters:
        trigger = IntervalTrigger(days=1, hours=6)

    2. Using interval_seconds (total seconds):
        trigger = IntervalTrigger(interval_seconds=3600)  # 1 hour

    3. Using timedelta directly:
        from datetime import timedelta
        trigger = IntervalTrigger(interval=timedelta(hours=6))

    Example:
        # Every 6 hours
        trigger = IntervalTrigger(hours=6)

        # Every 30 minutes
        trigger = IntervalTrigger(minutes=30)

        # Daily
        trigger = IntervalTrigger(days=1)

        # Using total seconds
        trigger = IntervalTrigger(interval_seconds=21600)  # 6 hours
    """

    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    interval_seconds: float | None = None  # Alternative: specify total seconds

    def __post_init__(self) -> None:
        """Initialize trigger."""
        super().__init__(f"interval({self.interval})")

    @property
    def interval(self) -> timedelta:
        """Get interval as timedelta.

        If interval_seconds is specified, it takes precedence.
        Otherwise, uses days/hours/minutes/seconds components.
        """
        if self.interval_seconds is not None:
            return timedelta(seconds=self.interval_seconds)

        return timedelta(
            days=self.days,
            hours=self.hours,
            minutes=self.minutes,
            seconds=self.seconds,
        )

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if interval has passed since last run."""
        if last_run is None:
            return True

        now = datetime.now()
        next_run = last_run + self.interval
        return now >= next_run

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Get next scheduled run time."""
        base = last_run or datetime.now()
        return base + self.interval


@dataclass
class DataChangeTrigger(BaseTrigger):
    """Trigger based on data changes.

    Profiles when data has changed beyond a threshold.

    Example:
        # Profile when 5% or more of data changes
        trigger = DataChangeTrigger(change_threshold=0.05)

        # Profile when row count changes significantly
        trigger = DataChangeTrigger(
            change_threshold=0.10,
            change_type="row_count",
        )
    """

    change_threshold: float = 0.05
    change_type: str = "content"  # 'content', 'row_count', 'schema'
    min_interval_seconds: float = 60.0  # Minimum time between runs

    def __post_init__(self) -> None:
        """Initialize trigger."""
        super().__init__(f"data_change({self.change_threshold})")

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if data has changed enough to warrant profiling."""
        # Check minimum interval
        if last_run is not None:
            elapsed = (datetime.now() - last_run).total_seconds()
            if elapsed < self.min_interval_seconds:
                return False

        # Check for change indicators in context
        if self.change_type == "content":
            current_hash = context.get("data_hash")
            last_hash = context.get("last_data_hash")
            if current_hash and last_hash:
                return current_hash != last_hash
            # If hashes not available, check row counts
            current_count = context.get("row_count")
            last_count = context.get("last_row_count")
            if current_count and last_count:
                return self._calculate_change(current_count, last_count) > self.change_threshold

        elif self.change_type == "row_count":
            current_count = context.get("row_count")
            last_count = context.get("last_row_count")
            if current_count and last_count:
                return self._calculate_change(current_count, last_count) > self.change_threshold

        elif self.change_type == "schema":
            current_schema = context.get("schema")
            last_schema = context.get("last_schema")
            if current_schema and last_schema:
                return current_schema != last_schema

        # Default to running if no change data available
        return last_run is None

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Cannot predict next run time for data-based trigger."""
        return None

    def _calculate_change(self, current: int, previous: int) -> float:
        """Calculate percentage change."""
        if previous == 0:
            return 1.0 if current > 0 else 0.0
        return abs(current - previous) / previous


@dataclass
class EventTrigger(BaseTrigger):
    """Trigger based on external events.

    Profiles when a specific event is signaled.

    Example:
        trigger = EventTrigger(event_name="data_updated")

        # Signal the event
        trigger.signal()

        # Or signal via context
        trigger.should_run(last_run, {"event_triggered": True})
    """

    event_name: str = "profile_requested"

    def __post_init__(self) -> None:
        """Initialize trigger."""
        super().__init__(f"event({self.event_name})")
        self._triggered = False

    def signal(self) -> None:
        """Signal that the event has occurred."""
        self._triggered = True

    def reset(self) -> None:
        """Reset the trigger."""
        self._triggered = False

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if event has been triggered."""
        # Check internal flag
        if self._triggered:
            self._triggered = False
            return True

        # Check context
        return context.get("event_triggered", False) or context.get(self.event_name, False)

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Cannot predict next run time for event-based trigger."""
        return None


@dataclass
class CompositeTrigger(BaseTrigger):
    """Composite trigger combining multiple triggers.

    Can use AND or OR logic to combine triggers.

    Example:
        # Run if cron time OR data changed
        trigger = CompositeTrigger(
            triggers=[CronTrigger("0 2 * * *"), DataChangeTrigger(0.05)],
            mode="any",  # OR logic
        )

        # Run if both time passed AND data changed
        trigger = CompositeTrigger(
            triggers=[IntervalTrigger(hours=1), DataChangeTrigger(0.05)],
            mode="all",  # AND logic
        )
    """

    triggers: list[ProfileTrigger] = field(default_factory=list)
    mode: str = "any"  # 'any' (OR) or 'all' (AND)

    def __post_init__(self) -> None:
        """Initialize trigger."""
        super().__init__(f"composite({self.mode})")

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if composite condition is met."""
        if not self.triggers:
            return False

        results = [t.should_run(last_run, context) for t in self.triggers]

        if self.mode == "any":
            return any(results)
        else:  # "all"
            return all(results)

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Get earliest next run time from all triggers."""
        next_times = [
            t.get_next_run_time(last_run)
            for t in self.triggers
            if t.get_next_run_time(last_run) is not None
        ]

        if not next_times:
            return None

        if self.mode == "any":
            return min(next_times)  # Earliest of any trigger
        else:
            return max(next_times)  # Latest (when all conditions met)


class AlwaysTrigger(BaseTrigger):
    """Trigger that always runs.

    Useful for testing or when you want explicit control.
    """

    def __init__(self) -> None:
        """Initialize trigger."""
        super().__init__("always")

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Always returns True."""
        return True

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Next run is always now."""
        return datetime.now()


class ManualTrigger(BaseTrigger):
    """Trigger that only runs when explicitly requested.

    Useful when you want full manual control.
    """

    def __init__(self) -> None:
        """Initialize trigger."""
        super().__init__("manual")
        self._should_run = False

    def trigger(self) -> None:
        """Request a run."""
        self._should_run = True

    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Only returns True when explicitly triggered."""
        if self._should_run:
            self._should_run = False
            return True
        return context.get("manual_trigger", False)

    def get_next_run_time(self, last_run: datetime | None) -> datetime | None:
        """Cannot predict next run time for manual trigger."""
        return None
