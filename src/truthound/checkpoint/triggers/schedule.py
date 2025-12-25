"""Schedule-based triggers.

These triggers execute checkpoints based on time intervals or cron expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.triggers.base import (
    BaseTrigger,
    TriggerConfig,
    TriggerResult,
)

if TYPE_CHECKING:
    pass


@dataclass
class ScheduleConfig(TriggerConfig):
    """Configuration for schedule trigger.

    Attributes:
        interval_seconds: Run interval in seconds.
        interval_minutes: Run interval in minutes (convenience).
        interval_hours: Run interval in hours (convenience).
        start_time: First run time (None = start immediately).
        end_time: Stop running after this time (None = run forever).
        run_on_weekdays: Only run on specific weekdays (0=Mon, 6=Sun).
        timezone: Timezone for scheduling (default: local).
    """

    interval_seconds: int = 0
    interval_minutes: int = 0
    interval_hours: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    run_on_weekdays: list[int] | None = None
    timezone: str | None = None

    def get_interval(self) -> timedelta:
        """Get total interval as timedelta."""
        return timedelta(
            seconds=self.interval_seconds,
            minutes=self.interval_minutes,
            hours=self.interval_hours,
        )


class ScheduleTrigger(BaseTrigger[ScheduleConfig]):
    """Trigger based on time intervals.

    Executes checkpoints at regular intervals, optionally restricted
    to specific times or weekdays.

    Example:
        >>> # Run every hour
        >>> trigger = ScheduleTrigger(interval_hours=1)
        >>>
        >>> # Run every 30 minutes during business hours
        >>> trigger = ScheduleTrigger(
        ...     interval_minutes=30,
        ...     run_on_weekdays=[0, 1, 2, 3, 4],  # Mon-Fri
        ... )
    """

    trigger_type = "schedule"

    def __init__(self, config: ScheduleConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._next_run: datetime | None = None

    @classmethod
    def _default_config(cls) -> ScheduleConfig:
        return ScheduleConfig()

    def should_trigger(self) -> TriggerResult:
        """Check if scheduled time has arrived."""
        now = datetime.now()

        # Check if within valid time range
        if self._config.start_time and now < self._config.start_time:
            return TriggerResult(
                should_run=False,
                reason=f"Before start time: {self._config.start_time}",
                next_run=self._config.start_time,
            )

        if self._config.end_time and now > self._config.end_time:
            return TriggerResult(
                should_run=False,
                reason=f"After end time: {self._config.end_time}",
            )

        # Check weekday restriction
        if self._config.run_on_weekdays is not None:
            if now.weekday() not in self._config.run_on_weekdays:
                # Find next valid weekday
                next_day = now + timedelta(days=1)
                while next_day.weekday() not in self._config.run_on_weekdays:
                    next_day += timedelta(days=1)
                return TriggerResult(
                    should_run=False,
                    reason=f"Not a valid weekday. Current: {now.weekday()}",
                    next_run=next_day.replace(hour=0, minute=0, second=0),
                )

        # Check interval
        interval = self._config.get_interval()
        if interval.total_seconds() == 0:
            return TriggerResult(
                should_run=False,
                reason="No interval configured",
            )

        # First run or check interval
        if self._last_run is None:
            self._next_run = now
            return TriggerResult(
                should_run=True,
                reason="First scheduled run",
                next_run=now + interval,
            )

        time_since_last = now - self._last_run
        if time_since_last >= interval:
            self._next_run = now + interval
            return TriggerResult(
                should_run=True,
                reason=f"Interval elapsed: {time_since_last}",
                next_run=self._next_run,
            )

        self._next_run = self._last_run + interval
        return TriggerResult(
            should_run=False,
            reason=f"Waiting for interval. Remaining: {interval - time_since_last}",
            next_run=self._next_run,
        )

    def get_next_run(self) -> datetime | None:
        """Get the next scheduled run time."""
        if self._next_run:
            return self._next_run

        if self._last_run:
            return self._last_run + self._config.get_interval()

        return self._config.start_time or datetime.now()

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        interval = self._config.get_interval()
        if interval.total_seconds() == 0:
            errors.append("At least one interval (seconds, minutes, hours) must be > 0")

        if self._config.run_on_weekdays:
            for day in self._config.run_on_weekdays:
                if day < 0 or day > 6:
                    errors.append(f"Invalid weekday: {day}. Must be 0-6 (Mon-Sun)")

        return errors


@dataclass
class CronConfig(TriggerConfig):
    """Configuration for cron trigger.

    Attributes:
        expression: Cron expression (5 or 6 fields).
        timezone: Timezone for cron evaluation.
    """

    expression: str = ""
    timezone: str | None = None


class CronTrigger(BaseTrigger[CronConfig]):
    """Trigger based on cron expressions.

    Executes checkpoints according to standard cron schedules.
    Supports 5-field (minute to day-of-week) and 6-field (with seconds)
    cron expressions.

    Example:
        >>> # Every day at midnight
        >>> trigger = CronTrigger(expression="0 0 * * *")
        >>>
        >>> # Every Monday at 9am
        >>> trigger = CronTrigger(expression="0 9 * * 1")
        >>>
        >>> # Every 15 minutes
        >>> trigger = CronTrigger(expression="*/15 * * * *")
    """

    trigger_type = "cron"

    def __init__(self, config: CronConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._next_run: datetime | None = None
        self._parsed_cron: dict[str, list[int]] | None = None

    @classmethod
    def _default_config(cls) -> CronConfig:
        return CronConfig()

    def _parse_cron(self) -> dict[str, list[int]]:
        """Parse cron expression into usable format."""
        if self._parsed_cron is not None:
            return self._parsed_cron

        expression = self._config.expression.strip()
        parts = expression.split()

        if len(parts) == 5:
            minute, hour, day, month, dow = parts
            second = "0"
        elif len(parts) == 6:
            second, minute, hour, day, month, dow = parts
        else:
            raise ValueError(f"Invalid cron expression: {expression}")

        def parse_field(field: str, min_val: int, max_val: int) -> list[int]:
            """Parse a single cron field."""
            values: set[int] = set()

            for part in field.split(","):
                if part == "*":
                    values.update(range(min_val, max_val + 1))
                elif "/" in part:
                    range_part, step = part.split("/")
                    step = int(step)
                    if range_part == "*":
                        start, end = min_val, max_val
                    elif "-" in range_part:
                        start, end = map(int, range_part.split("-"))
                    else:
                        start = int(range_part)
                        end = max_val
                    values.update(range(start, end + 1, step))
                elif "-" in part:
                    start, end = map(int, part.split("-"))
                    values.update(range(start, end + 1))
                else:
                    values.add(int(part))

            return sorted(values)

        self._parsed_cron = {
            "second": parse_field(second, 0, 59),
            "minute": parse_field(minute, 0, 59),
            "hour": parse_field(hour, 0, 23),
            "day": parse_field(day, 1, 31),
            "month": parse_field(month, 1, 12),
            "dow": parse_field(dow, 0, 6),
        }

        return self._parsed_cron

    def _matches_cron(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        cron = self._parse_cron()

        return (
            dt.second in cron["second"]
            and dt.minute in cron["minute"]
            and dt.hour in cron["hour"]
            and dt.day in cron["day"]
            and dt.month in cron["month"]
            and dt.weekday() in cron["dow"]
        )

    def _find_next_run(self, after: datetime) -> datetime:
        """Find the next run time after the given datetime."""
        # Start from next minute (or second if 6-field cron)
        cron = self._parse_cron()
        current = after.replace(microsecond=0)

        # Simple approach: iterate forward until we find a match
        # This could be optimized for production use
        for _ in range(366 * 24 * 60 * 60):  # Max 1 year of seconds
            current += timedelta(seconds=1)
            if self._matches_cron(current):
                return current

        # Fallback (should not reach here with valid cron)
        return current

    def should_trigger(self) -> TriggerResult:
        """Check if cron schedule matches current time."""
        if not self._config.expression:
            return TriggerResult(
                should_run=False,
                reason="No cron expression configured",
            )

        try:
            cron = self._parse_cron()
        except ValueError as e:
            return TriggerResult(
                should_run=False,
                reason=f"Invalid cron expression: {e}",
            )

        now = datetime.now()

        # Check if now matches cron
        if self._matches_cron(now):
            # Don't run if we just ran this minute/second
            if self._last_run:
                time_since = now - self._last_run
                if time_since.total_seconds() < 60:
                    next_run = self._find_next_run(now)
                    return TriggerResult(
                        should_run=False,
                        reason="Already ran this period",
                        next_run=next_run,
                    )

            next_run = self._find_next_run(now)
            return TriggerResult(
                should_run=True,
                reason="Cron schedule matched",
                next_run=next_run,
            )

        next_run = self._find_next_run(now)
        return TriggerResult(
            should_run=False,
            reason="Waiting for cron schedule",
            next_run=next_run,
        )

    def get_next_run(self) -> datetime | None:
        """Get the next scheduled run time."""
        return self._find_next_run(datetime.now())

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if not self._config.expression:
            errors.append("Cron expression is required")
        else:
            try:
                self._parse_cron()
            except ValueError as e:
                errors.append(str(e))

        return errors
