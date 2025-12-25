"""Checkpoint triggers for automated execution.

Triggers define when and how checkpoints should be executed,
supporting scheduled runs, event-based triggers, and CI/CD integration.

Example:
    >>> from truthound.checkpoint.triggers import (
    ...     ScheduleTrigger,
    ...     EventTrigger,
    ...     CronTrigger,
    ... )
    >>>
    >>> # Run every hour
    >>> schedule = ScheduleTrigger(interval_minutes=60)
    >>>
    >>> # Run on file changes
    >>> event = EventTrigger(
    ...     event_type="file_change",
    ...     patterns=["data/*.csv"],
    ... )
"""

from truthound.checkpoint.triggers.base import (
    BaseTrigger,
    TriggerConfig,
    TriggerResult,
    TriggerStatus,
)
from truthound.checkpoint.triggers.schedule import (
    ScheduleTrigger,
    ScheduleConfig,
    CronTrigger,
    CronConfig,
)
from truthound.checkpoint.triggers.event import (
    EventTrigger,
    EventConfig,
    FileWatchTrigger,
    FileWatchConfig,
)

__all__ = [
    # Base
    "BaseTrigger",
    "TriggerConfig",
    "TriggerResult",
    "TriggerStatus",
    # Schedule
    "ScheduleTrigger",
    "ScheduleConfig",
    "CronTrigger",
    "CronConfig",
    # Event
    "EventTrigger",
    "EventConfig",
    "FileWatchTrigger",
    "FileWatchConfig",
]
