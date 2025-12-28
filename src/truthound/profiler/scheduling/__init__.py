"""Scheduling module for incremental profiling.

This module provides scheduling capabilities for automated and
incremental profiling runs, with various trigger types and storage.

Key Features:
- Multiple trigger types (cron, interval, data change, event)
- Profile history storage and management
- Automatic incremental profiling
- Integration with checkpoint system

Example:
    from truthound.profiler.scheduling import (
        IncrementalProfileScheduler,
        CronTrigger,
        ProfileHistoryStorage,
    )

    # Create scheduler with cron trigger
    scheduler = IncrementalProfileScheduler(
        trigger=CronTrigger("0 2 * * *"),  # Daily at 2 AM
        storage=ProfileHistoryStorage("./profiles"),
    )

    # Run if scheduled
    result = scheduler.run_if_needed(data)

    # Or force run
    result = scheduler.run(data)
"""

from truthound.profiler.scheduling.protocols import (
    ProfileTrigger,
    ProfileStorage,
    SchedulerProtocol,
)
from truthound.profiler.scheduling.triggers import (
    CronTrigger,
    IntervalTrigger,
    DataChangeTrigger,
    EventTrigger,
    CompositeTrigger,
    AlwaysTrigger,
    ManualTrigger,
)
from truthound.profiler.scheduling.storage import (
    ProfileHistoryEntry,
    ProfileHistoryStorage,
    InMemoryProfileStorage,
    FileProfileStorage,
)
from truthound.profiler.scheduling.scheduler import (
    SchedulerConfig,
    SchedulerMetrics,
    IncrementalProfileScheduler,
    create_scheduler,
)

__all__ = [
    # Protocols
    "ProfileTrigger",
    "ProfileStorage",
    "SchedulerProtocol",
    # Triggers
    "CronTrigger",
    "IntervalTrigger",
    "DataChangeTrigger",
    "EventTrigger",
    "CompositeTrigger",
    "AlwaysTrigger",
    "ManualTrigger",
    # Storage
    "ProfileHistoryEntry",
    "ProfileHistoryStorage",
    "InMemoryProfileStorage",
    "FileProfileStorage",
    # Scheduler
    "SchedulerConfig",
    "SchedulerMetrics",
    "IncrementalProfileScheduler",
    "create_scheduler",
]
