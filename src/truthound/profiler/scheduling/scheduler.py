"""Incremental profiling scheduler implementation.

This module provides the main scheduler that coordinates triggers,
storage, and profiling execution.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import polars as pl

from truthound.profiler.scheduling.protocols import (
    ProfileTrigger,
    ProfileStorage,
    SchedulerProtocol,
)
from truthound.profiler.scheduling.triggers import IntervalTrigger, AlwaysTrigger
from truthound.profiler.scheduling.storage import InMemoryProfileStorage

if TYPE_CHECKING:
    from truthound.profiler.base import TableProfile
    from truthound.profiler.table_profiler import DataProfiler
    from truthound.profiler.incremental import IncrementalProfiler

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for the profiling scheduler.

    Attributes:
        enable_incremental: Use incremental profiling when possible.
        compute_data_hash: Compute hash for data change detection.
        save_history: Save profiles to storage.
        on_profile_complete: Callback after profiling completes.
        on_profile_skip: Callback when profiling is skipped.
        max_history_age_days: Maximum age of historical profiles.
        context_providers: Functions that provide context data.
    """

    enable_incremental: bool = True
    compute_data_hash: bool = True
    save_history: bool = True
    on_profile_complete: Callable[["TableProfile"], None] | None = None
    on_profile_skip: Callable[[str], None] | None = None
    max_history_age_days: int = 30
    context_providers: list[Callable[[], dict[str, Any]]] = field(default_factory=list)


@dataclass
class SchedulerMetrics:
    """Metrics for the profiling scheduler.

    Attributes:
        total_runs: Total number of profiling runs.
        incremental_runs: Number of incremental runs.
        full_runs: Number of full profiling runs.
        skipped_runs: Number of skipped runs.
        total_profile_time_ms: Total time spent profiling.
        last_run_time: Timestamp of last run.
        last_run_duration_ms: Duration of last run.
        errors: Number of errors encountered.
    """

    total_runs: int = 0
    incremental_runs: int = 0
    full_runs: int = 0
    skipped_runs: int = 0
    total_profile_time_ms: float = 0.0
    last_run_time: datetime | None = None
    last_run_duration_ms: float = 0.0
    errors: int = 0

    def record_run(
        self,
        duration_ms: float,
        incremental: bool = False,
        error: bool = False,
    ) -> None:
        """Record a profiling run."""
        self.total_runs += 1
        self.total_profile_time_ms += duration_ms
        self.last_run_time = datetime.now()
        self.last_run_duration_ms = duration_ms

        if incremental:
            self.incremental_runs += 1
        else:
            self.full_runs += 1

        if error:
            self.errors += 1

    def record_skip(self) -> None:
        """Record a skipped run."""
        self.skipped_runs += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "incremental_runs": self.incremental_runs,
            "full_runs": self.full_runs,
            "skipped_runs": self.skipped_runs,
            "total_profile_time_ms": self.total_profile_time_ms,
            "average_run_time_ms": (
                self.total_profile_time_ms / self.total_runs
                if self.total_runs > 0
                else 0
            ),
            "last_run_time": (
                self.last_run_time.isoformat() if self.last_run_time else None
            ),
            "last_run_duration_ms": self.last_run_duration_ms,
            "errors": self.errors,
        }


class IncrementalProfileScheduler(SchedulerProtocol):
    """Scheduler for automated incremental profiling.

    Coordinates triggers, storage, and profiling to provide
    automated, scheduled profiling with incremental updates.

    Example:
        from truthound.profiler.scheduling import (
            IncrementalProfileScheduler,
            CronTrigger,
            FileProfileStorage,
        )

        scheduler = IncrementalProfileScheduler(
            trigger=CronTrigger("0 2 * * *"),  # Daily at 2 AM
            storage=FileProfileStorage("./profiles"),
        )

        # Run if scheduled
        result = scheduler.run_if_needed(data)

        # Or check when next run is scheduled
        next_run = scheduler.get_next_run_time()
    """

    def __init__(
        self,
        trigger: ProfileTrigger | None = None,
        storage: ProfileStorage | None = None,
        config: SchedulerConfig | None = None,
    ):
        """Initialize the scheduler.

        Args:
            trigger: Trigger that determines when to run.
            storage: Storage backend for profile history.
            config: Configuration options.
        """
        self._trigger = trigger or IntervalTrigger(hours=1)
        self._storage = storage or InMemoryProfileStorage()
        self._config = config or SchedulerConfig()
        self._metrics = SchedulerMetrics()
        self._run_history: list[dict[str, Any]] = []

        # Profiler instances (lazy initialized)
        self._profiler: "DataProfiler | None" = None
        self._incremental_profiler: "IncrementalProfiler | None" = None

    @property
    def trigger(self) -> ProfileTrigger:
        """Get the trigger."""
        return self._trigger

    @property
    def storage(self) -> ProfileStorage:
        """Get the storage backend."""
        return self._storage

    @property
    def metrics(self) -> SchedulerMetrics:
        """Get scheduler metrics."""
        return self._metrics

    def run_if_needed(
        self,
        data: pl.LazyFrame,
        context: dict[str, Any] | None = None,
    ) -> "TableProfile | None":
        """Run profiling if the trigger conditions are met.

        Args:
            data: Data to profile.
            context: Additional context for trigger evaluation.

        Returns:
            Profile result if run, None if skipped.
        """
        # Build context
        ctx = self._build_context(data, context)

        # Get last run time
        last_run = self._storage.get_last_run_time()

        # Check trigger
        if not self._trigger.should_run(last_run, ctx):
            self._metrics.record_skip()

            if self._config.on_profile_skip:
                self._config.on_profile_skip(
                    f"Trigger '{self._trigger.name}' not satisfied"
                )

            logger.debug(
                f"Profiling skipped - trigger '{self._trigger.name}' not satisfied"
            )
            return None

        # Run profiling
        use_incremental = self._config.enable_incremental and last_run is not None
        return self.run(data, incremental=use_incremental)

    def run(
        self,
        data: pl.LazyFrame,
        incremental: bool = True,
    ) -> "TableProfile":
        """Force run profiling.

        Args:
            data: Data to profile.
            incremental: Whether to use incremental profiling.

        Returns:
            Profile result.
        """
        start_time = time.perf_counter()
        error_occurred = False
        profile: "TableProfile | None" = None

        try:
            if incremental:
                last_profile = self._storage.get_last_profile()
                if last_profile:
                    profile = self._run_incremental(data, last_profile)
                else:
                    profile = self._run_full(data)
            else:
                profile = self._run_full(data)

        except Exception as e:
            error_occurred = True
            logger.error(f"Profiling failed: {e}")
            raise

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.record_run(
                duration_ms=duration_ms,
                incremental=incremental and not error_occurred,
                error=error_occurred,
            )

        # Save to storage
        if profile and self._config.save_history:
            metadata = {
                "incremental": incremental,
                "duration_ms": duration_ms,
            }
            profile_id = self._storage.save(profile, metadata)

            # Record in history
            self._run_history.append({
                "profile_id": profile_id,
                "timestamp": datetime.now().isoformat(),
                "incremental": incremental,
                "duration_ms": duration_ms,
            })

        # Callback
        if profile and self._config.on_profile_complete:
            self._config.on_profile_complete(profile)

        return profile

    def get_next_run_time(self) -> datetime | None:
        """Get the next scheduled run time."""
        last_run = self._storage.get_last_run_time()
        return self._trigger.get_next_run_time(last_run)

    def get_run_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get history of profiling runs."""
        history = self._run_history.copy()
        history.reverse()  # Most recent first

        if limit:
            history = history[:limit]

        return history

    def _build_context(
        self,
        data: pl.LazyFrame,
        user_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build context for trigger evaluation."""
        ctx: dict[str, Any] = {}

        # Add user context
        if user_context:
            ctx.update(user_context)

        # Add data info
        try:
            df = data.collect()
            ctx["row_count"] = len(df)
            ctx["column_count"] = len(df.columns)

            if self._config.compute_data_hash:
                ctx["data_hash"] = self._compute_data_hash(df)
        except Exception as e:
            logger.warning(f"Failed to compute data context: {e}")

        # Get last profile info
        last_profile = self._storage.get_last_profile()
        if last_profile:
            ctx["last_row_count"] = getattr(last_profile, "row_count", None)
            ctx["last_schema"] = getattr(last_profile, "schema", None)

            # Get hash from last profile if available
            history = self._storage.list_profiles(limit=1)
            if history:
                profile_entry = next(
                    (e for e in self._run_history if e.get("profile_id") == history[0][0]),
                    None,
                )
                if profile_entry:
                    ctx["last_data_hash"] = profile_entry.get("data_hash")

        # Add from context providers
        for provider in self._config.context_providers:
            try:
                ctx.update(provider())
            except Exception as e:
                logger.warning(f"Context provider failed: {e}")

        return ctx

    def _compute_data_hash(self, df: pl.DataFrame) -> str:
        """Compute a hash of the data for change detection."""
        # Hash based on shape and sample of data
        hasher = hashlib.sha256()

        # Include shape
        hasher.update(f"{len(df)}x{len(df.columns)}".encode())

        # Include column names
        hasher.update(",".join(df.columns).encode())

        # Include sample of data (first and last rows)
        if len(df) > 0:
            sample_size = min(100, len(df))
            sample = df.head(sample_size // 2).vstack(df.tail(sample_size // 2))
            hasher.update(str(sample).encode())

        return hasher.hexdigest()[:16]

    def _run_full(self, data: pl.LazyFrame) -> "TableProfile":
        """Run full profiling."""
        if self._profiler is None:
            from truthound.profiler.table_profiler import DataProfiler
            self._profiler = DataProfiler()

        return self._profiler.profile(data)

    def _run_incremental(
        self,
        data: pl.LazyFrame,
        baseline: "TableProfile",
    ) -> "TableProfile":
        """Run incremental profiling."""
        if self._incremental_profiler is None:
            from truthound.profiler.incremental import IncrementalProfiler
            self._incremental_profiler = IncrementalProfiler()

        return self._incremental_profiler.profile(data, baseline)


def create_scheduler(
    trigger_type: str = "interval",
    storage_type: str = "memory",
    **kwargs: Any,
) -> IncrementalProfileScheduler:
    """Factory function for creating schedulers.

    Args:
        trigger_type: Type of trigger ('interval', 'cron', 'manual', 'always').
        storage_type: Type of storage ('memory', 'file').
        **kwargs: Additional configuration.

    Returns:
        Configured scheduler instance.

    Example:
        # Hourly profiling with file storage
        scheduler = create_scheduler(
            trigger_type="interval",
            storage_type="file",
            hours=1,
            storage_path="./profiles",
        )
    """
    from truthound.profiler.scheduling.triggers import (
        CronTrigger,
        IntervalTrigger,
        ManualTrigger,
        AlwaysTrigger,
    )
    from truthound.profiler.scheduling.storage import (
        InMemoryProfileStorage,
        FileProfileStorage,
    )

    # Create trigger
    if trigger_type == "interval":
        trigger = IntervalTrigger(
            days=kwargs.get("days", 0),
            hours=kwargs.get("hours", 1),
            minutes=kwargs.get("minutes", 0),
        )
    elif trigger_type == "cron":
        trigger = CronTrigger(
            expression=kwargs.get("expression", "0 * * * *"),
        )
    elif trigger_type == "manual":
        trigger = ManualTrigger()
    elif trigger_type == "always":
        trigger = AlwaysTrigger()
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    # Create storage
    if storage_type == "memory":
        storage = InMemoryProfileStorage(
            max_profiles=kwargs.get("max_profiles", 100),
        )
    elif storage_type == "file":
        storage = FileProfileStorage(
            base_path=kwargs.get("storage_path", "./profiles"),
            max_profiles=kwargs.get("max_profiles", 100),
            compress=kwargs.get("compress", False),
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

    # Create config
    config = SchedulerConfig(
        enable_incremental=kwargs.get("enable_incremental", True),
        compute_data_hash=kwargs.get("compute_data_hash", True),
        save_history=kwargs.get("save_history", True),
    )

    return IncrementalProfileScheduler(
        trigger=trigger,
        storage=storage,
        config=config,
    )
