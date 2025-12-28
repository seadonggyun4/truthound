"""Retention scheduler for automatic cleanup.

This module provides a scheduler that can run retention cleanup
on a configurable schedule.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.stores.retention.base import RetentionResult, RetentionSchedule
from truthound.stores.retention.store import RetentionStore

logger = logging.getLogger(__name__)


class RetentionScheduler:
    """Scheduler for automatic retention cleanup.

    This scheduler runs retention cleanup on a configurable schedule
    in a background thread.

    Example:
        >>> from truthound.stores.retention import (
        ...     RetentionStore,
        ...     RetentionScheduler,
        ... )
        >>>
        >>> # Create retention store
        >>> store = RetentionStore(base_store, config)
        >>>
        >>> # Create and start scheduler
        >>> scheduler = RetentionScheduler(store)
        >>> scheduler.start()
        >>>
        >>> # ... application runs ...
        >>>
        >>> # Stop scheduler on shutdown
        >>> scheduler.stop()
    """

    def __init__(
        self,
        store: RetentionStore[Any],
        schedule: RetentionSchedule | None = None,
        on_cleanup: Callable[[RetentionResult], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            store: The retention store to clean up.
            schedule: Override schedule (uses store config if None).
            on_cleanup: Callback after successful cleanup.
            on_error: Callback on cleanup error.
        """
        self._store = store
        self._schedule = schedule or store.retention_config.schedule
        self._on_cleanup = on_cleanup
        self._on_error = on_error

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_run: datetime | None = None
        self._run_count = 0
        self._error_count = 0

    def start(self) -> None:
        """Start the scheduler.

        Runs cleanup in a background thread based on schedule.
        """
        if self._running:
            return

        if not self._schedule.enabled:
            logger.info("Retention scheduler disabled in config")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Retention scheduler started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the scheduler.

        Args:
            timeout: Maximum seconds to wait for thread to stop.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        logger.info("Retention scheduler stopped")

    def run_now(self) -> RetentionResult:
        """Run cleanup immediately.

        Returns:
            Result of the cleanup.
        """
        return self._run_cleanup()

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            try:
                if self._should_run():
                    self._run_cleanup()

                # Sleep in small intervals to respond to stop quickly
                for _ in range(60):  # Check every second for up to a minute
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self._error_count += 1
                logger.error(f"Scheduler error: {e}")
                if self._on_error:
                    try:
                        self._on_error(e)
                    except Exception:
                        pass

                # Back off on errors
                time.sleep(60)

    def _should_run(self) -> bool:
        """Check if cleanup should run now."""
        now = datetime.now()

        # Check specific hour if configured
        if self._schedule.run_at_hour is not None:
            if now.hour != self._schedule.run_at_hour:
                return False

        # Check specific days if configured
        if self._schedule.run_on_days is not None:
            if now.weekday() not in self._schedule.run_on_days:
                return False

        # Check interval
        if self._last_run is not None:
            elapsed = now - self._last_run
            interval = timedelta(hours=self._schedule.interval_hours)
            if elapsed < interval:
                return False

        return True

    def _run_cleanup(self) -> RetentionResult:
        """Execute cleanup."""
        logger.info("Starting scheduled retention cleanup")
        start_time = datetime.now()

        try:
            result = self._store.run_cleanup()
            self._last_run = datetime.now()
            self._run_count += 1

            logger.info(
                f"Retention cleanup completed: "
                f"{result.items_deleted} deleted, "
                f"{result.items_preserved} preserved, "
                f"{result.bytes_freed} bytes freed"
            )

            if self._on_cleanup:
                try:
                    self._on_cleanup(result)
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Retention cleanup failed: {e}")

            if self._on_error:
                try:
                    self._on_error(e)
                except Exception:
                    pass

            # Return error result
            return RetentionResult(
                start_time=start_time,
                end_time=datetime.now(),
                errors=[str(e)],
            )

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler status.
        """
        return {
            "running": self._running,
            "enabled": self._schedule.enabled,
            "interval_hours": self._schedule.interval_hours,
            "run_at_hour": self._schedule.run_at_hour,
            "run_on_days": self._schedule.run_on_days,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "next_run": self._get_next_run_time(),
        }

    def _get_next_run_time(self) -> str | None:
        """Calculate next scheduled run time."""
        if not self._schedule.enabled:
            return None

        now = datetime.now()

        if self._last_run is None:
            # First run - check if we should run now based on hour/day
            if self._should_run():
                return now.isoformat()

        # Calculate next run based on interval
        if self._last_run:
            next_run = self._last_run + timedelta(hours=self._schedule.interval_hours)
        else:
            next_run = now

        # Adjust for specific hour if configured
        if self._schedule.run_at_hour is not None:
            next_run = next_run.replace(
                hour=self._schedule.run_at_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            if next_run <= now:
                next_run += timedelta(days=1)

        # Adjust for specific days if configured
        if self._schedule.run_on_days is not None:
            while next_run.weekday() not in self._schedule.run_on_days:
                next_run += timedelta(days=1)

        return next_run.isoformat()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def last_run(self) -> datetime | None:
        """Get last run time."""
        return self._last_run

    @property
    def run_count(self) -> int:
        """Get number of completed runs."""
        return self._run_count

    @property
    def error_count(self) -> int:
        """Get number of errors."""
        return self._error_count
