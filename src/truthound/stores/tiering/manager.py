"""Tiering manager for automated tier migrations.

This module provides a manager that runs tier migrations on a schedule
in a background thread.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.stores.tiering.base import TieringConfig, TieringResult
from truthound.stores.tiering.store import TieredStore

logger = logging.getLogger(__name__)


class TieringManager:
    """Manager for automated tier migrations.

    This manager runs tier migrations on a configurable schedule
    in a background thread.

    Example:
        >>> from truthound.stores.tiering import TieredStore, TieringManager
        >>>
        >>> # Create tiered store
        >>> store = TieredStore(tiers, config)
        >>>
        >>> # Create and start manager
        >>> manager = TieringManager(store)
        >>> manager.start()
        >>>
        >>> # ... application runs ...
        >>>
        >>> # Stop manager on shutdown
        >>> manager.stop()
    """

    def __init__(
        self,
        store: TieredStore[Any],
        check_interval_hours: int | None = None,
        on_migration: Callable[[TieringResult], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            store: The tiered store to manage.
            check_interval_hours: Override interval (uses config if None).
            on_migration: Callback after migration completes.
            on_error: Callback on error.
        """
        self._store = store
        self._check_interval_hours = (
            check_interval_hours or store._config.check_interval_hours
        )
        self._on_migration = on_migration
        self._on_error = on_error

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_run: datetime | None = None
        self._run_count = 0
        self._error_count = 0
        self._total_items_migrated = 0
        self._total_bytes_migrated = 0

    def start(self) -> None:
        """Start the tiering manager.

        Runs migrations in a background thread based on schedule.
        """
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Tiering manager started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the tiering manager.

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

        logger.info("Tiering manager stopped")

    def run_now(self, dry_run: bool = False) -> TieringResult:
        """Run tiering immediately.

        Args:
            dry_run: If True, only report what would be migrated.

        Returns:
            Result of the tiering operation.
        """
        return self._run_tiering(dry_run=dry_run)

    def _run_loop(self) -> None:
        """Main manager loop."""
        while self._running and not self._stop_event.is_set():
            try:
                if self._should_run():
                    self._run_tiering()

                # Sleep in small intervals to respond to stop quickly
                for _ in range(60):  # Check every second for up to a minute
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                self._error_count += 1
                logger.error(f"Tiering manager error: {e}")
                if self._on_error:
                    try:
                        self._on_error(e)
                    except Exception:
                        pass

                # Back off on errors
                time.sleep(60)

    def _should_run(self) -> bool:
        """Check if tiering should run now."""
        if self._last_run is None:
            return True

        elapsed = datetime.now() - self._last_run
        interval = timedelta(hours=self._check_interval_hours)
        return elapsed >= interval

    def _run_tiering(self, dry_run: bool = False) -> TieringResult:
        """Execute tiering."""
        logger.info("Starting scheduled tier migration")

        try:
            result = self._store.run_tiering(dry_run=dry_run)
            self._last_run = datetime.now()
            self._run_count += 1

            if not dry_run:
                self._total_items_migrated += result.items_migrated
                self._total_bytes_migrated += result.bytes_migrated

            logger.info(
                f"Tier migration completed: "
                f"{result.items_migrated} items migrated, "
                f"{result.bytes_migrated} bytes"
            )

            if self._on_migration:
                try:
                    self._on_migration(result)
                except Exception as e:
                    logger.warning(f"Migration callback failed: {e}")

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Tier migration failed: {e}")

            if self._on_error:
                try:
                    self._on_error(e)
                except Exception:
                    pass

            # Return error result
            return TieringResult(
                start_time=datetime.now(),
                end_time=datetime.now(),
                errors=[str(e)],
            )

    def get_status(self) -> dict[str, Any]:
        """Get manager status.

        Returns:
            Dictionary with manager status.
        """
        return {
            "running": self._running,
            "check_interval_hours": self._check_interval_hours,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "total_items_migrated": self._total_items_migrated,
            "total_bytes_migrated": self._total_bytes_migrated,
            "next_run": self._get_next_run_time(),
        }

    def _get_next_run_time(self) -> str | None:
        """Calculate next scheduled run time."""
        if self._last_run is None:
            return datetime.now().isoformat()

        next_run = self._last_run + timedelta(hours=self._check_interval_hours)
        return next_run.isoformat()

    def get_tier_stats(self) -> dict[str, Any]:
        """Get current tier statistics.

        Returns:
            Dictionary with tier statistics.
        """
        return self._store.get_tier_stats()

    def estimate_migrations(self) -> TieringResult:
        """Estimate what migrations would occur.

        Returns:
            Dry-run result showing planned migrations.
        """
        return self._store.run_tiering(dry_run=True)

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def last_run(self) -> datetime | None:
        """Get last run time."""
        return self._last_run

    @property
    def run_count(self) -> int:
        """Get number of completed runs."""
        return self._run_count
