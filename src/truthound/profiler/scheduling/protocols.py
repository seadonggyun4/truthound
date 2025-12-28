"""Protocol definitions for scheduling components.

This module defines the interfaces for triggers, storage, and schedulers.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    import polars as pl
    from truthound.profiler.base import TableProfile


@runtime_checkable
class ProfileTrigger(Protocol):
    """Protocol for profiling triggers.

    Triggers determine when profiling should run based on various conditions
    such as time, data changes, or external events.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get trigger name."""
        ...

    @abstractmethod
    def should_run(
        self,
        last_run: datetime | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if profiling should run.

        Args:
            last_run: Timestamp of last profiling run.
            context: Additional context data.

        Returns:
            True if profiling should run now.
        """
        ...

    @abstractmethod
    def get_next_run_time(
        self,
        last_run: datetime | None,
    ) -> datetime | None:
        """Get the next scheduled run time.

        Args:
            last_run: Timestamp of last run.

        Returns:
            Next run timestamp, or None if not determinable.
        """
        ...


@runtime_checkable
class ProfileStorage(Protocol):
    """Protocol for profile history storage.

    Storage backends manage profile history, enabling incremental
    profiling and trend analysis.
    """

    @abstractmethod
    def get_last_profile(self) -> "TableProfile | None":
        """Get the most recent profile.

        Returns:
            Last profile, or None if no history.
        """
        ...

    @abstractmethod
    def get_last_run_time(self) -> datetime | None:
        """Get timestamp of last profiling run.

        Returns:
            Last run timestamp, or None if never run.
        """
        ...

    @abstractmethod
    def save(self, profile: "TableProfile", metadata: dict[str, Any] | None = None) -> str:
        """Save a profile to storage.

        Args:
            profile: Profile to save.
            metadata: Additional metadata.

        Returns:
            Unique identifier for the saved profile.
        """
        ...

    @abstractmethod
    def get_profile(self, profile_id: str) -> "TableProfile | None":
        """Get a specific profile by ID.

        Args:
            profile_id: Profile identifier.

        Returns:
            The profile, or None if not found.
        """
        ...

    @abstractmethod
    def list_profiles(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, datetime]]:
        """List available profiles.

        Args:
            limit: Maximum number of profiles to return.
            since: Only include profiles after this time.

        Returns:
            List of (profile_id, timestamp) tuples.
        """
        ...

    @abstractmethod
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile.

        Args:
            profile_id: Profile identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def get_baseline_schema(self) -> Any:
        """Get the baseline schema from the first profile.

        Returns:
            Schema definition, or None if no profiles exist.
        """
        ...


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for profile schedulers."""

    @abstractmethod
    def run_if_needed(
        self,
        data: "pl.LazyFrame",
        context: dict[str, Any] | None = None,
    ) -> "TableProfile | None":
        """Run profiling if the trigger conditions are met.

        Args:
            data: Data to profile.
            context: Additional context for trigger evaluation.

        Returns:
            Profile result if run, None if skipped.
        """
        ...

    @abstractmethod
    def run(
        self,
        data: "pl.LazyFrame",
        incremental: bool = True,
    ) -> "TableProfile":
        """Force run profiling.

        Args:
            data: Data to profile.
            incremental: Whether to use incremental profiling.

        Returns:
            Profile result.
        """
        ...

    @abstractmethod
    def get_next_run_time(self) -> datetime | None:
        """Get the next scheduled run time.

        Returns:
            Next run timestamp, or None if not scheduled.
        """
        ...

    @abstractmethod
    def get_run_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get history of profiling runs.

        Args:
            limit: Maximum number of entries.

        Returns:
            List of run history entries.
        """
        ...
