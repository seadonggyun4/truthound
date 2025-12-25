"""Progress tracking and callbacks for profiling operations.

This module provides a comprehensive progress tracking system with:
- Typed progress events
- Hierarchical progress (table -> column -> analyzer)
- Multiple callback support
- Progress aggregation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol, Sequence


# =============================================================================
# Progress Events
# =============================================================================


class ProgressStage(str, Enum):
    """Stages of profiling progress."""

    INITIALIZING = "initializing"
    LOADING = "loading"
    PROFILING_TABLE = "profiling_table"
    PROFILING_COLUMN = "profiling_column"
    ANALYZING = "analyzing"
    GENERATING_RULES = "generating_rules"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable event representing progress update.

    Attributes:
        stage: Current profiling stage
        progress: Progress ratio (0.0 to 1.0)
        message: Human-readable status message
        column: Current column being processed (if applicable)
        analyzer: Current analyzer running (if applicable)
        rows_processed: Number of rows processed so far
        total_rows: Total rows to process (if known)
        elapsed_seconds: Time elapsed since start
        estimated_remaining: Estimated seconds remaining (if calculable)
        metadata: Additional event-specific data
    """

    stage: ProgressStage
    progress: float  # 0.0 to 1.0
    message: str = ""
    column: str | None = None
    analyzer: str | None = None
    rows_processed: int = 0
    total_rows: int | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def percent(self) -> float:
        """Get progress as percentage."""
        return self.progress * 100

    @property
    def is_complete(self) -> bool:
        """Check if profiling is complete."""
        return self.stage in {ProgressStage.COMPLETED, ProgressStage.FAILED}

    def with_column(self, column: str, column_progress: float) -> "ProgressEvent":
        """Create new event for column progress."""
        return ProgressEvent(
            stage=ProgressStage.PROFILING_COLUMN,
            progress=self.progress,
            message=f"Profiling column: {column}",
            column=column,
            analyzer=self.analyzer,
            rows_processed=self.rows_processed,
            total_rows=self.total_rows,
            elapsed_seconds=self.elapsed_seconds,
            metadata={**self.metadata, "column_progress": column_progress},
        )


# Type alias for progress callback
ProgressCallback = Callable[[ProgressEvent], None]


# =============================================================================
# Progress Tracker
# =============================================================================


class ProgressTracker:
    """Tracks and reports profiling progress.

    This class manages progress state and notifies registered callbacks
    when progress updates occur.

    Example:
        tracker = ProgressTracker()

        # Register callbacks
        tracker.on_progress(lambda e: print(f"{e.percent:.1f}%"))
        tracker.on_column_start(lambda col: print(f"Starting: {col}"))

        # Use in profiler
        tracker.start(total_columns=10, total_rows=1000000)
        for col in columns:
            tracker.column_start(col)
            # ... profile column
            tracker.column_complete(col)
        tracker.complete()
    """

    def __init__(self) -> None:
        self._callbacks: list[ProgressCallback] = []
        self._column_callbacks: list[Callable[[str, float], None]] = []
        self._completion_callbacks: list[Callable[[float], None]] = []

        # State
        self._start_time: datetime | None = None
        self._total_columns: int = 0
        self._total_rows: int | None = None
        self._completed_columns: int = 0
        self._current_column: str | None = None
        self._current_column_progress: float = 0.0
        self._rows_processed: int = 0

    def on_progress(self, callback: ProgressCallback) -> "ProgressTracker":
        """Register a general progress callback.

        Args:
            callback: Function called on each progress update

        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)
        return self

    def on_column(
        self,
        callback: Callable[[str, float], None],
    ) -> "ProgressTracker":
        """Register a column progress callback.

        Args:
            callback: Function called with (column_name, progress_pct)

        Returns:
            Self for chaining
        """
        self._column_callbacks.append(callback)
        return self

    def on_complete(
        self,
        callback: Callable[[float], None],
    ) -> "ProgressTracker":
        """Register a completion callback.

        Args:
            callback: Function called with total_seconds on completion

        Returns:
            Self for chaining
        """
        self._completion_callbacks.append(callback)
        return self

    def start(
        self,
        *,
        total_columns: int,
        total_rows: int | None = None,
        message: str = "Starting profiling",
    ) -> None:
        """Signal profiling start.

        Args:
            total_columns: Total number of columns to profile
            total_rows: Total rows (if known)
            message: Status message
        """
        self._start_time = datetime.now()
        self._total_columns = total_columns
        self._total_rows = total_rows
        self._completed_columns = 0
        self._rows_processed = 0

        self._emit(ProgressEvent(
            stage=ProgressStage.INITIALIZING,
            progress=0.0,
            message=message,
            total_rows=total_rows,
        ))

    def column_start(self, column: str) -> None:
        """Signal start of column profiling.

        Args:
            column: Column name
        """
        self._current_column = column
        self._current_column_progress = 0.0

        progress = self._calculate_progress()

        self._emit(ProgressEvent(
            stage=ProgressStage.PROFILING_COLUMN,
            progress=progress,
            message=f"Profiling column: {column}",
            column=column,
            rows_processed=self._rows_processed,
            total_rows=self._total_rows,
            elapsed_seconds=self._elapsed_seconds(),
        ))

        # Emit column-specific callbacks
        for cb in self._column_callbacks:
            cb(column, 0.0)

    def column_progress(
        self,
        column: str,
        progress: float,
        *,
        analyzer: str | None = None,
        rows: int = 0,
    ) -> None:
        """Update progress within a column.

        Args:
            column: Column name
            progress: Column progress (0.0 to 1.0)
            analyzer: Current analyzer name
            rows: Rows processed in this update
        """
        self._current_column_progress = progress
        self._rows_processed += rows

        overall_progress = self._calculate_progress()

        self._emit(ProgressEvent(
            stage=ProgressStage.PROFILING_COLUMN,
            progress=overall_progress,
            message=f"Profiling {column}" + (f" ({analyzer})" if analyzer else ""),
            column=column,
            analyzer=analyzer,
            rows_processed=self._rows_processed,
            total_rows=self._total_rows,
            elapsed_seconds=self._elapsed_seconds(),
            estimated_remaining=self._estimate_remaining(overall_progress),
            metadata={"column_progress": progress},
        ))

        # Emit column-specific callbacks
        for cb in self._column_callbacks:
            cb(column, progress * 100)

    def column_complete(self, column: str) -> None:
        """Signal completion of column profiling.

        Args:
            column: Column name
        """
        self._completed_columns += 1
        self._current_column = None
        self._current_column_progress = 0.0

        progress = self._calculate_progress()

        self._emit(ProgressEvent(
            stage=ProgressStage.PROFILING_COLUMN,
            progress=progress,
            message=f"Completed column: {column}",
            column=column,
            rows_processed=self._rows_processed,
            total_rows=self._total_rows,
            elapsed_seconds=self._elapsed_seconds(),
            estimated_remaining=self._estimate_remaining(progress),
        ))

        # Emit column-specific callbacks with 100%
        for cb in self._column_callbacks:
            cb(column, 100.0)

    def complete(self, message: str = "Profiling complete") -> None:
        """Signal profiling completion."""
        elapsed = self._elapsed_seconds()

        self._emit(ProgressEvent(
            stage=ProgressStage.COMPLETED,
            progress=1.0,
            message=message,
            rows_processed=self._rows_processed,
            total_rows=self._total_rows,
            elapsed_seconds=elapsed,
        ))

        # Emit completion callbacks
        for cb in self._completion_callbacks:
            cb(elapsed)

    def fail(self, message: str, error: Exception | None = None) -> None:
        """Signal profiling failure.

        Args:
            message: Error message
            error: Optional exception
        """
        self._emit(ProgressEvent(
            stage=ProgressStage.FAILED,
            progress=self._calculate_progress(),
            message=message,
            rows_processed=self._rows_processed,
            total_rows=self._total_rows,
            elapsed_seconds=self._elapsed_seconds(),
            metadata={"error": str(error)} if error else {},
        ))

    def _emit(self, event: ProgressEvent) -> None:
        """Emit progress event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors stop profiling

    def _calculate_progress(self) -> float:
        """Calculate overall progress."""
        if self._total_columns == 0:
            return 0.0

        completed_progress = self._completed_columns / self._total_columns
        current_progress = self._current_column_progress / self._total_columns

        return min(1.0, completed_progress + current_progress)

    def _elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def _estimate_remaining(self, progress: float) -> float | None:
        """Estimate remaining time."""
        if progress <= 0:
            return None

        elapsed = self._elapsed_seconds()
        if elapsed <= 0:
            return None

        total_estimated = elapsed / progress
        return max(0, total_estimated - elapsed)


# =============================================================================
# Progress Aggregator
# =============================================================================


class ProgressAggregator:
    """Aggregates progress from multiple sources.

    Useful for parallel profiling where multiple columns are
    processed simultaneously.

    Example:
        aggregator = ProgressAggregator(total_items=10)
        aggregator.on_progress(lambda p, m: print(f"{p:.1f}%: {m}"))

        # From different threads
        aggregator.update("col1", 0.5)
        aggregator.update("col2", 0.3)
        aggregator.complete("col1")
    """

    def __init__(
        self,
        total_items: int,
        *,
        callback: Callable[[float, str], None] | None = None,
    ):
        """Initialize aggregator.

        Args:
            total_items: Total number of items to track
            callback: Optional callback(progress_pct, message)
        """
        self._total = total_items
        self._progress: dict[str, float] = {}
        self._completed: set[str] = set()
        self._callback = callback

    def on_progress(
        self,
        callback: Callable[[float, str], None],
    ) -> "ProgressAggregator":
        """Set progress callback.

        Args:
            callback: Function called with (progress_pct, message)
        """
        self._callback = callback
        return self

    def update(self, item: str, progress: float) -> None:
        """Update progress for an item.

        Args:
            item: Item identifier
            progress: Progress ratio (0.0 to 1.0)
        """
        self._progress[item] = min(1.0, max(0.0, progress))
        self._notify(f"Processing: {item}")

    def complete(self, item: str) -> None:
        """Mark an item as complete.

        Args:
            item: Item identifier
        """
        self._completed.add(item)
        self._progress[item] = 1.0
        self._notify(f"Completed: {item}")

    def get_progress(self) -> float:
        """Get overall progress ratio."""
        if self._total == 0:
            return 1.0

        total_progress = sum(self._progress.values())
        return total_progress / self._total

    def get_percent(self) -> float:
        """Get overall progress percentage."""
        return self.get_progress() * 100

    def _notify(self, message: str) -> None:
        """Notify callback of progress."""
        if self._callback:
            try:
                self._callback(self.get_percent(), message)
            except Exception:
                pass


# =============================================================================
# Console Progress Reporter
# =============================================================================


class ConsoleProgressReporter:
    """Reports progress to console with progress bar.

    Example:
        reporter = ConsoleProgressReporter(show_eta=True)
        tracker.on_progress(reporter)
    """

    def __init__(
        self,
        *,
        width: int = 40,
        show_eta: bool = True,
        show_column: bool = True,
    ):
        """Initialize reporter.

        Args:
            width: Progress bar width in characters
            show_eta: Whether to show estimated time remaining
            show_column: Whether to show current column name
        """
        self.width = width
        self.show_eta = show_eta
        self.show_column = show_column
        self._last_line_length = 0

    def __call__(self, event: ProgressEvent) -> None:
        """Handle progress event."""
        # Build progress bar
        filled = int(event.progress * self.width)
        bar = "█" * filled + "░" * (self.width - filled)

        # Build status line
        parts = [f"\r[{bar}] {event.percent:5.1f}%"]

        if self.show_column and event.column:
            parts.append(f" | {event.column}")

        if self.show_eta and event.estimated_remaining is not None:
            eta = self._format_time(event.estimated_remaining)
            parts.append(f" | ETA: {eta}")

        line = "".join(parts)

        # Pad to overwrite previous line
        if len(line) < self._last_line_length:
            line += " " * (self._last_line_length - len(line))
        self._last_line_length = len(line)

        # Print (with newline on completion)
        if event.is_complete:
            print(line)
        else:
            print(line, end="", flush=True)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# =============================================================================
# Convenience Functions
# =============================================================================


def create_progress_callback(
    on_column: Callable[[str, float], None] | None = None,
    on_complete: Callable[[float], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> ProgressTracker:
    """Create a progress tracker with callbacks.

    Args:
        on_column: Called with (column_name, progress_percent)
        on_complete: Called with total_seconds on completion
        on_progress: Called with ProgressEvent on any update

    Returns:
        Configured ProgressTracker

    Example:
        tracker = create_progress_callback(
            on_column=lambda col, pct: print(f"{col}: {pct:.1f}%"),
            on_complete=lambda secs: print(f"Done in {secs:.2f}s"),
        )
    """
    tracker = ProgressTracker()

    if on_column:
        tracker.on_column(on_column)
    if on_complete:
        tracker.on_complete(on_complete)
    if on_progress:
        tracker.on_progress(on_progress)

    return tracker
