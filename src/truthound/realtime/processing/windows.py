"""Window processing for streaming data.

Provides window-based aggregation and processing:
- Tumbling windows (fixed-size, non-overlapping)
- Sliding windows (fixed-size, overlapping)
- Session windows (gap-based)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Generic, TypeVar
import asyncio
import logging
from collections import defaultdict

from truthound.realtime.protocols import StreamMessage


logger = logging.getLogger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
K = TypeVar("K")  # Key type


class WindowType(str, Enum):
    """Types of windows for stream processing."""

    TUMBLING = "tumbling"  # Fixed-size, non-overlapping
    SLIDING = "sliding"  # Fixed-size, overlapping
    SESSION = "session"  # Gap-based, key-specific
    GLOBAL = "global"  # Single global window


@dataclass
class WindowConfig:
    """Configuration for window processing.

    Attributes:
        window_type: Type of window
        size_ms: Window size in milliseconds
        slide_ms: Slide interval for sliding windows
        gap_ms: Gap timeout for session windows
        allowed_lateness_ms: Allowed lateness for late data
        watermark_delay_ms: Watermark delay
    """

    window_type: WindowType = WindowType.TUMBLING
    size_ms: int = 60000  # 1 minute
    slide_ms: int | None = None  # For sliding windows
    gap_ms: int | None = None  # For session windows
    allowed_lateness_ms: int = 0
    watermark_delay_ms: int = 0


@dataclass
class WindowResult(Generic[R]):
    """Result of window aggregation.

    Attributes:
        window_id: Unique window identifier
        window_start: Window start time
        window_end: Window end time
        key: Window key (for keyed windows)
        result: Aggregated result
        message_count: Number of messages in window
        first_timestamp: Timestamp of first message
        last_timestamp: Timestamp of last message
        is_final: Whether this is the final result
    """

    window_id: str
    window_start: datetime
    window_end: datetime
    key: str | None
    result: R
    message_count: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    is_final: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_id": self.window_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "key": self.key,
            "result": self.result,
            "message_count": self.message_count,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "is_final": self.is_final,
            "metadata": self.metadata,
        }


# =============================================================================
# Window Aggregators
# =============================================================================


class WindowAggregator(ABC, Generic[T, R]):
    """Base class for window aggregators."""

    @abstractmethod
    def add(self, value: T) -> None:
        """Add a value to the aggregation."""
        ...

    @abstractmethod
    def get_result(self) -> R:
        """Get the aggregation result."""
        ...

    @abstractmethod
    def merge(self, other: "WindowAggregator[T, R]") -> None:
        """Merge another aggregator into this one."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the aggregator."""
        ...

    @abstractmethod
    def copy(self) -> "WindowAggregator[T, R]":
        """Create a copy of this aggregator."""
        ...


class CountAggregator(WindowAggregator[Any, int]):
    """Count aggregator."""

    def __init__(self):
        self._count = 0

    def add(self, value: Any) -> None:
        self._count += 1

    def get_result(self) -> int:
        return self._count

    def merge(self, other: "CountAggregator") -> None:
        self._count += other._count

    def reset(self) -> None:
        self._count = 0

    def copy(self) -> "CountAggregator":
        agg = CountAggregator()
        agg._count = self._count
        return agg


class SumAggregator(WindowAggregator[float, float]):
    """Sum aggregator."""

    def __init__(self, field: str | None = None):
        self._sum = 0.0
        self._field = field

    def add(self, value: Any) -> None:
        if self._field and isinstance(value, dict):
            v = value.get(self._field, 0)
        else:
            v = value
        self._sum += float(v) if v is not None else 0

    def get_result(self) -> float:
        return self._sum

    def merge(self, other: "SumAggregator") -> None:
        self._sum += other._sum

    def reset(self) -> None:
        self._sum = 0.0

    def copy(self) -> "SumAggregator":
        agg = SumAggregator(self._field)
        agg._sum = self._sum
        return agg


class AvgAggregator(WindowAggregator[float, float]):
    """Average aggregator."""

    def __init__(self, field: str | None = None):
        self._sum = 0.0
        self._count = 0
        self._field = field

    def add(self, value: Any) -> None:
        if self._field and isinstance(value, dict):
            v = value.get(self._field, 0)
        else:
            v = value
        if v is not None:
            self._sum += float(v)
            self._count += 1

    def get_result(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def merge(self, other: "AvgAggregator") -> None:
        self._sum += other._sum
        self._count += other._count

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def copy(self) -> "AvgAggregator":
        agg = AvgAggregator(self._field)
        agg._sum = self._sum
        agg._count = self._count
        return agg


class MinAggregator(WindowAggregator[float, float | None]):
    """Minimum aggregator."""

    def __init__(self, field: str | None = None):
        self._min: float | None = None
        self._field = field

    def add(self, value: Any) -> None:
        if self._field and isinstance(value, dict):
            v = value.get(self._field)
        else:
            v = value
        if v is not None:
            v = float(v)
            if self._min is None or v < self._min:
                self._min = v

    def get_result(self) -> float | None:
        return self._min

    def merge(self, other: "MinAggregator") -> None:
        if other._min is not None:
            if self._min is None or other._min < self._min:
                self._min = other._min

    def reset(self) -> None:
        self._min = None

    def copy(self) -> "MinAggregator":
        agg = MinAggregator(self._field)
        agg._min = self._min
        return agg


class MaxAggregator(WindowAggregator[float, float | None]):
    """Maximum aggregator."""

    def __init__(self, field: str | None = None):
        self._max: float | None = None
        self._field = field

    def add(self, value: Any) -> None:
        if self._field and isinstance(value, dict):
            v = value.get(self._field)
        else:
            v = value
        if v is not None:
            v = float(v)
            if self._max is None or v > self._max:
                self._max = v

    def get_result(self) -> float | None:
        return self._max

    def merge(self, other: "MaxAggregator") -> None:
        if other._max is not None:
            if self._max is None or other._max > self._max:
                self._max = other._max

    def reset(self) -> None:
        self._max = None

    def copy(self) -> "MaxAggregator":
        agg = MaxAggregator(self._field)
        agg._max = self._max
        return agg


# =============================================================================
# Window State
# =============================================================================


@dataclass
class WindowState(Generic[T, R]):
    """State for a single window."""

    window_start: datetime
    window_end: datetime
    key: str | None
    aggregator: WindowAggregator[T, R]
    message_count: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    is_closed: bool = False


# =============================================================================
# Window Processors
# =============================================================================


class WindowProcessor(ABC, Generic[T, R]):
    """Base class for window processors.

    Window processors accumulate messages and emit results
    when windows close.
    """

    def __init__(
        self,
        config: WindowConfig,
        aggregator_factory: Callable[[], WindowAggregator[T, R]],
        key_extractor: Callable[[T], str] | None = None,
    ):
        """Initialize window processor.

        Args:
            config: Window configuration
            aggregator_factory: Factory for creating aggregators
            key_extractor: Function to extract key from message (for keyed windows)
        """
        self._config = config
        self._aggregator_factory = aggregator_factory
        self._key_extractor = key_extractor
        self._windows: dict[str, WindowState[T, R]] = {}
        self._watermark: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._lock = asyncio.Lock()

    @property
    def config(self) -> WindowConfig:
        """Get window configuration."""
        return self._config

    @property
    def watermark(self) -> datetime:
        """Get current watermark."""
        return self._watermark

    async def add(self, message: StreamMessage[T]) -> list[WindowResult[R]]:
        """Add a message to appropriate window(s).

        Args:
            message: Message to add

        Returns:
            List of closed window results
        """
        async with self._lock:
            # Update watermark
            self._update_watermark(message.timestamp)

            # Get windows for this message
            windows = self._get_windows_for_message(message)

            # Add to each window
            for window in windows:
                self._add_to_window(window, message)

            # Check for closed windows
            return self._emit_closed_windows()

    async def force_emit(self) -> list[WindowResult[R]]:
        """Force emit all current windows.

        Returns:
            List of all window results
        """
        async with self._lock:
            results = []
            for window_id, state in list(self._windows.items()):
                results.append(self._create_result(state, is_final=True))
                del self._windows[window_id]
            return results

    def get_state(self) -> dict[str, Any]:
        """Get current processor state for checkpointing."""
        return {
            "watermark": self._watermark.isoformat(),
            "windows": {
                wid: {
                    "window_start": state.window_start.isoformat(),
                    "window_end": state.window_end.isoformat(),
                    "key": state.key,
                    "message_count": state.message_count,
                    "first_timestamp": state.first_timestamp.isoformat() if state.first_timestamp else None,
                    "last_timestamp": state.last_timestamp.isoformat() if state.last_timestamp else None,
                }
                for wid, state in self._windows.items()
            },
        }

    @abstractmethod
    def _get_windows_for_message(self, message: StreamMessage[T]) -> list[WindowState[T, R]]:
        """Get windows that should receive this message."""
        ...

    @abstractmethod
    def _should_close_window(self, state: WindowState[T, R]) -> bool:
        """Check if a window should be closed."""
        ...

    def _update_watermark(self, timestamp: datetime) -> None:
        """Update watermark based on message timestamp."""
        ts = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        delay = timedelta(milliseconds=self._config.watermark_delay_ms)
        new_watermark = ts - delay
        if new_watermark > self._watermark:
            self._watermark = new_watermark

    def _add_to_window(self, state: WindowState[T, R], message: StreamMessage[T]) -> None:
        """Add message to window state."""
        state.aggregator.add(message.value)
        state.message_count += 1

        ts = message.timestamp
        if state.first_timestamp is None or ts < state.first_timestamp:
            state.first_timestamp = ts
        if state.last_timestamp is None or ts > state.last_timestamp:
            state.last_timestamp = ts

    def _emit_closed_windows(self) -> list[WindowResult[R]]:
        """Emit results for closed windows."""
        results = []
        closed_ids = []

        for window_id, state in self._windows.items():
            if self._should_close_window(state):
                results.append(self._create_result(state, is_final=True))
                closed_ids.append(window_id)

        for window_id in closed_ids:
            del self._windows[window_id]

        return results

    def _create_result(self, state: WindowState[T, R], is_final: bool) -> WindowResult[R]:
        """Create window result from state."""
        window_id = f"{state.window_start.timestamp()}-{state.window_end.timestamp()}"
        if state.key:
            window_id = f"{state.key}:{window_id}"

        return WindowResult(
            window_id=window_id,
            window_start=state.window_start,
            window_end=state.window_end,
            key=state.key,
            result=state.aggregator.get_result(),
            message_count=state.message_count,
            first_timestamp=state.first_timestamp,
            last_timestamp=state.last_timestamp,
            is_final=is_final,
        )

    def _get_window_key(self, window_start: datetime, key: str | None) -> str:
        """Generate unique window key."""
        ts = int(window_start.timestamp() * 1000)
        return f"{key or 'global'}:{ts}"


class TumblingWindowProcessor(WindowProcessor[T, R]):
    """Tumbling window processor.

    Creates fixed-size, non-overlapping windows.
    """

    def _get_windows_for_message(self, message: StreamMessage[T]) -> list[WindowState[T, R]]:
        """Get tumbling window for message."""
        ts = message.timestamp.replace(tzinfo=timezone.utc) if message.timestamp.tzinfo is None else message.timestamp
        ts_ms = int(ts.timestamp() * 1000)

        # Calculate window boundaries
        window_start_ms = (ts_ms // self._config.size_ms) * self._config.size_ms
        window_start = datetime.fromtimestamp(window_start_ms / 1000, tz=timezone.utc)
        window_end = window_start + timedelta(milliseconds=self._config.size_ms)

        # Extract key if keyed
        key = self._key_extractor(message.value) if self._key_extractor else None
        window_key = self._get_window_key(window_start, key)

        # Get or create window
        if window_key not in self._windows:
            self._windows[window_key] = WindowState(
                window_start=window_start,
                window_end=window_end,
                key=key,
                aggregator=self._aggregator_factory(),
            )

        return [self._windows[window_key]]

    def _should_close_window(self, state: WindowState[T, R]) -> bool:
        """Check if tumbling window should close."""
        lateness = timedelta(milliseconds=self._config.allowed_lateness_ms)
        return self._watermark >= state.window_end + lateness


class SlidingWindowProcessor(WindowProcessor[T, R]):
    """Sliding window processor.

    Creates fixed-size, overlapping windows with configurable slide interval.
    """

    def _get_windows_for_message(self, message: StreamMessage[T]) -> list[WindowState[T, R]]:
        """Get sliding windows for message."""
        ts = message.timestamp.replace(tzinfo=timezone.utc) if message.timestamp.tzinfo is None else message.timestamp
        ts_ms = int(ts.timestamp() * 1000)

        size_ms = self._config.size_ms
        slide_ms = self._config.slide_ms or size_ms

        # Extract key
        key = self._key_extractor(message.value) if self._key_extractor else None

        windows = []

        # Find all windows that contain this timestamp
        # Windows that end after ts and start before or at ts
        first_window_start_ms = ((ts_ms - size_ms) // slide_ms + 1) * slide_ms
        last_window_start_ms = (ts_ms // slide_ms) * slide_ms

        current_start_ms = first_window_start_ms
        while current_start_ms <= last_window_start_ms:
            window_start = datetime.fromtimestamp(current_start_ms / 1000, tz=timezone.utc)
            window_end = window_start + timedelta(milliseconds=size_ms)
            window_key = self._get_window_key(window_start, key)

            if window_key not in self._windows:
                self._windows[window_key] = WindowState(
                    window_start=window_start,
                    window_end=window_end,
                    key=key,
                    aggregator=self._aggregator_factory(),
                )

            windows.append(self._windows[window_key])
            current_start_ms += slide_ms

        return windows

    def _should_close_window(self, state: WindowState[T, R]) -> bool:
        """Check if sliding window should close."""
        lateness = timedelta(milliseconds=self._config.allowed_lateness_ms)
        return self._watermark >= state.window_end + lateness


class SessionWindowProcessor(WindowProcessor[T, R]):
    """Session window processor.

    Creates dynamic windows based on activity gaps.
    """

    def __init__(
        self,
        config: WindowConfig,
        aggregator_factory: Callable[[], WindowAggregator[T, R]],
        key_extractor: Callable[[T], str],
    ):
        """Initialize session window processor.

        Args:
            config: Window configuration (gap_ms is required)
            aggregator_factory: Factory for creating aggregators
            key_extractor: Function to extract key (required for sessions)
        """
        if config.gap_ms is None:
            raise ValueError("gap_ms is required for session windows")
        if key_extractor is None:
            raise ValueError("key_extractor is required for session windows")

        super().__init__(config, aggregator_factory, key_extractor)
        self._sessions: dict[str, WindowState[T, R]] = {}

    def _get_windows_for_message(self, message: StreamMessage[T]) -> list[WindowState[T, R]]:
        """Get or create session for message."""
        ts = message.timestamp.replace(tzinfo=timezone.utc) if message.timestamp.tzinfo is None else message.timestamp
        key = self._key_extractor(message.value)  # type: ignore
        gap = timedelta(milliseconds=self._config.gap_ms or 30000)

        if key in self._sessions:
            session = self._sessions[key]

            # Check if message extends the session
            if session.last_timestamp and ts <= session.last_timestamp + gap:
                # Extend session
                if ts > session.window_end - gap:
                    session.window_end = ts + gap
                return [session]
            else:
                # Gap exceeded, start new session
                # First, close the old session
                self._windows[self._get_window_key(session.window_start, key)] = session

        # Create new session
        new_session = WindowState(
            window_start=ts,
            window_end=ts + gap,
            key=key,
            aggregator=self._aggregator_factory(),
        )
        self._sessions[key] = new_session

        return [new_session]

    def _should_close_window(self, state: WindowState[T, R]) -> bool:
        """Check if session should close."""
        gap = timedelta(milliseconds=self._config.gap_ms or 30000)
        lateness = timedelta(milliseconds=self._config.allowed_lateness_ms)

        # Session closes when watermark passes window_end + lateness
        if state.last_timestamp:
            close_time = state.last_timestamp + gap + lateness
            return self._watermark >= close_time

        return False

    async def force_emit(self) -> list[WindowResult[R]]:
        """Force emit all sessions."""
        async with self._lock:
            results = []

            # Emit active sessions
            for key, session in self._sessions.items():
                results.append(self._create_result(session, is_final=True))

            # Emit closed sessions in _windows
            for window_id, state in self._windows.items():
                results.append(self._create_result(state, is_final=True))

            self._sessions.clear()
            self._windows.clear()

            return results
