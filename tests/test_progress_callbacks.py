"""Comprehensive tests for progress callback standardization.

Tests cover:
- Event types and levels
- Context and metrics
- Callback adapters (console, logging, file)
- Callback chain
- Filtering and throttling
- Buffering
- Registry pattern
- Emitter
- Presets
- Integration scenarios
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from truthound.profiler.progress_callbacks import (
    # Event types and levels
    EventLevel,
    EventType,
    # Context and metrics
    ProgressContext,
    ProgressMetrics,
    StandardProgressEvent,
    # Protocols
    ProgressCallback,
    AsyncProgressCallback,
    LifecycleCallback,
    # Base adapter
    CallbackAdapter,
    # Console adapters
    ConsoleStyle,
    ConsoleAdapter,
    MinimalConsoleAdapter,
    # Logging adapter
    LoggingAdapter,
    # File adapter
    FileOutputConfig,
    FileAdapter,
    # Callback chain
    CallbackChain,
    # Filtering and throttling
    FilterConfig,
    FilteringAdapter,
    ThrottleConfig,
    ThrottlingAdapter,
    # Buffering
    BufferConfig,
    BufferingAdapter,
    # Async
    AsyncAdapter,
    # Registry
    CallbackRegistry,
    # Emitter
    ProgressEmitter,
    # Presets
    CallbackPresets,
    # Convenience functions
    create_callback_chain,
    create_console_callback,
    create_logging_callback,
    create_file_callback,
    with_throttling,
    with_filtering,
    with_buffering,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> ProgressContext:
    """Create sample progress context."""
    return ProgressContext(
        operation_id="op_12345",
        table_name="users",
        column_name="email",
        tags=("test", "validation"),
    )


@pytest.fixture
def sample_metrics() -> ProgressMetrics:
    """Create sample progress metrics."""
    return ProgressMetrics(
        elapsed_seconds=10.5,
        estimated_remaining_seconds=5.0,
        rows_processed=50000,
        rows_per_second=4761.9,
        columns_completed=5,
        columns_total=10,
    )


@pytest.fixture
def sample_event(sample_context: ProgressContext, sample_metrics: ProgressMetrics) -> StandardProgressEvent:
    """Create sample progress event."""
    return StandardProgressEvent(
        event_type=EventType.PROGRESS,
        level=EventLevel.INFO,
        progress=0.5,
        message="Processing halfway",
        context=sample_context,
        metrics=sample_metrics,
    )


@pytest.fixture
def captured_events() -> list[StandardProgressEvent]:
    """Create list to capture events."""
    return []


class CapturingAdapter(CallbackAdapter):
    """Adapter that captures events for testing."""

    def __init__(self, events: list[StandardProgressEvent], **kwargs):
        super().__init__(**kwargs)
        self.events = events

    def _handle_event(self, event: StandardProgressEvent) -> None:
        self.events.append(event)


# =============================================================================
# Event Types and Levels Tests
# =============================================================================


class TestEventLevel:
    """Test EventLevel enum."""

    def test_all_levels_exist(self):
        """Test that all expected levels exist."""
        assert EventLevel.DEBUG
        assert EventLevel.INFO
        assert EventLevel.NOTICE
        assert EventLevel.WARNING
        assert EventLevel.ERROR
        assert EventLevel.CRITICAL

    def test_level_ordering(self):
        """Test that levels are ordered correctly."""
        assert EventLevel.DEBUG.value < EventLevel.INFO.value
        assert EventLevel.INFO.value < EventLevel.NOTICE.value
        assert EventLevel.NOTICE.value < EventLevel.WARNING.value
        assert EventLevel.WARNING.value < EventLevel.ERROR.value
        assert EventLevel.ERROR.value < EventLevel.CRITICAL.value


class TestEventType:
    """Test EventType enum."""

    def test_lifecycle_events(self):
        """Test lifecycle event types exist."""
        assert EventType.START
        assert EventType.COMPLETE
        assert EventType.FAIL
        assert EventType.CANCEL

    def test_progress_events(self):
        """Test progress event types exist."""
        assert EventType.PROGRESS
        assert EventType.COLUMN_START
        assert EventType.COLUMN_COMPLETE
        assert EventType.COLUMN_PROGRESS

    def test_event_values_are_strings(self):
        """Test that event values are strings."""
        assert isinstance(EventType.START.value, str)
        assert EventType.START.value == "start"


# =============================================================================
# Context and Metrics Tests
# =============================================================================


class TestProgressContext:
    """Test ProgressContext dataclass."""

    def test_create_basic_context(self):
        """Test creating basic context."""
        ctx = ProgressContext(operation_id="op1", table_name="users")
        assert ctx.operation_id == "op1"
        assert ctx.table_name == "users"
        assert ctx.column_name == ""

    def test_with_column(self):
        """Test creating child context for column."""
        parent = ProgressContext(operation_id="op1", table_name="users")
        child = parent.with_column("email")

        assert child.column_name == "email"
        assert child.table_name == "users"
        assert child.parent_context == parent

    def test_with_analyzer(self):
        """Test creating child context for analyzer."""
        ctx = ProgressContext(operation_id="op1", table_name="users", column_name="email")
        child = ctx.with_analyzer("pattern_matcher")

        assert child.analyzer_name == "pattern_matcher"
        assert child.column_name == "email"

    def test_get_path(self):
        """Test hierarchical path generation."""
        ctx = ProgressContext(
            table_name="users",
            column_name="email",
            analyzer_name="validator",
        )
        assert ctx.get_path() == "users/email/validator"

    def test_empty_path(self):
        """Test empty path for empty context."""
        ctx = ProgressContext()
        assert ctx.get_path() == ""


class TestProgressMetrics:
    """Test ProgressMetrics dataclass."""

    def test_columns_remaining(self):
        """Test columns remaining calculation."""
        metrics = ProgressMetrics(columns_completed=3, columns_total=10)
        assert metrics.columns_remaining == 7

    def test_throughput_string_low(self):
        """Test throughput string for low values."""
        metrics = ProgressMetrics(rows_per_second=500)
        assert "500 rows/s" in metrics.throughput_string

    def test_throughput_string_thousands(self):
        """Test throughput string for thousands."""
        metrics = ProgressMetrics(rows_per_second=5000)
        assert "5.0K rows/s" in metrics.throughput_string

    def test_throughput_string_millions(self):
        """Test throughput string for millions."""
        metrics = ProgressMetrics(rows_per_second=2_500_000)
        assert "2.5M rows/s" in metrics.throughput_string


class TestStandardProgressEvent:
    """Test StandardProgressEvent dataclass."""

    def test_create_basic_event(self):
        """Test creating basic event."""
        event = StandardProgressEvent(
            event_type=EventType.START,
            level=EventLevel.INFO,
            progress=0.0,
            message="Starting",
        )
        assert event.event_type == EventType.START
        assert event.progress == 0.0

    def test_percent_property(self):
        """Test percent property conversion."""
        event = StandardProgressEvent(event_type=EventType.PROGRESS, progress=0.5)
        assert event.percent == 50.0

    def test_is_complete_true(self):
        """Test is_complete for completion events."""
        for event_type in [EventType.COMPLETE, EventType.FAIL, EventType.CANCEL]:
            event = StandardProgressEvent(event_type=event_type)
            assert event.is_complete is True

    def test_is_complete_false(self):
        """Test is_complete for non-completion events."""
        event = StandardProgressEvent(event_type=EventType.PROGRESS)
        assert event.is_complete is False

    def test_is_error_true(self):
        """Test is_error for error levels."""
        for level in [EventLevel.ERROR, EventLevel.CRITICAL]:
            event = StandardProgressEvent(event_type=EventType.PROGRESS, level=level)
            assert event.is_error is True

    def test_is_error_false(self):
        """Test is_error for non-error levels."""
        event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.INFO)
        assert event.is_error is False

    def test_to_dict(self, sample_event: StandardProgressEvent):
        """Test serialization to dictionary."""
        d = sample_event.to_dict()

        assert d["event_type"] == "progress"
        assert d["level"] == "INFO"
        assert d["progress"] == 0.5
        assert d["context"]["table_name"] == "users"
        assert d["metrics"]["rows_processed"] == 50000


# =============================================================================
# Callback Adapter Tests
# =============================================================================


class TestCallbackAdapter:
    """Test base CallbackAdapter class."""

    def test_should_handle_enabled(self, captured_events: list):
        """Test filtering when enabled."""
        adapter = CapturingAdapter(captured_events, enabled=True)
        event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.INFO)

        assert adapter.should_handle(event) is True

    def test_should_handle_disabled(self, captured_events: list):
        """Test filtering when disabled."""
        adapter = CapturingAdapter(captured_events, enabled=False)
        event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.INFO)

        assert adapter.should_handle(event) is False

    def test_should_handle_min_level(self, captured_events: list):
        """Test minimum level filtering."""
        adapter = CapturingAdapter(captured_events, min_level=EventLevel.WARNING)

        info_event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.INFO)
        warning_event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.WARNING)

        assert adapter.should_handle(info_event) is False
        assert adapter.should_handle(warning_event) is True

    def test_should_handle_event_types(self, captured_events: list):
        """Test event type filtering."""
        adapter = CapturingAdapter(
            captured_events,
            event_types={EventType.START, EventType.COMPLETE},
        )

        start_event = StandardProgressEvent(event_type=EventType.START)
        progress_event = StandardProgressEvent(event_type=EventType.PROGRESS)

        assert adapter.should_handle(start_event) is True
        assert adapter.should_handle(progress_event) is False

    def test_lifecycle(self, captured_events: list):
        """Test adapter lifecycle."""
        adapter = CapturingAdapter(captured_events)

        assert adapter._started is False
        adapter.start()
        assert adapter._started is True
        adapter.stop()
        assert adapter._started is False


# =============================================================================
# Console Adapter Tests
# =============================================================================


class TestConsoleAdapter:
    """Test ConsoleAdapter class."""

    def test_create_with_defaults(self):
        """Test creating adapter with defaults."""
        adapter = ConsoleAdapter()
        assert adapter.style.bar_width == 40

    def test_create_with_custom_style(self):
        """Test creating adapter with custom style."""
        style = ConsoleStyle(bar_width=50, show_eta=False)
        adapter = ConsoleAdapter(style=style)

        assert adapter.style.bar_width == 50
        assert adapter.style.show_eta is False

    def test_render_progress_bar(self):
        """Test progress bar rendering."""
        stream = io.StringIO()
        adapter = ConsoleAdapter(stream=stream)

        event = StandardProgressEvent(
            event_type=EventType.PROGRESS,
            progress=0.5,
            context=ProgressContext(column_name="test_col"),
            metrics=ProgressMetrics(estimated_remaining_seconds=10.0),
        )

        adapter.on_progress(event)
        output = stream.getvalue()

        assert "50.0%" in output
        assert "test_col" in output

    def test_render_completion(self):
        """Test completion message rendering."""
        stream = io.StringIO()
        adapter = ConsoleAdapter(stream=stream)

        event = StandardProgressEvent(
            event_type=EventType.COMPLETE,
            progress=1.0,
            metrics=ProgressMetrics(elapsed_seconds=30.0),
        )

        adapter.on_progress(event)
        output = stream.getvalue()

        assert "Complete" in output or "✓" in output


class TestMinimalConsoleAdapter:
    """Test MinimalConsoleAdapter class."""

    def test_milestone_output(self, capsys):
        """Test milestone-based output."""
        adapter = MinimalConsoleAdapter(milestone_interval=25)

        # First milestone
        event1 = StandardProgressEvent(event_type=EventType.PROGRESS, progress=0.25)
        adapter.on_progress(event1)
        output1 = capsys.readouterr().out
        assert "25%" in output1

        # Same milestone (should not output)
        event2 = StandardProgressEvent(event_type=EventType.PROGRESS, progress=0.26)
        adapter.on_progress(event2)
        output2 = capsys.readouterr().out
        assert output2 == ""

        # Next milestone
        event3 = StandardProgressEvent(event_type=EventType.PROGRESS, progress=0.50)
        adapter.on_progress(event3)
        output3 = capsys.readouterr().out
        assert "50%" in output3


# =============================================================================
# Logging Adapter Tests
# =============================================================================


class TestLoggingAdapter:
    """Test LoggingAdapter class."""

    def test_level_mapping(self):
        """Test event level to logging level mapping."""
        logger = MagicMock()
        adapter = LoggingAdapter(logger=logger)

        event = StandardProgressEvent(
            event_type=EventType.PROGRESS,
            level=EventLevel.WARNING,
            message="Test warning",
        )

        adapter.on_progress(event)
        logger.log.assert_called_once()
        call_args = logger.log.call_args
        assert call_args[0][0] == logging.WARNING

    def test_message_formatting(self):
        """Test message formatting."""
        logger = MagicMock()
        adapter = LoggingAdapter(logger=logger)

        event = StandardProgressEvent(
            event_type=EventType.COLUMN_COMPLETE,
            context=ProgressContext(column_name="email"),
        )

        adapter.on_progress(event)
        call_args = logger.log.call_args
        assert "email" in call_args[0][1]

    def test_includes_context_extras(self):
        """Test that context is included in extras."""
        logger = MagicMock()
        adapter = LoggingAdapter(logger=logger, include_context=True)

        event = StandardProgressEvent(
            event_type=EventType.PROGRESS,
            context=ProgressContext(table_name="users"),
        )

        adapter.on_progress(event)
        extra = logger.log.call_args[1]["extra"]
        assert extra["context"]["table"] == "users"


# =============================================================================
# File Adapter Tests
# =============================================================================


class TestFileAdapter:
    """Test FileAdapter class."""

    def test_jsonl_output(self, tmp_path: Path):
        """Test JSONL file output."""
        output_file = tmp_path / "progress.jsonl"
        adapter = FileAdapter(output_file)

        adapter.start()

        event = StandardProgressEvent(
            event_type=EventType.START,
            progress=0.0,
            message="Starting",
        )
        adapter.on_progress(event)

        adapter.stop()

        content = output_file.read_text()
        data = json.loads(content.strip())
        assert data["event_type"] == "start"
        assert data["message"] == "Starting"

    def test_json_output(self, tmp_path: Path):
        """Test JSON file output (array format)."""
        output_file = tmp_path / "progress.json"
        config = FileOutputConfig(format="json")
        adapter = FileAdapter(output_file, config=config)

        adapter.start()

        for i in range(3):
            event = StandardProgressEvent(
                event_type=EventType.PROGRESS,
                progress=i * 0.33,
            )
            adapter.on_progress(event)

        adapter.stop()

        content = output_file.read_text()
        data = json.loads(content)
        assert isinstance(data, list)
        assert len(data) == 3


# =============================================================================
# Callback Chain Tests
# =============================================================================


class TestCallbackChain:
    """Test CallbackChain class."""

    def test_dispatch_to_all(self, captured_events: list):
        """Test dispatching to all callbacks."""
        events1 = []
        events2 = []

        chain = CallbackChain([
            CapturingAdapter(events1),
            CapturingAdapter(events2),
        ])

        event = StandardProgressEvent(event_type=EventType.PROGRESS, progress=0.5)
        chain.on_progress(event)

        assert len(events1) == 1
        assert len(events2) == 1

    def test_add_callback(self, captured_events: list):
        """Test adding callback to chain."""
        chain = CallbackChain()
        assert len(chain) == 0

        chain.add(CapturingAdapter(captured_events))
        assert len(chain) == 1

    def test_remove_callback(self, captured_events: list):
        """Test removing callback from chain."""
        adapter = CapturingAdapter(captured_events)
        chain = CallbackChain([adapter])

        assert chain.remove(adapter) is True
        assert len(chain) == 0

    def test_error_handling(self, captured_events: list):
        """Test error handling in chain."""
        class FailingAdapter(CallbackAdapter):
            def _handle_event(self, event):
                raise ValueError("Test error")

        chain = CallbackChain([
            FailingAdapter(),
            CapturingAdapter(captured_events),
        ])

        event = StandardProgressEvent(event_type=EventType.PROGRESS)
        chain.on_progress(event)  # Should not raise

        assert len(captured_events) == 1
        assert len(chain.errors) == 1

    def test_stop_on_error(self, captured_events: list):
        """Test stopping chain on error."""
        class FailingAdapter(CallbackAdapter):
            def _handle_event(self, event):
                raise ValueError("Test error")

        chain = CallbackChain([
            FailingAdapter(),
            CapturingAdapter(captured_events),
        ], stop_on_error=True)

        event = StandardProgressEvent(event_type=EventType.PROGRESS)

        with pytest.raises(ValueError):
            chain.on_progress(event)

    def test_lifecycle(self):
        """Test chain lifecycle management."""
        mock_adapter = MagicMock(spec=CallbackAdapter)
        chain = CallbackChain([mock_adapter])

        chain.start()
        mock_adapter.start.assert_called_once()

        chain.stop()
        mock_adapter.stop.assert_called_once()


# =============================================================================
# Filtering Adapter Tests
# =============================================================================


class TestFilteringAdapter:
    """Test FilteringAdapter class."""

    def test_min_level_filtering(self, captured_events: list):
        """Test minimum level filtering."""
        config = FilterConfig(min_level=EventLevel.WARNING)
        adapter = FilteringAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        info_event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.INFO)
        warning_event = StandardProgressEvent(event_type=EventType.PROGRESS, level=EventLevel.WARNING)

        adapter.on_progress(info_event)
        adapter.on_progress(warning_event)

        assert len(captured_events) == 1
        assert captured_events[0].level == EventLevel.WARNING

    def test_event_type_filtering(self, captured_events: list):
        """Test event type filtering."""
        config = FilterConfig(event_types={EventType.START, EventType.COMPLETE})
        adapter = FilteringAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        adapter.on_progress(StandardProgressEvent(event_type=EventType.START))
        adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS))
        adapter.on_progress(StandardProgressEvent(event_type=EventType.COMPLETE))

        assert len(captured_events) == 2
        assert captured_events[0].event_type == EventType.START
        assert captured_events[1].event_type == EventType.COMPLETE

    def test_tag_filtering(self, captured_events: list):
        """Test tag-based filtering."""
        config = FilterConfig(include_tags={"important"})
        adapter = FilteringAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        ctx_with_tag = ProgressContext(tags=("important", "other"))
        ctx_without_tag = ProgressContext(tags=("other",))

        adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS, context=ctx_with_tag))
        adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS, context=ctx_without_tag))

        assert len(captured_events) == 1


# =============================================================================
# Throttling Adapter Tests
# =============================================================================


class TestThrottlingAdapter:
    """Test ThrottlingAdapter class."""

    def test_lifecycle_events_always_pass(self, captured_events: list):
        """Test that lifecycle events always pass through."""
        config = ThrottleConfig(min_interval_ms=1000)
        adapter = ThrottlingAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        # All lifecycle events should pass
        for event_type in [EventType.START, EventType.COMPLETE, EventType.FAIL]:
            adapter.on_progress(StandardProgressEvent(event_type=event_type))

        assert len(captured_events) == 3

    def test_rate_limiting(self, captured_events: list):
        """Test rate limiting."""
        config = ThrottleConfig(min_interval_ms=100, max_events_per_second=2)
        adapter = ThrottlingAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        # Send many events quickly
        for i in range(10):
            adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS, progress=i * 0.1))

        # Should be throttled
        assert len(captured_events) < 10


# =============================================================================
# Buffering Adapter Tests
# =============================================================================


class TestBufferingAdapter:
    """Test BufferingAdapter class."""

    def test_buffer_flush_on_size(self, captured_events: list):
        """Test buffer flushes when max size reached."""
        config = BufferConfig(max_size=3, flush_interval_seconds=1000)  # Very long interval
        adapter = BufferingAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        # Send 4 events - the 3rd should trigger flush, 4th stays in buffer
        for i in range(4):
            adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS, progress=i * 0.1))

        # First 3 events should be flushed when buffer hits max_size
        assert len(captured_events) >= 3

    def test_buffer_flush_on_complete(self, captured_events: list):
        """Test buffer flushes on completion event."""
        config = BufferConfig(max_size=100, flush_on_complete=True)
        adapter = BufferingAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS))
        adapter.on_progress(StandardProgressEvent(event_type=EventType.COMPLETE))

        assert len(captured_events) == 2

    def test_stop_flushes_buffer(self, captured_events: list):
        """Test that stop() flushes remaining buffer."""
        config = BufferConfig(max_size=100, flush_interval_seconds=1000)  # Very long interval
        adapter = BufferingAdapter(
            wrapped=CapturingAdapter(captured_events),
            config=config,
        )

        # Record initial state before adding events
        initial_count = len(captured_events)

        adapter.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS))
        # Buffer may or may not have flushed yet depending on timing
        count_before_stop = len(captured_events)

        adapter.stop()
        # After stop, all buffered events should be flushed
        assert len(captured_events) >= initial_count + 1


# =============================================================================
# Registry Tests
# =============================================================================


class TestCallbackRegistry:
    """Test CallbackRegistry class."""

    def test_singleton_pattern(self):
        """Test singleton instance."""
        r1 = CallbackRegistry.get_instance()
        r2 = CallbackRegistry.get_instance()
        assert r1 is r2

    def test_builtin_adapters(self):
        """Test built-in adapters are registered."""
        registry = CallbackRegistry()
        adapters = registry.list_adapters()

        assert "console" in adapters
        assert "logging" in adapters
        assert "file" in adapters

    def test_create_by_name(self):
        """Test creating adapter by name."""
        registry = CallbackRegistry()
        adapter = registry.create("console")
        assert isinstance(adapter, ConsoleAdapter)

    def test_create_with_kwargs(self, tmp_path: Path):
        """Test creating adapter with keyword arguments."""
        registry = CallbackRegistry()
        adapter = registry.create("file", path=tmp_path / "test.jsonl")
        assert isinstance(adapter, FileAdapter)

    def test_register_custom(self, captured_events: list):
        """Test registering custom adapter."""
        registry = CallbackRegistry()

        @registry.register("custom")
        class CustomAdapter(CallbackAdapter):
            def _handle_event(self, event):
                pass

        adapter = registry.create("custom")
        assert adapter is not None

    def test_unknown_adapter_raises(self):
        """Test that unknown adapter raises KeyError."""
        registry = CallbackRegistry()
        with pytest.raises(KeyError):
            registry.create("unknown_adapter")


# =============================================================================
# Emitter Tests
# =============================================================================


class TestProgressEmitter:
    """Test ProgressEmitter class."""

    def test_emit_start(self, captured_events: list):
        """Test emitting start event."""
        adapter = CapturingAdapter(captured_events)
        emitter = ProgressEmitter(callback=adapter)

        emitter.start()

        assert len(captured_events) == 1
        assert captured_events[0].event_type == EventType.START

    def test_emit_column_lifecycle(self, captured_events: list):
        """Test emitting column lifecycle events."""
        adapter = CapturingAdapter(captured_events)
        emitter = ProgressEmitter(callback=adapter, total_columns=2)

        emitter.start()
        emitter.column_start("col1")
        emitter.column_complete("col1")
        emitter.column_start("col2")
        emitter.column_complete("col2")
        emitter.complete()

        event_types = [e.event_type for e in captured_events]
        assert EventType.START in event_types
        assert EventType.COLUMN_START in event_types
        assert EventType.COLUMN_COMPLETE in event_types
        assert EventType.COMPLETE in event_types

    def test_progress_calculation(self, captured_events: list):
        """Test progress calculation."""
        adapter = CapturingAdapter(captured_events)
        emitter = ProgressEmitter(callback=adapter, total_columns=4)

        emitter.start()
        emitter.column_start("col1")
        emitter.column_complete("col1")

        # After completing 1 of 4 columns, progress should be 25%
        complete_event = next(e for e in captured_events if e.event_type == EventType.COLUMN_COMPLETE)
        assert complete_event.progress == pytest.approx(0.25, abs=0.01)

    def test_checkpoint(self, captured_events: list):
        """Test checkpoint emission."""
        adapter = CapturingAdapter(captured_events)
        emitter = ProgressEmitter(callback=adapter)

        emitter.start()
        emitter.checkpoint("before_analysis", count=100)

        checkpoint = next(e for e in captured_events if e.event_type == EventType.CHECKPOINT)
        assert checkpoint.metadata.get("count") == 100

    def test_fail(self, captured_events: list):
        """Test failure emission."""
        adapter = CapturingAdapter(captured_events)
        emitter = ProgressEmitter(callback=adapter)

        emitter.start()
        emitter.fail("Something went wrong", error=ValueError("Test"))

        fail_event = next(e for e in captured_events if e.event_type == EventType.FAIL)
        assert fail_event.level == EventLevel.ERROR
        assert "wrong" in fail_event.message


# =============================================================================
# Presets Tests
# =============================================================================


class TestCallbackPresets:
    """Test CallbackPresets class."""

    def test_console_only(self):
        """Test console-only preset."""
        chain = CallbackPresets.console_only()
        assert len(chain) == 1
        assert isinstance(list(chain)[0], ConsoleAdapter)

    def test_logging_only(self):
        """Test logging-only preset."""
        chain = CallbackPresets.logging_only()
        assert len(chain) == 1
        assert isinstance(list(chain)[0], LoggingAdapter)

    def test_console_and_logging(self):
        """Test console and logging preset."""
        chain = CallbackPresets.console_and_logging()
        assert len(chain) == 2

    def test_full_observability(self, tmp_path: Path):
        """Test full observability preset."""
        chain = CallbackPresets.full_observability(log_file=tmp_path / "log.jsonl")
        assert len(chain) == 3

    def test_production(self):
        """Test production preset."""
        chain = CallbackPresets.production()
        assert len(chain) == 1
        # Should be throttled
        adapter = list(chain)[0]
        assert isinstance(adapter, ThrottlingAdapter)

    def test_silent(self):
        """Test silent preset."""
        chain = CallbackPresets.silent()
        assert len(chain) == 0


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_callback_chain(self, captured_events: list):
        """Test create_callback_chain."""
        chain = create_callback_chain(
            CapturingAdapter(captured_events),
            ConsoleAdapter(),
        )
        assert len(chain) == 2

    def test_create_console_callback(self):
        """Test create_console_callback."""
        adapter = create_console_callback(bar_width=50, show_eta=False)
        assert isinstance(adapter, ConsoleAdapter)
        assert adapter.style.bar_width == 50

    def test_create_logging_callback(self):
        """Test create_logging_callback."""
        adapter = create_logging_callback("test.logger")
        assert isinstance(adapter, LoggingAdapter)

    def test_create_file_callback(self, tmp_path: Path):
        """Test create_file_callback."""
        adapter = create_file_callback(tmp_path / "out.jsonl")
        assert isinstance(adapter, FileAdapter)

    def test_with_throttling(self, captured_events: list):
        """Test with_throttling."""
        base = CapturingAdapter(captured_events)
        throttled = with_throttling(base, max_per_second=5)
        assert isinstance(throttled, ThrottlingAdapter)

    def test_with_filtering(self, captured_events: list):
        """Test with_filtering."""
        base = CapturingAdapter(captured_events)
        filtered = with_filtering(base, min_level=EventLevel.WARNING)
        assert isinstance(filtered, FilteringAdapter)

    def test_with_buffering(self, captured_events: list):
        """Test with_buffering."""
        base = CapturingAdapter(captured_events)
        buffered = with_buffering(base, max_size=50)
        assert isinstance(buffered, BufferingAdapter)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for progress callback system."""

    def test_full_profiling_simulation(self, captured_events: list, tmp_path: Path):
        """Test simulating full profiling workflow."""
        # Create chain with multiple adapters
        file_events = []

        chain = CallbackChain([
            CapturingAdapter(captured_events),
            CapturingAdapter(file_events),
        ])

        # Create emitter
        emitter = ProgressEmitter(
            callback=chain,
            operation_id="test_op",
            table_name="test_table",
            total_columns=3,
        )

        # Simulate profiling
        emitter.start("Starting profiling")

        for col in ["col1", "col2", "col3"]:
            emitter.column_start(col)
            for progress in [0.25, 0.5, 0.75, 1.0]:
                emitter.column_progress(col, progress, rows=100)
            emitter.column_complete(col)

        emitter.complete("Profiling finished")

        # Verify events
        assert len(captured_events) == len(file_events)
        assert captured_events[0].event_type == EventType.START
        assert captured_events[-1].event_type == EventType.COMPLETE

        # Verify context propagation
        column_events = [e for e in captured_events if e.context.column_name]
        assert all(e.context.table_name == "test_table" for e in column_events)

    def test_nested_contexts(self, captured_events: list):
        """Test nested context handling."""
        # Use DEBUG level to capture column_progress events
        adapter = CapturingAdapter(captured_events, min_level=EventLevel.DEBUG)
        emitter = ProgressEmitter(
            callback=adapter,
            table_name="users",
            total_columns=1,
        )

        emitter.start()
        emitter.column_start("email")
        emitter.column_progress("email", 0.5, analyzer="pattern_matcher")
        emitter.column_complete("email")
        emitter.complete()

        # Find the column progress event with analyzer (it has DEBUG level)
        progress_events = [e for e in captured_events if e.event_type == EventType.COLUMN_PROGRESS]
        assert len(progress_events) == 1
        assert progress_events[0].context.analyzer_name == "pattern_matcher"

    def test_error_recovery(self, captured_events: list):
        """Test error recovery in callback chain."""
        class FailingAdapter(CallbackAdapter):
            fail_count = 0

            def _handle_event(self, event):
                self.fail_count += 1
                if self.fail_count < 3:
                    raise ValueError("Transient error")

        chain = CallbackChain([
            FailingAdapter(),
            CapturingAdapter(captured_events),
        ])

        # All events should reach the capturing adapter despite failures
        for i in range(5):
            chain.on_progress(StandardProgressEvent(event_type=EventType.PROGRESS))

        assert len(captured_events) == 5

    def test_thread_safety(self, captured_events: list):
        """Test thread-safe event emission."""
        adapter = CapturingAdapter(captured_events)
        chain = CallbackChain([adapter])

        def emit_events():
            for i in range(50):
                chain.on_progress(StandardProgressEvent(
                    event_type=EventType.PROGRESS,
                    progress=i / 50,
                ))

        threads = [threading.Thread(target=emit_events) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All events should be captured (200 total)
        assert len(captured_events) == 200


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Test protocol compliance."""

    def test_progress_callback_protocol(self):
        """Test ProgressCallback protocol compliance."""
        adapter = ConsoleAdapter()
        assert isinstance(adapter, ProgressCallback)

    def test_custom_callback_protocol(self, captured_events: list):
        """Test custom class protocol compliance."""
        class CustomCallback:
            def on_progress(self, event: StandardProgressEvent) -> None:
                captured_events.append(event)

        callback = CustomCallback()
        assert isinstance(callback, ProgressCallback)

        # Test it works
        event = StandardProgressEvent(event_type=EventType.PROGRESS)
        callback.on_progress(event)
        assert len(captured_events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
