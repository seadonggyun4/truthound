"""Tests for DistributedProgressAggregator."""

import pytest

from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    MonitorEvent,
    MonitorEventType,
)
from truthound.profiler.distributed.monitoring.progress_aggregator import (
    DistributedProgressAggregator,
    StreamingProgressAggregator,
)


class TestDistributedProgressAggregator:
    """Tests for DistributedProgressAggregator."""

    def test_set_total_partitions(self) -> None:
        """Test setting total partitions."""
        aggregator = DistributedProgressAggregator()

        aggregator.set_total_partitions(10)

        progress = aggregator.get_progress()
        assert progress.total_partitions == 10
        assert progress.completed_partitions == 0

    def test_update_partition(self) -> None:
        """Test partition update."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(10)

        aggregator.update_partition(0, progress=0.5, rows_processed=500)

        progress = aggregator.get_progress()
        assert progress.partition_progress[0] == 0.5
        assert progress.processed_rows == 500

    def test_complete_partition(self) -> None:
        """Test partition completion."""
        aggregator = DistributedProgressAggregator()
        aggregator.set_total_partitions(10)
        aggregator.start_partition(0, total_rows=1000)

        aggregator.complete_partition(0, rows_processed=1000)

        progress = aggregator.get_progress()
        assert progress.completed_partitions == 1
        assert progress.partition_progress[0] == 1.0

    def test_fail_partition(self) -> None:
        """Test partition failure."""
        aggregator = DistributedProgressAggregator()
        aggregator.set_total_partitions(10)
        aggregator.start_partition(0)

        aggregator.fail_partition(0, error="Test error")

        progress = aggregator.get_progress()
        assert progress.failed_partitions == 1

    def test_overall_progress_calculation(self) -> None:
        """Test overall progress calculation."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(4)

        # Complete 2 partitions
        aggregator.complete_partition(0)
        aggregator.complete_partition(1)

        # Update 1 partition to 50%
        aggregator.update_partition(2, progress=0.5)

        # Leave 1 partition at 0%

        progress = aggregator.get_progress()
        # (1.0 + 1.0 + 0.5 + 0.0) / 4 = 0.625
        assert progress.overall_progress == pytest.approx(0.625, rel=0.01)

    def test_rows_accumulation(self) -> None:
        """Test rows accumulation."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(2)

        aggregator.update_partition(0, progress=0.5, rows_processed=500)
        aggregator.update_partition(1, progress=0.3, rows_processed=300)

        progress = aggregator.get_progress()
        assert progress.processed_rows == 800

    def test_progress_callback(self) -> None:
        """Test progress callback."""
        progress_updates: list[AggregatedProgress] = []
        aggregator = DistributedProgressAggregator(
            on_progress=lambda p: progress_updates.append(p),
            emit_interval_seconds=0,
        )
        aggregator.set_total_partitions(2)

        aggregator.complete_partition(0)

        assert len(progress_updates) >= 1

    def test_event_callback(self) -> None:
        """Test event callback."""
        events: list[MonitorEvent] = []
        aggregator = DistributedProgressAggregator(
            on_event=lambda e: events.append(e),
            emit_interval_seconds=0,
        )
        aggregator.set_total_partitions(2)

        aggregator.start_partition(0)
        aggregator.complete_partition(0)

        # Should have start and complete events
        event_types = [e.event_type for e in events]
        assert MonitorEventType.PARTITION_START in event_types
        assert MonitorEventType.PARTITION_COMPLETE in event_types

    def test_milestone_events(self) -> None:
        """Test milestone events."""
        events: list[MonitorEvent] = []
        aggregator = DistributedProgressAggregator(
            on_event=lambda e: events.append(e),
            milestone_interval=25,  # Every 25%
            emit_interval_seconds=0,
        )
        aggregator.set_total_partitions(4)

        # Complete 1/4 = 25%
        aggregator.complete_partition(0)

        milestone_events = [
            e for e in events if e.event_type == MonitorEventType.PROGRESS_MILESTONE
        ]
        assert len(milestone_events) >= 1

    def test_completion_event(self) -> None:
        """Test completion event."""
        events: list[MonitorEvent] = []
        aggregator = DistributedProgressAggregator(
            on_event=lambda e: events.append(e),
        )
        aggregator.set_total_partitions(2)

        aggregator.complete_partition(0)
        aggregator.complete_partition(1)

        complete_events = [
            e for e in events if e.event_type == MonitorEventType.AGGREGATION_COMPLETE
        ]
        assert len(complete_events) == 1

    def test_completion_with_failures(self) -> None:
        """Test completion with failures."""
        events: list[MonitorEvent] = []
        aggregator = DistributedProgressAggregator(
            on_event=lambda e: events.append(e),
        )
        aggregator.set_total_partitions(2)

        aggregator.complete_partition(0)
        aggregator.fail_partition(1, error="Error")

        progress = aggregator.get_progress()
        assert progress.is_complete
        assert progress.success_rate == 0.5

        complete_events = [
            e for e in events if e.event_type == MonitorEventType.AGGREGATION_COMPLETE
        ]
        assert len(complete_events) == 1

    def test_get_partition_progress(self) -> None:
        """Test getting individual partition progress."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(2)
        aggregator.update_partition(0, progress=0.75)

        assert aggregator.get_partition_progress(0) == 0.75
        assert aggregator.get_partition_progress(1) == 0.0
        assert aggregator.get_partition_progress(999) == 0.0  # Unknown

    def test_get_slowest_partitions(self) -> None:
        """Test getting slowest partitions."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(5)

        aggregator.update_partition(0, progress=0.9)
        aggregator.update_partition(1, progress=0.1)
        aggregator.update_partition(2, progress=0.5)
        aggregator.update_partition(3, progress=0.3)
        aggregator.complete_partition(4)

        slowest = aggregator.get_slowest_partitions(3)
        assert len(slowest) == 3
        # Should be sorted by progress ascending
        assert slowest[0][1] <= slowest[1][1] <= slowest[2][1]

    def test_get_failed_partitions(self) -> None:
        """Test getting failed partitions."""
        aggregator = DistributedProgressAggregator()
        aggregator.set_total_partitions(3)

        aggregator.complete_partition(0)
        aggregator.fail_partition(1, error="Error 1")
        aggregator.fail_partition(2, error="Error 2")

        failed = aggregator.get_failed_partitions()
        assert len(failed) == 2
        assert (1, "Error 1") in failed
        assert (2, "Error 2") in failed

    def test_reset(self) -> None:
        """Test reset."""
        aggregator = DistributedProgressAggregator()
        aggregator.set_total_partitions(10)
        aggregator.complete_partition(0)
        aggregator.complete_partition(1)

        aggregator.reset()

        progress = aggregator.get_progress()
        assert progress.total_partitions == 0
        assert progress.completed_partitions == 0

    def test_estimated_remaining(self) -> None:
        """Test ETA estimation."""
        aggregator = DistributedProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(4)

        # Complete half
        aggregator.complete_partition(0)
        aggregator.complete_partition(1)

        progress = aggregator.get_progress()
        # ETA should be available if elapsed > 0 and progress > 0
        # Depends on timing, so just check it doesn't error
        assert progress.estimated_remaining_seconds is None or progress.estimated_remaining_seconds >= 0

    def test_total_rows_tracking(self) -> None:
        """Test total rows tracking."""
        aggregator = DistributedProgressAggregator()
        aggregator.set_total_partitions(3)

        aggregator.set_partition_rows(0, 1000)
        aggregator.set_partition_rows(1, 2000)
        aggregator.set_partition_rows(2, 1500)

        progress = aggregator.get_progress()
        assert progress.total_rows == 4500


class TestStreamingProgressAggregator:
    """Tests for StreamingProgressAggregator."""

    def test_only_increasing_progress(self) -> None:
        """Test that only increasing progress is accepted."""
        aggregator = StreamingProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(2)

        # First update
        aggregator.update_partition(0, progress=0.5)
        assert aggregator.get_partition_progress(0) == 0.5

        # Higher progress should work
        aggregator.update_partition(0, progress=0.7)
        assert aggregator.get_partition_progress(0) == 0.7

        # Lower progress should be ignored
        aggregator.update_partition(0, progress=0.3)
        assert aggregator.get_partition_progress(0) == 0.7

    def test_reset_clears_reported_progress(self) -> None:
        """Test that reset clears reported progress tracking."""
        aggregator = StreamingProgressAggregator(emit_interval_seconds=0)
        aggregator.set_total_partitions(2)

        aggregator.update_partition(0, progress=0.9)
        aggregator.reset()

        # After reset, should accept lower progress
        aggregator.set_total_partitions(2)
        aggregator.update_partition(0, progress=0.1)
        assert aggregator.get_partition_progress(0) == 0.1
