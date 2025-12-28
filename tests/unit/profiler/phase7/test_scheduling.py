"""Tests for incremental profiling scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pytest
import polars as pl

from truthound.profiler.scheduling import (
    # Triggers
    CronTrigger,
    IntervalTrigger,
    DataChangeTrigger,
    EventTrigger,
    CompositeTrigger,
    AlwaysTrigger,
    ManualTrigger,
    # Storage
    ProfileHistoryEntry,
    InMemoryProfileStorage,
    FileProfileStorage,
    # Scheduler
    SchedulerConfig,
    SchedulerMetrics,
    IncrementalProfileScheduler,
    create_scheduler,
)


class TestIntervalTrigger:
    """Tests for IntervalTrigger."""

    def test_should_run_on_first_call(self) -> None:
        """Test that trigger runs on first call (no last_run)."""
        trigger = IntervalTrigger(hours=1)

        assert trigger.should_run(None, {})

    def test_should_not_run_before_interval(self) -> None:
        """Test that trigger doesn't run before interval passes."""
        trigger = IntervalTrigger(hours=1)
        last_run = datetime.now() - timedelta(minutes=30)

        assert not trigger.should_run(last_run, {})

    def test_should_run_after_interval(self) -> None:
        """Test that trigger runs after interval passes."""
        trigger = IntervalTrigger(hours=1)
        last_run = datetime.now() - timedelta(hours=2)

        assert trigger.should_run(last_run, {})

    def test_get_next_run_time(self) -> None:
        """Test calculating next run time."""
        trigger = IntervalTrigger(hours=1)
        last_run = datetime.now()

        next_run = trigger.get_next_run_time(last_run)

        expected = last_run + timedelta(hours=1)
        assert next_run is not None
        assert abs((next_run - expected).total_seconds()) < 1


class TestCronTrigger:
    """Tests for CronTrigger."""

    def test_valid_cron_expression(self) -> None:
        """Test valid cron expression."""
        trigger = CronTrigger("0 2 * * *")  # Daily at 2 AM
        assert trigger.expression == "0 2 * * *"

    def test_invalid_cron_expression(self) -> None:
        """Test invalid cron expression."""
        with pytest.raises(ValueError):
            CronTrigger("invalid")

    def test_should_run_on_first_call(self) -> None:
        """Test that cron trigger runs on first call."""
        trigger = CronTrigger("0 * * * *")

        assert trigger.should_run(None, {})


class TestDataChangeTrigger:
    """Tests for DataChangeTrigger."""

    def test_should_run_on_first_call(self) -> None:
        """Test that trigger runs on first call."""
        trigger = DataChangeTrigger(change_threshold=0.05)

        assert trigger.should_run(None, {})

    def test_should_run_on_significant_change(self) -> None:
        """Test that trigger runs on significant data change."""
        trigger = DataChangeTrigger(change_threshold=0.05)
        last_run = datetime.now() - timedelta(minutes=5)

        context = {
            "row_count": 1000,
            "last_row_count": 900,  # 10% change
        }

        assert trigger.should_run(last_run, context)

    def test_should_not_run_on_small_change(self) -> None:
        """Test that trigger doesn't run on small change."""
        trigger = DataChangeTrigger(change_threshold=0.10)
        last_run = datetime.now() - timedelta(minutes=5)

        context = {
            "row_count": 1000,
            "last_row_count": 980,  # 2% change
        }

        assert not trigger.should_run(last_run, context)

    def test_should_run_on_hash_change(self) -> None:
        """Test that trigger runs when data hash changes."""
        trigger = DataChangeTrigger()
        last_run = datetime.now() - timedelta(minutes=5)

        context = {
            "data_hash": "abc123",
            "last_data_hash": "xyz789",
        }

        assert trigger.should_run(last_run, context)


class TestEventTrigger:
    """Tests for EventTrigger."""

    def test_signal_triggers_run(self) -> None:
        """Test that signaling triggers a run."""
        trigger = EventTrigger("data_updated")

        trigger.signal()
        assert trigger.should_run(None, {})

        # After running, should not trigger again
        assert not trigger.should_run(None, {})

    def test_context_triggers_run(self) -> None:
        """Test that context can trigger run."""
        trigger = EventTrigger("data_updated")

        context = {"event_triggered": True}
        assert trigger.should_run(None, context)


class TestCompositeTrigger:
    """Tests for CompositeTrigger."""

    def test_any_mode(self) -> None:
        """Test 'any' mode (OR logic)."""
        trigger1 = AlwaysTrigger()
        trigger2 = ManualTrigger()

        composite = CompositeTrigger(triggers=[trigger1, trigger2], mode="any")

        # AlwaysTrigger should trigger
        assert composite.should_run(None, {})

    def test_all_mode(self) -> None:
        """Test 'all' mode (AND logic)."""
        trigger1 = AlwaysTrigger()
        trigger2 = ManualTrigger()

        composite = CompositeTrigger(triggers=[trigger1, trigger2], mode="all")

        # ManualTrigger is not triggered, so composite should not trigger
        assert not composite.should_run(None, {})

        # Now trigger the manual one
        trigger2.trigger()
        assert composite.should_run(None, {})


class TestManualTrigger:
    """Tests for ManualTrigger."""

    def test_does_not_run_without_trigger(self) -> None:
        """Test that manual trigger doesn't run automatically."""
        trigger = ManualTrigger()

        assert not trigger.should_run(None, {})

    def test_runs_after_trigger(self) -> None:
        """Test that manual trigger runs after being triggered."""
        trigger = ManualTrigger()

        trigger.trigger()
        assert trigger.should_run(None, {})


class TestInMemoryProfileStorage:
    """Tests for InMemoryProfileStorage."""

    def test_save_and_get_last_profile(self) -> None:
        """Test saving and retrieving last profile."""
        storage = InMemoryProfileStorage()

        mock_profile = Mock()
        mock_profile.row_count = 100
        mock_profile.columns = ["a", "b"]

        profile_id = storage.save(mock_profile)

        assert profile_id is not None
        assert storage.get_last_profile() is mock_profile

    def test_get_last_run_time(self) -> None:
        """Test getting last run time."""
        storage = InMemoryProfileStorage()

        assert storage.get_last_run_time() is None

        mock_profile = Mock()
        mock_profile.row_count = 100
        mock_profile.columns = []
        storage.save(mock_profile)

        assert storage.get_last_run_time() is not None

    def test_list_profiles(self) -> None:
        """Test listing profiles."""
        storage = InMemoryProfileStorage()

        mock_profile1 = Mock()
        mock_profile1.row_count = 100
        mock_profile1.columns = []

        mock_profile2 = Mock()
        mock_profile2.row_count = 200
        mock_profile2.columns = []

        storage.save(mock_profile1)
        storage.save(mock_profile2)

        profiles = storage.list_profiles()
        assert len(profiles) == 2

    def test_max_profiles_limit(self) -> None:
        """Test that max profiles limit is enforced."""
        storage = InMemoryProfileStorage(max_profiles=2)

        for i in range(5):
            mock_profile = Mock()
            mock_profile.row_count = i * 100
            mock_profile.columns = []
            storage.save(mock_profile)

        profiles = storage.list_profiles()
        assert len(profiles) == 2

    def test_delete_profile(self) -> None:
        """Test deleting a profile."""
        storage = InMemoryProfileStorage()

        mock_profile = Mock()
        mock_profile.row_count = 100
        mock_profile.columns = []

        profile_id = storage.save(mock_profile)
        assert storage.delete_profile(profile_id)
        assert storage.get_profile(profile_id) is None


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics."""

    def test_record_run(self) -> None:
        """Test recording a run."""
        metrics = SchedulerMetrics()

        metrics.record_run(duration_ms=100.0, incremental=True)

        assert metrics.total_runs == 1
        assert metrics.incremental_runs == 1
        assert metrics.total_profile_time_ms == 100.0

    def test_record_skip(self) -> None:
        """Test recording a skip."""
        metrics = SchedulerMetrics()

        metrics.record_skip()

        assert metrics.skipped_runs == 1

    def test_to_dict(self) -> None:
        """Test converting metrics to dict."""
        metrics = SchedulerMetrics()
        metrics.record_run(100.0)

        data = metrics.to_dict()

        assert "total_runs" in data
        assert "average_run_time_ms" in data


class TestIncrementalProfileScheduler:
    """Tests for IncrementalProfileScheduler."""

    @pytest.fixture
    def sample_data(self) -> pl.LazyFrame:
        """Create sample data for testing."""
        return pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        }).lazy()

    def test_run_if_needed_first_time(self, sample_data: pl.LazyFrame) -> None:
        """Test run_if_needed on first run."""
        trigger = AlwaysTrigger()
        storage = InMemoryProfileStorage()
        config = SchedulerConfig(save_history=False)

        scheduler = IncrementalProfileScheduler(
            trigger=trigger,
            storage=storage,
            config=config,
        )

        with patch.object(scheduler, '_run_full') as mock_run:
            mock_profile = Mock()
            mock_run.return_value = mock_profile

            result = scheduler.run_if_needed(sample_data)

            assert result is mock_profile

    def test_run_if_needed_skips_when_not_triggered(self, sample_data: pl.LazyFrame) -> None:
        """Test that run_if_needed skips when not triggered."""
        trigger = ManualTrigger()  # Never triggers automatically
        storage = InMemoryProfileStorage()

        scheduler = IncrementalProfileScheduler(
            trigger=trigger,
            storage=storage,
        )

        result = scheduler.run_if_needed(sample_data)

        assert result is None
        assert scheduler.metrics.skipped_runs == 1

    def test_get_next_run_time(self) -> None:
        """Test getting next run time."""
        trigger = IntervalTrigger(hours=1)
        scheduler = IncrementalProfileScheduler(trigger=trigger)

        next_run = scheduler.get_next_run_time()

        assert next_run is not None

    def test_run_history(self, sample_data: pl.LazyFrame) -> None:
        """Test run history tracking."""
        trigger = AlwaysTrigger()
        storage = InMemoryProfileStorage()
        config = SchedulerConfig(save_history=True)

        scheduler = IncrementalProfileScheduler(
            trigger=trigger,
            storage=storage,
            config=config,
        )

        with patch.object(scheduler, '_run_full') as mock_run:
            mock_profile = Mock()
            mock_profile.row_count = 100
            mock_profile.columns = []
            mock_profile.source = "test"
            mock_run.return_value = mock_profile

            scheduler.run(sample_data, incremental=False)

        history = scheduler.get_run_history()
        assert len(history) == 1


class TestCreateScheduler:
    """Tests for create_scheduler factory."""

    def test_create_interval_scheduler(self) -> None:
        """Test creating interval-based scheduler."""
        scheduler = create_scheduler(
            trigger_type="interval",
            storage_type="memory",
            hours=1,
        )

        assert isinstance(scheduler, IncrementalProfileScheduler)

    def test_create_cron_scheduler(self) -> None:
        """Test creating cron-based scheduler."""
        scheduler = create_scheduler(
            trigger_type="cron",
            storage_type="memory",
            expression="0 2 * * *",
        )

        assert isinstance(scheduler, IncrementalProfileScheduler)

    def test_create_with_file_storage(self, tmp_path) -> None:
        """Test creating scheduler with file storage."""
        scheduler = create_scheduler(
            trigger_type="interval",
            storage_type="file",
            storage_path=str(tmp_path),
        )

        assert isinstance(scheduler, IncrementalProfileScheduler)

    def test_invalid_trigger_type(self) -> None:
        """Test that invalid trigger type raises error."""
        with pytest.raises(ValueError):
            create_scheduler(trigger_type="invalid")

    def test_invalid_storage_type(self) -> None:
        """Test that invalid storage type raises error."""
        with pytest.raises(ValueError):
            create_scheduler(storage_type="invalid")
