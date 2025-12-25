"""Tests for P1 profiler improvements.

Tests for:
- P1 #5: Progress callbacks
- P1 #6: Profile comparison/drift detection
- P1 #7: Incremental profiling
- P1 #8: Timeout settings
"""

from datetime import datetime, timedelta
import time

import polars as pl
import pytest

from truthound.profiler import (
    # Base types
    DataType,
    ColumnProfile,
    TableProfile,
    DistributionStats,
    profile_dataframe,
    # Progress
    ProgressStage,
    ProgressEvent,
    ProgressTracker,
    ProgressAggregator,
    ConsoleProgressReporter,
    create_progress_callback,
    # Comparison
    DriftType,
    DriftSeverity,
    ChangeDirection,
    DriftResult,
    ColumnComparison,
    ProfileComparison,
    DriftThresholds,
    ProfileComparator,
    compare_profiles,
    detect_drift,
    # Incremental
    ChangeReason,
    ColumnFingerprint,
    IncrementalConfig,
    IncrementalProfiler,
    ProfileMerger,
    profile_incrementally,
    # Timeout
    TimeoutAction,
    TimeoutConfig,
    TimeoutResult,
    TimeoutExecutor,
    DeadlineTracker,
    with_timeout,
    create_timeout_config,
)


# =============================================================================
# P1 #5: Progress Callback Tests
# =============================================================================


class TestProgressEvent:
    """Tests for ProgressEvent."""

    def test_create_event(self):
        """Test creating a progress event."""
        event = ProgressEvent(
            stage=ProgressStage.PROFILING_COLUMN,
            progress=0.5,
            message="Profiling column",
            column="test_col",
        )

        assert event.stage == ProgressStage.PROFILING_COLUMN
        assert event.progress == 0.5
        assert event.percent == 50.0
        assert event.column == "test_col"
        assert not event.is_complete

    def test_completed_event(self):
        """Test completed event detection."""
        completed = ProgressEvent(
            stage=ProgressStage.COMPLETED,
            progress=1.0,
        )
        failed = ProgressEvent(
            stage=ProgressStage.FAILED,
            progress=0.5,
        )

        assert completed.is_complete
        assert failed.is_complete


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_basic_tracking(self):
        """Test basic progress tracking."""
        tracker = ProgressTracker()
        events = []
        tracker.on_progress(lambda e: events.append(e))

        tracker.start(total_columns=3)
        tracker.column_start("col1")
        tracker.column_complete("col1")
        tracker.column_start("col2")
        tracker.column_complete("col2")
        tracker.complete()

        assert len(events) > 0
        assert events[-1].stage == ProgressStage.COMPLETED

    def test_column_callbacks(self):
        """Test column-specific callbacks."""
        tracker = ProgressTracker()
        column_updates = []

        tracker.on_column(lambda col, pct: column_updates.append((col, pct)))

        tracker.start(total_columns=2)
        tracker.column_start("col1")
        tracker.column_progress("col1", 0.5)
        tracker.column_complete("col1")

        assert ("col1", 0.0) in column_updates
        assert ("col1", 50.0) in column_updates
        assert ("col1", 100.0) in column_updates

    def test_completion_callback(self):
        """Test completion callback."""
        tracker = ProgressTracker()
        completion_times = []

        tracker.on_complete(lambda secs: completion_times.append(secs))

        tracker.start(total_columns=1)
        tracker.column_start("col1")
        tracker.column_complete("col1")
        tracker.complete()

        assert len(completion_times) == 1
        assert completion_times[0] >= 0


class TestProgressAggregator:
    """Tests for ProgressAggregator."""

    def test_aggregate_progress(self):
        """Test aggregating progress from multiple sources."""
        aggregator = ProgressAggregator(total_items=4)

        aggregator.update("item1", 0.5)
        aggregator.update("item2", 0.25)
        assert aggregator.get_progress() == pytest.approx(0.1875, rel=0.01)

        aggregator.complete("item1")
        aggregator.complete("item2")
        assert aggregator.get_progress() == pytest.approx(0.5, rel=0.01)


# =============================================================================
# P1 #6: Profile Comparison Tests
# =============================================================================


class TestDriftResult:
    """Tests for DriftResult."""

    def test_create_drift_result(self):
        """Test creating a drift result."""
        drift = DriftResult(
            drift_type=DriftType.COMPLETENESS,
            severity=DriftSeverity.WARNING,
            column="test_col",
            metric="null_ratio",
            old_value=0.1,
            new_value=0.2,
            change_ratio=0.1,
            direction=ChangeDirection.INCREASED,
            message="Null ratio increased",
        )

        assert drift.drift_type == DriftType.COMPLETENESS
        assert drift.severity == DriftSeverity.WARNING
        assert drift.column == "test_col"

    def test_to_dict(self):
        """Test serialization."""
        drift = DriftResult(
            drift_type=DriftType.SCHEMA,
            severity=DriftSeverity.CRITICAL,
            column="col1",
            metric="type",
            old_value="Int64",
            new_value="String",
        )

        d = drift.to_dict()
        assert d["drift_type"] == "schema"
        assert d["severity"] == "critical"


class TestDriftThresholds:
    """Tests for DriftThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = DriftThresholds()

        assert thresholds.null_ratio_warning == 0.05
        assert thresholds.null_ratio_critical == 0.20

    def test_strict_thresholds(self):
        """Test strict thresholds."""
        strict = DriftThresholds.strict()

        assert strict.null_ratio_warning < DriftThresholds().null_ratio_warning

    def test_loose_thresholds(self):
        """Test loose thresholds."""
        loose = DriftThresholds.loose()

        assert loose.null_ratio_warning > DriftThresholds().null_ratio_warning


class TestProfileComparator:
    """Tests for ProfileComparator."""

    def test_compare_identical_profiles(self):
        """Test comparing identical profiles."""
        profile = TableProfile(
            name="test",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64", null_ratio=0.1),
                ColumnProfile(name="col2", physical_type="String", null_ratio=0.0),
            ),
            profiled_at=datetime.now(),
        )

        comparison = compare_profiles(profile, profile)

        assert not comparison.has_drift or all(
            d.severity == DriftSeverity.INFO for d in comparison.all_drifts
        )

    def test_detect_schema_change(self):
        """Test detecting schema changes."""
        old = TableProfile(
            name="old",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64"),
                ColumnProfile(name="col2", physical_type="String"),
            ),
            profiled_at=datetime.now(),
        )

        new = TableProfile(
            name="new",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(name="col1", physical_type="String"),  # Type changed
                ColumnProfile(name="col3", physical_type="Float64"),  # New column
            ),
            profiled_at=datetime.now(),
        )

        comparison = compare_profiles(old, new)

        assert comparison.has_schema_changes
        schema_drifts = comparison.get_by_type(DriftType.SCHEMA)
        assert len(schema_drifts) >= 2  # Type change + removed + added

    def test_detect_null_ratio_drift(self):
        """Test detecting null ratio drift."""
        old = TableProfile(
            name="old",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64", null_ratio=0.05),
            ),
            profiled_at=datetime.now(),
        )

        new = TableProfile(
            name="new",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64", null_ratio=0.30),
            ),
            profiled_at=datetime.now(),
        )

        comparison = compare_profiles(old, new)

        completeness_drifts = comparison.get_by_type(DriftType.COMPLETENESS)
        assert len(completeness_drifts) > 0
        assert any(d.severity == DriftSeverity.CRITICAL for d in completeness_drifts)

    def test_comparison_report(self):
        """Test generating comparison report."""
        old = TableProfile(
            name="baseline",
            row_count=1000,
            column_count=1,
            columns=(
                ColumnProfile(name="value", physical_type="Float64", null_ratio=0.1),
            ),
            profiled_at=datetime.now() - timedelta(days=1),
        )

        new = TableProfile(
            name="current",
            row_count=1000,
            column_count=1,
            columns=(
                ColumnProfile(name="value", physical_type="Float64", null_ratio=0.35),
            ),
            profiled_at=datetime.now(),
        )

        comparison = compare_profiles(old, new)
        report = comparison.to_report()

        assert "PROFILE COMPARISON REPORT" in report
        assert "baseline" in report or "current" in report


class TestDetectDrift:
    """Tests for detect_drift convenience function."""

    def test_filter_by_severity(self):
        """Test filtering drifts by severity."""
        old = TableProfile(
            name="old",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64", null_ratio=0.0),
            ),
            profiled_at=datetime.now(),
        )

        new = TableProfile(
            name="new",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64", null_ratio=0.25),
            ),
            profiled_at=datetime.now(),
        )

        # Get only critical drifts
        critical_drifts = detect_drift(old, new, min_severity=DriftSeverity.CRITICAL)

        # All returned drifts should be critical
        for drift in critical_drifts:
            assert drift.severity == DriftSeverity.CRITICAL


# =============================================================================
# P1 #7: Incremental Profiling Tests
# =============================================================================


class TestColumnFingerprint:
    """Tests for ColumnFingerprint."""

    def test_create_fingerprint(self):
        """Test creating a column fingerprint."""
        fp = ColumnFingerprint(
            column_name="test",
            dtype="Int64",
            row_count=1000,
            null_count=10,
            sample_hash="abc123",
        )

        assert fp.column_name == "test"
        assert fp.row_count == 1000

    def test_fingerprint_serialization(self):
        """Test fingerprint serialization."""
        fp = ColumnFingerprint(
            column_name="test",
            dtype="Int64",
            row_count=100,
            null_count=5,
            sample_hash="xyz789",
        )

        d = fp.to_dict()
        restored = ColumnFingerprint.from_dict(d)

        assert restored.column_name == fp.column_name
        assert restored.sample_hash == fp.sample_hash


class TestIncrementalConfig:
    """Tests for IncrementalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IncrementalConfig()

        assert config.max_age is None
        assert config.sample_size == 1000

    def test_aggressive_config(self):
        """Test aggressive configuration."""
        config = IncrementalConfig.aggressive()

        assert config.max_age is not None
        assert config.sample_size < 1000

    def test_conservative_config(self):
        """Test conservative configuration."""
        config = IncrementalConfig.conservative()

        assert config.max_age is not None
        assert config.sample_size > 1000


class TestIncrementalProfiler:
    """Tests for IncrementalProfiler."""

    def test_first_run_profiles_all(self):
        """Test that first run profiles all columns."""
        df = pl.DataFrame({
            "col1": range(100),
            "col2": ["a"] * 100,
        })

        profiler = IncrementalProfiler()
        profile = profiler.profile(df, name="test")

        assert len(profiler.last_profiled_columns) == 2
        assert len(profiler.last_skipped_columns) == 0

    def test_incremental_skips_unchanged(self):
        """Test that incremental profiling skips unchanged columns."""
        df = pl.DataFrame({
            "col1": range(100),
            "col2": ["a"] * 100,
        })

        profiler = IncrementalProfiler()

        # First run
        profile1 = profiler.profile(df, name="test")

        # Second run with same data
        profile2 = profiler.profile(df, name="test", previous=profile1)

        # Should skip columns that haven't changed
        assert len(profiler.last_skipped_columns) > 0

    def test_force_refresh(self):
        """Test force refresh profiles all columns."""
        df = pl.DataFrame({
            "col1": range(100),
        })

        profiler = IncrementalProfiler()
        profile1 = profiler.profile(df, name="test")

        # Force refresh
        profile2 = profiler.profile(df, name="test", previous=profile1, force_refresh=True)

        assert "col1" in profiler.last_profiled_columns
        assert profiler.last_change_reasons.get("col1") == ChangeReason.FORCED


class TestProfileMerger:
    """Tests for ProfileMerger."""

    def test_merge_profiles(self):
        """Test merging multiple profiles."""
        profile1 = TableProfile(
            name="p1",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(name="col1", physical_type="Int64"),
            ),
            profiled_at=datetime.now() - timedelta(hours=1),
        )

        profile2 = TableProfile(
            name="p2",
            row_count=200,
            column_count=1,
            columns=(
                ColumnProfile(name="col2", physical_type="String"),
            ),
            profiled_at=datetime.now(),
        )

        merger = ProfileMerger()
        merged = merger.merge([profile1, profile2], name="merged")

        assert merged.name == "merged"
        assert merged.column_count == 2


# =============================================================================
# P1 #8: Timeout Settings Tests
# =============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_default_config(self):
        """Test default timeout configuration."""
        config = TimeoutConfig()

        assert config.column_timeout == timedelta(seconds=60)
        assert config.analyzer_timeout == timedelta(seconds=10)

    def test_strict_config(self):
        """Test strict timeout configuration."""
        config = TimeoutConfig.strict()

        assert config.table_timeout is not None
        assert config.default_action == TimeoutAction.FAIL

    def test_no_timeout_config(self):
        """Test no-timeout configuration."""
        config = TimeoutConfig.no_timeout()

        assert config.table_timeout is None
        assert config.column_timeout is None


class TestTimeoutResult:
    """Tests for TimeoutResult."""

    def test_ok_result(self):
        """Test successful result."""
        result = TimeoutResult.ok(42, elapsed=1.5)

        assert result.success
        assert result.value == 42
        assert not result.timed_out
        assert result.elapsed_seconds == 1.5

    def test_timeout_result(self):
        """Test timeout result."""
        result = TimeoutResult.timeout(10.0, retries=2)

        assert not result.success
        assert result.timed_out
        assert result.retries == 2

    def test_failure_result(self):
        """Test failure result."""
        error = ValueError("test error")
        result = TimeoutResult.failure(error, elapsed=0.5)

        assert not result.success
        assert not result.timed_out
        assert result.error == error


class TestTimeoutExecutor:
    """Tests for TimeoutExecutor."""

    def test_successful_execution(self):
        """Test successful execution within timeout."""
        executor = TimeoutExecutor()

        result = executor.run(
            lambda: 42,
            timeout=5.0,
        )

        assert result.success
        assert result.value == 42

    def test_timeout_on_slow_function(self):
        """Test timeout on slow function."""
        executor = TimeoutExecutor()

        def slow_func():
            time.sleep(10)
            return 42

        result = executor.run(
            slow_func,
            timeout=0.1,
        )

        assert not result.success
        assert result.timed_out

    def test_exception_handling(self):
        """Test exception handling."""
        executor = TimeoutExecutor()

        def failing_func():
            raise ValueError("test error")

        result = executor.run(
            failing_func,
            timeout=5.0,
        )

        assert not result.success
        assert not result.timed_out
        assert isinstance(result.error, ValueError)


class TestDeadlineTracker:
    """Tests for DeadlineTracker."""

    def test_deadline_tracking(self):
        """Test deadline tracking."""
        tracker = DeadlineTracker(total_seconds=1.0)

        assert not tracker.is_expired
        assert tracker.remaining_seconds > 0
        assert tracker.progress < 1.0

    def test_deadline_expiration(self):
        """Test deadline expiration."""
        tracker = DeadlineTracker(total_seconds=0.1)

        time.sleep(0.15)

        assert tracker.is_expired
        assert tracker.remaining_seconds == 0


class TestWithTimeout:
    """Tests for with_timeout convenience function."""

    def test_successful_execution(self):
        """Test successful execution."""
        result = with_timeout(
            lambda: 42,
            seconds=5.0,
            default=None,
        )

        assert result == 42

    def test_timeout_returns_default(self):
        """Test that timeout returns default value."""
        result = with_timeout(
            lambda: time.sleep(10) or 42,
            seconds=0.1,
            default=-1,
        )

        assert result == -1


# =============================================================================
# Integration Tests
# =============================================================================


class TestP1Integration:
    """Integration tests for P1 improvements."""

    def test_progress_with_profiling(self):
        """Test progress tracking with actual profiling."""
        df = pl.DataFrame({
            "col1": range(100),
            "col2": [f"value_{i}" for i in range(100)],
        })

        column_progress = []

        tracker = create_progress_callback(
            on_column=lambda col, pct: column_progress.append((col, pct)),
        )

        # Note: Would need to integrate tracker with profiler
        # For now, test that tracker works standalone
        tracker.start(total_columns=2)
        tracker.column_start("col1")
        tracker.column_complete("col1")
        tracker.column_start("col2")
        tracker.column_complete("col2")
        tracker.complete()

        assert len(column_progress) > 0

    def test_comparison_after_data_change(self):
        """Test comparison detects changes after data modification."""
        df1 = pl.DataFrame({
            "value": [1.0, 2.0, 3.0, None],
        })

        df2 = pl.DataFrame({
            "value": [1.0, 2.0, 3.0, None, None, None],  # More nulls
        })

        profile1 = profile_dataframe(df1, name="before")
        profile2 = profile_dataframe(df2, name="after")

        comparison = compare_profiles(profile1, profile2)

        # Should detect changes in null ratio
        assert comparison.has_drift

    def test_incremental_with_timeout(self):
        """Test incremental profiling with timeout config."""
        df = pl.DataFrame({
            "col1": range(100),
        })

        config = IncrementalConfig()
        profiler = IncrementalProfiler(config=config)

        # Should complete without timeout
        profile = profiler.profile(df, name="test")

        assert profile.row_count == 100
