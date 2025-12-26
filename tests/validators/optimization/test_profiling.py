"""Tests for validator performance profiling framework.

Tests cover:
- TimingMetrics statistical calculations
- MemoryMetrics tracking
- ThroughputMetrics calculations
- ValidatorMetrics aggregation
- ProfilerConfig modes
- ProfileContext lifecycle
- ValidatorProfiler sessions and metrics
- ProfilingSession management
- ExecutionSnapshot recording
- Decorators and context managers
- ProfilingReport generation
- Prometheus export format
- Thread safety
"""

import gc
import json
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.validators.optimization.profiling import (
    # Enums
    MetricType,
    ProfilerMode,
    # Data classes
    TimingMetrics,
    MemoryMetrics,
    ThroughputMetrics,
    ValidatorMetrics,
    ExecutionSnapshot,
    ProfilingSession,
    ProfilerConfig,
    ProfileContext,
    # Main classes
    ValidatorProfiler,
    MemoryTracker,
    ProfilingReport,
    # Convenience functions
    get_default_profiler,
    set_default_profiler,
    reset_default_profiler,
    profile_validator,
    profiled,
    # Constants
    DEFAULT_TIMING_BUCKETS_MS,
    MEMORY_WARNING_THRESHOLD_MB,
    MEMORY_CRITICAL_THRESHOLD_MB,
    MAX_HISTORY_ENTRIES,
)
from truthound.types import Severity


# =============================================================================
# Test Fixtures
# =============================================================================

class MockValidator(Validator):
    """Mock validator for testing."""
    name = "mock_validator"
    category = "test"

    def __init__(self, delay_ms: float = 0, issues_count: int = 0):
        super().__init__()
        self.delay_ms = delay_ms
        self.issues_count = issues_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)
        return [
            ValidationIssue(
                column="col",
                issue_type="test_issue",
                count=1,
                severity=Severity.LOW,
            )
            for _ in range(self.issues_count)
        ]


class SlowValidator(Validator):
    """Slow validator for timing tests."""
    name = "slow_validator"
    category = "performance_test"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        time.sleep(0.005)  # 5ms delay
        return []


@pytest.fixture
def sample_lazyframe():
    """Create a sample LazyFrame for testing."""
    return pl.DataFrame({
        "id": range(1000),
        "value": [i * 10 for i in range(1000)],
    }).lazy()


@pytest.fixture
def profiler():
    """Create a fresh profiler for each test."""
    return ValidatorProfiler()


@pytest.fixture(autouse=True)
def reset_global_profiler():
    """Reset the global profiler before and after each test."""
    reset_default_profiler()
    yield
    reset_default_profiler()


# =============================================================================
# TimingMetrics Tests
# =============================================================================

class TestTimingMetrics:
    """Tests for TimingMetrics class."""

    def test_empty_metrics(self):
        """Test empty metrics returns zeros."""
        metrics = TimingMetrics()

        assert metrics.count == 0
        assert metrics.total_ms == 0.0
        assert metrics.mean_ms == 0.0
        assert metrics.median_ms == 0.0
        assert metrics.std_ms == 0.0
        assert metrics.min_ms == 0.0
        assert metrics.max_ms == 0.0
        assert metrics.p50_ms == 0.0
        assert metrics.p95_ms == 0.0
        assert metrics.p99_ms == 0.0

    def test_single_duration(self):
        """Test single duration observation."""
        metrics = TimingMetrics()
        metrics.add(100.0)

        assert metrics.count == 1
        assert metrics.total_ms == 100.0
        assert metrics.mean_ms == 100.0
        assert metrics.min_ms == 100.0
        assert metrics.max_ms == 100.0
        assert metrics.std_ms == 0.0  # Single value has no std

    def test_multiple_durations(self):
        """Test multiple duration observations."""
        metrics = TimingMetrics()
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]
        for d in durations:
            metrics.add(d)

        assert metrics.count == 5
        assert metrics.total_ms == 150.0
        assert metrics.mean_ms == 30.0
        assert metrics.median_ms == 30.0
        assert metrics.min_ms == 10.0
        assert metrics.max_ms == 50.0

    def test_percentiles(self):
        """Test percentile calculations."""
        metrics = TimingMetrics()
        # Add 100 values from 1 to 100
        for i in range(1, 101):
            metrics.add(float(i))

        # Percentiles use index-based calculation, so allow for rounding
        assert 49.0 <= metrics.p50_ms <= 51.0
        assert 89.0 <= metrics.p90_ms <= 91.0
        assert 94.0 <= metrics.p95_ms <= 96.0
        assert 98.0 <= metrics.p99_ms <= 100.0

    def test_trimming(self):
        """Test that old entries are trimmed."""
        metrics = TimingMetrics()
        for i in range(MAX_HISTORY_ENTRIES + 100):
            metrics.add(float(i))

        assert len(metrics.durations_ms) == MAX_HISTORY_ENTRIES
        # Should have trimmed oldest, keeping newest
        assert metrics.durations_ms[0] == 100.0  # First trimmed entry

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = TimingMetrics()
        metrics.add(10.0)
        metrics.add(20.0)

        result = metrics.to_dict()

        assert "count" in result
        assert "total_ms" in result
        assert "mean_ms" in result
        assert "median_ms" in result
        assert "std_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert result["count"] == 2


# =============================================================================
# MemoryMetrics Tests
# =============================================================================

class TestMemoryMetrics:
    """Tests for MemoryMetrics class."""

    def test_empty_metrics(self):
        """Test empty metrics returns zeros."""
        metrics = MemoryMetrics()

        assert metrics.count == 0
        assert metrics.mean_peak_mb == 0.0
        assert metrics.max_peak_mb == 0.0
        assert metrics.mean_delta_mb == 0.0
        assert metrics.total_gc_collections == 0

    def test_add_observation(self):
        """Test adding memory observations."""
        metrics = MemoryMetrics()

        # Add 100MB peak, 10MB delta, 5 GC collections
        metrics.add(100 * 1024 * 1024, 10 * 1024 * 1024, 5)

        assert metrics.count == 1
        assert metrics.mean_peak_mb == pytest.approx(100.0, rel=0.01)
        assert metrics.max_peak_mb == pytest.approx(100.0, rel=0.01)
        assert metrics.mean_delta_mb == pytest.approx(10.0, rel=0.01)
        assert metrics.total_gc_collections == 5

    def test_multiple_observations(self):
        """Test multiple memory observations."""
        metrics = MemoryMetrics()

        metrics.add(100 * 1024 * 1024, 10 * 1024 * 1024, 2)
        metrics.add(200 * 1024 * 1024, 20 * 1024 * 1024, 3)

        assert metrics.count == 2
        assert metrics.mean_peak_mb == pytest.approx(150.0, rel=0.01)
        assert metrics.max_peak_mb == pytest.approx(200.0, rel=0.01)
        assert metrics.mean_delta_mb == pytest.approx(15.0, rel=0.01)
        assert metrics.total_gc_collections == 5

    def test_trimming(self):
        """Test that old entries are trimmed."""
        metrics = MemoryMetrics()
        for i in range(MAX_HISTORY_ENTRIES + 100):
            metrics.add(i * 1024, i * 512, 1)

        assert len(metrics.peak_bytes) == MAX_HISTORY_ENTRIES
        assert len(metrics.delta_bytes) == MAX_HISTORY_ENTRIES
        assert len(metrics.gc_collections) == MAX_HISTORY_ENTRIES


# =============================================================================
# ThroughputMetrics Tests
# =============================================================================

class TestThroughputMetrics:
    """Tests for ThroughputMetrics class."""

    def test_empty_metrics(self):
        """Test empty metrics returns zeros."""
        metrics = ThroughputMetrics()

        assert metrics.count == 0
        assert metrics.total_rows == 0
        assert metrics.mean_rows_per_sec == 0.0

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = ThroughputMetrics()

        # 1000 rows in 100ms = 10,000 rows/sec
        metrics.add(1000, 100.0)

        assert metrics.count == 1
        assert metrics.total_rows == 1000
        assert metrics.mean_rows_per_sec == pytest.approx(10000.0, rel=0.01)

    def test_multiple_observations(self):
        """Test multiple throughput observations."""
        metrics = ThroughputMetrics()

        # 1000 rows in 100ms
        metrics.add(1000, 100.0)
        # 2000 rows in 200ms
        metrics.add(2000, 200.0)

        assert metrics.count == 2
        assert metrics.total_rows == 3000
        # 3000 rows in 300ms = 10,000 rows/sec
        assert metrics.mean_rows_per_sec == pytest.approx(10000.0, rel=0.01)


# =============================================================================
# ValidatorMetrics Tests
# =============================================================================

class TestValidatorMetrics:
    """Tests for ValidatorMetrics class."""

    def test_initial_state(self):
        """Test initial validator metrics state."""
        metrics = ValidatorMetrics(
            validator_name="test_validator",
            validator_category="test",
        )

        assert metrics.validator_name == "test_validator"
        assert metrics.validator_category == "test"
        assert metrics.execution_count == 0
        assert metrics.total_issues == 0
        assert metrics.mean_issues == 0.0
        assert metrics.error_counts == 0
        assert metrics.last_execution is None

    def test_record_execution(self):
        """Test recording an execution."""
        metrics = ValidatorMetrics(
            validator_name="test_validator",
            validator_category="test",
        )

        metrics.record_execution(
            duration_ms=100.0,
            issue_count=5,
            rows_processed=1000,
            peak_memory=100 * 1024 * 1024,
            memory_delta=10 * 1024 * 1024,
            gc_collections=2,
        )

        assert metrics.execution_count == 1
        assert metrics.timing.count == 1
        assert metrics.timing.mean_ms == 100.0
        assert metrics.total_issues == 5
        assert metrics.throughput.count == 1
        assert metrics.memory.count == 1
        assert metrics.last_execution is not None

    def test_record_error(self):
        """Test recording an error execution."""
        metrics = ValidatorMetrics(
            validator_name="test_validator",
            validator_category="test",
        )

        metrics.record_execution(duration_ms=50.0, error=True)

        assert metrics.execution_count == 1
        assert metrics.error_counts == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = ValidatorMetrics(
            validator_name="test_validator",
            validator_category="test",
        )
        metrics.record_execution(duration_ms=100.0, issue_count=3)

        result = metrics.to_dict()

        assert result["validator_name"] == "test_validator"
        assert result["validator_category"] == "test"
        assert result["execution_count"] == 1
        assert result["total_issues"] == 3
        assert "timing" in result
        assert "memory" in result
        assert "throughput" in result


# =============================================================================
# ProfilerConfig Tests
# =============================================================================

class TestProfilerConfig:
    """Tests for ProfilerConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProfilerConfig()

        assert config.mode == ProfilerMode.STANDARD
        assert config.track_memory is True
        assert config.track_gc is True
        assert config.track_throughput is True
        assert config.record_snapshots is False

    def test_disabled_config(self):
        """Test disabled configuration."""
        config = ProfilerConfig.disabled()

        assert config.mode == ProfilerMode.DISABLED

    def test_basic_config(self):
        """Test basic configuration."""
        config = ProfilerConfig.basic()

        assert config.mode == ProfilerMode.BASIC
        assert config.track_memory is False
        assert config.track_gc is False

    def test_detailed_config(self):
        """Test detailed configuration."""
        config = ProfilerConfig.detailed()

        assert config.mode == ProfilerMode.DETAILED
        assert config.record_snapshots is True

    def test_diagnostic_config(self):
        """Test diagnostic configuration."""
        config = ProfilerConfig.diagnostic()

        assert config.mode == ProfilerMode.DIAGNOSTIC
        assert config.record_snapshots is True
        assert config.max_snapshots == 10000


# =============================================================================
# ProfileContext Tests
# =============================================================================

class TestProfileContext:
    """Tests for ProfileContext class."""

    def test_context_lifecycle(self):
        """Test context start/stop lifecycle."""
        config = ProfilerConfig.basic()
        ctx = ProfileContext(
            validator_name="test",
            validator_category="test",
            config=config,
        )

        ctx.start()
        time.sleep(0.01)  # 10ms
        ctx.stop()

        assert ctx._completed is True
        assert ctx.metrics is not None
        assert ctx.metrics.timing.count == 1
        assert ctx.duration_ms >= 10.0

    def test_set_issue_count(self):
        """Test setting issue count."""
        config = ProfilerConfig.basic()
        ctx = ProfileContext(
            validator_name="test",
            validator_category="test",
            config=config,
        )

        ctx.start()
        ctx.set_issue_count(10)
        ctx.stop()

        assert ctx._issue_count == 10
        assert ctx.metrics is not None
        assert ctx.metrics.total_issues == 10

    def test_set_error(self):
        """Test marking error."""
        config = ProfilerConfig.basic()
        ctx = ProfileContext(
            validator_name="test",
            validator_category="test",
            config=config,
        )

        ctx.start()
        ctx.set_error("Test error message")
        ctx.stop()

        assert ctx._error is True
        assert ctx._error_message == "Test error message"
        assert ctx.metrics is not None
        assert ctx.metrics.error_counts == 1

    def test_disabled_mode(self):
        """Test disabled mode skips profiling."""
        config = ProfilerConfig.disabled()
        ctx = ProfileContext(
            validator_name="test",
            validator_category="test",
            config=config,
        )

        ctx.start()
        ctx.stop()

        assert ctx._completed is True
        # Metrics should be None in disabled mode

    def test_snapshot_recording(self):
        """Test snapshot recording when configured."""
        config = ProfilerConfig.detailed()  # Has record_snapshots=True
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )
        ctx = ProfileContext(
            validator_name="test",
            validator_category="test",
            config=config,
            session=session,
        )

        ctx.start()
        ctx.set_issue_count(5)
        ctx.stop()

        assert ctx.snapshot is not None
        assert ctx.snapshot.validator_name == "test"
        assert ctx.snapshot.issue_count == 5
        assert len(session.snapshots) == 1


# =============================================================================
# ValidatorProfiler Tests
# =============================================================================

class TestValidatorProfiler:
    """Tests for ValidatorProfiler class."""

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.is_enabled is True
        assert profiler.current_session is None
        assert len(profiler.global_metrics) == 0

    def test_disabled_profiler(self):
        """Test disabled profiler."""
        config = ProfilerConfig.disabled()
        profiler = ValidatorProfiler(config)

        assert profiler.is_enabled is False

    def test_start_session(self, profiler):
        """Test starting a session."""
        session = profiler.start_session("test_session")

        assert session is not None
        assert session.session_id == "test_session"
        assert profiler.current_session is session

    def test_end_session(self, profiler):
        """Test ending a session."""
        profiler.start_session("test_session")
        session = profiler.end_session()

        assert session is not None
        assert session.end_time is not None
        assert profiler.current_session is None

    def test_auto_end_previous_session(self, profiler):
        """Test that starting new session ends previous."""
        session1 = profiler.start_session("session1")
        session2 = profiler.start_session("session2")

        assert session1.end_time is not None
        assert profiler.current_session is session2

    def test_profile_context_manager(self, profiler, sample_lazyframe):
        """Test profile context manager."""
        validator = MockValidator(delay_ms=1, issues_count=3)
        profiler.start_session()

        with profiler.profile(validator, rows_processed=1000) as ctx:
            issues = validator.validate(sample_lazyframe)
            ctx.set_issue_count(len(issues))

        metrics = profiler.get_metrics("mock_validator")
        assert metrics is not None
        assert metrics.execution_count == 1
        assert metrics.total_issues == 3

    def test_get_slowest_validators(self, profiler, sample_lazyframe):
        """Test getting slowest validators."""
        profiler.start_session()

        # Profile multiple validators with different delays (small delays for fast tests)
        for delay in [1, 2, 3, 4, 5]:
            validator = MockValidator(delay_ms=delay)
            validator.name = f"validator_{delay}ms"
            with profiler.profile(validator) as ctx:
                validator.validate(sample_lazyframe)

        profiler.end_session()

        slowest = profiler.get_slowest_validators(3)
        assert len(slowest) == 3
        assert "validator_5ms" in slowest[0][0]

    def test_get_memory_intensive_validators(self, profiler, sample_lazyframe):
        """Test getting memory-intensive validators."""
        profiler.start_session()

        validator = MockValidator()
        with profiler.profile(validator) as ctx:
            validator.validate(sample_lazyframe)

        profiler.end_session()

        memory_intensive = profiler.get_memory_intensive_validators(5)
        assert isinstance(memory_intensive, list)

    def test_summary(self, profiler, sample_lazyframe):
        """Test summary generation."""
        profiler.start_session()

        validator = MockValidator(issues_count=5)
        with profiler.profile(validator) as ctx:
            issues = validator.validate(sample_lazyframe)
            ctx.set_issue_count(len(issues))

        profiler.end_session()

        summary = profiler.summary()
        assert summary["total_validators"] == 1
        assert summary["total_executions"] == 1
        assert summary["total_issues"] == 5

    def test_reset(self, profiler, sample_lazyframe):
        """Test resetting profiler."""
        profiler.start_session()

        validator = MockValidator()
        with profiler.profile(validator) as ctx:
            validator.validate(sample_lazyframe)

        profiler.end_session()
        profiler.reset()

        assert profiler.current_session is None
        assert len(profiler.global_metrics) == 0

    def test_to_json(self, profiler, sample_lazyframe):
        """Test JSON export."""
        profiler.start_session()

        validator = MockValidator()
        with profiler.profile(validator) as ctx:
            validator.validate(sample_lazyframe)

        profiler.end_session()

        json_str = profiler.to_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert "global_metrics" in data

    def test_to_prometheus(self, profiler, sample_lazyframe):
        """Test Prometheus export."""
        profiler.start_session()

        validator = MockValidator(issues_count=2)
        with profiler.profile(validator) as ctx:
            issues = validator.validate(sample_lazyframe)
            ctx.set_issue_count(len(issues))

        profiler.end_session()

        prometheus = profiler.to_prometheus()

        assert "validator_execution_duration_ms" in prometheus
        assert "validator_execution_count" in prometheus
        assert "validator_issues_total" in prometheus
        assert 'validator="mock_validator"' in prometheus


# =============================================================================
# ProfilingSession Tests
# =============================================================================

class TestProfilingSession:
    """Tests for ProfilingSession class."""

    def test_initial_state(self):
        """Test initial session state."""
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )

        assert session.session_id == "test"
        assert session.end_time is None
        assert session.total_validators == 0
        assert session.total_executions == 0
        assert session.total_issues == 0

    def test_get_or_create_metrics(self):
        """Test getting or creating validator metrics."""
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )

        # Create new
        metrics1 = session.get_or_create_metrics("validator1", "category1")
        assert metrics1.validator_name == "validator1"
        assert len(session.validator_metrics) == 1

        # Get existing
        metrics2 = session.get_or_create_metrics("validator1", "category1")
        assert metrics1 is metrics2
        assert len(session.validator_metrics) == 1

    def test_add_snapshot(self):
        """Test adding execution snapshots."""
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )

        snapshot = ExecutionSnapshot(
            validator_name="test",
            timestamp=datetime.now(),
            duration_ms=100.0,
            rows_processed=1000,
            issue_count=5,
            peak_memory_bytes=100 * 1024 * 1024,
            memory_delta_bytes=10 * 1024 * 1024,
            gc_before=(0, 0, 0),
            gc_after=(1, 0, 0),
            success=True,
        )

        session.add_snapshot(snapshot)
        assert len(session.snapshots) == 1

    def test_duration_calculation(self):
        """Test duration calculation."""
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )

        time.sleep(0.01)  # 10ms
        session.end_time = datetime.now()

        assert session.duration_ms >= 10.0

    def test_to_json(self):
        """Test JSON export."""
        session = ProfilingSession(
            session_id="test",
            start_time=datetime.now(),
        )

        metrics = session.get_or_create_metrics("test_validator", "test")
        metrics.record_execution(duration_ms=100.0)

        session.end_time = datetime.now()

        json_str = session.to_json()
        data = json.loads(json_str)

        assert data["session_id"] == "test"
        assert "validators" in data


# =============================================================================
# ExecutionSnapshot Tests
# =============================================================================

class TestExecutionSnapshot:
    """Tests for ExecutionSnapshot class."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = ExecutionSnapshot(
            validator_name="test",
            timestamp=datetime.now(),
            duration_ms=100.0,
            rows_processed=1000,
            issue_count=5,
            peak_memory_bytes=100 * 1024 * 1024,
            memory_delta_bytes=10 * 1024 * 1024,
            gc_before=(0, 0, 0),
            gc_after=(1, 0, 0),
            success=True,
        )

        assert snapshot.validator_name == "test"
        assert snapshot.duration_ms == 100.0
        assert snapshot.rows_processed == 1000
        assert snapshot.issue_count == 5
        assert snapshot.success is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        snapshot = ExecutionSnapshot(
            validator_name="test",
            timestamp=datetime.now(),
            duration_ms=100.0,
            rows_processed=1000,
            issue_count=5,
            peak_memory_bytes=100 * 1024 * 1024,
            memory_delta_bytes=10 * 1024 * 1024,
            gc_before=(0, 0, 0),
            gc_after=(2, 0, 0),
            success=True,
        )

        result = snapshot.to_dict()

        assert result["validator_name"] == "test"
        assert result["duration_ms"] == 100.0
        assert result["gc_collections"] == 2  # gc_after - gc_before
        assert result["peak_memory_mb"] == pytest.approx(100.0, rel=0.01)


# =============================================================================
# MemoryTracker Tests
# =============================================================================

class TestMemoryTracker:
    """Tests for MemoryTracker class."""

    def test_availability_check(self):
        """Test availability check for psutil."""
        result = MemoryTracker.is_available()
        assert isinstance(result, bool)

    def test_basic_tracking(self):
        """Test basic memory tracking lifecycle."""
        tracker = MemoryTracker()

        tracker.start()
        # Allocate some memory
        data = [0] * 100000
        tracker.update_peak()
        peak, delta, end, gc_before, gc_after = tracker.stop()

        # Results depend on psutil availability
        if MemoryTracker.is_available():
            assert peak >= 0
        else:
            assert peak == 0

    def test_not_tracking_without_start(self):
        """Test that stop returns zeros without start."""
        tracker = MemoryTracker()

        peak, delta, end, gc_before, gc_after = tracker.stop()

        assert peak == 0
        assert delta == 0


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_default_profiler(self):
        """Test getting default profiler."""
        profiler = get_default_profiler()

        assert isinstance(profiler, ValidatorProfiler)

        # Same instance returned on subsequent calls
        profiler2 = get_default_profiler()
        assert profiler is profiler2

    def test_set_default_profiler(self):
        """Test setting default profiler."""
        custom_profiler = ValidatorProfiler(ProfilerConfig.detailed())
        set_default_profiler(custom_profiler)

        retrieved = get_default_profiler()
        assert retrieved is custom_profiler

    def test_reset_default_profiler(self):
        """Test resetting default profiler."""
        profiler = get_default_profiler()
        profiler.start_session()

        reset_default_profiler()

        # New profiler should be created
        new_profiler = get_default_profiler()
        assert new_profiler is not profiler

    def test_profile_validator_context_manager(self, sample_lazyframe):
        """Test profile_validator context manager."""
        validator = MockValidator(issues_count=3)

        with profile_validator(validator, rows_processed=1000) as ctx:
            issues = validator.validate(sample_lazyframe)
            ctx.set_issue_count(len(issues))

        assert ctx.metrics is not None
        assert ctx.metrics.total_issues == 3


# =============================================================================
# Decorator Tests
# =============================================================================

class TestProfiledDecorator:
    """Tests for @profiled decorator."""

    def test_basic_decoration(self, sample_lazyframe):
        """Test basic decoration."""
        profiler = ValidatorProfiler()
        set_default_profiler(profiler)
        profiler.start_session()

        class DecoratedValidator(Validator):
            name = "decorated"
            category = "test"

            @profiled()
            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                time.sleep(0.01)
                return [
                    ValidationIssue(
                        column="col",
                        issue_type="test",
                        count=1,
                        severity=Severity.LOW,
                    )
                ]

        validator = DecoratedValidator()
        issues = validator.validate(sample_lazyframe)

        profiler.end_session()

        metrics = profiler.get_metrics("decorated")
        assert metrics is not None
        assert metrics.execution_count == 1
        assert metrics.total_issues == 1

    def test_custom_profiler(self, sample_lazyframe):
        """Test decoration with custom profiler."""
        custom_profiler = ValidatorProfiler()
        custom_profiler.start_session()

        class CustomProfiledValidator(Validator):
            name = "custom_profiled"
            category = "test"

            @profiled(profiler=custom_profiler)
            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                return []

        validator = CustomProfiledValidator()
        validator.validate(sample_lazyframe)

        custom_profiler.end_session()

        metrics = custom_profiler.get_metrics("custom_profiled")
        assert metrics is not None


# =============================================================================
# ProfilingReport Tests
# =============================================================================

class TestProfilingReport:
    """Tests for ProfilingReport class."""

    def test_text_summary(self, profiler, sample_lazyframe):
        """Test text summary generation."""
        profiler.start_session()

        validator = MockValidator(issues_count=5)
        with profiler.profile(validator) as ctx:
            issues = validator.validate(sample_lazyframe)
            ctx.set_issue_count(len(issues))

        profiler.end_session()

        report = ProfilingReport(profiler)
        text = report.text_summary()

        assert "VALIDATOR PROFILING REPORT" in text
        assert "Total Validators" in text
        assert "SLOWEST VALIDATORS" in text

    def test_html_report(self, profiler, sample_lazyframe):
        """Test HTML report generation."""
        profiler.start_session()

        validator = MockValidator()
        with profiler.profile(validator) as ctx:
            validator.validate(sample_lazyframe)

        profiler.end_session()

        report = ProfilingReport(profiler)
        html = report.html_report()

        assert "<html>" in html
        assert "Validator Profiling Report" in html
        assert "<table>" in html


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_profiling(self, sample_lazyframe):
        """Test concurrent profiling from multiple threads."""
        profiler = ValidatorProfiler()
        profiler.start_session()

        errors = []

        def profile_validator_thread(idx):
            try:
                validator = MockValidator(delay_ms=1)
                validator.name = f"validator_{idx}"
                with profiler.profile(validator) as ctx:
                    validator.validate(sample_lazyframe)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=profile_validator_thread, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        profiler.end_session()

        assert len(errors) == 0
        assert profiler.summary()["total_validators"] == 10

    def test_concurrent_session_management(self):
        """Test concurrent session operations."""
        profiler = ValidatorProfiler()
        errors = []

        def session_operation(idx):
            try:
                if idx % 2 == 0:
                    profiler.start_session(f"session_{idx}")
                else:
                    profiler.end_session()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            t = threading.Thread(target=session_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors (race conditions handled by locking)
        assert len(errors) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for profiling with real validators."""

    def test_full_profiling_workflow(self, sample_lazyframe):
        """Test complete profiling workflow."""
        profiler = ValidatorProfiler(ProfilerConfig.detailed())

        # Start session
        session = profiler.start_session("integration_test")

        # Profile multiple validators (small delays for fast tests)
        validator1 = MockValidator(delay_ms=1, issues_count=2)
        validator1.name = "mock_validator_1"
        validator2 = MockValidator(delay_ms=2, issues_count=3)
        validator2.name = "mock_validator_2"
        validators = [validator1, validator2, SlowValidator()]

        for validator in validators:
            with profiler.profile(validator, rows_processed=1000) as ctx:
                issues = validator.validate(sample_lazyframe)
                ctx.set_issue_count(len(issues))

        # End session
        session = profiler.end_session()

        # Verify results
        assert session.total_validators == 3
        assert session.total_executions == 3
        assert session.total_issues == 5  # 2 + 3 + 0

        # Check snapshots were recorded
        assert len(session.snapshots) == 3

        # Generate report
        report = ProfilingReport(profiler)
        text = report.text_summary()
        assert "slow_validator" in text

    def test_error_handling_in_profiled_validator(self, sample_lazyframe):
        """Test error handling when validator raises exception."""
        profiler = ValidatorProfiler()
        profiler.start_session()

        class FailingValidator(Validator):
            name = "failing_validator"
            category = "test"

            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                raise ValueError("Test error")

        validator = FailingValidator()

        with pytest.raises(ValueError):
            with profiler.profile(validator) as ctx:
                validator.validate(sample_lazyframe)

        profiler.end_session()

        # Error should be recorded
        metrics = profiler.get_metrics("failing_validator")
        assert metrics is not None
        assert metrics.error_counts == 1

    def test_profiling_disabled_mode(self, sample_lazyframe):
        """Test that disabled mode has minimal overhead."""
        config = ProfilerConfig.disabled()
        profiler = ValidatorProfiler(config)

        validator = MockValidator()

        start = time.time()
        for _ in range(10):  # Reduced iterations for faster tests
            with profiler.profile(validator) as ctx:
                validator.validate(sample_lazyframe)
        disabled_time = time.time() - start

        # Disabled mode should be very fast
        assert disabled_time < 1.0  # Should complete 10 iterations in < 1s


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_default_timing_buckets(self):
        """Test default timing buckets are sensible."""
        assert len(DEFAULT_TIMING_BUCKETS_MS) > 0
        assert DEFAULT_TIMING_BUCKETS_MS == sorted(DEFAULT_TIMING_BUCKETS_MS)
        assert DEFAULT_TIMING_BUCKETS_MS[0] > 0

    def test_memory_thresholds(self):
        """Test memory thresholds are sensible."""
        assert MEMORY_WARNING_THRESHOLD_MB < MEMORY_CRITICAL_THRESHOLD_MB
        assert MEMORY_WARNING_THRESHOLD_MB > 0

    def test_max_history_entries(self):
        """Test max history entries is reasonable."""
        assert MAX_HISTORY_ENTRIES > 0
        assert MAX_HISTORY_ENTRIES <= 10000


# =============================================================================
# MetricType Enum Tests
# =============================================================================

class TestMetricTypeEnum:
    """Tests for MetricType enum."""

    def test_metric_types(self):
        """Test all metric types exist."""
        assert MetricType.TIMING is not None
        assert MetricType.MEMORY is not None
        assert MetricType.THROUGHPUT is not None
        assert MetricType.ISSUE_COUNT is not None
        assert MetricType.GC_IMPACT is not None


# =============================================================================
# ProfilerMode Enum Tests
# =============================================================================

class TestProfilerModeEnum:
    """Tests for ProfilerMode enum."""

    def test_profiler_modes(self):
        """Test all profiler modes exist."""
        assert ProfilerMode.DISABLED is not None
        assert ProfilerMode.BASIC is not None
        assert ProfilerMode.STANDARD is not None
        assert ProfilerMode.DETAILED is not None
        assert ProfilerMode.DIAGNOSTIC is not None
