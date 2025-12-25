"""Integration tests for the Realtime module."""

from __future__ import annotations

import pytest
import polars as pl
import tempfile
from pathlib import Path

from truthound.realtime import (
    StreamingConfig,
    StreamingValidator,
    MockStreamingSource,
    IncrementalValidator,
    MemoryStateStore,
    CheckpointManager,
    BatchResult,
    StreamingMode,
    WindowType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_source() -> MockStreamingSource:
    """Mock streaming source for testing."""
    return MockStreamingSource(n_records=100)


@pytest.fixture
def memory_store() -> MemoryStateStore:
    """In-memory state store."""
    return MemoryStateStore()


# =============================================================================
# Test StreamingConfig
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_creation(self):
        """Test config creation."""
        config = StreamingConfig()
        assert config is not None

    def test_custom_values(self):
        """Test custom config values."""
        config = StreamingConfig(
            batch_size=500,
            mode=StreamingMode.MICRO_BATCH,
        )
        assert config.batch_size == 500
        assert config.mode == StreamingMode.MICRO_BATCH


# =============================================================================
# Test MockStreamingSource
# =============================================================================


class TestMockStreamingSource:
    """Tests for MockStreamingSource."""

    def test_creation(self):
        """Test source creation."""
        source = MockStreamingSource(n_records=50)
        assert source is not None

    def test_connect_disconnect(self):
        """Test connection lifecycle."""
        source = MockStreamingSource(n_records=50)

        source.connect()
        assert source.is_connected

        source.disconnect()
        assert not source.is_connected

    def test_read_batch(self, mock_source: MockStreamingSource):
        """Test reading batches."""
        mock_source.connect()

        batch = mock_source.read_batch(max_records=20)

        assert isinstance(batch, pl.DataFrame)
        assert len(batch) == 20

    def test_context_manager(self):
        """Test using as context manager."""
        source = MockStreamingSource(n_records=50)

        with source as s:
            assert s.is_connected
            batch = s.read_batch(max_records=10)
            assert len(batch) == 10

        assert not source.is_connected


# =============================================================================
# Test StreamingValidator
# =============================================================================


class TestStreamingValidator:
    """Tests for StreamingValidator."""

    def test_creation(self):
        """Test validator creation."""
        validator = StreamingValidator()
        assert validator is not None

    def test_validate_batch(self, mock_source: MockStreamingSource):
        """Test validating a batch."""
        mock_source.connect()
        batch = mock_source.read_batch(max_records=20)

        validator = StreamingValidator()
        result = validator.validate_batch(batch)

        assert isinstance(result, BatchResult)
        assert result.record_count == 20

    def test_get_stats(self):
        """Test getting validation statistics."""
        validator = StreamingValidator()
        stats = validator.get_stats()

        assert isinstance(stats, dict)

    def test_reset_stats(self):
        """Test resetting validation statistics."""
        validator = StreamingValidator()
        validator.reset_stats()
        stats = validator.get_stats()

        assert stats.get("total_records", 0) == 0


# =============================================================================
# Test MemoryStateStore
# =============================================================================


class TestMemoryStateStore:
    """Tests for MemoryStateStore."""

    def test_creation(self, memory_store: MemoryStateStore):
        """Test store creation."""
        assert memory_store is not None

    def test_set_and_get(self, memory_store: MemoryStateStore):
        """Test setting and getting state."""
        state = {"count": 100, "sum": 5000.0}

        memory_store.set("test_key", state)
        loaded = memory_store.get("test_key")

        assert loaded == state

    def test_get_nonexistent(self, memory_store: MemoryStateStore):
        """Test getting nonexistent key."""
        result = memory_store.get("nonexistent")
        assert result is None

    def test_delete(self, memory_store: MemoryStateStore):
        """Test deleting state."""
        memory_store.set("key", {"data": "value"})
        memory_store.delete("key")

        assert memory_store.get("key") is None

    def test_keys(self, memory_store: MemoryStateStore):
        """Test listing keys."""
        memory_store.set("key1", {"a": 1})
        memory_store.set("key2", {"b": 2})

        keys = memory_store.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_clear(self, memory_store: MemoryStateStore):
        """Test clearing all state."""
        memory_store.set("key1", {"a": 1})
        memory_store.set("key2", {"b": 2})
        memory_store.clear()

        assert memory_store.get("key1") is None
        assert memory_store.get("key2") is None


# =============================================================================
# Test IncrementalValidator
# =============================================================================


class TestIncrementalValidator:
    """Tests for IncrementalValidator."""

    def test_creation(self, memory_store: MemoryStateStore):
        """Test validator creation."""
        validator = IncrementalValidator(state_store=memory_store)
        assert validator is not None

    def test_creation_with_defaults(self):
        """Test validator creation with defaults."""
        validator = IncrementalValidator()
        assert validator is not None

    def test_validate_batch(self, memory_store: MemoryStateStore):
        """Test validating a batch."""
        validator = IncrementalValidator(state_store=memory_store)

        batch = pl.DataFrame({
            "id": list(range(20)),
            "value": [float(i) for i in range(20)],
        })

        result = validator.validate_batch(batch)

        assert isinstance(result, BatchResult)
        assert result.record_count == 20

    def test_with_duplicate_tracking(self, memory_store: MemoryStateStore):
        """Test with duplicate tracking enabled."""
        validator = IncrementalValidator(
            state_store=memory_store,
            track_duplicates=True,
            duplicate_columns=["id"],
        )

        batch = pl.DataFrame({
            "id": [1, 2, 3, 1],  # Has duplicate id=1
            "value": [10.0, 20.0, 30.0, 40.0],
        })

        result = validator.validate_batch(batch)
        assert result.record_count == 4


# =============================================================================
# Test CheckpointManager
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_creation(self):
        """Test manager creation with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            assert manager is not None

    def test_create_checkpoint(self, memory_store: MemoryStateStore):
        """Test creating a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            checkpoint = manager.create_checkpoint(
                state=memory_store,
                batch_count=5,
                total_records=500,
                total_issues=10,
            )

            assert checkpoint is not None
            assert checkpoint.batch_count == 5
            assert checkpoint.total_records == 500

    def test_get_checkpoint(self, memory_store: MemoryStateStore):
        """Test getting a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            checkpoint = manager.create_checkpoint(
                state=memory_store,
                batch_count=10,
                total_records=1000,
                total_issues=20,
            )

            restored = manager.get_checkpoint(checkpoint.checkpoint_id)

            assert restored is not None
            assert restored.batch_count == 10

    def test_list_checkpoints(self, memory_store: MemoryStateStore):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            manager.create_checkpoint(memory_store, 1, 100, 5)
            manager.create_checkpoint(memory_store, 2, 200, 10)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 2

    def test_get_latest(self, memory_store: MemoryStateStore):
        """Test getting the latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            manager.create_checkpoint(memory_store, 1, 100, 5)
            manager.create_checkpoint(memory_store, 2, 200, 10)

            latest = manager.get_latest()
            assert latest is not None
            assert latest.batch_count == 2


# =============================================================================
# Test BatchResult
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult."""

    def test_creation(self):
        """Test batch result creation."""
        result = BatchResult(
            batch_id="batch-001",
            record_count=100,
            issue_count=5,
        )

        assert result.batch_id == "batch-001"
        assert result.record_count == 100
        assert result.issue_count == 5

    def test_has_issues(self):
        """Test has_issues property."""
        result_with_issues = BatchResult(
            batch_id="batch-001",
            record_count=100,
            issue_count=5,
        )
        result_no_issues = BatchResult(
            batch_id="batch-002",
            record_count=100,
            issue_count=0,
        )

        assert result_with_issues.has_issues is True
        assert result_no_issues.has_issues is False

    def test_issue_ratio(self):
        """Test issue_ratio property."""
        result = BatchResult(
            batch_id="batch-001",
            record_count=100,
            issue_count=10,
        )

        assert result.issue_ratio == 0.1  # 10/100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BatchResult(
            batch_id="batch-001",
            record_count=100,
            issue_count=5,
        )

        data = result.to_dict()
        assert data["batch_id"] == "batch-001"
        assert data["record_count"] == 100


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_streaming_mode(self):
        """Test StreamingMode values."""
        assert StreamingMode.CONTINUOUS.value == "continuous"
        assert StreamingMode.MICRO_BATCH.value == "micro_batch"
        assert StreamingMode.WINDOWED.value == "windowed"

    def test_window_type(self):
        """Test WindowType values."""
        assert WindowType.TUMBLING.value == "tumbling"
        assert WindowType.SLIDING.value == "sliding"
