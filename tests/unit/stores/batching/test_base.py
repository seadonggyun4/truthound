"""Tests for batching base module."""

from __future__ import annotations

import pytest
from datetime import datetime

from truthound.stores.batching.base import (
    Batch,
    BatchConfig,
    BatchMetrics,
    BatchState,
    BatchStatus,
)


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()
        assert config.batch_size == 1000
        assert config.flush_interval_seconds == 5.0
        assert config.max_buffer_memory_mb == 100
        assert config.max_pending_batches == 10
        assert config.parallelism == 4
        assert config.compression_enabled is True
        assert config.retry_enabled is True
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BatchConfig(
            batch_size=500,
            flush_interval_seconds=10.0,
            parallelism=8,
        )
        assert config.batch_size == 500
        assert config.flush_interval_seconds == 10.0
        assert config.parallelism == 8

    def test_validate_batch_size(self) -> None:
        """Test batch size validation."""
        with pytest.raises(ValueError, match="batch_size"):
            config = BatchConfig(batch_size=0)
            config.validate()

    def test_validate_flush_interval(self) -> None:
        """Test flush interval validation."""
        with pytest.raises(ValueError, match="flush_interval_seconds"):
            config = BatchConfig(flush_interval_seconds=-1.0)
            config.validate()

    def test_validate_memory(self) -> None:
        """Test memory limit validation."""
        with pytest.raises(ValueError, match="max_buffer_memory_mb"):
            config = BatchConfig(max_buffer_memory_mb=0)
            config.validate()

    def test_validate_parallelism(self) -> None:
        """Test parallelism validation."""
        with pytest.raises(ValueError, match="parallelism"):
            config = BatchConfig(parallelism=0)
            config.validate()

    def test_validate_compression_level(self) -> None:
        """Test compression level validation."""
        with pytest.raises(ValueError, match="compression_level"):
            config = BatchConfig(compression_level=0)
            config.validate()

        with pytest.raises(ValueError, match="compression_level"):
            config = BatchConfig(compression_level=10)
            config.validate()

    def test_validate_retry_settings(self) -> None:
        """Test retry settings validation."""
        with pytest.raises(ValueError, match="max_retries"):
            config = BatchConfig(max_retries=-1)
            config.validate()

        with pytest.raises(ValueError, match="retry_delay_seconds"):
            config = BatchConfig(retry_delay_seconds=-1.0)
            config.validate()

        with pytest.raises(ValueError, match="retry_backoff_multiplier"):
            config = BatchConfig(retry_backoff_multiplier=0.5)
            config.validate()


class TestBatchMetrics:
    """Tests for BatchMetrics."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        metrics = BatchMetrics()
        assert metrics.total_items == 0
        assert metrics.batches_created == 0
        assert metrics.batches_written == 0
        assert metrics.items_written == 0
        assert metrics.bytes_written == 0

    def test_record_item_added(self) -> None:
        """Test recording item addition."""
        metrics = BatchMetrics()
        metrics.record_item_added(1)
        assert metrics.total_items == 1

        metrics.record_item_added(5)
        assert metrics.total_items == 2
        assert metrics.peak_buffer_size == 5

    def test_record_batch_created(self) -> None:
        """Test recording batch creation."""
        metrics = BatchMetrics()
        metrics.record_batch_created(100)
        assert metrics.batches_created == 1

    def test_record_batch_written(self) -> None:
        """Test recording successful batch write."""
        metrics = BatchMetrics()
        metrics.record_batch_written(100, 1024, 50.0)
        assert metrics.batches_written == 1
        assert metrics.items_written == 100
        assert metrics.bytes_written == 1024
        assert metrics.total_write_time_ms == 50.0
        assert metrics.average_batch_size == 100.0
        assert metrics.average_write_time_ms == 50.0

    def test_record_batch_failed(self) -> None:
        """Test recording failed batch."""
        metrics = BatchMetrics()
        metrics.record_batch_failed(100)
        assert metrics.batches_failed == 1
        assert metrics.items_failed == 100

    def test_record_retry(self) -> None:
        """Test recording retry."""
        metrics = BatchMetrics()
        metrics.record_retry()
        assert metrics.retry_count == 1

    def test_record_flush(self) -> None:
        """Test recording flush."""
        metrics = BatchMetrics()
        metrics.record_flush()
        assert metrics.flush_count == 1
        assert metrics.last_flush_time is not None

    def test_compression_ratio(self) -> None:
        """Test compression ratio calculation."""
        metrics = BatchMetrics()
        assert metrics.get_compression_ratio() == 1.0

        metrics.bytes_written = 1000
        metrics.bytes_compressed = 500
        assert metrics.get_compression_ratio() == 0.5

    def test_throughput(self) -> None:
        """Test throughput calculation."""
        metrics = BatchMetrics()
        assert metrics.get_throughput() == 0.0

        metrics.items_written = 1000
        metrics.total_write_time_ms = 1000.0  # 1 second
        assert metrics.get_throughput() == 1000.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = BatchMetrics()
        metrics.record_batch_written(100, 1024, 50.0)

        d = metrics.to_dict()
        assert d["batches_written"] == 1
        assert d["items_written"] == 100
        assert d["throughput_items_per_sec"] == 2000.0  # 100 items in 50ms


class TestBatch:
    """Tests for Batch."""

    def test_creation(self) -> None:
        """Test batch creation."""
        batch = Batch(
            batch_id="test-1",
            items=[1, 2, 3],
        )
        assert batch.batch_id == "test-1"
        assert batch.items == [1, 2, 3]
        assert batch.status == BatchStatus.PENDING
        assert batch.size == 3
        assert batch.retry_count == 0

    def test_mark_flushing(self) -> None:
        """Test marking as flushing."""
        batch = Batch(batch_id="test-1", items=[1, 2, 3])
        batch.mark_flushing()
        assert batch.status == BatchStatus.FLUSHING

    def test_mark_completed(self) -> None:
        """Test marking as completed."""
        batch = Batch(batch_id="test-1", items=[1, 2, 3])
        batch.mark_completed()
        assert batch.status == BatchStatus.COMPLETED

    def test_mark_failed(self) -> None:
        """Test marking as failed."""
        batch = Batch(batch_id="test-1", items=[1, 2, 3])
        batch.mark_failed("Test error")
        assert batch.status == BatchStatus.FAILED
        assert batch.error == "Test error"

    def test_mark_retrying(self) -> None:
        """Test marking for retry."""
        batch = Batch(batch_id="test-1", items=[1, 2, 3])
        batch.mark_retrying()
        assert batch.status == BatchStatus.RETRYING
        assert batch.retry_count == 1

        batch.mark_retrying()
        assert batch.retry_count == 2


class TestBatchState:
    """Tests for BatchState."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = BatchState()
        assert state.is_active is False
        assert state.pending_batches == 0
        assert state.current_buffer_size == 0
        assert state.is_flushing is False

    def test_start_stop(self) -> None:
        """Test start and stop."""
        state = BatchState()
        state.start()
        assert state.is_active is True

        state.stop()
        assert state.is_active is False

    def test_flush_lifecycle(self) -> None:
        """Test flush begin/end."""
        state = BatchState()
        state.begin_flush()
        assert state.is_flushing is True

        state.end_flush()
        assert state.is_flushing is False
        assert state.last_flush is not None

    def test_pending_batches(self) -> None:
        """Test pending batch tracking."""
        state = BatchState()
        state.add_pending()
        state.add_pending()
        assert state.pending_batches == 2

        state.remove_pending()
        assert state.pending_batches == 1

        state.remove_pending()
        state.remove_pending()  # Should not go negative
        assert state.pending_batches == 0

    def test_buffer_size(self) -> None:
        """Test buffer size update."""
        state = BatchState()
        state.update_buffer_size(100)
        assert state.current_buffer_size == 100


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_values(self) -> None:
        """Test batch status values."""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.BUFFERING.value == "buffering"
        assert BatchStatus.FLUSHING.value == "flushing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"
        assert BatchStatus.RETRYING.value == "retrying"
