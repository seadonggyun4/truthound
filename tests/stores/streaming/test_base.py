"""Tests for streaming base classes and protocols."""

import pytest
from datetime import datetime

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamSession,
    StreamStatus,
)


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.format == StreamingFormat.JSONL
        assert config.compression == CompressionType.NONE
        assert config.chunk_size == 10000
        assert config.buffer_size == 1000
        assert config.max_memory_mb == 512
        assert config.enable_checkpoints is True
        assert config.enable_metrics is True

    def test_validate_success(self):
        """Test validation with valid values."""
        config = StreamingConfig(
            chunk_size=5000,
            buffer_size=500,
            max_memory_mb=256,
        )
        config.validate()  # Should not raise

    def test_validate_invalid_chunk_size(self):
        """Test validation with invalid chunk_size."""
        config = StreamingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()

    def test_validate_invalid_buffer_size(self):
        """Test validation with invalid buffer_size."""
        config = StreamingConfig(buffer_size=-1)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            config.validate()

    def test_validate_invalid_max_memory(self):
        """Test validation with invalid max_memory_mb."""
        config = StreamingConfig(max_memory_mb=0)
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            config.validate()


class TestStreamingMetrics:
    """Tests for StreamingMetrics."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = StreamingMetrics()

        assert metrics.records_written == 0
        assert metrics.records_read == 0
        assert metrics.bytes_written == 0
        assert metrics.chunks_written == 0
        assert metrics.errors == []

    def test_record_write(self):
        """Test recording write operations."""
        metrics = StreamingMetrics()
        metrics.record_write(count=100, bytes_count=5000)

        assert metrics.records_written == 100
        assert metrics.bytes_written == 5000

        metrics.record_write(count=50, bytes_count=2500)
        assert metrics.records_written == 150
        assert metrics.bytes_written == 7500

    def test_record_read(self):
        """Test recording read operations."""
        metrics = StreamingMetrics()
        metrics.record_read(count=100, bytes_count=5000)

        assert metrics.records_read == 100
        assert metrics.bytes_read == 5000

    def test_record_chunk(self):
        """Test recording chunk operations."""
        metrics = StreamingMetrics()
        metrics.record_chunk(is_write=True)
        metrics.record_chunk(is_write=True)
        metrics.record_chunk(is_write=False)

        assert metrics.chunks_written == 2
        assert metrics.chunks_read == 1

    def test_record_error(self):
        """Test recording errors."""
        metrics = StreamingMetrics()
        metrics.record_error("Error 1")
        metrics.record_error("Error 2")

        assert len(metrics.errors) == 2
        assert "Error 1" in metrics.errors
        assert "Error 2" in metrics.errors

    def test_start_and_finish(self):
        """Test start and finish timing."""
        metrics = StreamingMetrics()
        metrics.start()

        assert metrics.start_time is not None
        assert metrics.end_time is None

        # Simulate some work
        metrics.record_write(count=1000)

        metrics.finish()

        assert metrics.end_time is not None
        assert metrics.average_throughput > 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = StreamingMetrics()
        metrics.record_write(count=100, bytes_count=5000)
        metrics.record_chunk(is_write=True)

        data = metrics.to_dict()

        assert data["records_written"] == 100
        assert data["bytes_written"] == 5000
        assert data["chunks_written"] == 1


class TestChunkInfo:
    """Tests for ChunkInfo."""

    def test_creation(self):
        """Test chunk info creation."""
        chunk = ChunkInfo(
            chunk_id="chunk_001",
            chunk_index=0,
            record_count=1000,
            byte_size=50000,
            start_offset=0,
            end_offset=1000,
            checksum="abc123",
            path="/path/to/chunk",
        )

        assert chunk.chunk_id == "chunk_001"
        assert chunk.chunk_index == 0
        assert chunk.record_count == 1000
        assert chunk.byte_size == 50000
        assert chunk.start_offset == 0
        assert chunk.end_offset == 1000
        assert chunk.checksum == "abc123"
        assert chunk.path == "/path/to/chunk"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        chunk = ChunkInfo(
            chunk_id="chunk_001",
            chunk_index=0,
            record_count=1000,
            byte_size=50000,
            start_offset=0,
            end_offset=1000,
        )

        data = chunk.to_dict()

        assert data["chunk_id"] == "chunk_001"
        assert data["chunk_index"] == 0
        assert data["record_count"] == 1000
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "chunk_id": "chunk_002",
            "chunk_index": 1,
            "record_count": 500,
            "byte_size": 25000,
            "start_offset": 1000,
            "end_offset": 1500,
            "checksum": "def456",
            "created_at": "2025-01-01T00:00:00",
            "path": "/path/to/chunk2",
        }

        chunk = ChunkInfo.from_dict(data)

        assert chunk.chunk_id == "chunk_002"
        assert chunk.chunk_index == 1
        assert chunk.record_count == 500
        assert chunk.checksum == "def456"


class TestStreamSession:
    """Tests for StreamSession."""

    def test_creation(self):
        """Test session creation."""
        session = StreamSession(
            session_id="session_001",
            run_id="run_001",
            data_asset="test.csv",
        )

        assert session.session_id == "session_001"
        assert session.run_id == "run_001"
        assert session.data_asset == "test.csv"
        assert session.status == StreamStatus.PENDING
        assert isinstance(session.metrics, StreamingMetrics)
        assert session.chunks == []

    def test_to_dict(self):
        """Test serialization to dictionary."""
        session = StreamSession(
            session_id="session_001",
            run_id="run_001",
            data_asset="test.csv",
            status=StreamStatus.ACTIVE,
        )
        session.chunks.append(
            ChunkInfo(
                chunk_id="chunk_001",
                chunk_index=0,
                record_count=100,
                byte_size=5000,
                start_offset=0,
                end_offset=100,
            )
        )

        data = session.to_dict()

        assert data["session_id"] == "session_001"
        assert data["run_id"] == "run_001"
        assert data["status"] == "active"
        assert len(data["chunks"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "session_id": "session_002",
            "run_id": "run_002",
            "data_asset": "data.parquet",
            "status": "completed",
            "metrics": {"records_written": 1000},
            "chunks": [
                {
                    "chunk_id": "chunk_001",
                    "chunk_index": 0,
                    "record_count": 1000,
                    "byte_size": 50000,
                    "start_offset": 0,
                    "end_offset": 1000,
                    "created_at": "2025-01-01T00:00:00",
                }
            ],
            "metadata": {"key": "value"},
            "started_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "checkpoint_offset": 500,
        }

        session = StreamSession.from_dict(data)

        assert session.session_id == "session_002"
        assert session.status == StreamStatus.COMPLETED
        assert session.metrics.records_written == 1000
        assert len(session.chunks) == 1
        assert session.checkpoint_offset == 500


class TestStreamStatus:
    """Tests for StreamStatus enum."""

    def test_status_values(self):
        """Test all status values."""
        assert StreamStatus.PENDING.value == "pending"
        assert StreamStatus.ACTIVE.value == "active"
        assert StreamStatus.PAUSED.value == "paused"
        assert StreamStatus.COMPLETED.value == "completed"
        assert StreamStatus.FAILED.value == "failed"
        assert StreamStatus.ABORTED.value == "aborted"


class TestStreamingFormat:
    """Tests for StreamingFormat enum."""

    def test_format_values(self):
        """Test all format values."""
        assert StreamingFormat.JSONL.value == "jsonl"
        assert StreamingFormat.NDJSON.value == "ndjson"
        assert StreamingFormat.CSV.value == "csv"
        assert StreamingFormat.PARQUET.value == "parquet"


class TestCompressionType:
    """Tests for CompressionType enum."""

    def test_compression_values(self):
        """Test all compression values."""
        assert CompressionType.NONE.value == "none"
        assert CompressionType.GZIP.value == "gzip"
        assert CompressionType.ZSTD.value == "zstd"
        assert CompressionType.LZ4.value == "lz4"
        assert CompressionType.SNAPPY.value == "snappy"
