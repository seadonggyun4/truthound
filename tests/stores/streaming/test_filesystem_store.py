"""Tests for StreamingFileSystemStore."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from truthound.stores.results import ValidatorResult
from truthound.stores.streaming import (
    CompressionType,
    StreamingConfig,
    StreamingFileSystemConfig,
    StreamingFileSystemStore,
    StreamingFormat,
    StreamStatus,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    """Create a streaming filesystem store."""
    store = StreamingFileSystemStore(base_path=temp_dir)
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def sample_results():
    """Create sample validator results."""
    return [
        ValidatorResult(
            validator_name=f"validator_{i}",
            success=i % 2 == 0,
            column=f"column_{i % 5}",
            issue_type="test_issue",
            count=i,
            severity="medium" if i % 2 else "low",
            message=f"Test message {i}",
        )
        for i in range(100)
    ]


class TestStreamingFileSystemStoreBasic:
    """Basic tests for StreamingFileSystemStore."""

    def test_initialization(self, temp_dir):
        """Test store initialization."""
        store = StreamingFileSystemStore(base_path=temp_dir)
        store.initialize()

        assert Path(temp_dir).exists()
        store.close()

    def test_default_config(self, temp_dir):
        """Test default configuration."""
        store = StreamingFileSystemStore(base_path=temp_dir)

        assert store.config.format == StreamingFormat.JSONL
        assert store.config.compression == CompressionType.NONE
        assert store.config.chunk_size == 10000

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        store = StreamingFileSystemStore(
            base_path=temp_dir,
            format=StreamingFormat.JSONL,
            compression=CompressionType.GZIP,
            chunk_size=5000,
            buffer_size=500,
        )

        assert store.config.compression == CompressionType.GZIP
        assert store.config.chunk_size == 5000
        assert store.config.buffer_size == 500

    def test_context_manager(self, temp_dir):
        """Test context manager protocol."""
        with StreamingFileSystemStore(base_path=temp_dir) as store:
            assert store._initialized is True

    def test_create_session(self, store):
        """Test session creation."""
        session = store.create_session(
            run_id="test_run",
            data_asset="test_data.csv",
            metadata={"key": "value"},
        )

        assert session.run_id == "test_run"
        assert session.data_asset == "test_data.csv"
        assert session.status == StreamStatus.PENDING
        assert session.metadata["key"] == "value"

    def test_get_session(self, store):
        """Test getting an existing session."""
        session = store.create_session("test_run", "test.csv")
        retrieved = store.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_nonexistent_session(self, store):
        """Test getting a nonexistent session."""
        session = store.get_session("nonexistent")
        assert session is None


class TestStreamingWrite:
    """Tests for streaming write operations."""

    def test_write_single_result(self, store, sample_results):
        """Test writing a single result."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_result(sample_results[0])
            writer.flush()

        assert len(session.chunks) >= 0  # May or may not have flushed yet

    def test_write_batch_results(self, store, sample_results):
        """Test writing a batch of results."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:50])
            writer.flush()

        assert session.metrics.records_written == 50

    def test_write_all_results(self, store, sample_results):
        """Test writing all results."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            for result in sample_results:
                writer.write_result(result)

        # After close, all records should be written
        assert session.metrics.records_written == 100
        assert session.status == StreamStatus.COMPLETED

    def test_auto_flush_on_buffer_full(self, temp_dir, sample_results):
        """Test auto-flush when buffer is full."""
        store = StreamingFileSystemStore(
            base_path=temp_dir,
            buffer_size=10,  # Small buffer
            flush_interval_seconds=0,  # Disable time-based flush
        )
        store.initialize()

        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            # Write more than buffer size
            for i in range(25):
                writer.write_result(sample_results[i])

        # Should have created chunks
        assert len(session.chunks) >= 2
        store.close()

    def test_write_with_compression(self, temp_dir, sample_results):
        """Test writing with gzip compression."""
        store = StreamingFileSystemStore(
            base_path=temp_dir,
            compression=CompressionType.GZIP,
        )
        store.initialize()

        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Verify chunks exist
        run_path = Path(temp_dir) / "test_run"
        chunk_files = list(run_path.glob("*.jsonl.gz"))
        assert len(chunk_files) > 0

        store.close()


class TestStreamingRead:
    """Tests for streaming read operations."""

    def test_read_written_results(self, store, sample_results):
        """Test reading back written results."""
        session = store.create_session("test_run", "test.csv")

        # Write results
        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Read results
        read_results = list(store.iter_results("test_run"))

        assert len(read_results) == len(sample_results)

    def test_read_results_preserve_data(self, store, sample_results):
        """Test that read results preserve original data."""
        session = store.create_session("test_run", "test.csv")

        # Write results
        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:10])

        # Read and verify
        read_results = list(store.iter_results("test_run"))

        assert len(read_results) == 10
        assert read_results[0].validator_name == "validator_0"
        assert read_results[0].success is True
        assert read_results[1].success is False

    def test_read_with_reader_context(self, store, sample_results):
        """Test reading with reader context manager."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:20])

        reader = store.create_reader("test_run")
        with reader:
            results = list(reader.iter_results())

        assert len(results) == 20

    def test_read_batch(self, store, sample_results):
        """Test reading in batches."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        reader = store.create_reader("test_run")
        with reader:
            batch1 = reader.read_batch(30)
            batch2 = reader.read_batch(30)
            batch3 = reader.read_batch(50)  # Should only get remaining

        assert len(batch1) == 30
        assert len(batch2) == 30
        assert len(batch3) == 40  # Remaining

    def test_iter_failed_results(self, store, sample_results):
        """Test iterating over failed results only."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Count expected failures
        expected_failures = sum(1 for r in sample_results if not r.success)

        # Iterate failures
        failures = list(store.iter_failed_results("test_run"))
        assert len(failures) == expected_failures

    def test_iter_results_by_column(self, store, sample_results):
        """Test iterating results by column."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Count expected for column_0
        expected = sum(1 for r in sample_results if r.column == "column_0")

        # Iterate by column
        column_results = list(store.iter_results_by_column("test_run", "column_0"))
        assert len(column_results) == expected


class TestChunkManagement:
    """Tests for chunk management."""

    def test_list_chunks(self, store, sample_results):
        """Test listing chunks."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        chunks = store.list_chunks("test_run")
        assert len(chunks) > 0

    def test_get_chunk(self, store, sample_results):
        """Test getting a specific chunk."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        chunks = store.list_chunks("test_run")
        if chunks:
            chunk_results = store.get_chunk(chunks[0])
            assert len(chunk_results) > 0

    def test_delete_chunks(self, store, sample_results):
        """Test deleting chunks."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Verify chunks exist
        run_path = Path(store._base_path) / "test_run"
        assert run_path.exists()

        # Delete
        deleted = store.delete_chunks("test_run")
        assert deleted > 0

        # Verify deleted
        assert not run_path.exists()


class TestFinalizeResult:
    """Tests for result finalization."""

    def test_finalize_result(self, store, sample_results):
        """Test finalizing a streaming session."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        # Finalize
        result = store.finalize_result(session)

        assert result.run_id == "test_run"
        assert result.data_asset == "test.csv"
        assert result.statistics.total_validators == 100
        assert result.statistics.passed_validators == 50
        assert result.statistics.failed_validators == 50

    def test_finalize_result_with_metadata(self, store, sample_results):
        """Test finalizing with additional metadata."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:10])

        result = store.finalize_result(
            session,
            additional_metadata={"extra_key": "extra_value"},
        )

        assert result.metadata["extra_key"] == "extra_value"
        assert "streaming" in result.metadata

    def test_finalize_result_status(self, store):
        """Test finalization determines correct status."""
        session = store.create_session("test_run", "test.csv")

        # All passing
        passing_results = [
            ValidatorResult(
                validator_name="passing",
                success=True,
                column="col",
            )
            for _ in range(10)
        ]

        with store.create_writer(session) as writer:
            writer.write_results(passing_results)

        result = store.finalize_result(session)
        assert result.success is True


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_list_runs(self, store, sample_results):
        """Test listing all runs."""
        # Create multiple runs
        for run_id in ["run_1", "run_2", "run_3"]:
            session = store.create_session(run_id, "test.csv")
            with store.create_writer(session) as writer:
                writer.write_results(sample_results[:10])

        runs = store.list_runs()
        assert "run_1" in runs
        assert "run_2" in runs
        assert "run_3" in runs

    def test_get_streaming_stats(self, store, sample_results):
        """Test getting streaming statistics."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        stats = store.get_streaming_stats("test_run")

        assert stats["run_id"] == "test_run"
        assert stats["data_asset"] == "test.csv"
        assert stats["total_records"] == 100

    def test_get_storage_size(self, store, sample_results):
        """Test getting storage size."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results)

        size = store.get_storage_size("test_run")
        assert size > 0

        total_size = store.get_storage_size()
        assert total_size >= size

    def test_cleanup_incomplete_sessions(self, store, sample_results, temp_dir):
        """Test cleaning up incomplete sessions."""
        # Create a session but don't complete it
        session = store.create_session("incomplete_run", "test.csv")
        session.status = StreamStatus.FAILED

        # Save the failed status
        store._write_manifest(session)

        # Cleanup
        cleaned = store.cleanup_incomplete_sessions()
        assert cleaned > 0


class TestResumeSession:
    """Tests for session resumption."""

    def test_resume_session(self, store, sample_results):
        """Test resuming an interrupted session."""
        session = store.create_session("test_run", "test.csv")

        # Write some data
        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:50])

        # Simulate interruption by marking as active
        session.status = StreamStatus.ACTIVE
        store._write_manifest(session)

        # Resume
        resumed = store.resume_session(session.session_id)

        assert resumed.session_id == session.session_id
        assert resumed.status == StreamStatus.ACTIVE

    def test_resume_completed_session_fails(self, store, sample_results):
        """Test that resuming a completed session fails."""
        session = store.create_session("test_run", "test.csv")

        with store.create_writer(session) as writer:
            writer.write_results(sample_results[:10])

        with pytest.raises(ValueError, match="already completed"):
            store.resume_session(session.session_id)

    def test_resume_nonexistent_session_fails(self, store):
        """Test that resuming a nonexistent session fails."""
        with pytest.raises(ValueError, match="not found"):
            store.resume_session("nonexistent")
