"""Tests for streaming writers and readers."""

import gzip
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

from truthound.stores.results import ValidatorResult
from truthound.stores.streaming.base import (
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.writer import (
    BufferedStreamWriter,
    CSVSerializer,
    GzipCompressor,
    JSONLSerializer,
    NoCompressor,
    StreamingResultWriter,
    StreamWriteError,
    WriteBuffer,
    get_compressor,
    get_serializer,
)
from truthound.stores.streaming.reader import (
    ChunkedResultReader,
    CSVDeserializer,
    JSONLDeserializer,
    MemoryChunkLoader,
    StreamingResultReader,
    get_decompressor,
    get_deserializer,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_session():
    """Create a sample streaming session."""
    return StreamSession(
        session_id="test_session",
        run_id="test_run",
        data_asset="test.csv",
        status=StreamStatus.PENDING,
        config=StreamingConfig(),
    )


@pytest.fixture
def sample_records():
    """Create sample records."""
    return [
        {
            "validator_name": f"validator_{i}",
            "success": i % 2 == 0,
            "column": f"column_{i % 5}",
            "count": i,
        }
        for i in range(50)
    ]


class TestWriteBuffer:
    """Tests for WriteBuffer."""

    def test_add_record(self):
        """Test adding a record to buffer."""
        buffer = WriteBuffer(max_records=10)
        result = buffer.add({"key": "value"})

        assert len(buffer.records) == 1
        assert result is False  # Not yet full

    def test_add_triggers_flush(self):
        """Test that adding beyond max_records triggers flush."""
        buffer = WriteBuffer(max_records=3)

        buffer.add({"a": 1})
        buffer.add({"b": 2})
        result = buffer.add({"c": 3})

        assert result is True  # Should flush

    def test_add_batch(self):
        """Test adding a batch of records."""
        buffer = WriteBuffer(max_records=10)
        buffer.add_batch([{"a": 1}, {"b": 2}, {"c": 3}])

        assert len(buffer.records) == 3

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = WriteBuffer()
        buffer.add({"a": 1})
        buffer.add({"b": 2})

        records = buffer.clear()

        assert len(records) == 2
        assert buffer.is_empty()

    def test_byte_size_tracking(self):
        """Test byte size tracking."""
        buffer = WriteBuffer(max_bytes=100)
        record = {"key": "value"}
        buffer.add(record)

        assert buffer.byte_size > 0

    def test_byte_size_triggers_flush(self):
        """Test that exceeding max_bytes triggers flush."""
        buffer = WriteBuffer(max_bytes=50)  # Very small

        # Add a large record
        result = buffer.add({"key": "x" * 100})

        assert result is True


class TestSerializers:
    """Tests for serializers."""

    def test_jsonl_serializer_single(self):
        """Test JSONL serializer with single record."""
        serializer = JSONLSerializer()
        record = {"key": "value", "number": 42}

        data = serializer.serialize(record)

        assert data.endswith(b"\n")
        parsed = json.loads(data.decode())
        assert parsed["key"] == "value"

    def test_jsonl_serializer_batch(self):
        """Test JSONL serializer with batch."""
        serializer = JSONLSerializer()
        records = [{"a": 1}, {"b": 2}, {"c": 3}]

        data = serializer.serialize_batch(records)
        lines = data.decode().strip().split("\n")

        assert len(lines) == 3
        assert json.loads(lines[0])["a"] == 1

    def test_jsonl_content_type(self):
        """Test JSONL content type."""
        serializer = JSONLSerializer()
        assert serializer.get_content_type() == "application/x-ndjson"

    def test_csv_serializer_single(self):
        """Test CSV serializer with single record."""
        serializer = CSVSerializer(columns=["key", "value"])
        record = {"key": "a", "value": "b"}

        data = serializer.serialize(record)
        lines = data.decode().strip().split("\n")

        assert len(lines) == 2  # Header + row
        assert "key" in lines[0]

    def test_csv_serializer_batch(self):
        """Test CSV serializer with batch."""
        serializer = CSVSerializer()
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        data = serializer.serialize_batch(records)
        lines = data.decode().strip().split("\n")

        assert len(lines) == 3  # Header + 2 rows

    def test_get_serializer_jsonl(self):
        """Test getting JSONL serializer."""
        serializer = get_serializer(StreamingFormat.JSONL)
        assert isinstance(serializer, JSONLSerializer)

    def test_get_serializer_ndjson(self):
        """Test getting NDJSON serializer (alias for JSONL)."""
        serializer = get_serializer(StreamingFormat.NDJSON)
        assert isinstance(serializer, JSONLSerializer)

    def test_get_serializer_csv(self):
        """Test getting CSV serializer."""
        serializer = get_serializer(StreamingFormat.CSV)
        assert isinstance(serializer, CSVSerializer)


class TestCompressors:
    """Tests for compressors."""

    def test_no_compressor(self):
        """Test no compression."""
        compressor = NoCompressor()
        data = b"test data"

        compressed = compressor.compress(data)

        assert compressed == data
        assert compressor.get_extension() == ""

    def test_gzip_compressor(self):
        """Test gzip compression."""
        compressor = GzipCompressor(level=6)
        data = b"test data" * 100

        compressed = compressor.compress(data)

        # Should be smaller
        assert len(compressed) < len(data)
        assert compressor.get_extension() == ".gz"

        # Verify decompression
        decompressed = gzip.decompress(compressed)
        assert decompressed == data

    def test_get_compressor_none(self):
        """Test getting no compressor."""
        compressor = get_compressor(CompressionType.NONE)
        assert isinstance(compressor, NoCompressor)

    def test_get_compressor_gzip(self):
        """Test getting gzip compressor."""
        compressor = get_compressor(CompressionType.GZIP)
        assert isinstance(compressor, GzipCompressor)


class TestDeserializers:
    """Tests for deserializers."""

    def test_jsonl_deserializer(self):
        """Test JSONL deserializer."""
        deserializer = JSONLDeserializer()
        data = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'

        records = list(deserializer.deserialize(data))

        assert len(records) == 3
        assert records[0]["a"] == 1

    def test_jsonl_deserializer_handles_empty_lines(self):
        """Test JSONL deserializer handles empty lines."""
        deserializer = JSONLDeserializer()
        data = b'{"a": 1}\n\n{"b": 2}\n'

        records = list(deserializer.deserialize(data))

        assert len(records) == 2

    def test_csv_deserializer(self):
        """Test CSV deserializer."""
        deserializer = CSVDeserializer()
        data = b"a,b\n1,2\n3,4\n"

        records = list(deserializer.deserialize(data))

        assert len(records) == 2
        assert records[0]["a"] == "1"

    def test_get_deserializer_jsonl(self):
        """Test getting JSONL deserializer."""
        deserializer = get_deserializer(StreamingFormat.JSONL)
        assert isinstance(deserializer, JSONLDeserializer)


class TestDecompressors:
    """Tests for decompressors."""

    def test_get_decompressor_none(self):
        """Test getting no decompressor."""
        from truthound.stores.streaming.reader import NoDecompressor

        decompressor = get_decompressor(CompressionType.NONE)
        assert isinstance(decompressor, NoDecompressor)

    def test_get_decompressor_gzip(self):
        """Test getting gzip decompressor."""
        from truthound.stores.streaming.reader import GzipDecompressor

        decompressor = get_decompressor(CompressionType.GZIP)
        assert isinstance(decompressor, GzipDecompressor)

    def test_gzip_round_trip(self):
        """Test gzip compression/decompression round trip."""
        from truthound.stores.streaming.reader import GzipDecompressor

        compressor = GzipCompressor()
        decompressor = GzipDecompressor()

        original = b"test data for compression"
        compressed = compressor.compress(original)
        decompressed = decompressor.decompress(compressed)

        assert decompressed == original


class TestBufferedStreamWriter:
    """Tests for BufferedStreamWriter."""

    def test_write_single_record(self, sample_session):
        """Test writing a single record."""
        config = StreamingConfig(buffer_size=10, flush_interval_seconds=0)
        writer = BufferedStreamWriter(sample_session, config)

        writer.write({"key": "value"})

        assert len(writer.buffer.records) == 1

    def test_write_batch(self, sample_session, sample_records):
        """Test writing a batch of records."""
        config = StreamingConfig(buffer_size=100, flush_interval_seconds=0)
        writer = BufferedStreamWriter(sample_session, config)

        writer.write_batch(sample_records)

        assert len(writer.buffer.records) == 50

    def test_flush(self, sample_session, sample_records):
        """Test flushing buffer."""
        config = StreamingConfig(buffer_size=100, flush_interval_seconds=0)
        writer = BufferedStreamWriter(sample_session, config)

        writer.write_batch(sample_records)
        writer.flush()

        assert writer.buffer.is_empty()
        assert writer.get_output() != b""

    def test_context_manager(self, sample_session, sample_records):
        """Test context manager protocol."""
        config = StreamingConfig(buffer_size=100, flush_interval_seconds=0)

        with BufferedStreamWriter(sample_session, config) as writer:
            writer.write_batch(sample_records)

        assert sample_session.status == StreamStatus.COMPLETED

    def test_closed_writer_raises(self, sample_session):
        """Test that writing to closed writer raises error."""
        config = StreamingConfig(flush_interval_seconds=0)
        writer = BufferedStreamWriter(sample_session, config)
        writer.close()

        with pytest.raises(StreamWriteError):
            writer.write({"key": "value"})


class TestStreamingResultWriter:
    """Tests for StreamingResultWriter (filesystem)."""

    def test_write_creates_files(self, temp_dir, sample_session, sample_records):
        """Test that writing creates chunk files."""
        config = StreamingConfig(
            buffer_size=10,  # Small buffer to force chunks
            flush_interval_seconds=0,
        )
        writer = StreamingResultWriter(sample_session, config, temp_dir)

        with writer:
            for record in sample_records:
                writer.write(record)

        # Check files were created
        run_path = Path(temp_dir) / sample_session.run_id
        assert run_path.exists()

        chunk_files = list(run_path.glob("*.jsonl"))
        assert len(chunk_files) > 0

    def test_write_creates_manifest(self, temp_dir, sample_session, sample_records):
        """Test that manifest is created."""
        config = StreamingConfig(flush_interval_seconds=0)
        writer = StreamingResultWriter(sample_session, config, temp_dir)

        with writer:
            writer.write_batch(sample_records[:10])

        manifest_path = Path(temp_dir) / sample_session.run_id / "_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["run_id"] == sample_session.run_id
        assert manifest["status"] == "completed"

    def test_write_result_method(self, temp_dir, sample_session):
        """Test write_result method with ValidatorResult."""
        config = StreamingConfig(flush_interval_seconds=0)
        writer = StreamingResultWriter(sample_session, config, temp_dir)

        result = ValidatorResult(
            validator_name="test",
            success=True,
            column="col1",
        )

        with writer:
            writer.write_result(result)

        assert sample_session.metrics.records_written > 0


class TestStreamingResultReader:
    """Tests for StreamingResultReader (filesystem)."""

    def test_read_written_data(self, temp_dir, sample_session, sample_records):
        """Test reading back written data."""
        config = StreamingConfig(flush_interval_seconds=0)

        # Write data
        writer = StreamingResultWriter(sample_session, config, temp_dir)
        with writer:
            writer.write_batch(sample_records)

        # Read data
        run_path = Path(temp_dir) / sample_session.run_id
        reader = StreamingResultReader(run_path, config)

        with reader:
            records = list(reader)

        assert len(records) == len(sample_records)
        assert records[0]["validator_name"] == "validator_0"

    def test_read_batch(self, temp_dir, sample_session, sample_records):
        """Test reading in batches."""
        config = StreamingConfig(flush_interval_seconds=0)

        # Write data
        writer = StreamingResultWriter(sample_session, config, temp_dir)
        with writer:
            writer.write_batch(sample_records)

        # Read in batches
        run_path = Path(temp_dir) / sample_session.run_id
        reader = StreamingResultReader(run_path, config)

        with reader:
            batch1 = reader.read_batch(20)
            batch2 = reader.read_batch(20)
            batch3 = reader.read_batch(20)

        assert len(batch1) == 20
        assert len(batch2) == 20
        assert len(batch3) == 10

    def test_iter_results(self, temp_dir, sample_session, sample_records):
        """Test iterating as ValidatorResult objects."""
        config = StreamingConfig(flush_interval_seconds=0)

        # Write data
        writer = StreamingResultWriter(sample_session, config, temp_dir)
        with writer:
            writer.write_batch(sample_records)

        # Read as ValidatorResult
        run_path = Path(temp_dir) / sample_session.run_id
        reader = StreamingResultReader(run_path, config)

        with reader:
            results = list(reader.iter_results())

        assert len(results) == len(sample_records)
        assert isinstance(results[0], ValidatorResult)


class TestChunkedResultReader:
    """Tests for ChunkedResultReader."""

    def test_read_from_memory_loader(self):
        """Test reading from memory chunk loader."""
        from truthound.stores.streaming.base import ChunkInfo

        # Create chunks in memory
        loader = MemoryChunkLoader()

        records_1 = b'{"a": 1}\n{"b": 2}\n'
        records_2 = b'{"c": 3}\n{"d": 4}\n'

        loader.add_chunk("chunk_0", records_1)
        loader.add_chunk("chunk_1", records_2)

        chunks = [
            ChunkInfo(
                chunk_id="chunk_0",
                chunk_index=0,
                record_count=2,
                byte_size=len(records_1),
                start_offset=0,
                end_offset=2,
            ),
            ChunkInfo(
                chunk_id="chunk_1",
                chunk_index=1,
                record_count=2,
                byte_size=len(records_2),
                start_offset=2,
                end_offset=4,
            ),
        ]

        config = StreamingConfig()
        reader = ChunkedResultReader(chunks, loader, config)

        with reader:
            records = list(reader)

        assert len(records) == 4
        assert records[0]["a"] == 1
        assert records[3]["d"] == 4


class TestCompressedReadWrite:
    """Tests for compressed read/write operations."""

    def test_gzip_round_trip(self, temp_dir, sample_session, sample_records):
        """Test writing and reading with gzip compression."""
        config = StreamingConfig(
            compression=CompressionType.GZIP,
            flush_interval_seconds=0,
        )

        # Write with compression
        writer = StreamingResultWriter(sample_session, config, temp_dir)
        with writer:
            writer.write_batch(sample_records)

        # Verify .gz files exist
        run_path = Path(temp_dir) / sample_session.run_id
        gz_files = list(run_path.glob("*.jsonl.gz"))
        assert len(gz_files) > 0

        # Read back
        reader = StreamingResultReader(run_path, config)
        with reader:
            records = list(reader)

        assert len(records) == len(sample_records)
        assert records[0]["validator_name"] == sample_records[0]["validator_name"]
