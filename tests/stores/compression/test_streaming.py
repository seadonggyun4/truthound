"""Tests for streaming compression."""

import io
import pytest

from truthound.stores.compression import (
    CompressionAlgorithm,
    CompressionLevel,
    StreamingCompressor,
    StreamingDecompressor,
    ChunkedCompressor,
    ChunkInfo,
    ChunkIndex,
    StreamingMetrics,
    GzipStreamWriter,
    GzipStreamReader,
    get_compressor,
)


class TestStreamingMetrics:
    """Tests for StreamingMetrics."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = StreamingMetrics()

        assert metrics.bytes_in == 0
        assert metrics.bytes_out == 0
        assert metrics.chunks_processed == 0

    def test_duration_calculation(self):
        """Test duration calculation."""
        metrics = StreamingMetrics()
        metrics.start_time = 100.0
        metrics.end_time = 100.5

        assert metrics.duration_ms == 500.0

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = StreamingMetrics()
        metrics.bytes_in = 1024 * 1024  # 1MB
        metrics.start_time = 0.0
        metrics.end_time = 1.0

        assert metrics.throughput_mbps == 1.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = StreamingMetrics()
        metrics.bytes_in = 1000
        metrics.bytes_out = 250
        metrics.chunks_processed = 5

        data = metrics.to_dict()

        assert data["bytes_in"] == 1000
        assert data["bytes_out"] == 250
        assert data["chunks_processed"] == 5


class TestGzipStreamWriter:
    """Tests for GzipStreamWriter."""

    def test_basic_write(self):
        """Test basic write operation."""
        writer = GzipStreamWriter()

        writer.write(b"Hello, ")
        writer.write(b"World!")
        result = writer.close()

        assert len(result) > 0

    def test_write_with_flush(self):
        """Test write with flush."""
        writer = GzipStreamWriter()

        writer.write(b"First chunk")
        flushed = writer.flush()
        writer.write(b"Second chunk")
        final = writer.close()

        # Both should have data
        assert len(flushed) + len(final) > 0

    def test_bytes_written_tracking(self):
        """Test bytes written tracking."""
        writer = GzipStreamWriter()

        writer.write(b"Test data")
        writer.close()

        assert writer.bytes_written == 9


class TestGzipStreamReader:
    """Tests for GzipStreamReader."""

    def test_read_from_bytes(self):
        """Test reading from compressed bytes."""
        # First compress some data
        import gzip

        original = b"Test data for reading"
        compressed = gzip.compress(original)

        reader = GzipStreamReader(compressed)
        result = reader.read()

        assert result == original

    def test_read_chunk(self):
        """Test reading a chunk."""
        import gzip

        original = b"Chunk data"
        compressed = gzip.compress(original)

        reader = GzipStreamReader(b"")
        result = reader.read_chunk(compressed)

        assert result == original


class TestStreamingCompressor:
    """Tests for StreamingCompressor."""

    def test_basic_streaming(self):
        """Test basic streaming compression."""
        with StreamingCompressor() as compressor:
            compressor.write(b"First chunk of data " * 100)
            compressor.write(b"Second chunk of data " * 100)
            final = compressor.finalize()

        assert len(final) > 0

    def test_context_manager(self):
        """Test context manager protocol."""
        with StreamingCompressor() as compressor:
            assert compressor.metrics.start_time > 0

    def test_metrics_tracking(self):
        """Test metrics are tracked."""
        with StreamingCompressor() as compressor:
            compressor.write(b"Data " * 100)
            compressor.finalize()

            assert compressor.metrics.bytes_in > 0
            assert compressor.metrics.chunks_processed > 0

    def test_auto_flush(self):
        """Test automatic flush on threshold."""
        compressor = StreamingCompressor(flush_threshold=100)

        with compressor:
            # Write more than threshold
            result = compressor.write(b"X" * 200)

            # Should have auto-flushed
            assert len(result) > 0

    def test_empty_finalize(self):
        """Test finalizing empty stream."""
        with StreamingCompressor() as compressor:
            final = compressor.finalize()

        assert final == b""

    def test_multiple_algorithms(self):
        """Test with different algorithms."""
        data = b"Test data " * 100

        for algo in [CompressionAlgorithm.GZIP]:
            with StreamingCompressor(algorithm=algo) as compressor:
                compressor.write(data)
                result = compressor.finalize()

            assert len(result) > 0


class TestStreamingDecompressor:
    """Tests for StreamingDecompressor."""

    def test_basic_decompression(self):
        """Test basic streaming decompression."""
        # First compress some data
        import gzip

        original = b"Test data for decompression " * 100
        compressed = gzip.compress(original)

        with StreamingDecompressor() as decompressor:
            result = decompressor.write(compressed)

        assert result == original

    def test_metrics_tracking(self):
        """Test metrics are tracked during decompression."""
        import gzip

        original = b"Test data " * 100
        compressed = gzip.compress(original)

        with StreamingDecompressor() as decompressor:
            decompressor.write(compressed)

        assert decompressor.metrics.bytes_in > 0
        assert decompressor.metrics.bytes_out > 0


class TestChunkInfo:
    """Tests for ChunkInfo."""

    def test_creation(self):
        """Test chunk info creation."""
        info = ChunkInfo(
            index=0,
            original_size=1000,
            compressed_size=250,
            offset=0,
        )

        assert info.index == 0
        assert info.original_size == 1000
        assert info.compressed_size == 250


class TestChunkIndex:
    """Tests for ChunkIndex."""

    def test_creation(self):
        """Test chunk index creation."""
        index = ChunkIndex(
            algorithm=CompressionAlgorithm.GZIP,
            chunk_size=1024,
        )

        assert index.algorithm == CompressionAlgorithm.GZIP
        assert index.chunk_size == 1024
        assert len(index.chunks) == 0

    def test_serialization(self):
        """Test serialization and deserialization."""
        index = ChunkIndex(
            algorithm=CompressionAlgorithm.GZIP,
            chunk_size=1024,
        )
        index.chunks.append(
            ChunkInfo(index=0, original_size=1000, compressed_size=250, offset=0)
        )
        index.chunks.append(
            ChunkInfo(index=1, original_size=1000, compressed_size=260, offset=250)
        )

        serialized = index.to_bytes()
        deserialized = ChunkIndex.from_bytes(serialized)

        assert deserialized.algorithm == index.algorithm
        assert deserialized.chunk_size == index.chunk_size
        assert len(deserialized.chunks) == 2


class TestChunkedCompressor:
    """Tests for ChunkedCompressor."""

    def test_basic_chunked_compression(self):
        """Test basic chunked compression."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"X" * 500

        compressed, index = chunked.compress(data)

        assert len(compressed) > 0
        assert len(index.chunks) == 5  # 500 / 100

    def test_chunked_roundtrip(self):
        """Test chunked compression roundtrip."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"Test data for chunked compression! " * 50

        compressed, index = chunked.compress(data)
        decompressed = chunked.decompress(compressed, index)

        assert decompressed == data

    def test_compress_with_header(self):
        """Test compression with embedded header."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"Data with header " * 50

        result = chunked.compress_with_header(data)

        # Should be able to decompress without providing index
        decompressed = chunked.decompress(result)

        assert decompressed == data

    def test_decompress_specific_chunk(self):
        """Test decompressing a specific chunk."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"AAAA" * 25 + b"BBBB" * 25 + b"CCCC" * 25 + b"DDDD" * 25 + b"EEEE" * 25

        compressed, index = chunked.compress(data)

        # Decompress just chunk 0
        chunk_data = chunked.decompress_chunk(compressed, index, 0)

        assert len(chunk_data) == 100
        assert chunk_data == b"AAAA" * 25

    def test_chunk_checksum(self):
        """Test chunk checksums are verified."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"Checksum test " * 50

        compressed, index = chunked.compress(data)

        # Verify all chunks have checksums
        for chunk in index.chunks:
            assert len(chunk.checksum) == 16  # MD5

    def test_different_algorithms(self):
        """Test with different algorithms."""
        data = b"Algorithm test " * 100

        for algo in [CompressionAlgorithm.GZIP]:
            chunked = ChunkedCompressor(algorithm=algo)
            compressed, index = chunked.compress(data)
            decompressed = chunked.decompress(compressed, index)

            assert decompressed == data

    def test_small_data(self):
        """Test with data smaller than chunk size."""
        chunked = ChunkedCompressor(chunk_size=1000)
        data = b"Small"

        compressed, index = chunked.compress(data)
        decompressed = chunked.decompress(compressed, index)

        assert decompressed == data
        assert len(index.chunks) == 1

    def test_compress_iter(self):
        """Test compression with iterator."""
        chunked = ChunkedCompressor(chunk_size=100)

        def data_generator():
            for i in range(5):
                yield b"X" * 50

        chunks = list(chunked.compress_iter(data_generator()))

        assert len(chunks) >= 2  # 250 bytes / 100 chunk size

    def test_decompress_iter(self):
        """Test decompression with iterator."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"Iterator test " * 50

        compressed, index = chunked.compress(data)

        # Create iterator of compressed chunks
        def chunk_generator():
            for chunk_info in index.chunks:
                yield compressed[chunk_info.offset : chunk_info.offset + chunk_info.compressed_size]

        decompressed = b"".join(chunked.decompress_iter(chunk_generator(), index))

        assert decompressed == data


class TestStreamingRoundtrip:
    """Roundtrip tests for streaming compression."""

    def test_stream_compress_decompress(self):
        """Test streaming compression and decompression."""
        original = b"Streaming roundtrip test " * 100

        # Compress
        compressed_parts = []
        with StreamingCompressor() as compressor:
            compressed_parts.append(compressor.write(original[:500]))
            compressed_parts.append(compressor.write(original[500:]))
            compressed_parts.append(compressor.finalize())

        compressed = b"".join(p for p in compressed_parts if p)

        # Decompress
        compressor_obj = get_compressor(CompressionAlgorithm.GZIP)
        decompressed = compressor_obj.decompress(compressed)

        assert decompressed == original

    def test_large_data_chunked(self):
        """Test chunked compression with larger data."""
        chunked = ChunkedCompressor(chunk_size=1024)
        data = b"Large data test pattern! " * 1000

        compressed, index = chunked.compress(data)
        decompressed = chunked.decompress(compressed, index)

        assert decompressed == data
        assert index.total_compressed_size < index.total_original_size

    def test_parallel_chunk_decompression(self):
        """Test that chunks can be decompressed in parallel."""
        import concurrent.futures

        chunked = ChunkedCompressor(chunk_size=500)
        data = b"Parallel test " * 200

        compressed, index = chunked.compress(data)

        def decompress_chunk(chunk_idx):
            return chunked.decompress_chunk(compressed, index, chunk_idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(decompress_chunk, i): i for i in range(len(index.chunks))
            }
            results = {
                futures[f]: f.result() for f in concurrent.futures.as_completed(futures)
            }

        # Reassemble in order
        reassembled = b"".join(results[i] for i in range(len(index.chunks)))

        assert reassembled == data


class TestEdgeCases:
    """Edge case tests for streaming compression."""

    def test_empty_data_streaming(self):
        """Test streaming with empty data."""
        with StreamingCompressor() as compressor:
            final = compressor.finalize()

        assert final == b""

    def test_empty_data_chunked(self):
        """Test chunked with empty data."""
        chunked = ChunkedCompressor()
        data = b""

        compressed, index = chunked.compress(data)
        decompressed = chunked.decompress(compressed, index)

        assert decompressed == b""
        assert len(index.chunks) == 0

    def test_single_byte_streaming(self):
        """Test streaming with single byte."""
        with StreamingCompressor() as compressor:
            compressor.write(b"X")
            result = compressor.finalize()

        compressor_obj = get_compressor(CompressionAlgorithm.GZIP)
        assert compressor_obj.decompress(result) == b"X"

    def test_chunk_size_boundary(self):
        """Test data exactly at chunk boundary."""
        chunked = ChunkedCompressor(chunk_size=100)
        data = b"X" * 100  # Exactly one chunk

        compressed, index = chunked.compress(data)
        decompressed = chunked.decompress(compressed, index)

        assert decompressed == data
        assert len(index.chunks) == 1
