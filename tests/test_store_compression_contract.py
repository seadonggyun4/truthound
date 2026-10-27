"""Native codec metadata and configuration cannot silently change the result."""

import struct

import pytest

from truthound.stores.compression.adaptive import AdaptiveCompressor, AdaptiveConfig
from truthound.stores.compression.base import CompressionAlgorithm, CompressionError
from truthound.stores.compression.streaming import ChunkedCompressor

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("change", ["size", "flags", "algorithm"])
def test_adaptive_header_rejects_inconsistent_metadata(change):
    codec = AdaptiveCompressor()
    data = bytearray(codec.compress(b"short"))
    magic, algorithm, flags, size = struct.unpack(codec.HEADER_FORMAT, data[: codec.HEADER_SIZE])
    if change == "size":
        size += 10
    elif change == "flags":
        flags |= 0x80
    else:
        algorithm = 255
        flags = 0
    data[: codec.HEADER_SIZE] = struct.pack(codec.HEADER_FORMAT, magic, algorithm, flags, size)
    with pytest.raises(CompressionError):
        codec.decompress(bytes(data))


def test_adaptive_configured_sampling_is_observed_in_native_analysis():
    codec = AdaptiveCompressor(
        AdaptiveConfig(allowed_algorithms=[CompressionAlgorithm.GZIP], enable_sampling=True)
    )
    data = b"repeated-data" * 1000
    assert codec.decompress(codec.compress(data)) == data
    assert codec.last_analysis.sample_compression_ratios["gzip"] > 1


@pytest.mark.parametrize("change", ["version", "algorithm", "count", "offset"])
def test_chunked_header_is_checked_against_its_index(change):
    codec = ChunkedCompressor(chunk_size=4)
    data = bytearray(codec.compress_with_header(b"abcdefghijkl"))
    fields = list(struct.unpack(codec.HEADER_FORMAT, data[: codec.HEADER_SIZE]))
    position = {"version": 1, "algorithm": 2, "count": 4, "offset": 5}[change]
    fields[position] += 1
    data[: codec.HEADER_SIZE] = struct.pack(codec.HEADER_FORMAT, *fields)
    with pytest.raises(CompressionError):
        codec.decompress(bytes(data))


def test_chunked_random_access_checks_checksum_and_negative_index():
    codec = ChunkedCompressor(chunk_size=4, algorithm=CompressionAlgorithm.NONE)
    data, index = codec.compress(b"abcdefghijkl")
    altered = b"X" + data[1:]
    with pytest.raises(CompressionError):
        codec.decompress_chunk(altered, index, 0)
    with pytest.raises(CompressionError):
        codec.decompress_chunk(data, index, -1)


def test_chunked_iterator_rejects_missing_and_extra_chunks():
    codec = ChunkedCompressor(chunk_size=4)
    pieces = list(codec.compress_iter([b"abcdefghijkl"]))
    data, index = codec.compress(b"abcdefghijkl")
    with pytest.raises(CompressionError):
        list(codec.decompress_iter([piece[0] for piece in pieces[:-1]], index))
    with pytest.raises(CompressionError):
        list(codec.decompress_iter([piece[0] for piece in pieces] + [pieces[0][0]], index))


@pytest.mark.parametrize("kind", ["adaptive", "chunked_compression", "chunked_encryption"])
def test_native_declared_output_bound_is_checked_before_full_restore(kind):
    from truthound.stores.encryption.base import DecryptionError
    from truthound.stores.encryption.streaming import ChunkedEncryptor

    source = b"repeated" * 1000
    if kind == "adaptive":
        codec = AdaptiveCompressor(AdaptiveConfig(allowed_algorithms=[CompressionAlgorithm.GZIP]))
        data = codec.compress(source)
        restore = codec.decompress
    elif kind == "chunked_compression":
        codec = ChunkedCompressor(chunk_size=64)
        data = codec.compress_with_header(source)
        restore = codec.decompress
    else:
        codec = ChunkedEncryptor(bytes(range(32)), chunk_size=64)
        data = codec.encrypt_with_header(source)
        restore = codec.decrypt_with_header
    with pytest.raises((CompressionError, DecryptionError), match="size limit"):
        restore(data, max_output_bytes=1)
    assert restore(data, max_output_bytes=len(source)) == source


def test_stream_compressor_rejects_post_finalize_and_aborts_failed_context():
    from truthound.stores.compression.streaming import StreamingCompressor

    codec = StreamingCompressor()
    codec.write(b"pending")
    codec.finalize()
    with pytest.raises(CompressionError):
        codec.write(b"after final")
    aborted = StreamingCompressor()
    with pytest.raises(RuntimeError, match="abort"), aborted:
        aborted.write(b"unfinished")
        raise RuntimeError("abort")
    with pytest.raises(CompressionError):
        aborted.finalize()


def test_stream_compression_failure_latches_without_discard_and_retry(monkeypatch):
    from truthound.stores.compression.streaming import StreamingCompressor

    codec = StreamingCompressor()
    calls = []

    def failing(data):
        calls.append(data)
        raise CompressionError("native failure")

    monkeypatch.setattr(codec._compressor, "compress", failing)
    codec.write(b"unfinished")
    with pytest.raises(CompressionError):
        codec.flush()
    with pytest.raises(CompressionError):
        codec.flush()
    assert calls == [b"unfinished"]


def test_stream_decoder_cannot_resume_after_failed_frame_or_closed_context():
    from truthound.stores.compression.streaming import StreamingDecompressor

    codec = StreamingDecompressor()
    with pytest.raises(CompressionError):
        codec.write(b"invalid frame")
    with pytest.raises(CompressionError):
        codec.write(b"")
    with StreamingDecompressor() as closed:
        pass
    with pytest.raises(CompressionError):
        closed.write(b"")


def test_adaptive_allowed_algorithm_does_not_probe_unselected_optional_providers(monkeypatch):
    import truthound.stores.compression.adaptive as module

    def reject_unrelated_probe():
        raise RuntimeError("unselected provider discovery")

    monkeypatch.setattr(module, "list_available_algorithms", reject_unrelated_probe)
    codec = AdaptiveCompressor(AdaptiveConfig(allowed_algorithms=[CompressionAlgorithm.GZIP]))
    assert codec.analyze(b"repeated" * 100).sample_compression_ratios["gzip"] > 1
