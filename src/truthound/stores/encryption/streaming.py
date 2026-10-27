"""Streaming encryption for large data handling.

This module provides memory-efficient encryption for large datasets
that cannot fit entirely in memory. It uses chunked encryption with
proper nonce management and authentication.

Security Considerations:
    - Each chunk has a unique nonce derived from base nonce + counter
    - Chunk reordering/deletion is detected via AEAD authentication
    - Partial reads are not supported (must read all chunks)
    - Consider compress-then-encrypt for optimal security

Example:
    >>> from truthound.stores.encryption.streaming import (
    ...     StreamingEncryptor,
    ...     StreamingDecryptor,
    ... )
    >>>
    >>> # Encrypt large file
    >>> with StreamingEncryptor(key, "output.enc") as enc:
    ...     for chunk in read_chunks("large_file.dat"):
    ...         enc.write(chunk)
    >>>
    >>> # Decrypt
    >>> with StreamingDecryptor(key, "output.enc") as dec:
    ...     for chunk in dec:
    ...         process(chunk)
"""

from __future__ import annotations

import io
import struct
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    EncryptionError,
    IntegrityError,
    generate_nonce,
)

# =============================================================================
# Chunk Metadata
# =============================================================================


@dataclass
class ChunkMetadata:
    """Metadata for an encrypted chunk.

    Attributes:
        index: Chunk index (0-based).
        nonce: Nonce used for this chunk.
        plaintext_size: Size of plaintext chunk.
        ciphertext_size: Size of ciphertext (including tag).
        offset: Byte offset in output stream.
    """

    index: int
    nonce: bytes
    plaintext_size: int
    ciphertext_size: int
    offset: int


@dataclass
class StreamingMetrics:
    """Metrics for streaming encryption/decryption.

    Attributes:
        total_chunks: Number of chunks processed.
        total_plaintext_bytes: Total plaintext bytes.
        total_ciphertext_bytes: Total ciphertext bytes.
        total_time_ms: Total processing time.
        algorithm: Encryption algorithm used.
    """

    total_chunks: int = 0
    total_plaintext_bytes: int = 0
    total_ciphertext_bytes: int = 0
    total_time_ms: float = 0.0
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.NONE

    @property
    def overhead_bytes(self) -> int:
        """Total encryption overhead."""
        return self.total_ciphertext_bytes - self.total_plaintext_bytes

    @property
    def throughput_mbps(self) -> float:
        """Processing throughput in MB/s."""
        if self.total_time_ms == 0:
            return 0.0
        return (self.total_plaintext_bytes / 1024 / 1024) / (self.total_time_ms / 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "total_plaintext_bytes": self.total_plaintext_bytes,
            "total_ciphertext_bytes": self.total_ciphertext_bytes,
            "overhead_bytes": self.overhead_bytes,
            "throughput_mbps": round(self.throughput_mbps, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "algorithm": self.algorithm.value,
        }


# =============================================================================
# Nonce Derivation
# =============================================================================


MAX_STREAM_CHUNK_SIZE = 16 * 1024 * 1024
_STREAM_ALGORITHMS = {
    1: EncryptionAlgorithm.AES_128_GCM,
    2: EncryptionAlgorithm.AES_256_GCM,
    3: EncryptionAlgorithm.CHACHA20_POLY1305,
    4: EncryptionAlgorithm.XCHACHA20_POLY1305,
}


def derive_chunk_nonce(base_nonce: bytes, chunk_index: int, nonce_size: int = 12) -> bytes:
    """Derive a counter nonce; reject invalid sizes and counter overflow."""
    if type(nonce_size) is not int or nonce_size < 1 or nonce_size > 32:
        raise ValueError("Invalid nonce size")
    if not isinstance(base_nonce, bytes) or len(base_nonce) != nonce_size:
        raise ValueError("Invalid base nonce size")
    if type(chunk_index) is not int or not 0 <= chunk_index < 1 << (nonce_size * 8):
        raise ValueError("Invalid nonce counter")
    return (int.from_bytes(base_nonce, "big") ^ chunk_index).to_bytes(nonce_size, "big")


@contextmanager
def _owned_stream_file(path, mode):
    with open(path, mode) as stream:
        yield stream


def _read_exact(stream, size):
    output = bytearray()
    while len(output) < size:
        chunk = stream.read(size - len(output))
        if not isinstance(chunk, bytes) or not chunk or len(chunk) > size - len(output):
            raise DecryptionError("Incomplete encrypted stream")
        output.extend(chunk)
    return bytes(output)


def _write_all(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if type(written) is not int or not 0 < written <= len(view):
            raise EncryptionError("Incomplete encrypted stream write")
        view = view[written:]


def _validate_stream_options(key, algorithm, chunk_size, aad):
    if algorithm not in _STREAM_ALGORITHMS.values():
        raise EncryptionError("Unsupported streaming algorithm")
    if not isinstance(key, bytes) or len(key) != algorithm.key_size:
        raise EncryptionError("Invalid streaming key size")
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_STREAM_CHUNK_SIZE:
        raise ValueError("chunk_size must be between 1 and 16777216")
    if aad is not None and not isinstance(aad, bytes):
        raise ValueError("aad must be bytes")


def _stream_aad(header, aad, index, *, final=False, total=0):
    context = aad if aad is not None else b""
    result = b"THSE-v2\0" + header + struct.pack(">BQ", int(final), index)
    result += struct.pack(">Q", len(context)) + context
    return result + (struct.pack(">Q", total) if final else b"")


@dataclass
class StreamingHeader:
    """THSE header; v2 binds these bytes into every AEAD record and footer.

    Header CRC detects accidental corruption only. Authentication is verified
    by record tags. v1 is readable only through explicit legacy opt-in.
    """

    version: int = 2
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    chunk_size: int = 64 * 1024
    base_nonce: bytes = b""
    reserved: bytes = b"\x00" * 6
    MAGIC = b"THSE"

    def to_bytes(self) -> bytes:
        import zlib

        if self.version not in {1, 2} or self.algorithm not in _STREAM_ALGORITHMS.values():
            raise EncryptionError("Unsupported streaming header")
        if (
            type(self.chunk_size) is not int
            or not 1 <= self.chunk_size <= MAX_STREAM_CHUNK_SIZE
            or len(self.base_nonce) != self.algorithm.nonce_size
            or self.reserved != b"\0" * 6
        ):
            raise EncryptionError("Invalid streaming header")
        code = next(
            code for code, algorithm in _STREAM_ALGORITHMS.items() if algorithm == self.algorithm
        )
        header = (
            self.MAGIC
            + struct.pack(">BBIB", self.version, code, self.chunk_size, len(self.base_nonce))
            + self.base_nonce
            + self.reserved
        )
        return header + zlib.crc32(header).to_bytes(4, "big")

    @classmethod
    def from_bytes(cls, data: bytes) -> StreamingHeader:
        import zlib

        if len(data) < 11 or data[:4] != cls.MAGIC:
            raise DecryptionError("Invalid streaming encryption header")
        version, code, chunk_size, nonce_len = struct.unpack(">BBIB", data[4:11])
        size = 21 + nonce_len
        if len(data) < size:
            raise DecryptionError("Incomplete streaming header")
        if zlib.crc32(data[: size - 4]).to_bytes(4, "big") != data[size - 4 : size]:
            raise IntegrityError("Header checksum mismatch")
        algorithm = _STREAM_ALGORITHMS.get(code)
        if (
            version not in {1, 2}
            or algorithm is None
            or nonce_len != algorithm.nonce_size
            or not 1 <= chunk_size <= MAX_STREAM_CHUNK_SIZE
            or data[11 + nonce_len : size - 4] != b"\0" * 6
        ):
            raise DecryptionError("Unsupported or invalid streaming header")
        return cls(
            version=version,
            algorithm=algorithm,
            chunk_size=chunk_size,
            base_nonce=data[11 : 11 + nonce_len],
            reserved=data[11 + nonce_len : size - 4],
        )

    @property
    def size(self) -> int:
        return 21 + len(self.base_nonce)


class StreamingEncryptor:
    """Chunk AEAD with authenticated v2 header/order/final row counts.

    Completion requires the authenticated final empty record. Failed writes or
    exceptional context exits abort without publishing a valid completion tag.
    Caller-owned outputs stay open; owned files may contain incomplete bytes.
    """

    def __init__(
        self,
        key: bytes,
        output: str | Path | BinaryIO,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        chunk_size: int = 64 * 1024,
        aad: bytes | None = None,
    ) -> None:
        from truthound.stores.encryption.providers import get_encryptor

        _validate_stream_options(key, algorithm, chunk_size, aad)
        self._encryptor = get_encryptor(algorithm)
        self._key, self._algorithm, self._chunk_size, self._aad = key, algorithm, chunk_size, aad
        self._lock = RLock()
        self._base_nonce = generate_nonce(algorithm)
        self._header_bytes = StreamingHeader(
            algorithm=algorithm, chunk_size=chunk_size, base_nonce=self._base_nonce
        ).to_bytes()
        self._owns_output = isinstance(output, (str, Path))
        self._io_stack = ExitStack()
        self._output = (
            self._io_stack.enter_context(_owned_stream_file(output, "wb"))
            if self._owns_output
            else output
        )
        self._chunk_index = 0
        self._buffer = bytearray()
        self._metrics = StreamingMetrics(algorithm=algorithm)
        self._start_time = None
        self._header_written = self._finalized = self._failed = self._closed = False
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def _require_open(self):
        self._check_guard()
        if self._failed or self._closed or self._finalized:
            raise EncryptionError("Encrypted stream is no longer writable")

    def _write_header(self):
        if not self._header_written:
            self._check_guard()
            _write_all(self._output, self._header_bytes)
            self._header_written = True

    def _encrypt_chunk(self, plaintext):
        self._check_guard()
        nonce = derive_chunk_nonce(self._base_nonce, self._chunk_index, self._algorithm.nonce_size)
        aad = _stream_aad(self._header_bytes, self._aad, self._chunk_index)
        result = self._encryptor.encrypt_with_metrics(plaintext, self._key, nonce=nonce, aad=aad)
        self._check_guard()
        data = result.ciphertext + result.tag
        _write_all(self._output, struct.pack(">I", len(data)) + data)
        self._metrics.total_chunks += 1
        self._metrics.total_plaintext_bytes += len(plaintext)
        self._metrics.total_ciphertext_bytes += len(data) + 4
        self._chunk_index += 1

    def write(self, data: bytes) -> int:
        with self._lock:
            self._require_open()
            if not isinstance(data, bytes):
                raise ValueError("data must be bytes")
            try:
                if self._start_time is None:
                    self._start_time = time.perf_counter()
                self._write_header()
                # Retain only the unfinished chunk rather than copying all input.
                view = memoryview(data)
                while view:
                    self._check_guard()
                    take = min(self._chunk_size - len(self._buffer), len(view))
                    self._buffer.extend(view[:take])
                    view = view[take:]
                    if len(self._buffer) == self._chunk_size:
                        self._encrypt_chunk(bytes(self._buffer))
                        self._buffer.clear()
                return len(data)
            except BaseException:
                self._failed = True
                raise

    def flush(self) -> None:
        with self._lock:
            self._require_open()
            try:
                self._output.flush()
            except BaseException:
                self._failed = True
                raise

    def finalize(self) -> StreamingMetrics:
        with self._lock:
            if self._finalized:
                return self._metrics
            self._require_open()
            try:
                self._write_header()
                if self._buffer:
                    self._encrypt_chunk(bytes(self._buffer))
                    self._buffer.clear()
                nonce = derive_chunk_nonce(
                    self._base_nonce, self._chunk_index, self._algorithm.nonce_size
                )
                aad = _stream_aad(
                    self._header_bytes,
                    self._aad,
                    self._chunk_index,
                    final=True,
                    total=self._metrics.total_plaintext_bytes,
                )
                self._check_guard()
                final = self._encryptor.encrypt_with_metrics(b"", self._key, nonce=nonce, aad=aad)
                self._check_guard()
                _write_all(self._output, struct.pack(">I", 0) + final.ciphertext + final.tag)
                self._output.flush()
                self._metrics.total_ciphertext_bytes += 4 + len(final.ciphertext) + len(final.tag)
                if self._start_time is not None:
                    self._metrics.total_time_ms = (time.perf_counter() - self._start_time) * 1000
                self._finalized = True
                return self._metrics
            except BaseException:
                self._failed = True
                raise

    @property
    def metrics(self) -> StreamingMetrics:
        return self._metrics

    def abort(self) -> None:
        with self._lock:
            self._failed = True
            self._buffer.clear()
            if self._owns_output and not self._closed:
                self._io_stack.close()
            self._closed = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if not self._failed and not self._finalized:
                    self.finalize()
            finally:
                if self._owns_output:
                    self._io_stack.close()
                self._closed = True

    def __enter__(self) -> StreamingEncryptor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()


class StreamingDecryptor:
    """Read v2 records; authenticated_complete is true only after verified EOF.

    Individual yielded chunks authenticate their own content, not completion of
    the entire stream. Consume to EOF before committing an all-stream result.
    Legacy v1 opt-in cannot authenticate completeness or header integrity.
    """

    def __init__(
        self,
        key: bytes,
        input_source: str | Path | BinaryIO,
        aad: bytes | None = None,
        *,
        allow_legacy: bool = False,
    ) -> None:
        from truthound.stores.encryption.providers import get_encryptor

        self._key, self._aad = key, aad
        self._lock = RLock()
        self._owns_input = isinstance(input_source, (str, Path))
        self._io_stack = ExitStack()
        self._input = (
            self._io_stack.enter_context(_owned_stream_file(input_source, "rb"))
            if self._owns_input
            else input_source
        )
        self._execution_guard = None
        self._finished = self._failed = self._closed = False
        self.authenticated_complete = False
        try:
            self._header = self._read_header()
            if self._header.version == 1 and allow_legacy is not True:
                raise DecryptionError("Legacy v1 requires explicit allow_legacy=True")
            self._algorithm = self._header.algorithm
            _validate_stream_options(key, self._algorithm, self._header.chunk_size, aad)
            self._base_nonce = self._header.base_nonce
            self._header_bytes = self._header.to_bytes()
            self._decryptor = get_encryptor(self._algorithm)
        except BaseException:
            if self._owns_input:
                self._io_stack.close()
            raise
        self._chunk_index = 0
        self._metrics = StreamingMetrics(algorithm=self._algorithm)
        self._start_time = None

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def _read_header(self):
        prefix = _read_exact(self._input, 11)
        if prefix[:4] != StreamingHeader.MAGIC or prefix[10] not in {12, 24}:
            raise DecryptionError("Invalid streaming header")
        return StreamingHeader.from_bytes(prefix + _read_exact(self._input, prefix[10] + 10))

    def _decrypt_record(self, data, *, final=False):
        self._check_guard()
        nonce = derive_chunk_nonce(self._base_nonce, self._chunk_index, self._algorithm.nonce_size)
        aad = (
            self._aad
            if self._header.version == 1
            else _stream_aad(
                self._header_bytes,
                self._aad,
                self._chunk_index,
                final=final,
                total=self._metrics.total_plaintext_bytes,
            )
        )
        try:
            result = self._decryptor.decrypt(nonce + data, self._key, aad=aad)
        except Exception as error:
            raise IntegrityError("Encrypted stream authentication failed") from error
        self._check_guard()
        return result

    def read_chunk(self) -> bytes | None:
        with self._lock:
            self._check_guard()
            if self._failed or self._closed:
                raise DecryptionError("Encrypted stream is no longer readable")
            if self._finished:
                return None
            if self._start_time is None:
                self._start_time = time.perf_counter()
            try:
                size = struct.unpack(">I", _read_exact(self._input, 4))[0]
                if size == 0:
                    if self._header.version == 2:
                        tag = _read_exact(self._input, self._algorithm.tag_size)
                        if self._decrypt_record(tag, final=True) != b"":
                            raise IntegrityError("Invalid final record")
                    if self._input.read(1) != b"":
                        raise DecryptionError("Trailing encrypted stream bytes")
                    self._check_guard()
                    self._finished = True
                    self.authenticated_complete = self._header.version == 2
                    self._metrics.total_time_ms = (time.perf_counter() - self._start_time) * 1000
                    return None
                if (
                    not self._algorithm.tag_size
                    <= size
                    <= self._header.chunk_size + self._algorithm.tag_size
                ):
                    raise DecryptionError("Invalid encrypted chunk size")
                data = _read_exact(self._input, size)
                plaintext = self._decrypt_record(data)
                self._chunk_index += 1
                self._metrics.total_chunks += 1
                self._metrics.total_plaintext_bytes += len(plaintext)
                self._metrics.total_ciphertext_bytes += size + 4
                return plaintext
            except BaseException:
                self._failed = True
                raise

    def __iter__(self) -> Iterator[bytes]:
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk

    def read_all(self) -> bytes:
        """Materialize only after authenticating the final completion record."""
        return b"".join(self)

    @property
    def metrics(self) -> StreamingMetrics:
        return self._metrics

    def close(self) -> None:
        if not self._closed:
            if self._owns_input:
                self._io_stack.close()
            self._closed = True

    def __enter__(self) -> StreamingDecryptor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# =============================================================================
# Chunked Encryption for Random Access
# =============================================================================


@dataclass
class ChunkIndex:
    """Random-access metadata; v2 authenticates the complete canonical index."""

    chunks: list[ChunkMetadata] = field(default_factory=list)
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    base_nonce: bytes = b""
    chunk_size: int = 64 * 1024
    total_plaintext_size: int = 0
    version: int = 2
    authentication_tag: bytes = b""

    def get_chunk_for_offset(self, offset: int) -> int:
        return offset // self.chunk_size

    def _record(self):
        return {
            "version": self.version,
            "algorithm": self.algorithm.value,
            "base_nonce": self.base_nonce.hex(),
            "chunk_size": self.chunk_size,
            "total_plaintext_size": self.total_plaintext_size,
            "chunks": [
                {
                    "index": c.index,
                    "nonce": c.nonce.hex(),
                    "plaintext_size": c.plaintext_size,
                    "ciphertext_size": c.ciphertext_size,
                    "offset": c.offset,
                }
                for c in self.chunks
            ],
        }

    def _authentication_payload(self, aad):
        import json

        context = aad if aad is not None else b""
        return (
            b"THCI-v2\0"
            + json.dumps(self._record(), sort_keys=True, separators=(",", ":")).encode()
            + struct.pack(">Q", len(context))
            + context
        )

    def to_bytes(self) -> bytes:
        import json

        record = self._record()
        record["authentication_tag"] = self.authentication_tag.hex()
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> ChunkIndex:
        import json

        try:
            record = json.loads(data)
            return cls(
                algorithm=EncryptionAlgorithm(record["algorithm"]),
                base_nonce=bytes.fromhex(record["base_nonce"]),
                chunk_size=record["chunk_size"],
                total_plaintext_size=record["total_plaintext_size"],
                version=record.get("version", 1),
                authentication_tag=bytes.fromhex(record.get("authentication_tag", "")),
                chunks=[
                    ChunkMetadata(
                        index=c["index"],
                        nonce=bytes.fromhex(c["nonce"]),
                        plaintext_size=c["plaintext_size"],
                        ciphertext_size=c["ciphertext_size"],
                        offset=c["offset"],
                    )
                    for c in record["chunks"]
                ],
            )
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise DecryptionError("Invalid encrypted chunk index") from error

    def _validate(self):
        if self.version not in {1, 2} or self.algorithm not in _STREAM_ALGORITHMS.values():
            raise DecryptionError("Unsupported encrypted chunk index")
        if (
            type(self.chunk_size) is not int
            or not 1 <= self.chunk_size <= MAX_STREAM_CHUNK_SIZE
            or type(self.total_plaintext_size) is not int
            or self.total_plaintext_size < 0
            or len(self.base_nonce) != self.algorithm.nonce_size
        ):
            raise DecryptionError("Invalid encrypted chunk index")
        offset = total = 0
        for number, chunk in enumerate(self.chunks):
            if (
                type(chunk.index) is not int
                or chunk.index != number
                or type(chunk.offset) is not int
                or chunk.offset != offset
                or type(chunk.plaintext_size) is not int
                or not 1 <= chunk.plaintext_size <= self.chunk_size
                or type(chunk.ciphertext_size) is not int
                or chunk.ciphertext_size != 4 + chunk.plaintext_size + self.algorithm.tag_size
                or chunk.nonce
                != derive_chunk_nonce(self.base_nonce, number, self.algorithm.nonce_size)
                or (number < len(self.chunks) - 1 and chunk.plaintext_size != self.chunk_size)
            ):
                raise DecryptionError("Invalid encrypted chunk metadata")
            total += chunk.plaintext_size
            offset += chunk.ciphertext_size
        if total != self.total_plaintext_size:
            raise DecryptionError("Encrypted chunk index size mismatch")
        return offset


class ChunkedEncryptor:
    """Independent chunk AEAD plus an authenticated v2 random-access index."""

    def __init__(
        self,
        key: bytes,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        chunk_size: int = 64 * 1024,
        aad: bytes | None = None,
    ) -> None:
        from truthound.stores.encryption.providers import get_encryptor

        _validate_stream_options(key, algorithm, chunk_size, aad)
        self._key, self._algorithm, self._chunk_size, self._aad = key, algorithm, chunk_size, aad
        self._encryptor = get_encryptor(algorithm)
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def encrypt(self, plaintext: bytes) -> tuple[bytes, ChunkIndex]:
        self._check_guard()
        if not isinstance(plaintext, bytes):
            raise ValueError("plaintext must be bytes")
        base_nonce = generate_nonce(self._algorithm)
        index = ChunkIndex(
            algorithm=self._algorithm,
            base_nonce=base_nonce,
            chunk_size=self._chunk_size,
            total_plaintext_size=len(plaintext),
        )
        output = io.BytesIO()
        for number, start in enumerate(range(0, len(plaintext), self._chunk_size)):
            self._check_guard()
            data = plaintext[start : start + self._chunk_size]
            nonce = derive_chunk_nonce(base_nonce, number, self._algorithm.nonce_size)
            result = self._encryptor.encrypt_with_metrics(
                data, self._key, nonce=nonce, aad=self._aad
            )
            self._check_guard()
            encrypted = result.ciphertext + result.tag
            index.chunks.append(
                ChunkMetadata(
                    index=number,
                    nonce=nonce,
                    plaintext_size=len(data),
                    ciphertext_size=len(encrypted) + 4,
                    offset=output.tell(),
                )
            )
            output.write(struct.pack(">I", len(encrypted)) + encrypted)
        self._check_guard()
        nonce = derive_chunk_nonce(base_nonce, len(index.chunks), self._algorithm.nonce_size)
        signature = self._encryptor.encrypt_with_metrics(
            b"", self._key, nonce=nonce, aad=index._authentication_payload(self._aad)
        )
        self._check_guard()
        index.authentication_tag = signature.ciphertext + signature.tag
        return output.getvalue(), index

    def encrypt_with_header(self, plaintext: bytes) -> bytes:
        """Bundle the authenticated native index and ciphertext for persistence."""
        ciphertext, index = self.encrypt(plaintext)
        record = index.to_bytes()
        return b"THCE" + struct.pack(">I", len(record)) + record + ciphertext

    def decrypt_with_header(
        self,
        data: bytes,
        *,
        start: int | None = None,
        end: int | None = None,
        max_output_bytes: int | None = None,
    ) -> bytes:
        """Restore a native bundle, optionally reading a validated byte range."""
        self._check_guard()
        if len(data) < 8 or data[:4] != b"THCE":
            raise DecryptionError("Invalid encrypted chunk bundle")
        size = struct.unpack(">I", data[4:8])[0]
        if not 0 < size <= len(data) - 8:
            raise DecryptionError("Invalid encrypted index size")
        index = ChunkIndex.from_bytes(data[8 : 8 + size])
        if index.algorithm != self._algorithm or index.chunk_size != self._chunk_size:
            raise DecryptionError("Chunk bundle does not match selected codec")
        native = ChunkedDecryptor(self._key, index, aad=self._aad)
        native.set_execution_guard(self._execution_guard)
        payload = data[8 + size :]
        if start is None and end is None:
            return native.decrypt_all(payload, max_output_bytes=max_output_bytes)
        if start is None or end is None:
            raise DecryptionError("A chunk range requires both bounds")
        return native.decrypt_range(payload, start, end, max_output_bytes=max_output_bytes)

    def encrypt_to_file(
        self, plaintext: bytes, output_path: str | Path, index_path: str | Path | None = None
    ) -> ChunkIndex:
        ciphertext, index = self.encrypt(plaintext)
        self._check_guard()
        with open(output_path, "wb") as output:
            _write_all(output, ciphertext)
        if index_path is not None:
            self._check_guard()
            with open(index_path, "wb") as output:
                _write_all(output, index.to_bytes())
        self._check_guard()
        return index


class ChunkedDecryptor:
    """Authenticate the captured index before returning any selected chunk."""

    def __init__(
        self, key: bytes, index: ChunkIndex, aad: bytes | None = None, *, allow_legacy: bool = False
    ) -> None:
        from copy import deepcopy

        from truthound.stores.encryption.providers import get_encryptor

        self._index = deepcopy(index)
        self._ciphertext_size = self._index._validate()
        _validate_stream_options(key, self._index.algorithm, self._index.chunk_size, aad)
        self._key, self._aad = key, aad
        self._decryptor = get_encryptor(self._index.algorithm)
        self._execution_guard = None
        self.authenticated_index = False
        if self._index.version == 1:
            if allow_legacy is not True:
                raise DecryptionError("Legacy chunk index requires explicit allow_legacy=True")
        else:
            if len(self._index.authentication_tag) != self._index.algorithm.tag_size:
                raise DecryptionError("Missing encrypted index authentication")
            nonce = derive_chunk_nonce(
                self._index.base_nonce, len(self._index.chunks), self._index.algorithm.nonce_size
            )
            try:
                plaintext = self._decryptor.decrypt(
                    nonce + self._index.authentication_tag,
                    key,
                    aad=self._index._authentication_payload(aad),
                )
                if plaintext != b"":
                    raise IntegrityError("Invalid encrypted index authentication")
            except Exception as error:
                raise IntegrityError("Encrypted chunk index authentication failed") from error
            self.authenticated_index = True

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    @staticmethod
    def _check_output_limit(size, maximum):
        if maximum is not None:
            if type(maximum) is not int or maximum < 0:
                raise ValueError("Invalid output size limit")
            if size > maximum:
                raise DecryptionError("Chunk output size limit exceeded")

    def decrypt_chunk(self, ciphertext: bytes | BinaryIO, chunk_index: int) -> bytes:
        self._check_guard()
        if type(chunk_index) is not int or not 0 <= chunk_index < len(self._index.chunks):
            raise DecryptionError("Encrypted chunk index out of range")
        metadata = self._index.chunks[chunk_index]
        if isinstance(ciphertext, bytes):
            raw = ciphertext[metadata.offset : metadata.offset + metadata.ciphertext_size]
            source = io.BytesIO(raw)
        else:
            ciphertext.seek(metadata.offset)
            source = ciphertext
        size = struct.unpack(">I", _read_exact(source, 4))[0]
        if size != metadata.ciphertext_size - 4:
            raise DecryptionError("Encrypted chunk length mismatch")
        data = _read_exact(source, size)
        self._check_guard()
        try:
            plaintext = self._decryptor.decrypt(metadata.nonce + data, self._key, aad=self._aad)
        except Exception as error:
            raise IntegrityError("Encrypted chunk authentication failed") from error
        self._check_guard()
        if len(plaintext) != metadata.plaintext_size:
            raise DecryptionError("Decrypted chunk length mismatch")
        return plaintext

    def decrypt_range(
        self,
        ciphertext: bytes | BinaryIO,
        start: int,
        end: int,
        *,
        max_output_bytes: int | None = None,
    ) -> bytes:
        self._check_guard()
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= self._index.total_plaintext_size
        ):
            raise DecryptionError("Invalid encrypted plaintext range")
        self._check_output_limit(end - start, max_output_bytes)
        if start == end:
            return b""
        first = self._index.get_chunk_for_offset(start)
        last = self._index.get_chunk_for_offset(end - 1)
        combined = b"".join(
            self.decrypt_chunk(ciphertext, number) for number in range(first, last + 1)
        )
        offset = start - first * self._index.chunk_size
        return combined[offset : offset + end - start]

    def decrypt_all(
        self, ciphertext: bytes | BinaryIO, *, max_output_bytes: int | None = None
    ) -> bytes:
        self._check_guard()
        self._check_output_limit(self._index.total_plaintext_size, max_output_bytes)
        if isinstance(ciphertext, bytes) and len(ciphertext) != self._ciphertext_size:
            raise DecryptionError("Encrypted payload size mismatch")
        result = b"".join(
            self.decrypt_chunk(ciphertext, number) for number in range(len(self._index.chunks))
        )
        if not isinstance(ciphertext, bytes):
            ciphertext.seek(self._ciphertext_size)
            if ciphertext.read(1) != b"":
                raise DecryptionError("Trailing encrypted payload bytes")
        self._check_guard()
        return result
