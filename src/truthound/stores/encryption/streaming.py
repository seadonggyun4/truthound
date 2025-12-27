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
import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Iterator

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    EncryptionError,
    EncryptionHeader,
    EncryptionMetrics,
    IntegrityError,
    KeyDerivation,
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


def derive_chunk_nonce(base_nonce: bytes, chunk_index: int, nonce_size: int = 12) -> bytes:
    """Derive a unique nonce for a chunk.

    Uses counter mode: nonce = base_nonce XOR chunk_index

    Args:
        base_nonce: Base nonce (random).
        chunk_index: Chunk index (0-based).
        nonce_size: Required nonce size.

    Returns:
        Derived nonce for this chunk.
    """
    # Pad base_nonce if needed
    if len(base_nonce) < nonce_size:
        base_nonce = base_nonce + b"\x00" * (nonce_size - len(base_nonce))

    # Convert to integer, XOR with counter, convert back
    nonce_int = int.from_bytes(base_nonce[:nonce_size], "big")
    derived_int = nonce_int ^ chunk_index
    return derived_int.to_bytes(nonce_size, "big")


# =============================================================================
# Streaming Header Format
# =============================================================================


@dataclass
class StreamingHeader:
    """Header for streaming encrypted data.

    Format (binary):
        - Magic bytes: 4 bytes ("THSE")
        - Version: 1 byte
        - Algorithm: 1 byte
        - Chunk size: 4 bytes
        - Base nonce: 12-24 bytes
        - Reserved: 6 bytes
        - Header checksum: 4 bytes

    Total: 32-44 bytes
    """

    version: int = 1
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    chunk_size: int = 64 * 1024
    base_nonce: bytes = b""
    reserved: bytes = b"\x00" * 6

    MAGIC = b"THSE"  # TrutHound Streaming Encryption

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        import zlib

        # Algorithm mapping
        algo_map = {
            EncryptionAlgorithm.AES_128_GCM: 1,
            EncryptionAlgorithm.AES_256_GCM: 2,
            EncryptionAlgorithm.CHACHA20_POLY1305: 3,
            EncryptionAlgorithm.XCHACHA20_POLY1305: 4,
        }
        algo_byte = algo_map.get(self.algorithm, 2)

        header = (
            self.MAGIC
            + struct.pack("B", self.version)
            + struct.pack("B", algo_byte)
            + struct.pack(">I", self.chunk_size)
            + struct.pack("B", len(self.base_nonce))
            + self.base_nonce
            + self.reserved
        )

        # Add checksum
        checksum = zlib.crc32(header).to_bytes(4, "big")
        return header + checksum

    @classmethod
    def from_bytes(cls, data: bytes) -> "StreamingHeader":
        """Deserialize header from bytes."""
        import zlib

        if data[:4] != cls.MAGIC:
            raise DecryptionError("Invalid streaming encryption header")

        offset = 4
        version = data[offset]
        offset += 1

        algo_byte = data[offset]
        offset += 1
        algo_map = {
            1: EncryptionAlgorithm.AES_128_GCM,
            2: EncryptionAlgorithm.AES_256_GCM,
            3: EncryptionAlgorithm.CHACHA20_POLY1305,
            4: EncryptionAlgorithm.XCHACHA20_POLY1305,
        }
        algorithm = algo_map.get(algo_byte, EncryptionAlgorithm.AES_256_GCM)

        chunk_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4

        nonce_len = data[offset]
        offset += 1

        base_nonce = data[offset : offset + nonce_len]
        offset += nonce_len

        reserved = data[offset : offset + 6]
        offset += 6

        # Verify checksum
        expected_checksum = data[offset : offset + 4]
        actual_checksum = zlib.crc32(data[:offset]).to_bytes(4, "big")
        if expected_checksum != actual_checksum:
            raise IntegrityError("Header checksum mismatch")

        return cls(
            version=version,
            algorithm=algorithm,
            chunk_size=chunk_size,
            base_nonce=base_nonce,
            reserved=reserved,
        )

    @property
    def size(self) -> int:
        """Get header size in bytes."""
        return 4 + 1 + 1 + 4 + 1 + len(self.base_nonce) + 6 + 4


# =============================================================================
# Streaming Encryptor
# =============================================================================


class StreamingEncryptor:
    """Memory-efficient streaming encryption.

    Encrypts data in chunks, each with a unique nonce derived from
    a base nonce and chunk counter. Supports both file and stream output.

    Example:
        >>> key = generate_key(EncryptionAlgorithm.AES_256_GCM)
        >>>
        >>> # File output
        >>> with StreamingEncryptor(key, "output.enc") as enc:
        ...     enc.write(data_chunk_1)
        ...     enc.write(data_chunk_2)
        >>>
        >>> # Stream output
        >>> output = io.BytesIO()
        >>> with StreamingEncryptor(key, output) as enc:
        ...     enc.write(data)
    """

    def __init__(
        self,
        key: bytes,
        output: str | Path | BinaryIO,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        chunk_size: int = 64 * 1024,
        aad: bytes | None = None,
    ) -> None:
        """Initialize streaming encryptor.

        Args:
            key: Encryption key.
            output: Output file path or binary stream.
            algorithm: Encryption algorithm.
            chunk_size: Size of plaintext chunks.
            aad: Additional authenticated data (applied to each chunk).
        """
        self._key = key
        self._algorithm = algorithm
        self._chunk_size = chunk_size
        self._aad = aad
        self._lock = RLock()

        # Initialize output
        if isinstance(output, (str, Path)):
            self._output: BinaryIO = open(output, "wb")
            self._owns_output = True
        else:
            self._output = output
            self._owns_output = False

        # Generate base nonce
        self._base_nonce = generate_nonce(algorithm)
        self._chunk_index = 0
        self._buffer = io.BytesIO()

        # Metrics
        self._metrics = StreamingMetrics(algorithm=algorithm)
        self._start_time: float | None = None
        self._header_written = False
        self._finalized = False

        # Get encryptor
        self._encryptor = self._get_encryptor()

    def _get_encryptor(self) -> Any:
        """Get the appropriate encryptor."""
        from truthound.stores.encryption.providers import get_encryptor

        return get_encryptor(self._algorithm)

    def _write_header(self) -> None:
        """Write streaming header."""
        if self._header_written:
            return

        header = StreamingHeader(
            algorithm=self._algorithm,
            chunk_size=self._chunk_size,
            base_nonce=self._base_nonce,
        )
        self._output.write(header.to_bytes())
        self._header_written = True

    def _encrypt_chunk(self, plaintext: bytes) -> bytes:
        """Encrypt a single chunk."""
        nonce = derive_chunk_nonce(
            self._base_nonce,
            self._chunk_index,
            self._algorithm.nonce_size,
        )

        # Encrypt chunk
        result = self._encryptor.encrypt_with_metrics(
            plaintext, self._key, nonce=nonce, aad=self._aad
        )

        # Build chunk: [chunk_len (4 bytes)][ciphertext][tag]
        chunk_data = result.ciphertext + result.tag
        chunk_len = len(chunk_data)
        output = struct.pack(">I", chunk_len) + chunk_data

        # Update metrics
        self._metrics.total_chunks += 1
        self._metrics.total_plaintext_bytes += len(plaintext)
        self._metrics.total_ciphertext_bytes += len(output)

        self._chunk_index += 1
        return output

    def write(self, data: bytes) -> int:
        """Write data to encryption stream.

        Args:
            data: Plaintext data.

        Returns:
            Number of bytes written.
        """
        with self._lock:
            if self._finalized:
                raise EncryptionError("Cannot write to finalized stream")

            if self._start_time is None:
                self._start_time = time.perf_counter()

            self._write_header()

            # Add to buffer
            self._buffer.write(data)
            bytes_written = len(data)

            # Process complete chunks
            self._buffer.seek(0)
            buffer_data = self._buffer.read()

            while len(buffer_data) >= self._chunk_size:
                chunk = buffer_data[: self._chunk_size]
                buffer_data = buffer_data[self._chunk_size :]

                encrypted = self._encrypt_chunk(chunk)
                self._output.write(encrypted)

            # Keep remaining data in buffer
            self._buffer = io.BytesIO(buffer_data)
            self._buffer.seek(0, 2)  # Seek to end

            return bytes_written

    def flush(self) -> None:
        """Flush output stream."""
        self._output.flush()

    def finalize(self) -> StreamingMetrics:
        """Finalize encryption and get metrics.

        This writes any remaining buffered data and closes the stream.

        Returns:
            Streaming metrics.
        """
        with self._lock:
            if self._finalized:
                return self._metrics

            self._write_header()

            # Process remaining buffer
            self._buffer.seek(0)
            remaining = self._buffer.read()
            if remaining:
                encrypted = self._encrypt_chunk(remaining)
                self._output.write(encrypted)

            # Write end marker (zero-length chunk)
            self._output.write(struct.pack(">I", 0))

            self._output.flush()
            self._finalized = True

            # Calculate total time
            if self._start_time is not None:
                self._metrics.total_time_ms = (
                    time.perf_counter() - self._start_time
                ) * 1000

            return self._metrics

    def close(self) -> None:
        """Close the encryptor and output stream."""
        if not self._finalized:
            self.finalize()
        if self._owns_output:
            self._output.close()

    def __enter__(self) -> "StreamingEncryptor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# =============================================================================
# Streaming Decryptor
# =============================================================================


class StreamingDecryptor:
    """Memory-efficient streaming decryption.

    Decrypts data in chunks, verifying authentication for each chunk.
    Supports iteration over decrypted chunks.

    Example:
        >>> with StreamingDecryptor(key, "encrypted.dat") as dec:
        ...     for plaintext_chunk in dec:
        ...         process(plaintext_chunk)
    """

    def __init__(
        self,
        key: bytes,
        input_source: str | Path | BinaryIO,
        aad: bytes | None = None,
    ) -> None:
        """Initialize streaming decryptor.

        Args:
            key: Decryption key.
            input_source: Input file path or binary stream.
            aad: Additional authenticated data.
        """
        self._key = key
        self._aad = aad
        self._lock = RLock()

        # Initialize input
        if isinstance(input_source, (str, Path)):
            self._input: BinaryIO = open(input_source, "rb")
            self._owns_input = True
        else:
            self._input = input_source
            self._owns_input = False

        # Read header
        self._header = self._read_header()
        self._algorithm = self._header.algorithm
        self._base_nonce = self._header.base_nonce
        self._chunk_index = 0

        # Metrics
        self._metrics = StreamingMetrics(algorithm=self._algorithm)
        self._start_time: float | None = None
        self._finished = False

        # Get decryptor
        self._decryptor = self._get_decryptor()

    def _get_decryptor(self) -> Any:
        """Get the appropriate decryptor."""
        from truthound.stores.encryption.providers import get_encryptor

        return get_encryptor(self._algorithm)

    def _read_header(self) -> StreamingHeader:
        """Read and parse streaming header."""
        # Read enough for header (max ~44 bytes)
        header_data = self._input.read(64)
        if len(header_data) < 32:
            raise DecryptionError("Incomplete streaming header")

        header = StreamingHeader.from_bytes(header_data)

        # Seek to end of header
        self._input.seek(header.size)
        return header

    def _decrypt_chunk(self, chunk_data: bytes) -> bytes:
        """Decrypt a single chunk."""
        nonce = derive_chunk_nonce(
            self._base_nonce,
            self._chunk_index,
            self._algorithm.nonce_size,
        )

        # Reconstruct ciphertext format: nonce + ciphertext + tag
        ciphertext = nonce + chunk_data

        try:
            plaintext = self._decryptor.decrypt(ciphertext, self._key, aad=self._aad)
        except Exception as e:
            raise IntegrityError(
                f"Chunk {self._chunk_index} authentication failed: {e}",
                self._algorithm.value,
            ) from e

        # Update metrics
        self._metrics.total_chunks += 1
        self._metrics.total_ciphertext_bytes += len(chunk_data) + 4  # +4 for length
        self._metrics.total_plaintext_bytes += len(plaintext)

        self._chunk_index += 1
        return plaintext

    def read_chunk(self) -> bytes | None:
        """Read and decrypt the next chunk.

        Returns:
            Decrypted chunk or None if finished.
        """
        with self._lock:
            if self._finished:
                return None

            if self._start_time is None:
                self._start_time = time.perf_counter()

            # Read chunk length
            len_data = self._input.read(4)
            if len(len_data) < 4:
                self._finished = True
                return None

            chunk_len = struct.unpack(">I", len_data)[0]

            # Zero length = end marker
            if chunk_len == 0:
                self._finished = True
                self._metrics.total_time_ms = (
                    time.perf_counter() - self._start_time
                ) * 1000
                return None

            # Read chunk data
            chunk_data = self._input.read(chunk_len)
            if len(chunk_data) < chunk_len:
                raise DecryptionError("Incomplete chunk data")

            return self._decrypt_chunk(chunk_data)

    def __iter__(self) -> Iterator[bytes]:
        """Iterate over decrypted chunks."""
        while True:
            chunk = self.read_chunk()
            if chunk is None:
                break
            yield chunk

    def read_all(self) -> bytes:
        """Read and decrypt all data.

        WARNING: Loads all decrypted data into memory.

        Returns:
            Complete decrypted data.
        """
        chunks = []
        for chunk in self:
            chunks.append(chunk)
        return b"".join(chunks)

    @property
    def metrics(self) -> StreamingMetrics:
        """Get decryption metrics."""
        return self._metrics

    def close(self) -> None:
        """Close the decryptor and input stream."""
        if self._owns_input:
            self._input.close()

    def __enter__(self) -> "StreamingDecryptor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# =============================================================================
# Chunked Encryption for Random Access
# =============================================================================


@dataclass
class ChunkIndex:
    """Index for chunked encrypted data allowing random access.

    Attributes:
        chunks: List of chunk metadata.
        algorithm: Encryption algorithm.
        base_nonce: Base nonce for nonce derivation.
        chunk_size: Plaintext chunk size.
        total_plaintext_size: Total size of original data.
    """

    chunks: list[ChunkMetadata] = field(default_factory=list)
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    base_nonce: bytes = b""
    chunk_size: int = 64 * 1024
    total_plaintext_size: int = 0

    def get_chunk_for_offset(self, offset: int) -> int:
        """Get chunk index containing the given byte offset."""
        return offset // self.chunk_size

    def to_bytes(self) -> bytes:
        """Serialize index to bytes."""
        import json

        data = {
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
        return json.dumps(data).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChunkIndex":
        """Deserialize index from bytes."""
        import json

        d = json.loads(data.decode())
        return cls(
            algorithm=EncryptionAlgorithm(d["algorithm"]),
            base_nonce=bytes.fromhex(d["base_nonce"]),
            chunk_size=d["chunk_size"],
            total_plaintext_size=d["total_plaintext_size"],
            chunks=[
                ChunkMetadata(
                    index=c["index"],
                    nonce=bytes.fromhex(c["nonce"]),
                    plaintext_size=c["plaintext_size"],
                    ciphertext_size=c["ciphertext_size"],
                    offset=c["offset"],
                )
                for c in d["chunks"]
            ],
        )


class ChunkedEncryptor:
    """Chunked encryption with random access support.

    Unlike streaming encryption, chunked encryption maintains an index
    that allows decrypting individual chunks without reading the entire
    stream. Useful for large files where only portions need to be read.

    Example:
        >>> enc = ChunkedEncryptor(key)
        >>> ciphertext, index = enc.encrypt(large_data)
        >>>
        >>> # Later: decrypt only chunk 5
        >>> dec = ChunkedDecryptor(key, index)
        >>> chunk_5_data = dec.decrypt_chunk(ciphertext, 5)
    """

    def __init__(
        self,
        key: bytes,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        chunk_size: int = 64 * 1024,
        aad: bytes | None = None,
    ) -> None:
        """Initialize chunked encryptor.

        Args:
            key: Encryption key.
            algorithm: Encryption algorithm.
            chunk_size: Size of plaintext chunks.
            aad: Additional authenticated data.
        """
        self._key = key
        self._algorithm = algorithm
        self._chunk_size = chunk_size
        self._aad = aad

        from truthound.stores.encryption.providers import get_encryptor

        self._encryptor = get_encryptor(algorithm)

    def encrypt(self, plaintext: bytes) -> tuple[bytes, ChunkIndex]:
        """Encrypt data and return ciphertext with index.

        Args:
            plaintext: Data to encrypt.

        Returns:
            Tuple of (ciphertext, chunk_index).
        """
        base_nonce = generate_nonce(self._algorithm)
        index = ChunkIndex(
            algorithm=self._algorithm,
            base_nonce=base_nonce,
            chunk_size=self._chunk_size,
            total_plaintext_size=len(plaintext),
        )

        output = io.BytesIO()
        offset = 0
        chunk_idx = 0

        while offset < len(plaintext):
            chunk_data = plaintext[offset : offset + self._chunk_size]
            nonce = derive_chunk_nonce(
                base_nonce, chunk_idx, self._algorithm.nonce_size
            )

            result = self._encryptor.encrypt_with_metrics(
                chunk_data, self._key, nonce=nonce, aad=self._aad
            )

            # Write chunk: [len][ciphertext][tag]
            chunk_bytes = result.ciphertext + result.tag
            chunk_output = struct.pack(">I", len(chunk_bytes)) + chunk_bytes

            chunk_meta = ChunkMetadata(
                index=chunk_idx,
                nonce=nonce,
                plaintext_size=len(chunk_data),
                ciphertext_size=len(chunk_output),
                offset=output.tell(),
            )
            index.chunks.append(chunk_meta)

            output.write(chunk_output)
            offset += len(chunk_data)
            chunk_idx += 1

        return output.getvalue(), index

    def encrypt_to_file(
        self,
        plaintext: bytes,
        output_path: str | Path,
        index_path: str | Path | None = None,
    ) -> ChunkIndex:
        """Encrypt data to file.

        Args:
            plaintext: Data to encrypt.
            output_path: Output file path.
            index_path: Optional separate index file path.

        Returns:
            Chunk index.
        """
        ciphertext, index = self.encrypt(plaintext)

        with open(output_path, "wb") as f:
            f.write(ciphertext)

        if index_path:
            with open(index_path, "wb") as f:
                f.write(index.to_bytes())

        return index


class ChunkedDecryptor:
    """Chunked decryption with random access support.

    Example:
        >>> dec = ChunkedDecryptor(key, index)
        >>> chunk_data = dec.decrypt_chunk(ciphertext, chunk_index=5)
        >>> range_data = dec.decrypt_range(ciphertext, start=1000, end=2000)
    """

    def __init__(
        self,
        key: bytes,
        index: ChunkIndex,
        aad: bytes | None = None,
    ) -> None:
        """Initialize chunked decryptor.

        Args:
            key: Decryption key.
            index: Chunk index.
            aad: Additional authenticated data.
        """
        self._key = key
        self._index = index
        self._aad = aad

        from truthound.stores.encryption.providers import get_encryptor

        self._decryptor = get_encryptor(index.algorithm)

    def decrypt_chunk(
        self,
        ciphertext: bytes | BinaryIO,
        chunk_index: int,
    ) -> bytes:
        """Decrypt a single chunk.

        Args:
            ciphertext: Encrypted data or file-like object.
            chunk_index: Index of chunk to decrypt.

        Returns:
            Decrypted chunk data.
        """
        if chunk_index >= len(self._index.chunks):
            raise DecryptionError(f"Chunk index {chunk_index} out of range")

        chunk_meta = self._index.chunks[chunk_index]

        # Read chunk data
        if isinstance(ciphertext, bytes):
            offset = chunk_meta.offset
            chunk_len = struct.unpack(
                ">I", ciphertext[offset : offset + 4]
            )[0]
            chunk_data = ciphertext[offset + 4 : offset + 4 + chunk_len]
        else:
            ciphertext.seek(chunk_meta.offset)
            chunk_len = struct.unpack(">I", ciphertext.read(4))[0]
            chunk_data = ciphertext.read(chunk_len)

        # Decrypt
        nonce = chunk_meta.nonce
        full_ciphertext = nonce + chunk_data

        try:
            return self._decryptor.decrypt(full_ciphertext, self._key, aad=self._aad)
        except Exception as e:
            raise IntegrityError(
                f"Chunk {chunk_index} authentication failed",
                self._index.algorithm.value,
            ) from e

    def decrypt_range(
        self,
        ciphertext: bytes | BinaryIO,
        start: int,
        end: int,
    ) -> bytes:
        """Decrypt a byte range.

        Args:
            ciphertext: Encrypted data or file-like object.
            start: Start byte offset (inclusive).
            end: End byte offset (exclusive).

        Returns:
            Decrypted data for the range.
        """
        start_chunk = self._index.get_chunk_for_offset(start)
        end_chunk = self._index.get_chunk_for_offset(end - 1)

        chunks = []
        for i in range(start_chunk, end_chunk + 1):
            chunks.append(self.decrypt_chunk(ciphertext, i))

        # Combine chunks
        combined = b"".join(chunks)

        # Calculate offsets within combined data
        combined_start = start - (start_chunk * self._index.chunk_size)
        combined_end = combined_start + (end - start)

        return combined[combined_start:combined_end]

    def decrypt_all(self, ciphertext: bytes | BinaryIO) -> bytes:
        """Decrypt all chunks.

        Args:
            ciphertext: Encrypted data or file-like object.

        Returns:
            Complete decrypted data.
        """
        chunks = []
        for i in range(len(self._index.chunks)):
            chunks.append(self.decrypt_chunk(ciphertext, i))
        return b"".join(chunks)
