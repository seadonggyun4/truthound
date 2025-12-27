"""Encryption pipeline for combining compression and encryption.

This module provides a pipeline that combines compression and encryption
in the correct order (compress-then-encrypt) for optimal security and
efficiency.

Security Note:
    Always compress BEFORE encrypting. Encrypting first prevents effective
    compression (ciphertext appears random), while compressing ciphertext
    can leak information about the plaintext.

Example:
    >>> from truthound.stores.encryption.pipeline import (
    ...     EncryptionPipeline,
    ...     create_secure_pipeline,
    ... )
    >>>
    >>> # Create default secure pipeline
    >>> pipeline = create_secure_pipeline(key)
    >>> encrypted = pipeline.process(data)
    >>> decrypted = pipeline.reverse(encrypted)
    >>>
    >>> # Custom pipeline
    >>> pipeline = (
    ...     EncryptionPipeline()
    ...     .add_compression("zstd", level=CompressionLevel.HIGH)
    ...     .add_encryption("aes-256-gcm", key=key)
    ... )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    EncryptionHeader,
    EncryptionMetrics,
    KeyDerivation,
    generate_nonce,
    generate_salt,
)


# =============================================================================
# Pipeline Stage Types
# =============================================================================


class StageType(Enum):
    """Type of pipeline stage."""

    COMPRESSION = auto()
    ENCRYPTION = auto()
    TRANSFORM = auto()
    CHECKSUM = auto()


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        stage_type: Type of stage.
        input_size: Input data size.
        output_size: Output data size.
        time_ms: Processing time in milliseconds.
        extra: Additional stage-specific metrics.
    """

    stage_name: str
    stage_type: StageType
    input_size: int = 0
    output_size: int = 0
    time_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size_change_percent(self) -> float:
        """Calculate size change percentage."""
        if self.input_size == 0:
            return 0.0
        return ((self.output_size - self.input_size) / self.input_size) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type.name,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "size_change_percent": round(self.size_change_percent, 2),
            "time_ms": round(self.time_ms, 2),
            **self.extra,
        }


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire pipeline.

    Attributes:
        stages: Metrics for each stage.
        total_input_size: Original input size.
        total_output_size: Final output size.
        total_time_ms: Total processing time.
    """

    stages: list[StageMetrics] = field(default_factory=list)
    total_input_size: int = 0
    total_output_size: int = 0
    total_time_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio (before encryption overhead)."""
        compression_stages = [
            s for s in self.stages if s.stage_type == StageType.COMPRESSION
        ]
        if not compression_stages or compression_stages[0].input_size == 0:
            return 1.0
        return compression_stages[0].input_size / compression_stages[0].output_size

    @property
    def overhead_percent(self) -> float:
        """Encryption overhead percentage."""
        if self.total_input_size == 0:
            return 0.0
        # Find size after compression
        compression_stages = [
            s for s in self.stages if s.stage_type == StageType.COMPRESSION
        ]
        compressed_size = (
            compression_stages[-1].output_size
            if compression_stages
            else self.total_input_size
        )
        return ((self.total_output_size - compressed_size) / compressed_size) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_input_size": self.total_input_size,
            "total_output_size": self.total_output_size,
            "compression_ratio": round(self.compression_ratio, 2),
            "overhead_percent": round(self.overhead_percent, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "stages": [s.to_dict() for s in self.stages],
        }


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result from pipeline processing.

    Attributes:
        data: Processed data.
        metrics: Pipeline metrics.
        header: Pipeline header for decryption.
    """

    data: bytes
    metrics: PipelineMetrics
    header: "PipelineHeader"

    def to_bytes(self) -> bytes:
        """Serialize result with header."""
        header_bytes = self.header.to_bytes()
        return header_bytes + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "PipelineResult":
        """Deserialize result from bytes."""
        header, offset = PipelineHeader.from_bytes(data)
        return cls(
            data=data[offset:],
            metrics=PipelineMetrics(),  # Metrics not preserved
            header=header,
        )


# =============================================================================
# Pipeline Header
# =============================================================================


@dataclass
class PipelineHeader:
    """Header describing the pipeline configuration.

    This header enables self-describing encrypted data that can be
    decrypted without knowing the original pipeline configuration.

    Format:
        - Magic: 4 bytes ("THEP")
        - Version: 1 byte
        - Flags: 1 byte
        - Stage count: 1 byte
        - Reserved: 1 byte
        - For each stage:
            - Stage type: 1 byte
            - Config length: 2 bytes
            - Config data: variable
        - Header checksum: 4 bytes
    """

    version: int = 1
    stages: list[dict[str, Any]] = field(default_factory=list)
    flags: int = 0

    MAGIC = b"THEP"  # TrutHound Encryption Pipeline

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        import json
        import zlib

        header = (
            self.MAGIC
            + bytes([self.version, self.flags, len(self.stages), 0])
        )

        for stage in self.stages:
            stage_type = stage["type"]
            config = json.dumps(stage["config"]).encode()
            header += bytes([stage_type]) + len(config).to_bytes(2, "big") + config

        checksum = zlib.crc32(header).to_bytes(4, "big")
        return header + checksum

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["PipelineHeader", int]:
        """Deserialize header from bytes.

        Returns:
            Tuple of (header, bytes_consumed).
        """
        import json
        import zlib

        if data[:4] != cls.MAGIC:
            raise DecryptionError("Invalid pipeline header")

        version = data[4]
        flags = data[5]
        stage_count = data[6]

        offset = 8
        stages = []

        for _ in range(stage_count):
            stage_type = data[offset]
            offset += 1

            config_len = int.from_bytes(data[offset : offset + 2], "big")
            offset += 2

            config = json.loads(data[offset : offset + config_len].decode())
            offset += config_len

            stages.append({"type": stage_type, "config": config})

        # Verify checksum
        expected_checksum = data[offset : offset + 4]
        actual_checksum = zlib.crc32(data[:offset]).to_bytes(4, "big")
        if expected_checksum != actual_checksum:
            raise DecryptionError("Pipeline header checksum mismatch")

        return (
            cls(version=version, flags=flags, stages=stages),
            offset + 4,
        )


# =============================================================================
# Pipeline Stages
# =============================================================================


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    @property
    @abstractmethod
    def stage_type(self) -> StageType:
        """Get the stage type."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the stage name."""
        ...

    @abstractmethod
    def process(self, data: bytes) -> tuple[bytes, dict[str, Any]]:
        """Process data forward through the stage.

        Args:
            data: Input data.

        Returns:
            Tuple of (processed_data, config_for_reverse).
        """
        ...

    @abstractmethod
    def reverse(self, data: bytes, config: dict[str, Any]) -> bytes:
        """Process data backward through the stage.

        Args:
            data: Processed data.
            config: Configuration from forward pass.

        Returns:
            Original data.
        """
        ...


class CompressionStage(PipelineStage):
    """Pipeline stage for compression."""

    def __init__(
        self,
        algorithm: str = "gzip",
        level: Any = None,
    ) -> None:
        """Initialize compression stage.

        Args:
            algorithm: Compression algorithm name.
            level: Compression level.
        """
        self._algorithm = algorithm
        self._level = level
        self._compressor: Any = None

    @property
    def stage_type(self) -> StageType:
        return StageType.COMPRESSION

    @property
    def name(self) -> str:
        return f"compress_{self._algorithm}"

    def _get_compressor(self) -> Any:
        """Get compressor instance."""
        if self._compressor is None:
            from truthound.stores.compression import get_compressor

            self._compressor = get_compressor(self._algorithm, level=self._level)
        return self._compressor

    def process(self, data: bytes) -> tuple[bytes, dict[str, Any]]:
        """Compress data."""
        compressor = self._get_compressor()
        compressed = compressor.compress(data)
        config = {
            "algorithm": self._algorithm,
            "original_size": len(data),
        }
        return compressed, config

    def reverse(self, data: bytes, config: dict[str, Any]) -> bytes:
        """Decompress data."""
        compressor = self._get_compressor()
        return compressor.decompress(data)


class EncryptionStage(PipelineStage):
    """Pipeline stage for encryption."""

    def __init__(
        self,
        algorithm: EncryptionAlgorithm | str = EncryptionAlgorithm.AES_256_GCM,
        key: bytes | None = None,
        aad: bytes | None = None,
    ) -> None:
        """Initialize encryption stage.

        Args:
            algorithm: Encryption algorithm.
            key: Encryption key.
            aad: Additional authenticated data.
        """
        if isinstance(algorithm, str):
            algorithm = EncryptionAlgorithm(algorithm)
        self._algorithm = algorithm
        self._key = key
        self._aad = aad
        self._encryptor: Any = None

    @property
    def stage_type(self) -> StageType:
        return StageType.ENCRYPTION

    @property
    def name(self) -> str:
        return f"encrypt_{self._algorithm.value}"

    def set_key(self, key: bytes) -> None:
        """Set the encryption key."""
        self._key = key
        self._encryptor = None

    def _get_encryptor(self) -> Any:
        """Get encryptor instance."""
        if self._encryptor is None:
            from truthound.stores.encryption.providers import get_encryptor

            self._encryptor = get_encryptor(self._algorithm)
        return self._encryptor

    def process(self, data: bytes) -> tuple[bytes, dict[str, Any]]:
        """Encrypt data."""
        if self._key is None:
            raise EncryptionError("Encryption key not set")

        encryptor = self._get_encryptor()
        encrypted = encryptor.encrypt(data, self._key, aad=self._aad)

        config = {
            "algorithm": self._algorithm.value,
            "nonce_size": self._algorithm.nonce_size,
            "tag_size": self._algorithm.tag_size,
        }
        return encrypted, config

    def reverse(self, data: bytes, config: dict[str, Any]) -> bytes:
        """Decrypt data."""
        if self._key is None:
            raise DecryptionError("Decryption key not set")

        encryptor = self._get_encryptor()
        return encryptor.decrypt(data, self._key, aad=self._aad)


class ChecksumStage(PipelineStage):
    """Pipeline stage for checksum verification."""

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize checksum stage.

        Args:
            algorithm: Hash algorithm (sha256, sha512, blake2b).
        """
        self._algorithm = algorithm

    @property
    def stage_type(self) -> StageType:
        return StageType.CHECKSUM

    @property
    def name(self) -> str:
        return f"checksum_{self._algorithm}"

    def _hash(self, data: bytes) -> bytes:
        """Compute hash of data."""
        import hashlib

        if self._algorithm == "blake2b":
            return hashlib.blake2b(data, digest_size=32).digest()
        return hashlib.new(self._algorithm, data).digest()

    def process(self, data: bytes) -> tuple[bytes, dict[str, Any]]:
        """Add checksum to data."""
        checksum = self._hash(data)
        config = {
            "algorithm": self._algorithm,
            "checksum": checksum.hex(),
        }
        # Prepend checksum length and checksum
        return bytes([len(checksum)]) + checksum + data, config

    def reverse(self, data: bytes, config: dict[str, Any]) -> bytes:
        """Verify and remove checksum."""
        checksum_len = data[0]
        stored_checksum = data[1 : 1 + checksum_len]
        actual_data = data[1 + checksum_len :]

        computed_checksum = self._hash(actual_data)
        if stored_checksum != computed_checksum:
            raise DecryptionError(
                f"Checksum verification failed: expected {stored_checksum.hex()}, "
                f"got {computed_checksum.hex()}"
            )

        return actual_data


# =============================================================================
# Encryption Pipeline
# =============================================================================


class EncryptionPipeline:
    """Pipeline for combining compression and encryption.

    Provides a fluent interface for building secure data processing
    pipelines with compression, encryption, and integrity verification.

    Example:
        >>> pipeline = (
        ...     EncryptionPipeline()
        ...     .add_compression("zstd", level=CompressionLevel.HIGH)
        ...     .add_encryption("aes-256-gcm", key=my_key)
        ...     .add_checksum("sha256")
        ... )
        >>>
        >>> result = pipeline.process(sensitive_data)
        >>> original = pipeline.reverse(result.data, result.header)
    """

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self._stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> "EncryptionPipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add.

        Returns:
            Self for chaining.
        """
        self._stages.append(stage)
        return self

    def add_compression(
        self,
        algorithm: str = "gzip",
        level: Any = None,
    ) -> "EncryptionPipeline":
        """Add compression stage.

        Args:
            algorithm: Compression algorithm.
            level: Compression level.

        Returns:
            Self for chaining.
        """
        return self.add_stage(CompressionStage(algorithm, level))

    def add_encryption(
        self,
        algorithm: EncryptionAlgorithm | str = EncryptionAlgorithm.AES_256_GCM,
        key: bytes | None = None,
        aad: bytes | None = None,
    ) -> "EncryptionPipeline":
        """Add encryption stage.

        Args:
            algorithm: Encryption algorithm.
            key: Encryption key.
            aad: Additional authenticated data.

        Returns:
            Self for chaining.
        """
        return self.add_stage(EncryptionStage(algorithm, key, aad))

    def add_checksum(self, algorithm: str = "sha256") -> "EncryptionPipeline":
        """Add checksum verification stage.

        Args:
            algorithm: Hash algorithm.

        Returns:
            Self for chaining.
        """
        return self.add_stage(ChecksumStage(algorithm))

    def set_key(self, key: bytes) -> "EncryptionPipeline":
        """Set encryption key for all encryption stages.

        Args:
            key: Encryption key.

        Returns:
            Self for chaining.
        """
        for stage in self._stages:
            if isinstance(stage, EncryptionStage):
                stage.set_key(key)
        return self

    def process(self, data: bytes) -> PipelineResult:
        """Process data through all stages.

        Args:
            data: Input data.

        Returns:
            Pipeline result with encrypted data and metadata.
        """
        metrics = PipelineMetrics(total_input_size=len(data))
        header_stages: list[dict[str, Any]] = []

        current_data = data
        total_start = time.perf_counter()

        for stage in self._stages:
            start_time = time.perf_counter()
            input_size = len(current_data)

            current_data, config = stage.process(current_data)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            output_size = len(current_data)

            stage_metrics = StageMetrics(
                stage_name=stage.name,
                stage_type=stage.stage_type,
                input_size=input_size,
                output_size=output_size,
                time_ms=elapsed_ms,
            )
            metrics.stages.append(stage_metrics)

            header_stages.append({
                "type": stage.stage_type.value,
                "config": config,
            })

        metrics.total_output_size = len(current_data)
        metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

        header = PipelineHeader(stages=header_stages)

        return PipelineResult(
            data=current_data,
            metrics=metrics,
            header=header,
        )

    def reverse(self, data: bytes, header: PipelineHeader | None = None) -> bytes:
        """Reverse process data through all stages.

        Args:
            data: Processed data.
            header: Pipeline header (if not embedded in data).

        Returns:
            Original data.
        """
        # If header not provided, try to extract from data
        if header is None:
            header, offset = PipelineHeader.from_bytes(data)
            data = data[offset:]

        current_data = data

        # Process stages in reverse order
        for stage, stage_config in zip(
            reversed(self._stages),
            reversed(header.stages),
        ):
            current_data = stage.reverse(current_data, stage_config["config"])

        return current_data

    def process_to_bytes(self, data: bytes) -> bytes:
        """Process data and return bytes with embedded header.

        Args:
            data: Input data.

        Returns:
            Processed data with header.
        """
        result = self.process(data)
        return result.to_bytes()

    def reverse_from_bytes(self, data: bytes) -> bytes:
        """Reverse process data with embedded header.

        Args:
            data: Processed data with header.

        Returns:
            Original data.
        """
        result = PipelineResult.from_bytes(data)
        return self.reverse(result.data, result.header)


# =============================================================================
# Pre-built Pipelines
# =============================================================================


def create_secure_pipeline(
    key: bytes,
    compression: str = "gzip",
    encryption: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    include_checksum: bool = True,
) -> EncryptionPipeline:
    """Create a secure compress-then-encrypt pipeline.

    Args:
        key: Encryption key.
        compression: Compression algorithm.
        encryption: Encryption algorithm.
        include_checksum: Whether to include checksum verification.

    Returns:
        Configured pipeline.
    """
    pipeline = (
        EncryptionPipeline()
        .add_compression(compression)
        .add_encryption(encryption, key=key)
    )

    if include_checksum:
        pipeline.add_checksum()

    return pipeline


def create_fast_pipeline(
    key: bytes,
    compression: str = "lz4",
) -> EncryptionPipeline:
    """Create a fast pipeline optimized for speed.

    Uses LZ4 compression and ChaCha20-Poly1305 encryption
    for maximum throughput.

    Args:
        key: Encryption key.
        compression: Compression algorithm (default: lz4).

    Returns:
        Configured pipeline.
    """
    return (
        EncryptionPipeline()
        .add_compression(compression)
        .add_encryption(EncryptionAlgorithm.CHACHA20_POLY1305, key=key)
    )


def create_max_compression_pipeline(
    key: bytes,
    compression: str = "zstd",
) -> EncryptionPipeline:
    """Create a pipeline optimized for maximum compression.

    Uses Zstandard with high compression level.

    Args:
        key: Encryption key.
        compression: Compression algorithm (default: zstd).

    Returns:
        Configured pipeline.
    """
    from truthound.stores.compression import CompressionLevel

    return (
        EncryptionPipeline()
        .add_compression(compression, level=CompressionLevel.HIGH)
        .add_encryption(EncryptionAlgorithm.AES_256_GCM, key=key)
        .add_checksum()
    )


def create_password_pipeline(
    password: str,
    compression: str = "gzip",
    kdf: KeyDerivation = KeyDerivation.ARGON2ID,
) -> tuple[EncryptionPipeline, bytes]:
    """Create a pipeline with password-derived key.

    Args:
        password: Password for key derivation.
        compression: Compression algorithm.
        kdf: Key derivation function.

    Returns:
        Tuple of (pipeline, salt). Salt must be stored with the data.
    """
    from truthound.stores.encryption.keys import derive_key

    key, salt = derive_key(password, kdf=kdf)

    pipeline = (
        EncryptionPipeline()
        .add_compression(compression)
        .add_encryption(EncryptionAlgorithm.AES_256_GCM, key=key)
        .add_checksum()
    )

    return pipeline, salt


# =============================================================================
# Streaming Pipeline
# =============================================================================


class StreamingPipeline:
    """Streaming version of encryption pipeline for large data.

    Combines streaming compression and streaming encryption for
    memory-efficient processing of large datasets.

    Example:
        >>> pipeline = StreamingPipeline(key, "output.enc")
        >>> with pipeline as p:
        ...     for chunk in read_large_file():
        ...         p.write(chunk)
    """

    def __init__(
        self,
        key: bytes,
        output: Any,  # str | Path | BinaryIO
        compression: str = "gzip",
        encryption: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        chunk_size: int = 64 * 1024,
    ) -> None:
        """Initialize streaming pipeline.

        Args:
            key: Encryption key.
            output: Output file path or stream.
            compression: Compression algorithm.
            encryption: Encryption algorithm.
            chunk_size: Processing chunk size.
        """
        self._key = key
        self._output = output
        self._compression = compression
        self._encryption = encryption
        self._chunk_size = chunk_size

        self._compressor: Any = None
        self._encryptor: Any = None
        self._buffer = b""

    def __enter__(self) -> "StreamingPipeline":
        """Start the streaming pipeline."""
        from truthound.stores.compression import get_compressor
        from truthound.stores.encryption.streaming import StreamingEncryptor

        import io
        self._compressed_buffer = io.BytesIO()

        # Initialize compressor
        self._compressor = get_compressor(self._compression)

        # Initialize encryptor
        self._encryptor = StreamingEncryptor(
            self._key,
            self._output,
            self._encryption,
            self._chunk_size,
        )

        return self

    def write(self, data: bytes) -> int:
        """Write data to the pipeline.

        Args:
            data: Data to process.

        Returns:
            Number of bytes written.
        """
        # Compress data
        compressed = self._compressor.compress(data)

        # Write compressed data to encryptor
        self._encryptor.write(compressed)

        return len(data)

    def __exit__(self, *args: Any) -> None:
        """Finalize and close the pipeline."""
        # Finalize compression (flush any remaining data)
        # Note: Some compressors need explicit finalization
        self._encryptor.finalize()
        self._encryptor.close()
