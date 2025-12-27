"""Compression pipeline for chained transformations.

This module provides a pipeline architecture for composing multiple
compression stages and data transformations.

Example:
    >>> from truthound.stores.compression import (
    ...     CompressionPipeline,
    ...     PipelineStage,
    ...     DeduplicationTransform,
    ...     GzipCompressor,
    ... )
    >>>
    >>> pipeline = (
    ...     CompressionPipeline()
    ...     .add_transform(DeduplicationTransform())
    ...     .add_compression(GzipCompressor())
    ... )
    >>>
    >>> result = pipeline.process(data)
    >>> original = pipeline.reverse(result.data)
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

from truthound.stores.compression.base import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionError,
    CompressionMetrics,
    CompressionResult,
    Compressor,
    Decompressor,
)


# =============================================================================
# Exceptions
# =============================================================================


class PipelineError(CompressionError):
    """Error during pipeline processing."""

    def __init__(self, message: str, stage: str | None = None) -> None:
        self.stage = stage
        if stage:
            message = f"[Stage: {stage}] {message}"
        super().__init__(message)


class TransformError(PipelineError):
    """Error during transform operation."""

    pass


# =============================================================================
# Enums
# =============================================================================


class StageType(Enum):
    """Type of pipeline stage."""

    TRANSFORM = auto()  # Data transformation (dedup, delta, etc.)
    COMPRESS = auto()  # Compression stage
    ENCRYPT = auto()  # Encryption stage (placeholder)
    CHECKSUM = auto()  # Checksum/validation stage


class TransformDirection(Enum):
    """Direction of transformation."""

    FORWARD = auto()  # Apply transform
    REVERSE = auto()  # Reverse transform


# =============================================================================
# Transform Protocols and Base Classes
# =============================================================================


T = TypeVar("T", bytes, bytearray)


class Transform(ABC):
    """Base class for data transformations.

    Transforms are reversible operations that modify data before compression.
    Common transforms include deduplication, delta encoding, and dictionary
    preprocessing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get transform name."""
        ...

    @property
    def is_reversible(self) -> bool:
        """Check if transform is reversible."""
        return True

    @abstractmethod
    def apply(self, data: bytes) -> bytes:
        """Apply transformation to data.

        Args:
            data: Input data.

        Returns:
            Transformed data.
        """
        ...

    @abstractmethod
    def reverse(self, data: bytes) -> bytes:
        """Reverse transformation.

        Args:
            data: Transformed data.

        Returns:
            Original data.
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get transform statistics."""
        return {"name": self.name}


# =============================================================================
# Built-in Transforms
# =============================================================================


class IdentityTransform(Transform):
    """Identity transform that returns data unchanged."""

    @property
    def name(self) -> str:
        return "identity"

    def apply(self, data: bytes) -> bytes:
        return data

    def reverse(self, data: bytes) -> bytes:
        return data


class DeduplicationTransform(Transform):
    """Block-level deduplication transform.

    Identifies and removes duplicate blocks, replacing them with references.
    Effective for data with repeated patterns.

    Format:
        [4 bytes: block_size]
        [4 bytes: num_unique_blocks]
        [4 bytes: num_references]
        [unique_blocks...]
        [references: block_index for each original block position]
    """

    def __init__(self, block_size: int = 4096) -> None:
        """Initialize deduplication transform.

        Args:
            block_size: Size of blocks for deduplication.
        """
        self.block_size = block_size
        self._stats: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "deduplication"

    def apply(self, data: bytes) -> bytes:
        """Apply deduplication."""
        if len(data) < self.block_size * 2:
            # Too small to benefit from dedup
            return self._wrap_passthrough(data)

        # Split into blocks
        blocks: list[bytes] = []
        for i in range(0, len(data), self.block_size):
            blocks.append(data[i : i + self.block_size])

        # Find unique blocks
        block_hashes: dict[bytes, int] = {}  # hash -> index in unique_blocks
        unique_blocks: list[bytes] = []
        references: list[int] = []

        for block in blocks:
            block_hash = hashlib.md5(block).digest()
            if block_hash in block_hashes:
                references.append(block_hashes[block_hash])
            else:
                idx = len(unique_blocks)
                block_hashes[block_hash] = idx
                unique_blocks.append(block)
                references.append(idx)

        # Calculate stats
        original_size = len(data)
        dedup_ratio = len(unique_blocks) / len(blocks) if blocks else 1.0

        self._stats = {
            "original_blocks": len(blocks),
            "unique_blocks": len(unique_blocks),
            "dedup_ratio": dedup_ratio,
            "space_saved_percent": (1 - dedup_ratio) * 100,
        }

        # If no significant dedup, return passthrough
        if len(unique_blocks) >= len(blocks) * 0.9:
            return self._wrap_passthrough(data)

        # Build output
        output = bytearray()
        # Header: magic, block_size, num_unique, num_refs, last_block_size
        last_block_size = len(blocks[-1]) if blocks else 0
        output.extend(struct.pack("<4sIIII", b"DDUP", self.block_size, len(unique_blocks), len(references), last_block_size))

        # Unique blocks
        for block in unique_blocks:
            output.extend(struct.pack("<I", len(block)))
            output.extend(block)

        # References
        for ref in references:
            output.extend(struct.pack("<I", ref))

        return bytes(output)

    def reverse(self, data: bytes) -> bytes:
        """Reverse deduplication."""
        if len(data) < 4:
            raise TransformError("Invalid dedup data: too short", self.name)

        # Check for passthrough
        if data[:4] == b"PASS":
            size = struct.unpack("<I", data[4:8])[0]
            return data[8 : 8 + size]

        # Parse header
        if data[:4] != b"DDUP":
            raise TransformError("Invalid dedup magic", self.name)

        offset = 4
        block_size, num_unique, num_refs, last_block_size = struct.unpack("<IIII", data[offset : offset + 16])
        offset += 16

        # Read unique blocks
        unique_blocks: list[bytes] = []
        for _ in range(num_unique):
            block_len = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            unique_blocks.append(data[offset : offset + block_len])
            offset += block_len

        # Read references and reconstruct
        output = bytearray()
        for i in range(num_refs):
            ref = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4

            block = unique_blocks[ref]
            # Last block may be shorter
            if i == num_refs - 1 and last_block_size > 0:
                block = block[:last_block_size]
            output.extend(block)

        return bytes(output)

    def _wrap_passthrough(self, data: bytes) -> bytes:
        """Wrap data as passthrough (no dedup applied)."""
        return b"PASS" + struct.pack("<I", len(data)) + data

    def get_stats(self) -> dict[str, Any]:
        return {**super().get_stats(), **self._stats}


class DeltaEncodingTransform(Transform):
    """Delta encoding transform for numerical data.

    Stores differences between consecutive values instead of absolute values.
    Effective for time-series or sequential numerical data.

    Format:
        [4 bytes: magic 'DLTA']
        [4 bytes: original_length]
        [first_byte]
        [delta_bytes...]
    """

    @property
    def name(self) -> str:
        return "delta_encoding"

    def apply(self, data: bytes) -> bytes:
        """Apply delta encoding."""
        if len(data) < 2:
            return self._wrap_passthrough(data)

        output = bytearray()
        output.extend(b"DLTA")
        output.extend(struct.pack("<I", len(data)))
        output.append(data[0])

        for i in range(1, len(data)):
            delta = (data[i] - data[i - 1]) & 0xFF
            output.append(delta)

        return bytes(output)

    def reverse(self, data: bytes) -> bytes:
        """Reverse delta encoding."""
        if len(data) < 4:
            raise TransformError("Invalid delta data: too short", self.name)

        # Check for passthrough
        if data[:4] == b"PASS":
            size = struct.unpack("<I", data[4:8])[0]
            return data[8 : 8 + size]

        if data[:4] != b"DLTA":
            raise TransformError("Invalid delta magic", self.name)

        original_length = struct.unpack("<I", data[4:8])[0]
        output = bytearray()
        output.append(data[8])

        for i in range(1, original_length):
            value = (output[i - 1] + data[8 + i]) & 0xFF
            output.append(value)

        return bytes(output)

    def _wrap_passthrough(self, data: bytes) -> bytes:
        """Wrap data as passthrough."""
        return b"PASS" + struct.pack("<I", len(data)) + data


class RunLengthTransform(Transform):
    """Run-length encoding transform.

    Compresses consecutive repeated bytes.
    Effective for data with many repeated values.

    Format:
        [4 bytes: magic 'RLNC']
        [4 bytes: original_length]
        [encoded_data: count, value pairs]
    """

    def __init__(self, min_run: int = 4) -> None:
        """Initialize RLE transform.

        Args:
            min_run: Minimum run length to encode.
        """
        self.min_run = min_run

    @property
    def name(self) -> str:
        return "run_length"

    def apply(self, data: bytes) -> bytes:
        """Apply run-length encoding."""
        if len(data) < 2:
            return self._wrap_passthrough(data)

        output = bytearray()
        output.extend(b"RLNC")
        output.extend(struct.pack("<I", len(data)))

        i = 0
        while i < len(data):
            # Count consecutive bytes
            run_length = 1
            while i + run_length < len(data) and data[i + run_length] == data[i] and run_length < 255:
                run_length += 1

            if run_length >= self.min_run:
                # Encode as run
                output.append(0)  # Marker for run
                output.append(run_length)
                output.append(data[i])
            else:
                # Find literal sequence
                literal_start = i
                while i < len(data):
                    # Check if next is a run
                    next_run = 1
                    while i + next_run < len(data) and data[i + next_run] == data[i] and next_run < 255:
                        next_run += 1
                    if next_run >= self.min_run:
                        break
                    i += 1
                    if i - literal_start >= 127:
                        break

                literal_len = i - literal_start
                output.append(literal_len | 0x80)  # High bit = literal
                output.extend(data[literal_start:i])
                continue

            i += run_length

        return bytes(output)

    def reverse(self, data: bytes) -> bytes:
        """Reverse run-length encoding."""
        if len(data) < 4:
            raise TransformError("Invalid RLE data: too short", self.name)

        if data[:4] == b"PASS":
            size = struct.unpack("<I", data[4:8])[0]
            return data[8 : 8 + size]

        if data[:4] != b"RLNC":
            raise TransformError("Invalid RLE magic", self.name)

        original_length = struct.unpack("<I", data[4:8])[0]
        output = bytearray()
        i = 8

        while i < len(data) and len(output) < original_length:
            marker = data[i]
            i += 1

            if marker == 0:
                # Run
                run_length = data[i]
                value = data[i + 1]
                output.extend([value] * run_length)
                i += 2
            elif marker & 0x80:
                # Literal
                literal_len = marker & 0x7F
                output.extend(data[i : i + literal_len])
                i += literal_len

        return bytes(output[:original_length])

    def _wrap_passthrough(self, data: bytes) -> bytes:
        """Wrap data as passthrough."""
        return b"PASS" + struct.pack("<I", len(data)) + data


# =============================================================================
# Pipeline Stage
# =============================================================================


@dataclass
class PipelineStage:
    """A single stage in the compression pipeline.

    Attributes:
        name: Stage identifier.
        stage_type: Type of stage (transform, compress, etc.).
        processor: The actual processor (Transform or Compressor).
        enabled: Whether this stage is enabled.
        config: Optional stage-specific configuration.
    """

    name: str
    stage_type: StageType
    processor: Transform | Compressor | Any
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def process(self, data: bytes) -> bytes:
        """Process data through this stage."""
        if not self.enabled:
            return data

        if self.stage_type == StageType.TRANSFORM:
            return self.processor.apply(data)
        elif self.stage_type == StageType.COMPRESS:
            return self.processor.compress(data)
        elif self.stage_type == StageType.CHECKSUM:
            return self._add_checksum(data)
        else:
            return data

    def reverse(self, data: bytes) -> bytes:
        """Reverse the stage processing."""
        if not self.enabled:
            return data

        if self.stage_type == StageType.TRANSFORM:
            return self.processor.reverse(data)
        elif self.stage_type == StageType.COMPRESS:
            return self.processor.decompress(data)
        elif self.stage_type == StageType.CHECKSUM:
            return self._verify_checksum(data)
        else:
            return data

    def _add_checksum(self, data: bytes) -> bytes:
        """Add checksum to data."""
        checksum = hashlib.sha256(data).digest()
        return checksum + data

    def _verify_checksum(self, data: bytes) -> bytes:
        """Verify and remove checksum."""
        if len(data) < 32:
            raise PipelineError("Data too short for checksum", self.name)

        expected = data[:32]
        actual_data = data[32:]
        actual = hashlib.sha256(actual_data).digest()

        if expected != actual:
            raise PipelineError("Checksum verification failed", self.name)

        return actual_data


# =============================================================================
# Pipeline Metrics
# =============================================================================


@dataclass
class PipelineMetrics:
    """Metrics for pipeline processing.

    Attributes:
        total_time_ms: Total processing time.
        stage_metrics: Per-stage metrics.
        input_size: Original input size.
        output_size: Final output size.
        overall_ratio: Overall compression ratio.
    """

    total_time_ms: float = 0.0
    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    input_size: int = 0
    output_size: int = 0
    overall_ratio: float = 0.0

    def add_stage_metric(self, stage_name: str, input_size: int, output_size: int, time_ms: float) -> None:
        """Add metrics for a stage."""
        self.stage_metrics[stage_name] = {
            "input_size": input_size,
            "output_size": output_size,
            "time_ms": time_ms,
            "ratio": input_size / output_size if output_size > 0 else 0.0,
        }

    def update_totals(self) -> None:
        """Update total metrics."""
        if self.output_size > 0:
            self.overall_ratio = self.input_size / self.output_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time_ms": round(self.total_time_ms, 2),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "overall_ratio": round(self.overall_ratio, 2),
            "space_savings_percent": round((1 - self.output_size / self.input_size) * 100, 2) if self.input_size > 0 else 0.0,
            "stages": self.stage_metrics,
        }


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result of pipeline processing.

    Attributes:
        data: Processed data.
        metrics: Processing metrics.
        stage_order: Order of stages applied.
        config_snapshot: Configuration at time of processing.
    """

    data: bytes
    metrics: PipelineMetrics
    stage_order: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Compression Pipeline
# =============================================================================


class CompressionPipeline:
    """Composable compression pipeline.

    Allows chaining multiple transforms and compression stages for
    optimal compression of different data types.

    Example:
        >>> pipeline = (
        ...     CompressionPipeline("json_pipeline")
        ...     .add_transform(DeduplicationTransform())
        ...     .add_compression(get_compressor(CompressionAlgorithm.ZSTD))
        ...     .add_checksum()
        ... )
        >>>
        >>> result = pipeline.process(json_data)
        >>> original = pipeline.reverse(result.data)
    """

    def __init__(self, name: str = "default") -> None:
        """Initialize pipeline.

        Args:
            name: Pipeline name for identification.
        """
        self.name = name
        self._stages: list[PipelineStage] = []
        self._enabled = True

    def add_stage(self, stage: PipelineStage) -> "CompressionPipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add.

        Returns:
            Self for chaining.
        """
        self._stages.append(stage)
        return self

    def add_transform(self, transform: Transform, name: str | None = None) -> "CompressionPipeline":
        """Add a transform stage.

        Args:
            transform: Transform to add.
            name: Optional stage name.

        Returns:
            Self for chaining.
        """
        stage_name = name or transform.name
        stage = PipelineStage(name=stage_name, stage_type=StageType.TRANSFORM, processor=transform)
        return self.add_stage(stage)

    def add_compression(self, compressor: Compressor, name: str | None = None) -> "CompressionPipeline":
        """Add a compression stage.

        Args:
            compressor: Compressor to add.
            name: Optional stage name.

        Returns:
            Self for chaining.
        """
        stage_name = name or f"compress_{compressor.algorithm.value}"
        stage = PipelineStage(name=stage_name, stage_type=StageType.COMPRESS, processor=compressor)
        return self.add_stage(stage)

    def add_checksum(self, name: str = "checksum") -> "CompressionPipeline":
        """Add a checksum verification stage.

        Args:
            name: Stage name.

        Returns:
            Self for chaining.
        """
        stage = PipelineStage(name=name, stage_type=StageType.CHECKSUM, processor=None)
        return self.add_stage(stage)

    def process(self, data: bytes) -> PipelineResult:
        """Process data through the pipeline.

        Args:
            data: Input data.

        Returns:
            Pipeline result with processed data and metrics.
        """
        import time

        metrics = PipelineMetrics()
        metrics.input_size = len(data)
        stage_order: list[str] = []

        start_total = time.perf_counter()
        current_data = data

        for stage in self._stages:
            if not stage.enabled:
                continue

            stage_start = time.perf_counter()
            input_size = len(current_data)

            try:
                current_data = stage.process(current_data)
            except Exception as e:
                raise PipelineError(f"Stage failed: {e}", stage.name)

            stage_time = (time.perf_counter() - stage_start) * 1000
            metrics.add_stage_metric(stage.name, input_size, len(current_data), stage_time)
            stage_order.append(stage.name)

        metrics.total_time_ms = (time.perf_counter() - start_total) * 1000
        metrics.output_size = len(current_data)
        metrics.update_totals()

        return PipelineResult(
            data=current_data,
            metrics=metrics,
            stage_order=stage_order,
            config_snapshot=self._get_config_snapshot(),
        )

    def reverse(self, data: bytes) -> bytes:
        """Reverse pipeline processing.

        Args:
            data: Processed data.

        Returns:
            Original data.
        """
        current_data = data

        # Reverse stages in reverse order
        for stage in reversed(self._stages):
            if not stage.enabled:
                continue

            try:
                current_data = stage.reverse(current_data)
            except Exception as e:
                raise PipelineError(f"Reverse stage failed: {e}", stage.name)

        return current_data

    def enable_stage(self, name: str, enabled: bool = True) -> "CompressionPipeline":
        """Enable or disable a stage by name.

        Args:
            name: Stage name.
            enabled: Whether to enable the stage.

        Returns:
            Self for chaining.
        """
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = enabled
                break
        return self

    def remove_stage(self, name: str) -> "CompressionPipeline":
        """Remove a stage by name.

        Args:
            name: Stage name to remove.

        Returns:
            Self for chaining.
        """
        self._stages = [s for s in self._stages if s.name != name]
        return self

    @property
    def stages(self) -> list[PipelineStage]:
        """Get all stages."""
        return self._stages.copy()

    def _get_config_snapshot(self) -> dict[str, Any]:
        """Get current configuration snapshot."""
        return {
            "name": self.name,
            "stages": [
                {
                    "name": s.name,
                    "type": s.stage_type.name,
                    "enabled": s.enabled,
                }
                for s in self._stages
            ],
        }

    def clone(self, new_name: str | None = None) -> "CompressionPipeline":
        """Create a copy of this pipeline.

        Args:
            new_name: Name for the cloned pipeline.

        Returns:
            Cloned pipeline.
        """
        import copy

        cloned = CompressionPipeline(new_name or f"{self.name}_clone")
        cloned._stages = copy.deepcopy(self._stages)
        return cloned


# =============================================================================
# Pre-built Pipelines
# =============================================================================


def create_text_pipeline() -> CompressionPipeline:
    """Create a pipeline optimized for text data."""
    from truthound.stores.compression.providers import get_compressor

    return CompressionPipeline("text").add_compression(get_compressor(CompressionAlgorithm.GZIP))


def create_json_pipeline() -> CompressionPipeline:
    """Create a pipeline optimized for JSON data."""
    from truthound.stores.compression.providers import get_compressor

    return (
        CompressionPipeline("json").add_transform(DeduplicationTransform(block_size=1024)).add_compression(get_compressor(CompressionAlgorithm.GZIP))
    )


def create_binary_pipeline() -> CompressionPipeline:
    """Create a pipeline optimized for binary data."""
    from truthound.stores.compression.providers import get_compressor, is_algorithm_available

    # Prefer ZSTD, fall back to GZIP
    algo = CompressionAlgorithm.ZSTD if is_algorithm_available(CompressionAlgorithm.ZSTD) else CompressionAlgorithm.GZIP

    return (
        CompressionPipeline("binary")
        .add_transform(RunLengthTransform())
        .add_transform(DeduplicationTransform())
        .add_compression(get_compressor(algo))
    )


def create_timeseries_pipeline() -> CompressionPipeline:
    """Create a pipeline optimized for time-series data."""
    from truthound.stores.compression.providers import get_compressor, is_algorithm_available

    # Prefer ZSTD, fall back to GZIP
    algo = CompressionAlgorithm.ZSTD if is_algorithm_available(CompressionAlgorithm.ZSTD) else CompressionAlgorithm.GZIP

    return (
        CompressionPipeline("timeseries")
        .add_transform(DeltaEncodingTransform())
        .add_compression(get_compressor(algo))
    )
