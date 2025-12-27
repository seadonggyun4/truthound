"""Adaptive compression with automatic algorithm selection.

This module provides intelligent compression that automatically selects
the best algorithm based on data characteristics.

Example:
    >>> from truthound.stores.compression import (
    ...     AdaptiveCompressor,
    ...     AdaptiveConfig,
    ... )
    >>>
    >>> compressor = AdaptiveCompressor()
    >>> result = compressor.compress_with_metrics(data)
    >>> print(f"Selected: {result.metrics.algorithm}")
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from truthound.stores.compression.base import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionError,
    CompressionLevel,
    CompressionMetrics,
    CompressionResult,
)
from truthound.stores.compression.providers import (
    get_compressor,
    list_available_algorithms,
)


# =============================================================================
# Data Characteristics Analysis
# =============================================================================


class DataType(Enum):
    """Detected data type."""

    UNKNOWN = auto()
    TEXT = auto()
    JSON = auto()
    BINARY = auto()
    NUMERIC = auto()
    STRUCTURED = auto()
    RANDOM = auto()
    HIGHLY_COMPRESSIBLE = auto()


class CompressionGoal(Enum):
    """Compression optimization goal."""

    BALANCED = auto()  # Balance speed and ratio
    BEST_RATIO = auto()  # Maximum compression ratio
    BEST_SPEED = auto()  # Maximum speed
    LOW_MEMORY = auto()  # Minimize memory usage


@dataclass
class DataCharacteristics:
    """Analyzed characteristics of input data.

    Attributes:
        size: Data size in bytes.
        entropy: Shannon entropy (0-8 bits per byte).
        is_text: Whether data appears to be text.
        is_binary: Whether data appears to be binary.
        repetition_score: Score indicating repetition (0-1).
        pattern_score: Score indicating patterns (0-1).
        detected_type: Detected data type.
        sample_compression_ratios: Quick compression test results.
    """

    size: int = 0
    entropy: float = 0.0
    is_text: bool = False
    is_binary: bool = False
    repetition_score: float = 0.0
    pattern_score: float = 0.0
    detected_type: DataType = DataType.UNKNOWN
    sample_compression_ratios: dict[str, float] = field(default_factory=dict)

    @property
    def compressibility_estimate(self) -> float:
        """Estimate compressibility from 0 (not compressible) to 1 (highly compressible)."""
        # Lower entropy = more compressible
        # Max entropy is 8 bits per byte
        entropy_factor = 1 - (self.entropy / 8)

        # Higher repetition = more compressible
        rep_factor = self.repetition_score

        # Combine factors
        return (entropy_factor * 0.6 + rep_factor * 0.4)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "size": self.size,
            "entropy": round(self.entropy, 3),
            "is_text": self.is_text,
            "is_binary": self.is_binary,
            "repetition_score": round(self.repetition_score, 3),
            "pattern_score": round(self.pattern_score, 3),
            "detected_type": self.detected_type.name,
            "compressibility_estimate": round(self.compressibility_estimate, 3),
            "sample_ratios": self.sample_compression_ratios,
        }


class DataAnalyzer:
    """Analyzes data characteristics for compression optimization."""

    def __init__(self, sample_size: int = 65536) -> None:
        """Initialize analyzer.

        Args:
            sample_size: Maximum bytes to sample for analysis.
        """
        self.sample_size = sample_size

    def analyze(self, data: bytes) -> DataCharacteristics:
        """Analyze data characteristics.

        Args:
            data: Data to analyze.

        Returns:
            Data characteristics.
        """
        chars = DataCharacteristics(size=len(data))

        # Use sample for analysis if data is large
        sample = data[: self.sample_size] if len(data) > self.sample_size else data

        # Calculate entropy
        chars.entropy = self._calculate_entropy(sample)

        # Check if text
        chars.is_text = self._is_text(sample)
        chars.is_binary = not chars.is_text

        # Calculate repetition score
        chars.repetition_score = self._calculate_repetition(sample)

        # Calculate pattern score
        chars.pattern_score = self._calculate_pattern_score(sample)

        # Detect type
        chars.detected_type = self._detect_type(sample, chars)

        return chars

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy in bits per byte."""
        if len(data) == 0:
            return 0.0

        # Count byte frequencies
        freq = [0] * 256
        for byte in data:
            freq[byte] += 1

        # Calculate entropy
        entropy = 0.0
        length = len(data)
        for count in freq:
            if count > 0:
                prob = count / length
                entropy -= prob * math.log2(prob)

        return entropy

    def _is_text(self, data: bytes) -> bool:
        """Check if data appears to be text."""
        if len(data) == 0:
            return False

        # Count printable ASCII and common control characters
        text_chars = 0
        for byte in data:
            if 32 <= byte <= 126 or byte in (9, 10, 13):  # Printable or tab/newline/cr
                text_chars += 1

        return text_chars / len(data) > 0.85

    def _calculate_repetition(self, data: bytes) -> float:
        """Calculate repetition score (0-1)."""
        if len(data) < 4:
            return 0.0

        # Count consecutive repeated bytes
        repeated = 0
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                repeated += 1

        return repeated / (len(data) - 1)

    def _calculate_pattern_score(self, data: bytes) -> float:
        """Calculate pattern detection score."""
        if len(data) < 8:
            return 0.0

        # Look for 4-byte patterns
        patterns: dict[bytes, int] = {}
        for i in range(len(data) - 3):
            pattern = data[i : i + 4]
            patterns[pattern] = patterns.get(pattern, 0) + 1

        # Score based on repeated patterns
        total_patterns = len(data) - 3
        repeated = sum(c - 1 for c in patterns.values() if c > 1)

        return min(repeated / total_patterns, 1.0) if total_patterns > 0 else 0.0

    def _detect_type(self, data: bytes, chars: DataCharacteristics) -> DataType:
        """Detect data type from characteristics."""
        if len(data) == 0:
            return DataType.UNKNOWN

        # Check for JSON
        if chars.is_text and (data.startswith(b"{") or data.startswith(b"[")):
            return DataType.JSON

        # Check for high entropy (random/encrypted)
        if chars.entropy > 7.5:
            return DataType.RANDOM

        # Check for highly compressible
        if chars.entropy < 3.0 or chars.repetition_score > 0.5:
            return DataType.HIGHLY_COMPRESSIBLE

        # Check for structured binary
        if chars.is_binary and chars.pattern_score > 0.3:
            return DataType.STRUCTURED

        # Text
        if chars.is_text:
            return DataType.TEXT

        # Binary
        if chars.is_binary:
            return DataType.BINARY

        return DataType.UNKNOWN


# =============================================================================
# Compression Profile
# =============================================================================


@dataclass
class CompressionProfile:
    """Profile for a compression algorithm's suitability.

    Attributes:
        algorithm: Compression algorithm.
        score: Suitability score (0-1).
        expected_ratio: Expected compression ratio.
        expected_speed_mbps: Expected speed in MB/s.
        reason: Reason for this score.
    """

    algorithm: CompressionAlgorithm
    score: float = 0.0
    expected_ratio: float = 1.0
    expected_speed_mbps: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "score": round(self.score, 3),
            "expected_ratio": round(self.expected_ratio, 2),
            "expected_speed_mbps": round(self.expected_speed_mbps, 2),
            "reason": self.reason,
        }


class AlgorithmSelector:
    """Selects best compression algorithm based on data and goals."""

    # Algorithm characteristics (approximate)
    ALGORITHM_PROFILES = {
        CompressionAlgorithm.LZ4: {
            "speed_compress": 400,  # MB/s
            "speed_decompress": 2000,
            "ratio_text": 2.0,
            "ratio_binary": 1.5,
            "memory": "low",
        },
        CompressionAlgorithm.SNAPPY: {
            "speed_compress": 350,
            "speed_decompress": 1500,
            "ratio_text": 1.8,
            "ratio_binary": 1.4,
            "memory": "low",
        },
        CompressionAlgorithm.GZIP: {
            "speed_compress": 50,
            "speed_decompress": 300,
            "ratio_text": 3.0,
            "ratio_binary": 2.0,
            "memory": "medium",
        },
        CompressionAlgorithm.ZSTD: {
            "speed_compress": 150,
            "speed_decompress": 500,
            "ratio_text": 3.2,
            "ratio_binary": 2.5,
            "memory": "medium",
        },
        CompressionAlgorithm.BROTLI: {
            "speed_compress": 30,
            "speed_decompress": 300,
            "ratio_text": 3.5,
            "ratio_binary": 2.2,
            "memory": "high",
        },
        CompressionAlgorithm.LZMA: {
            "speed_compress": 10,
            "speed_decompress": 100,
            "ratio_text": 4.0,
            "ratio_binary": 3.0,
            "memory": "high",
        },
        CompressionAlgorithm.BZ2: {
            "speed_compress": 15,
            "speed_decompress": 50,
            "ratio_text": 3.8,
            "ratio_binary": 2.8,
            "memory": "medium",
        },
    }

    def __init__(self, available_algorithms: list[CompressionAlgorithm] | None = None) -> None:
        """Initialize selector.

        Args:
            available_algorithms: List of available algorithms.
        """
        if available_algorithms:
            self.available = available_algorithms
        else:
            self.available = list_available_algorithms()

    def select(
        self,
        characteristics: DataCharacteristics,
        goal: CompressionGoal = CompressionGoal.BALANCED,
    ) -> CompressionProfile:
        """Select best algorithm for data.

        Args:
            characteristics: Data characteristics.
            goal: Optimization goal.

        Returns:
            Best compression profile.
        """
        profiles = self.rank_algorithms(characteristics, goal)
        return profiles[0] if profiles else CompressionProfile(CompressionAlgorithm.GZIP)

    def rank_algorithms(
        self,
        characteristics: DataCharacteristics,
        goal: CompressionGoal = CompressionGoal.BALANCED,
    ) -> list[CompressionProfile]:
        """Rank all algorithms by suitability.

        Args:
            characteristics: Data characteristics.
            goal: Optimization goal.

        Returns:
            List of profiles sorted by score (descending).
        """
        profiles: list[CompressionProfile] = []

        for algo in self.available:
            if algo == CompressionAlgorithm.NONE:
                continue

            profile = self._score_algorithm(algo, characteristics, goal)
            profiles.append(profile)

        # Sort by score descending
        profiles.sort(key=lambda p: p.score, reverse=True)
        return profiles

    def _score_algorithm(
        self,
        algorithm: CompressionAlgorithm,
        characteristics: DataCharacteristics,
        goal: CompressionGoal,
    ) -> CompressionProfile:
        """Score an algorithm for given data and goal."""
        props = self.ALGORITHM_PROFILES.get(algorithm, {})

        if not props:
            return CompressionProfile(
                algorithm=algorithm,
                score=0.1,
                reason="Unknown algorithm properties",
            )

        # Get base metrics
        is_text = characteristics.is_text
        base_ratio = props.get("ratio_text", 2.0) if is_text else props.get("ratio_binary", 1.5)
        speed = props.get("speed_compress", 50)

        # Adjust for entropy
        entropy_factor = 1 - (characteristics.entropy / 8)
        adjusted_ratio = 1 + (base_ratio - 1) * entropy_factor

        # Calculate score based on goal
        if goal == CompressionGoal.BEST_SPEED:
            score = speed / 500  # Normalize speed
            reason = f"High speed: {speed} MB/s"
        elif goal == CompressionGoal.BEST_RATIO:
            score = adjusted_ratio / 4  # Normalize ratio
            reason = f"Best ratio: {adjusted_ratio:.1f}x"
        elif goal == CompressionGoal.LOW_MEMORY:
            memory = props.get("memory", "medium")
            memory_score = {"low": 1.0, "medium": 0.6, "high": 0.3}.get(memory, 0.5)
            score = memory_score
            reason = f"Low memory ({memory})"
        else:  # BALANCED
            speed_score = min(speed / 200, 1.0)
            ratio_score = min(adjusted_ratio / 3, 1.0)
            score = (speed_score + ratio_score) / 2
            reason = f"Balanced: {speed} MB/s, {adjusted_ratio:.1f}x"

        # Penalize random/high-entropy data
        if characteristics.detected_type == DataType.RANDOM:
            score *= 0.3
            reason = "Random data, poor compression expected"

        return CompressionProfile(
            algorithm=algorithm,
            score=min(score, 1.0),
            expected_ratio=adjusted_ratio,
            expected_speed_mbps=speed,
            reason=reason,
        )


# =============================================================================
# Adaptive Configuration
# =============================================================================


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive compression.

    Attributes:
        goal: Optimization goal.
        enable_sampling: Whether to test compression on samples.
        sample_size: Size of sample for testing.
        fallback_algorithm: Algorithm to use if selection fails.
        min_size_for_compression: Minimum data size to compress.
        max_analysis_time_ms: Maximum time for analysis.
        allowed_algorithms: Limit to specific algorithms.
        level: Compression level to use.
    """

    goal: CompressionGoal = CompressionGoal.BALANCED
    enable_sampling: bool = True
    sample_size: int = 65536
    fallback_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    min_size_for_compression: int = 128
    max_analysis_time_ms: float = 50.0
    allowed_algorithms: list[CompressionAlgorithm] | None = None
    level: CompressionLevel = CompressionLevel.BALANCED


# =============================================================================
# Adaptive Compressor
# =============================================================================


class AdaptiveCompressor:
    """Compressor that automatically selects the best algorithm.

    Analyzes input data characteristics and selects the optimal
    compression algorithm based on the configured goal.

    Note: This class doesn't inherit from BaseCompressor because it
    dynamically selects the algorithm rather than using a fixed one.

    Example:
        >>> compressor = AdaptiveCompressor(
        ...     config=AdaptiveConfig(goal=CompressionGoal.BEST_RATIO)
        ... )
        >>> result = compressor.compress_with_metrics(data)
        >>> print(f"Used: {result.metrics.algorithm}")
    """

    # Header format: magic (4) + algorithm (1) + flags (1) + original_size (4)
    HEADER_FORMAT = "<4sBBI"
    HEADER_SIZE = 10
    MAGIC = b"ADPT"

    def __init__(
        self,
        config: AdaptiveConfig | None = None,
        analyzer: DataAnalyzer | None = None,
        selector: AlgorithmSelector | None = None,
    ) -> None:
        """Initialize adaptive compressor.

        Args:
            config: Adaptive configuration.
            analyzer: Data analyzer instance.
            selector: Algorithm selector instance.
        """
        self.config = config or AdaptiveConfig()
        self._analyzer = analyzer or DataAnalyzer(sample_size=self.config.sample_size)

        # Filter available algorithms if specified
        available = None
        if self.config.allowed_algorithms:
            available = [a for a in self.config.allowed_algorithms if a != CompressionAlgorithm.NONE]

        self._selector = selector or AlgorithmSelector(available_algorithms=available)
        self._last_analysis: DataCharacteristics | None = None
        self._last_profile: CompressionProfile | None = None

    @property
    def algorithm(self) -> CompressionAlgorithm:
        """Get algorithm (returns NONE as placeholder for adaptive)."""
        return CompressionAlgorithm.NONE

    @property
    def last_analysis(self) -> DataCharacteristics | None:
        """Get last data analysis result."""
        return self._last_analysis

    @property
    def last_profile(self) -> CompressionProfile | None:
        """Get last algorithm profile used."""
        return self._last_profile

    def compress(self, data: bytes) -> bytes:
        """Compress data with adaptive algorithm selection.

        Args:
            data: Data to compress.

        Returns:
            Compressed data with header indicating algorithm used.
        """
        if len(data) < self.config.min_size_for_compression:
            return self._wrap_uncompressed(data)

        # Analyze and select algorithm
        self._last_analysis = self._analyzer.analyze(data)
        self._last_profile = self._selector.select(self._last_analysis, self.config.goal)

        # Compress with selected algorithm
        try:
            compressor = get_compressor(self._last_profile.algorithm, level=self.config.level)
            compressed = compressor.compress(data)
        except Exception:
            # Fallback to default
            compressor = get_compressor(self.config.fallback_algorithm, level=self.config.level)
            compressed = compressor.compress(data)
            self._last_profile = CompressionProfile(
                algorithm=self.config.fallback_algorithm,
                reason="Fallback after error",
            )

        # Check if compression helped
        if len(compressed) >= len(data):
            return self._wrap_uncompressed(data)

        return self._wrap_compressed(compressed, self._last_profile.algorithm, len(data))

    def decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed data with header.

        Returns:
            Decompressed data.
        """
        if len(data) < self.HEADER_SIZE:
            raise CompressionError("Data too short for adaptive header")

        # Parse header
        magic, algo_byte, flags, original_size = struct.unpack(self.HEADER_FORMAT, data[: self.HEADER_SIZE])

        if magic != self.MAGIC:
            raise CompressionError("Invalid adaptive compression header")

        # Check if uncompressed
        if flags & 0x01:
            return data[self.HEADER_SIZE : self.HEADER_SIZE + original_size]

        # Decompress
        algorithm = CompressionAlgorithm(list(CompressionAlgorithm)[algo_byte].value)
        compressor = get_compressor(algorithm)
        return compressor.decompress(data[self.HEADER_SIZE :])

    def compress_with_metrics(self, data: bytes) -> CompressionResult:
        """Compress with full metrics.

        Args:
            data: Data to compress.

        Returns:
            Compression result with metrics.
        """
        start = time.perf_counter()
        compressed = self.compress(data)
        compress_time = (time.perf_counter() - start) * 1000

        # Get actual algorithm used
        if len(compressed) >= self.HEADER_SIZE:
            _, algo_byte, flags, _ = struct.unpack(self.HEADER_FORMAT, compressed[: self.HEADER_SIZE])
            if flags & 0x01:
                algo = CompressionAlgorithm.NONE
            else:
                algo = CompressionAlgorithm(list(CompressionAlgorithm)[algo_byte].value)
        else:
            algo = CompressionAlgorithm.NONE

        metrics = CompressionMetrics(
            original_size=len(data),
            compressed_size=len(compressed),
            compression_time_ms=compress_time,
            algorithm=algo,
            level=self.config.level.get_level(algo) if algo != CompressionAlgorithm.NONE else 0,
        )
        metrics.update_ratio()

        return CompressionResult(
            data=compressed,
            metrics=metrics,
            header={
                "adaptive": True,
                "analysis": self._last_analysis.to_dict() if self._last_analysis else None,
                "profile": self._last_profile.to_dict() if self._last_profile else None,
            },
        )

    def _wrap_uncompressed(self, data: bytes) -> bytes:
        """Wrap uncompressed data with header."""
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            0,  # No algorithm
            0x01,  # Flag: uncompressed
            len(data),
        )
        return header + data

    def _wrap_compressed(self, data: bytes, algorithm: CompressionAlgorithm, original_size: int) -> bytes:
        """Wrap compressed data with header."""
        algo_index = list(CompressionAlgorithm).index(algorithm)
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            algo_index,
            0x00,  # Flags: compressed
            original_size,
        )
        return header + data

    def get_recommendation(self, data: bytes) -> list[CompressionProfile]:
        """Get algorithm recommendations without compressing.

        Args:
            data: Data to analyze.

        Returns:
            List of algorithm profiles ranked by suitability.
        """
        characteristics = self._analyzer.analyze(data)
        return self._selector.rank_algorithms(characteristics, self.config.goal)
