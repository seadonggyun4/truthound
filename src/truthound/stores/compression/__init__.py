"""Compression module for validation result storage.

This module provides a flexible, extensible compression system for handling
large validation results. It supports multiple compression algorithms with
automatic selection based on data characteristics.

Features:
    - Multiple compression algorithms (gzip, zstd, lz4, snappy, brotli)
    - Adaptive compression with automatic algorithm selection
    - Compression pipelines for chained transformations
    - Streaming compression for memory efficiency
    - Compression metrics and statistics
    - Encryption-ready architecture

Example:
    >>> from truthound.stores.compression import (
    ...     get_compressor,
    ...     CompressionLevel,
    ...     AdaptiveCompressor,
    ... )
    >>>
    >>> # Simple compression
    >>> compressor = get_compressor("gzip", level=CompressionLevel.BALANCED)
    >>> compressed = compressor.compress(data)
    >>> original = compressor.decompress(compressed)
    >>>
    >>> # Adaptive compression (auto-selects best algorithm)
    >>> adaptive = AdaptiveCompressor()
    >>> compressed = adaptive.compress(data)  # Automatically chooses best algorithm
    >>>
    >>> # Compression pipeline
    >>> from truthound.stores.compression import CompressionPipeline
    >>> pipeline = CompressionPipeline([
    ...     ("dedupe", DeduplicationTransform()),
    ...     ("compress", get_compressor("zstd")),
    ... ])
    >>> result = pipeline.process(data)

Streaming Example:
    >>> from truthound.stores.compression import StreamingCompressor
    >>>
    >>> with StreamingCompressor("gzip") as stream:
    ...     for chunk in large_data_chunks:
    ...         stream.write(chunk)
    ...     compressed = stream.finalize()
"""

from truthound.stores.compression.base import (
    # Protocols
    Compressor,
    Decompressor,
    StreamingCompressor as StreamingCompressorProtocol,
    # Enums
    CompressionAlgorithm,
    CompressionLevel,
    CompressionMode,
    # Data classes
    CompressionConfig,
    CompressionResult,
    CompressionMetrics,
    CompressionStats,
    # Exceptions
    CompressionError,
    DecompressionError,
    UnsupportedAlgorithmError,
)
from truthound.stores.compression.providers import (
    # Base
    BaseCompressor,
    # Providers
    GzipCompressor,
    ZstdCompressor,
    LZ4Compressor,
    SnappyCompressor,
    BrotliCompressor,
    NoopCompressor,
    # Factory
    get_compressor,
    get_decompressor,
    register_compressor,
    list_available_algorithms,
    is_algorithm_available,
)
from truthound.stores.compression.adaptive import (
    AdaptiveCompressor,
    AdaptiveConfig,
    CompressionGoal,
    CompressionProfile,
    DataCharacteristics,
    DataType,
    DataAnalyzer,
    AlgorithmSelector,
)
from truthound.stores.compression.pipeline import (
    CompressionPipeline,
    PipelineStage,
    PipelineResult,
    PipelineMetrics,
    Transform,
    IdentityTransform,
    DeduplicationTransform,
    DeltaEncodingTransform,
    RunLengthTransform,
    PipelineError,
    TransformError,
    StageType,
    # Pre-built pipelines
    create_text_pipeline,
    create_json_pipeline,
    create_binary_pipeline,
    create_timeseries_pipeline,
)
from truthound.stores.compression.streaming import (
    StreamingCompressor,
    StreamingDecompressor,
    ChunkedCompressor,
    ChunkInfo,
    ChunkIndex,
    StreamingMetrics,
    GzipStreamWriter,
    GzipStreamReader,
)

__all__ = [
    # Protocols
    "Compressor",
    "Decompressor",
    "StreamingCompressorProtocol",
    # Enums
    "CompressionAlgorithm",
    "CompressionLevel",
    "CompressionMode",
    "CompressionGoal",
    "DataType",
    "StageType",
    # Data classes
    "CompressionConfig",
    "CompressionResult",
    "CompressionMetrics",
    "CompressionStats",
    # Exceptions
    "CompressionError",
    "DecompressionError",
    "UnsupportedAlgorithmError",
    "PipelineError",
    "TransformError",
    # Base
    "BaseCompressor",
    # Providers
    "GzipCompressor",
    "ZstdCompressor",
    "LZ4Compressor",
    "SnappyCompressor",
    "BrotliCompressor",
    "NoopCompressor",
    # Factory
    "get_compressor",
    "get_decompressor",
    "register_compressor",
    "list_available_algorithms",
    "is_algorithm_available",
    # Adaptive
    "AdaptiveCompressor",
    "AdaptiveConfig",
    "CompressionProfile",
    "DataCharacteristics",
    "DataAnalyzer",
    "AlgorithmSelector",
    # Pipeline
    "CompressionPipeline",
    "PipelineStage",
    "PipelineResult",
    "PipelineMetrics",
    "Transform",
    "IdentityTransform",
    "DeduplicationTransform",
    "DeltaEncodingTransform",
    "RunLengthTransform",
    # Pre-built pipelines
    "create_text_pipeline",
    "create_json_pipeline",
    "create_binary_pipeline",
    "create_timeseries_pipeline",
    # Streaming
    "StreamingCompressor",
    "StreamingDecompressor",
    "ChunkedCompressor",
    "ChunkInfo",
    "ChunkIndex",
    "StreamingMetrics",
    "GzipStreamWriter",
    "GzipStreamReader",
]
