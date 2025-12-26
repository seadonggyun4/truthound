"""Streaming validators for memory-efficient large dataset processing.

This module provides validators optimized for processing very large datasets
that don't fit in memory. Key features:

1. Chunked Processing: Process data in configurable chunks
2. Result Aggregation: Combine results across chunks accurately
3. Memory Bounded: Predictable memory usage regardless of data size
4. Compatible: Works with standard LazyFrame/DataFrame inputs
5. File-Based Streaming: Direct streaming from Parquet, CSV, Arrow IPC files
6. Arrow Flight Streaming: Distributed data streaming from remote servers

Usage:
    from truthound.validators.streaming import (
        StreamingNullValidator,
        StreamingRangeValidator,
        StreamingValidationPipeline,
        # New streaming sources
        ParquetStreamingSource,
        CSVStreamingSource,
        ArrowIPCStreamingSource,
        ArrowFlightStreamingSource,
        # Convenience functions
        stream_validate,
        stream_validate_many,
    )

    # Single streaming validator
    validator = StreamingNullValidator(chunk_size=50_000)
    issues = validator.validate(very_large_lazyframe)

    # Pipeline of streaming validators
    pipeline = StreamingValidationPipeline(
        validators=[
            StreamingNullValidator(),
            StreamingRangeValidator(min_value=0),
        ],
        chunk_size=100_000,
    )
    issues = pipeline.validate(very_large_lazyframe)

    # File-based streaming (new)
    with ParquetStreamingSource("huge_file.parquet", chunk_size=100_000) as source:
        for chunk_df in source:
            issues = validator.validate(chunk_df.lazy())

    # One-liner streaming validation (new)
    issues = stream_validate(validator, "huge_file.parquet")

    # Multi-validator streaming (new)
    results = stream_validate_many(
        [validator1, validator2],
        "huge_file.parquet",
    )
"""

from truthound.validators.streaming.base import (
    StreamingValidator,
    StreamingValidationPipeline,
    StreamingState,
    ChunkResult,
)
from truthound.validators.streaming.completeness import (
    StreamingNullValidator,
    StreamingCompletenessValidator,
)
from truthound.validators.streaming.range import (
    StreamingRangeValidator,
    StreamingOutlierValidator,
)

# New streaming sources
from truthound.validators.streaming.sources import (
    StreamingSource,
    StreamingSourceConfig,
    ParquetStreamingSource,
    CSVStreamingSource,
    JSONLStreamingSource,
    ArrowIPCStreamingSource,
    ArrowFlightStreamingSource,
    LazyFrameStreamingSource,
    create_streaming_source,
)

# New streaming mixin and utilities
from truthound.validators.streaming.mixin import (
    StreamingValidatorMixin,
    StreamingValidatorAdapter,
    StreamingAccumulator,
    CountingAccumulator,
    SamplingAccumulator,
    stream_validate,
    stream_validate_many,
)

__all__ = [
    # Base classes
    "StreamingValidator",
    "StreamingValidationPipeline",
    "StreamingState",
    "ChunkResult",
    # Completeness validators
    "StreamingNullValidator",
    "StreamingCompletenessValidator",
    # Range validators
    "StreamingRangeValidator",
    "StreamingOutlierValidator",
    # Streaming sources
    "StreamingSource",
    "StreamingSourceConfig",
    "ParquetStreamingSource",
    "CSVStreamingSource",
    "JSONLStreamingSource",
    "ArrowIPCStreamingSource",
    "ArrowFlightStreamingSource",
    "LazyFrameStreamingSource",
    "create_streaming_source",
    # Streaming mixin and utilities
    "StreamingValidatorMixin",
    "StreamingValidatorAdapter",
    "StreamingAccumulator",
    "CountingAccumulator",
    "SamplingAccumulator",
    "stream_validate",
    "stream_validate_many",
]
