"""Streaming validators for memory-efficient large dataset processing.

This module provides validators optimized for processing very large datasets
that don't fit in memory. Key features:

1. Chunked Processing: Process data in configurable chunks
2. Result Aggregation: Combine results across chunks accurately
3. Memory Bounded: Predictable memory usage regardless of data size
4. Compatible: Works with standard LazyFrame/DataFrame inputs

Usage:
    from truthound.validators.streaming import (
        StreamingNullValidator,
        StreamingRangeValidator,
        StreamingValidationPipeline,
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
"""

from truthound.validators.streaming.base import (
    StreamingValidator,
    StreamingValidationPipeline,
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

__all__ = [
    # Base classes
    "StreamingValidator",
    "StreamingValidationPipeline",
    "ChunkResult",
    # Completeness
    "StreamingNullValidator",
    "StreamingCompletenessValidator",
    # Range
    "StreamingRangeValidator",
    "StreamingOutlierValidator",
]
