"""Streaming validation mixin for adding streaming capabilities to validators.

Provides a mixin class that adds streaming validation capabilities
to any validator, enabling memory-efficient processing of large datasets.

Usage:
    # Add streaming to any validator:
    class MyStreamingValidator(MyValidator, StreamingValidatorMixin):
        pass

    # Or use the adapter for existing validators:
    streaming_validator = StreamingValidatorAdapter(existing_validator)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Callable, TypeVar, Generic
from pathlib import Path

import polars as pl

from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.streaming.base import StreamingState, ChunkResult
from truthound.validators.streaming.sources import (
    StreamingSource,
    create_streaming_source,
    ParquetStreamingSource,
    CSVStreamingSource,
    ArrowIPCStreamingSource,
)


T = TypeVar("T")


# =============================================================================
# Streaming Accumulator Protocol
# =============================================================================


class StreamingAccumulator(ABC, Generic[T]):
    """Abstract base for streaming result accumulators.

    Accumulators aggregate partial results from chunks into
    final validation results.

    Subclasses implement different aggregation strategies:
    - CountingAccumulator: Sum counts across chunks
    - SamplingAccumulator: Keep a sample of issues
    - ThresholdAccumulator: Track threshold violations
    """

    @abstractmethod
    def initialize(self) -> T:
        """Create initial accumulator state."""
        pass

    @abstractmethod
    def accumulate(self, state: T, chunk_issues: list[ValidationIssue]) -> T:
        """Accumulate chunk issues into state."""
        pass

    @abstractmethod
    def finalize(self, state: T, total_rows: int) -> list[ValidationIssue]:
        """Generate final issues from accumulated state."""
        pass


@dataclass
class CountingState:
    """State for counting accumulator."""

    issue_counts: dict[str, int] = field(default_factory=dict)
    first_issues: dict[str, ValidationIssue] = field(default_factory=dict)
    total_count: int = 0


class CountingAccumulator(StreamingAccumulator[CountingState]):
    """Accumulator that counts issues across chunks.

    Aggregates issues by type, keeping count and a sample issue
    for each type.
    """

    def initialize(self) -> CountingState:
        return CountingState()

    def accumulate(
        self, state: CountingState, chunk_issues: list[ValidationIssue]
    ) -> CountingState:
        for issue in chunk_issues:
            key = f"{issue.column}:{issue.issue_type}"

            if key not in state.issue_counts:
                state.issue_counts[key] = 0
                state.first_issues[key] = issue

            state.issue_counts[key] += issue.count
            state.total_count += issue.count

        return state

    def finalize(
        self, state: CountingState, total_rows: int
    ) -> list[ValidationIssue]:
        issues = []

        for key, count in state.issue_counts.items():
            template = state.first_issues[key]

            # Create aggregated issue
            issues.append(
                ValidationIssue(
                    column=template.column,
                    issue_type=template.issue_type,
                    count=count,
                    severity=template.severity,
                    details=f"{template.details} (aggregated from streaming)",
                    expected=template.expected,
                )
            )

        return issues


@dataclass
class SamplingState:
    """State for sampling accumulator."""

    sampled_issues: list[ValidationIssue] = field(default_factory=list)
    total_count: int = 0
    issue_type_counts: dict[str, int] = field(default_factory=dict)


class SamplingAccumulator(StreamingAccumulator[SamplingState]):
    """Accumulator that keeps a sample of issues.

    Useful when you want to see example issues rather than
    just counts.
    """

    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples

    def initialize(self) -> SamplingState:
        return SamplingState()

    def accumulate(
        self, state: SamplingState, chunk_issues: list[ValidationIssue]
    ) -> SamplingState:
        for issue in chunk_issues:
            state.total_count += issue.count

            key = f"{issue.column}:{issue.issue_type}"
            state.issue_type_counts[key] = state.issue_type_counts.get(key, 0) + 1

            # Sample issues up to max
            if len(state.sampled_issues) < self.max_samples:
                state.sampled_issues.append(issue)

        return state

    def finalize(
        self, state: SamplingState, total_rows: int
    ) -> list[ValidationIssue]:
        return state.sampled_issues


# =============================================================================
# Streaming Validator Mixin
# =============================================================================


class StreamingValidatorMixin:
    """Mixin that adds streaming validation capabilities to any validator.

    This mixin provides methods for processing data from streaming sources,
    enabling memory-efficient validation of datasets larger than RAM.

    Features:
        - File-based streaming (Parquet, CSV, Arrow IPC)
        - Arrow Flight streaming for distributed data
        - Configurable chunk sizes
        - Multiple accumulation strategies
        - Progress callbacks

    Usage:
        class MyStreamingValidator(MyValidator, StreamingValidatorMixin):
            pass

        validator = MyStreamingValidator(...)

        # Stream from file
        issues = validator.validate_streaming("huge_file.parquet")

        # Stream with custom chunk size
        issues = validator.validate_streaming(
            "data.parquet",
            chunk_size=50_000,
        )

        # Stream with progress callback
        issues = validator.validate_streaming(
            "data.parquet",
            on_chunk=lambda i, n: print(f"Chunk {i}/{n}"),
        )
    """

    # Default chunk size for streaming
    default_streaming_chunk_size: int = 100_000

    def validate_streaming(
        self,
        source: str | Path | pl.LazyFrame | StreamingSource,
        chunk_size: int | None = None,
        accumulator: StreamingAccumulator | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        **source_kwargs: Any,
    ) -> list[ValidationIssue]:
        """Validate data from a streaming source.

        Processes data in chunks, accumulating results to produce
        final validation issues.

        Args:
            source: File path, LazyFrame, or StreamingSource
            chunk_size: Rows per chunk (None = use default)
            accumulator: Strategy for aggregating chunk results
            on_chunk: Callback(chunk_index, total_chunks) for progress
            columns: Specific columns to validate (None = all)
            max_rows: Maximum rows to process (None = all)
            **source_kwargs: Additional source-specific options

        Returns:
            Aggregated list of validation issues

        Example:
            # Basic streaming validation
            issues = validator.validate_streaming("data.parquet")

            # With progress tracking
            def progress(i, n):
                print(f"Processing chunk {i}/{n}")

            issues = validator.validate_streaming(
                "data.parquet",
                chunk_size=50_000,
                on_chunk=progress,
            )
        """
        chunk_size = chunk_size or self.default_streaming_chunk_size
        accumulator = accumulator or CountingAccumulator()

        # Create or use streaming source
        if isinstance(source, StreamingSource):
            streaming_source = source
            own_source = False
        else:
            streaming_source = create_streaming_source(
                source,
                chunk_size=chunk_size,
                columns=columns,
                max_rows=max_rows,
                **source_kwargs,
            )
            own_source = True

        # Initialize accumulator
        state = accumulator.initialize()
        total_rows = 0
        chunk_index = 0

        try:
            if own_source:
                streaming_source.open()

            # Estimate total chunks if possible
            try:
                estimated_rows = len(streaming_source)
                estimated_chunks = (estimated_rows + chunk_size - 1) // chunk_size
            except Exception:
                estimated_chunks = -1

            # Process chunks
            for chunk_df in streaming_source:
                # Call progress callback
                if on_chunk:
                    on_chunk(chunk_index, estimated_chunks)

                # Validate chunk
                chunk_lf = chunk_df.lazy()
                chunk_issues = self.validate(chunk_lf)

                # Accumulate results
                state = accumulator.accumulate(state, chunk_issues)
                total_rows += len(chunk_df)
                chunk_index += 1

        finally:
            if own_source:
                streaming_source.close()

        # Finalize and return
        return accumulator.finalize(state, total_rows)

    def validate_streaming_iter(
        self,
        source: str | Path | pl.LazyFrame | StreamingSource,
        chunk_size: int | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        **source_kwargs: Any,
    ) -> Iterator[tuple[int, list[ValidationIssue]]]:
        """Iterate over validation results chunk by chunk.

        Yields results as they're processed, useful for real-time
        monitoring or early termination.

        Args:
            source: File path, LazyFrame, or StreamingSource
            chunk_size: Rows per chunk
            columns: Specific columns to validate
            max_rows: Maximum rows to process
            **source_kwargs: Additional source options

        Yields:
            Tuples of (chunk_index, chunk_issues)

        Example:
            for chunk_idx, issues in validator.validate_streaming_iter("data.parquet"):
                if issues:
                    print(f"Chunk {chunk_idx}: {len(issues)} issues")
                    if len(issues) > 100:
                        break  # Early termination
        """
        chunk_size = chunk_size or self.default_streaming_chunk_size

        if isinstance(source, StreamingSource):
            streaming_source = source
            own_source = False
        else:
            streaming_source = create_streaming_source(
                source,
                chunk_size=chunk_size,
                columns=columns,
                max_rows=max_rows,
                **source_kwargs,
            )
            own_source = True

        try:
            if own_source:
                streaming_source.open()

            chunk_index = 0
            for chunk_df in streaming_source:
                chunk_lf = chunk_df.lazy()
                chunk_issues = self.validate(chunk_lf)
                yield chunk_index, chunk_issues
                chunk_index += 1

        finally:
            if own_source:
                streaming_source.close()


# =============================================================================
# Streaming Validator Adapter
# =============================================================================


class StreamingValidatorAdapter(StreamingValidatorMixin):
    """Adapter that adds streaming capabilities to any validator.

    Wraps an existing validator instance and provides streaming
    validation methods.

    Example:
        # Wrap any validator
        base_validator = NullValidator(column="id")
        streaming_validator = StreamingValidatorAdapter(base_validator)

        # Use streaming methods
        issues = streaming_validator.validate_streaming("huge_file.parquet")
    """

    def __init__(
        self,
        validator: Validator,
        chunk_size: int = 100_000,
    ):
        self._validator = validator
        self.default_streaming_chunk_size = chunk_size

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Delegate to wrapped validator."""
        return self._validator.validate(lf)

    @property
    def name(self) -> str:
        return f"streaming_{self._validator.name}"

    @property
    def category(self) -> str:
        return self._validator.category


# =============================================================================
# Multi-Validator Streaming Pipeline
# =============================================================================


class StreamingValidationPipelineMixin:
    """Mixin for running multiple validators in a single streaming pass.

    More efficient than running validators separately as it shares
    the data loading overhead.

    Example:
        pipeline = StreamingValidationPipeline([
            NullValidator(column="id"),
            RangeValidator(column="value", min_value=0),
        ])

        all_issues = pipeline.validate_streaming("data.parquet")
    """

    def __init__(
        self,
        validators: list[Validator],
        chunk_size: int = 100_000,
    ):
        self.validators = validators
        self.default_streaming_chunk_size = chunk_size

    def validate_streaming(
        self,
        source: str | Path | pl.LazyFrame | StreamingSource,
        chunk_size: int | None = None,
        on_chunk: Callable[[int, int], None] | None = None,
        **source_kwargs: Any,
    ) -> dict[str, list[ValidationIssue]]:
        """Run all validators on streaming data.

        Args:
            source: Data source
            chunk_size: Rows per chunk
            on_chunk: Progress callback
            **source_kwargs: Source options

        Returns:
            Dict mapping validator name to issues
        """
        chunk_size = chunk_size or self.default_streaming_chunk_size

        if isinstance(source, StreamingSource):
            streaming_source = source
            own_source = False
        else:
            streaming_source = create_streaming_source(
                source,
                chunk_size=chunk_size,
                **source_kwargs,
            )
            own_source = True

        # Initialize accumulators for each validator
        accumulators = {v.name: CountingAccumulator() for v in self.validators}
        states = {v.name: accumulators[v.name].initialize() for v in self.validators}
        total_rows = 0
        chunk_index = 0

        try:
            if own_source:
                streaming_source.open()

            try:
                estimated_rows = len(streaming_source)
                estimated_chunks = (estimated_rows + chunk_size - 1) // chunk_size
            except Exception:
                estimated_chunks = -1

            for chunk_df in streaming_source:
                if on_chunk:
                    on_chunk(chunk_index, estimated_chunks)

                chunk_lf = chunk_df.lazy()

                # Run all validators on this chunk
                for validator in self.validators:
                    chunk_issues = validator.validate(chunk_lf)
                    states[validator.name] = accumulators[validator.name].accumulate(
                        states[validator.name], chunk_issues
                    )

                total_rows += len(chunk_df)
                chunk_index += 1

        finally:
            if own_source:
                streaming_source.close()

        # Finalize all
        results = {}
        for validator in self.validators:
            results[validator.name] = accumulators[validator.name].finalize(
                states[validator.name], total_rows
            )

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def stream_validate(
    validator: Validator,
    source: str | Path | pl.LazyFrame | StreamingSource,
    chunk_size: int = 100_000,
    **kwargs: Any,
) -> list[ValidationIssue]:
    """Convenience function for one-off streaming validation.

    Args:
        validator: Any validator instance
        source: Data source (file path, LazyFrame, or StreamingSource)
        chunk_size: Rows per chunk
        **kwargs: Additional source options

    Returns:
        Aggregated validation issues

    Example:
        from truthound.validators import NullValidator
        from truthound.validators.streaming.mixin import stream_validate

        validator = NullValidator(column="id")
        issues = stream_validate(validator, "huge_file.parquet")
    """
    adapter = StreamingValidatorAdapter(validator, chunk_size=chunk_size)
    return adapter.validate_streaming(source, **kwargs)


def stream_validate_many(
    validators: list[Validator],
    source: str | Path | pl.LazyFrame | StreamingSource,
    chunk_size: int = 100_000,
    **kwargs: Any,
) -> dict[str, list[ValidationIssue]]:
    """Convenience function for multi-validator streaming.

    Args:
        validators: List of validators
        source: Data source
        chunk_size: Rows per chunk
        **kwargs: Additional source options

    Returns:
        Dict mapping validator name to issues

    Example:
        validators = [
            NullValidator(column="id"),
            RangeValidator(column="value", min_value=0),
        ]
        results = stream_validate_many(validators, "data.parquet")
    """
    pipeline = StreamingValidationPipelineMixin(validators, chunk_size=chunk_size)
    return pipeline.validate_streaming(source, **kwargs)
