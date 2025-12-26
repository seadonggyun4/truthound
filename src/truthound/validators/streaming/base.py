"""Base classes for streaming validators.

Provides abstract base classes and utilities for implementing validators
that process data in chunks to support very large datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Generic, TypeVar

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    ValidatorConfig,
)


T = TypeVar("T")


@dataclass
class ChunkResult(Generic[T]):
    """Result from processing a single chunk.

    Attributes:
        chunk_index: Index of the processed chunk
        row_offset: Starting row offset in the original dataset
        row_count: Number of rows in this chunk
        data: Chunk-specific aggregated data (e.g., counts, sums)
        issues: Issues found in this chunk (before aggregation)
    """

    chunk_index: int
    row_offset: int
    row_count: int
    data: T
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class StreamingState:
    """State maintained across chunks during streaming validation.

    Attributes:
        total_rows: Total rows processed so far
        chunk_count: Number of chunks processed
        column_stats: Per-column accumulated statistics
    """

    total_rows: int = 0
    chunk_count: int = 0
    column_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def update_column_stat(
        self,
        column: str,
        key: str,
        value: Any,
        aggregator: str = "sum",
    ) -> None:
        """Update a column statistic with aggregation.

        Args:
            column: Column name
            key: Statistic key
            value: Value to aggregate
            aggregator: Aggregation method ('sum', 'max', 'min', 'last')
        """
        if column not in self.column_stats:
            self.column_stats[column] = {}

        if key not in self.column_stats[column]:
            self.column_stats[column][key] = value
        else:
            current = self.column_stats[column][key]
            if aggregator == "sum":
                self.column_stats[column][key] = current + value
            elif aggregator == "max":
                self.column_stats[column][key] = max(current, value)
            elif aggregator == "min":
                self.column_stats[column][key] = min(current, value)
            elif aggregator == "last":
                self.column_stats[column][key] = value

    def get_column_stat(self, column: str, key: str, default: Any = None) -> Any:
        """Get a column statistic."""
        return self.column_stats.get(column, {}).get(key, default)


class StreamingValidator(Validator, ABC):
    """Abstract base class for streaming validators.

    Streaming validators process data in chunks, maintaining state across
    chunks and aggregating results at the end. This enables validation of
    datasets larger than available memory.

    Subclasses must implement:
    - process_chunk(): Process a single chunk and update state
    - finalize(): Generate final issues from accumulated state

    Example:
        class MyStreamingValidator(StreamingValidator):
            def process_chunk(self, chunk_df, state):
                # Count something in this chunk
                count = len(chunk_df.filter(...))
                state.update_column_stat("col", "count", count, "sum")

            def finalize(self, state, total_rows):
                issues = []
                count = state.get_column_stat("col", "count", 0)
                if count > threshold:
                    issues.append(ValidationIssue(...))
                return issues
    """

    name = "streaming_base"
    category = "streaming"

    default_chunk_size: int = 100_000

    def __init__(
        self,
        chunk_size: int | None = None,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self.chunk_size = chunk_size or self.default_chunk_size

    def _iter_chunks(
        self,
        lf: pl.LazyFrame,
        total_rows: int,
    ) -> Iterator[tuple[int, int, pl.DataFrame]]:
        """Iterate over chunks of the LazyFrame.

        Yields:
            Tuples of (chunk_index, row_offset, chunk_dataframe)
        """
        chunk_index = 0
        for offset in range(0, total_rows, self.chunk_size):
            chunk_df = lf.slice(offset, self.chunk_size).collect()
            yield chunk_index, offset, chunk_df
            chunk_index += 1

    @abstractmethod
    def process_chunk(
        self,
        chunk_df: pl.DataFrame,
        state: StreamingState,
        chunk_index: int,
        row_offset: int,
    ) -> None:
        """Process a single chunk and update state.

        Args:
            chunk_df: The chunk DataFrame to process
            state: Streaming state to update
            chunk_index: Index of current chunk
            row_offset: Row offset of this chunk in original data
        """
        pass

    @abstractmethod
    def finalize(
        self,
        state: StreamingState,
        total_rows: int,
    ) -> list[ValidationIssue]:
        """Generate final validation issues from accumulated state.

        Args:
            state: Final streaming state after all chunks
            total_rows: Total rows in the dataset

        Returns:
            List of validation issues
        """
        pass

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run streaming validation on the LazyFrame.

        Processes data in chunks to limit memory usage.

        Args:
            lf: LazyFrame to validate

        Returns:
            List of aggregated validation issues
        """
        # Get total rows
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return []

        # Initialize state
        state = StreamingState()

        # Process chunks
        for chunk_index, row_offset, chunk_df in self._iter_chunks(lf, total_rows):
            self.process_chunk(chunk_df, state, chunk_index, row_offset)
            state.chunk_count += 1
            state.total_rows += len(chunk_df)

        # Generate final issues
        return self.finalize(state, total_rows)


class StreamingValidationPipeline:
    """Pipeline for running multiple streaming validators efficiently.

    Processes all validators in a single pass through the data chunks,
    sharing the chunking overhead across validators.

    Example:
        pipeline = StreamingValidationPipeline(
            validators=[
                StreamingNullValidator(),
                StreamingRangeValidator(min_value=0, max_value=100),
            ],
            chunk_size=50_000,
        )
        all_issues = pipeline.validate(large_lazyframe)
    """

    def __init__(
        self,
        validators: list[StreamingValidator],
        chunk_size: int = 100_000,
    ):
        self.validators = validators
        self.chunk_size = chunk_size

        # Override chunk size for all validators
        for v in self.validators:
            v.chunk_size = chunk_size

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run all validators in a single pass through the data.

        Args:
            lf: LazyFrame to validate

        Returns:
            Combined list of issues from all validators
        """
        total_rows = lf.select(pl.len()).collect().item()

        if total_rows == 0:
            return []

        # Initialize state for each validator
        states = [StreamingState() for _ in self.validators]

        # Single pass through chunks
        chunk_index = 0
        for offset in range(0, total_rows, self.chunk_size):
            chunk_df = lf.slice(offset, self.chunk_size).collect()

            # Process chunk with each validator
            for validator, state in zip(self.validators, states):
                validator.process_chunk(chunk_df, state, chunk_index, offset)
                state.chunk_count += 1
                state.total_rows += len(chunk_df)

            chunk_index += 1

        # Finalize all validators
        all_issues: list[ValidationIssue] = []
        for validator, state in zip(self.validators, states):
            issues = validator.finalize(state, total_rows)
            all_issues.extend(issues)

        return all_issues

    def add_validator(self, validator: StreamingValidator) -> "StreamingValidationPipeline":
        """Add a validator to the pipeline.

        Args:
            validator: Streaming validator to add

        Returns:
            Self for chaining
        """
        validator.chunk_size = self.chunk_size
        self.validators.append(validator)
        return self
