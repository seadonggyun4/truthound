"""Streaming Pattern Matching with Chunk Integration.

This module provides chunk-aware pattern matching for streaming data processing.
It solves the problem of pattern detection across chunk boundaries by maintaining
state and aggregating pattern statistics across multiple chunks.

Key features:
- Chunk-aware pattern state management
- Pluggable aggregation strategies
- Cross-chunk pattern boundary detection
- Statistical confidence tracking across chunks
- Memory-efficient incremental processing
- Integration with existing streaming profiler

Design Principles:
- Strategy Pattern: Aggregation strategies are pluggable
- Observer Pattern: Callbacks for pattern events
- State Pattern: Chunk state management
- Template Method: Customizable aggregation pipeline

Example:
    from truthound.profiler.streaming_patterns import (
        StreamingPatternMatcher,
        IncrementalAggregation,
    )

    matcher = StreamingPatternMatcher(
        aggregation_strategy=IncrementalAggregation(),
    )

    # Process chunks
    for chunk in chunks:
        matcher.process_chunk(chunk, "column_name")

    # Get final results
    results = matcher.finalize()
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
)

import polars as pl

from truthound.profiler.base import DataType, PatternMatch
from truthound.profiler.native_patterns import (
    BUILTIN_PATTERNS,
    NativePatternMatcher,
    PatternMatchResult,
    PatternRegistry,
    PatternSpec,
)
from truthound.profiler.sampling import (
    SamplingConfig,
    SamplingMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class AggregationMethod(str, Enum):
    """Methods for aggregating pattern statistics across chunks."""

    INCREMENTAL = "incremental"      # Running totals
    WEIGHTED = "weighted"            # Size-weighted averages
    SLIDING_WINDOW = "sliding_window"  # Recent chunks only
    EXPONENTIAL = "exponential"      # Exponential moving average
    RESERVOIR = "reservoir"          # Reservoir-based sampling
    CONSENSUS = "consensus"          # Agreement across chunks
    ADAPTIVE = "adaptive"            # Auto-select based on data


class ChunkProcessingStatus(str, Enum):
    """Status of chunk processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Pattern State Management
# =============================================================================


@dataclass
class PatternChunkStats:
    """Statistics for a pattern in a single chunk.

    This is the basic unit of pattern statistics captured per chunk.
    Immutable after creation to ensure thread-safety.
    """

    pattern_name: str
    match_count: int
    total_count: int
    chunk_index: int
    processing_time_ms: float = 0.0

    @property
    def match_ratio(self) -> float:
        """Calculate match ratio for this chunk."""
        return self.match_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "match_count": self.match_count,
            "total_count": self.total_count,
            "match_ratio": self.match_ratio,
            "chunk_index": self.chunk_index,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class PatternState:
    """Mutable state for a pattern across all chunks.

    Maintains running statistics and history for aggregation.
    """

    pattern: PatternSpec
    chunk_stats: list[PatternChunkStats] = field(default_factory=list)

    # Running totals
    total_matches: int = 0
    total_rows: int = 0
    chunks_processed: int = 0

    # Sample collection
    sample_matches: list[str] = field(default_factory=list)
    max_samples: int = 10

    # Timing
    total_processing_time_ms: float = 0.0

    def add_chunk_stats(self, stats: PatternChunkStats) -> None:
        """Add statistics from a new chunk."""
        self.chunk_stats.append(stats)
        self.total_matches += stats.match_count
        self.total_rows += stats.total_count
        self.chunks_processed += 1
        self.total_processing_time_ms += stats.processing_time_ms

    def add_samples(self, samples: Sequence[str]) -> None:
        """Add sample matches (up to max_samples)."""
        remaining = self.max_samples - len(self.sample_matches)
        if remaining > 0:
            self.sample_matches.extend(samples[:remaining])

    @property
    def overall_match_ratio(self) -> float:
        """Calculate overall match ratio across all chunks."""
        return self.total_matches / self.total_rows if self.total_rows > 0 else 0.0

    @property
    def chunk_ratios(self) -> list[float]:
        """Get match ratios for each chunk."""
        return [s.match_ratio for s in self.chunk_stats]

    @property
    def variance(self) -> float:
        """Calculate variance of match ratios across chunks."""
        if len(self.chunk_stats) < 2:
            return 0.0
        ratios = self.chunk_ratios
        mean = sum(ratios) / len(ratios)
        return sum((r - mean) ** 2 for r in ratios) / (len(ratios) - 1)

    @property
    def std_deviation(self) -> float:
        """Calculate standard deviation of match ratios."""
        return self.variance ** 0.5

    @property
    def is_consistent(self) -> bool:
        """Check if pattern is consistent across chunks."""
        if len(self.chunk_stats) < 2:
            return True
        return self.std_deviation < 0.1  # Less than 10% variation

    def to_pattern_match(self) -> PatternMatch:
        """Convert to legacy PatternMatch format."""
        return PatternMatch(
            pattern=self.pattern.name,
            regex=self.pattern.regex,
            match_ratio=self.overall_match_ratio,
            sample_matches=tuple(self.sample_matches),
        )


@dataclass
class ColumnPatternState:
    """Complete pattern state for a single column."""

    column_name: str
    pattern_states: dict[str, PatternState] = field(default_factory=dict)
    chunks_processed: int = 0
    total_rows: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def get_or_create_pattern_state(self, pattern: PatternSpec) -> PatternState:
        """Get or create pattern state for a pattern."""
        if pattern.name not in self.pattern_states:
            self.pattern_states[pattern.name] = PatternState(pattern=pattern)
        return self.pattern_states[pattern.name]

    def add_chunk(self, chunk_rows: int) -> None:
        """Register a chunk was processed."""
        self.chunks_processed += 1
        self.total_rows += chunk_rows

    def finalize(self) -> None:
        """Mark processing as complete."""
        self.completed_at = datetime.now()

    @property
    def processing_duration_ms(self) -> float:
        """Get total processing duration."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds() * 1000


# =============================================================================
# Aggregation Strategies
# =============================================================================


class AggregationStrategy(ABC):
    """Abstract base class for pattern aggregation strategies.

    Aggregation strategies determine how pattern statistics from
    multiple chunks are combined into final results.

    Subclass this to create custom aggregation behavior.
    """

    name: str = "base"

    @abstractmethod
    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate pattern statistics into final result.

        Args:
            state: Pattern state with all chunk statistics
            min_match_ratio: Minimum ratio to consider a match

        Returns:
            Aggregated PatternMatchResult or None if not matching
        """
        pass

    @abstractmethod
    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Determine if pattern should be included in results.

        Args:
            state: Pattern state
            min_match_ratio: Minimum ratio threshold

        Returns:
            True if pattern should be included
        """
        pass


class IncrementalAggregation(AggregationStrategy):
    """Simple incremental aggregation using running totals.

    The most straightforward aggregation: sum all matches and
    divide by total rows. Works well for uniform data.
    """

    name = "incremental"

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate using simple totals."""
        if not self.should_include_pattern(state, min_match_ratio):
            return None

        return PatternMatchResult(
            pattern=state.pattern,
            match_count=state.total_matches,
            total_count=state.total_rows,
            match_ratio=state.overall_match_ratio,
            sample_matches=tuple(state.sample_matches),
        )

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Include if overall ratio meets threshold."""
        return state.overall_match_ratio >= min_match_ratio


class WeightedAggregation(AggregationStrategy):
    """Weighted aggregation based on chunk sizes.

    Gives more weight to larger chunks. Useful when chunk
    sizes vary significantly.
    """

    name = "weighted"

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate using size-weighted average."""
        if not self.should_include_pattern(state, min_match_ratio):
            return None

        # Weighted average is the same as simple total ratio
        # when weights are proportional to counts
        return PatternMatchResult(
            pattern=state.pattern,
            match_count=state.total_matches,
            total_count=state.total_rows,
            match_ratio=state.overall_match_ratio,
            sample_matches=tuple(state.sample_matches),
        )

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Include based on weighted ratio."""
        return state.overall_match_ratio >= min_match_ratio


class SlidingWindowAggregation(AggregationStrategy):
    """Aggregation using only recent chunks.

    Useful for detecting patterns in recent data when older
    data may have different characteristics.
    """

    name = "sliding_window"

    def __init__(self, window_size: int = 5):
        """Initialize with window size.

        Args:
            window_size: Number of recent chunks to consider
        """
        self.window_size = window_size

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate using recent chunks only."""
        if not self.should_include_pattern(state, min_match_ratio):
            return None

        # Get recent chunks
        recent = state.chunk_stats[-self.window_size:]
        if not recent:
            return None

        total_matches = sum(s.match_count for s in recent)
        total_rows = sum(s.total_count for s in recent)
        match_ratio = total_matches / total_rows if total_rows > 0 else 0.0

        return PatternMatchResult(
            pattern=state.pattern,
            match_count=total_matches,
            total_count=total_rows,
            match_ratio=match_ratio,
            sample_matches=tuple(state.sample_matches),
        )

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Include based on recent chunks."""
        recent = state.chunk_stats[-self.window_size:]
        if not recent:
            return False

        total_matches = sum(s.match_count for s in recent)
        total_rows = sum(s.total_count for s in recent)
        ratio = total_matches / total_rows if total_rows > 0 else 0.0
        return ratio >= min_match_ratio


class ExponentialAggregation(AggregationStrategy):
    """Exponential moving average aggregation.

    Gives exponentially more weight to recent chunks.
    Alpha controls the decay rate (higher = more weight to recent).
    """

    name = "exponential"

    def __init__(self, alpha: float = 0.3):
        """Initialize with smoothing factor.

        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate using exponential moving average."""
        if not self.should_include_pattern(state, min_match_ratio):
            return None

        ema_ratio = self._calculate_ema(state.chunk_ratios)

        return PatternMatchResult(
            pattern=state.pattern,
            match_count=state.total_matches,
            total_count=state.total_rows,
            match_ratio=ema_ratio,
            sample_matches=tuple(state.sample_matches),
        )

    def _calculate_ema(self, ratios: list[float]) -> float:
        """Calculate exponential moving average of ratios."""
        if not ratios:
            return 0.0

        ema = ratios[0]
        for ratio in ratios[1:]:
            ema = self.alpha * ratio + (1 - self.alpha) * ema
        return ema

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Include based on EMA ratio."""
        if not state.chunk_ratios:
            return False
        ema = self._calculate_ema(state.chunk_ratios)
        return ema >= min_match_ratio


class ConsensusAggregation(AggregationStrategy):
    """Consensus-based aggregation requiring agreement across chunks.

    Pattern is included only if it matches in a minimum fraction
    of chunks. Useful for detecting consistent patterns.
    """

    name = "consensus"

    def __init__(self, consensus_threshold: float = 0.8):
        """Initialize with consensus threshold.

        Args:
            consensus_threshold: Fraction of chunks that must match (0-1)
        """
        if not 0 < consensus_threshold <= 1:
            raise ValueError(
                f"consensus_threshold must be between 0 and 1, got {consensus_threshold}"
            )
        self.consensus_threshold = consensus_threshold

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate requiring consensus across chunks."""
        if not self.should_include_pattern(state, min_match_ratio):
            return None

        return PatternMatchResult(
            pattern=state.pattern,
            match_count=state.total_matches,
            total_count=state.total_rows,
            match_ratio=state.overall_match_ratio,
            sample_matches=tuple(state.sample_matches),
        )

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Include if consensus threshold is met."""
        if not state.chunk_stats:
            return False

        # Count chunks where pattern matches
        matching_chunks = sum(
            1 for s in state.chunk_stats if s.match_ratio >= min_match_ratio
        )

        consensus_ratio = matching_chunks / len(state.chunk_stats)
        return consensus_ratio >= self.consensus_threshold


class AdaptiveAggregation(AggregationStrategy):
    """Adaptive aggregation that selects strategy based on data characteristics.

    Automatically chooses the best aggregation method based on:
    - Variance in chunk ratios
    - Number of chunks processed
    - Pattern consistency
    """

    name = "adaptive"

    def __init__(self) -> None:
        """Initialize with sub-strategies."""
        self._strategies = {
            "incremental": IncrementalAggregation(),
            "exponential": ExponentialAggregation(alpha=0.3),
            "consensus": ConsensusAggregation(consensus_threshold=0.7),
        }

    def aggregate(
        self,
        state: PatternState,
        min_match_ratio: float = 0.8,
    ) -> PatternMatchResult | None:
        """Aggregate using adaptively selected strategy."""
        strategy = self._select_strategy(state)
        logger.debug(
            f"Adaptive aggregation selected '{strategy.name}' for pattern '{state.pattern.name}'"
        )
        return strategy.aggregate(state, min_match_ratio)

    def should_include_pattern(
        self,
        state: PatternState,
        min_match_ratio: float,
    ) -> bool:
        """Check using adaptively selected strategy."""
        strategy = self._select_strategy(state)
        return strategy.should_include_pattern(state, min_match_ratio)

    def _select_strategy(self, state: PatternState) -> AggregationStrategy:
        """Select best strategy based on state characteristics."""
        if len(state.chunk_stats) < 3:
            # Too few chunks for sophisticated analysis
            return self._strategies["incremental"]

        if state.is_consistent:
            # Consistent pattern: simple aggregation is fine
            return self._strategies["incremental"]

        if state.std_deviation > 0.2:
            # High variance: use consensus to require agreement
            return self._strategies["consensus"]

        # Default: exponential for balanced handling
        return self._strategies["exponential"]


# =============================================================================
# Aggregation Strategy Registry
# =============================================================================


class AggregationStrategyRegistry:
    """Registry for aggregation strategies.

    Allows registration of custom strategies and creation by name.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, AggregationStrategy] = {}
        self._lock = threading.RLock()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in strategies."""
        self.register(IncrementalAggregation())
        self.register(WeightedAggregation())
        self.register(SlidingWindowAggregation())
        self.register(ExponentialAggregation())
        self.register(ConsensusAggregation())
        self.register(AdaptiveAggregation())

    def register(self, strategy: AggregationStrategy) -> None:
        """Register an aggregation strategy."""
        with self._lock:
            self._strategies[strategy.name] = strategy
            logger.debug(f"Registered aggregation strategy: {strategy.name}")

    def get(self, name: str) -> AggregationStrategy:
        """Get a strategy by name."""
        with self._lock:
            if name not in self._strategies:
                available = list(self._strategies.keys())
                raise KeyError(
                    f"Unknown aggregation strategy: '{name}'. Available: {available}"
                )
            return self._strategies[name]

    def get_or_default(
        self,
        name: str,
        default: AggregationStrategy | None = None,
    ) -> AggregationStrategy:
        """Get strategy by name with fallback."""
        try:
            return self.get(name)
        except KeyError:
            return default or AdaptiveAggregation()

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        with self._lock:
            return list(self._strategies.keys())

    def create_from_method(self, method: AggregationMethod) -> AggregationStrategy:
        """Create strategy from AggregationMethod enum."""
        return self.get(method.value)


# Global registry instance
aggregation_strategy_registry = AggregationStrategyRegistry()


# =============================================================================
# Streaming Pattern Matcher Result
# =============================================================================


@dataclass
class StreamingPatternResult:
    """Result of streaming pattern matching for a column.

    Contains aggregated pattern matches and metadata about
    the streaming process.
    """

    column: str
    matches: list[PatternMatchResult]
    chunks_processed: int
    total_rows: int
    processing_time_ms: float
    aggregation_method: str
    inferred_type: DataType | None = None

    # Per-pattern statistics
    pattern_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def has_matches(self) -> bool:
        """Check if any patterns matched."""
        return len(self.matches) > 0

    @property
    def best_match(self) -> PatternMatchResult | None:
        """Get the best (highest ratio) match."""
        return self.matches[0] if self.matches else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "matches": [
                {
                    "pattern_name": m.pattern.name,
                    "match_ratio": m.match_ratio,
                    "match_count": m.match_count,
                    "total_count": m.total_count,
                }
                for m in self.matches
            ],
            "chunks_processed": self.chunks_processed,
            "total_rows": self.total_rows,
            "processing_time_ms": self.processing_time_ms,
            "aggregation_method": self.aggregation_method,
            "inferred_type": self.inferred_type.value if self.inferred_type else None,
            "pattern_stats": self.pattern_stats,
        }


# =============================================================================
# Pattern Event Callbacks
# =============================================================================


@dataclass
class PatternEvent:
    """Event emitted during pattern processing."""

    event_type: str  # "chunk_processed", "pattern_detected", "processing_complete"
    column: str
    chunk_index: int
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


PatternEventCallback = Callable[[PatternEvent], None]


# =============================================================================
# Streaming Pattern Matcher Configuration
# =============================================================================


@dataclass
class StreamingPatternConfig:
    """Configuration for streaming pattern matching.

    Attributes:
        aggregation_method: Method for aggregating chunk statistics
        min_match_ratio: Minimum ratio to consider a pattern matched
        sample_size_per_chunk: Max samples to collect per chunk
        patterns: Pattern registry to use
        enable_early_termination: Stop if pattern definitely matched/not matched
        early_termination_chunks: Chunks after which early termination is checked
        collect_statistics: Collect detailed per-chunk statistics
    """

    aggregation_method: AggregationMethod = AggregationMethod.ADAPTIVE
    min_match_ratio: float = 0.8
    sample_size_per_chunk: int = 3
    patterns: PatternRegistry | None = None
    enable_early_termination: bool = True
    early_termination_chunks: int = 3
    early_termination_confidence: float = 0.95
    collect_statistics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.min_match_ratio <= 1.0:
            raise ValueError(
                f"min_match_ratio must be between 0 and 1, got {self.min_match_ratio}"
            )
        if self.sample_size_per_chunk < 0:
            raise ValueError(
                f"sample_size_per_chunk must be non-negative, got {self.sample_size_per_chunk}"
            )

    @classmethod
    def fast(cls) -> "StreamingPatternConfig":
        """Create config optimized for speed."""
        return cls(
            aggregation_method=AggregationMethod.INCREMENTAL,
            min_match_ratio=0.7,
            sample_size_per_chunk=2,
            enable_early_termination=True,
            early_termination_chunks=2,
            collect_statistics=False,
        )

    @classmethod
    def accurate(cls) -> "StreamingPatternConfig":
        """Create config optimized for accuracy."""
        return cls(
            aggregation_method=AggregationMethod.CONSENSUS,
            min_match_ratio=0.85,
            sample_size_per_chunk=5,
            enable_early_termination=False,
            collect_statistics=True,
        )

    @classmethod
    def balanced(cls) -> "StreamingPatternConfig":
        """Create balanced config (default)."""
        return cls(
            aggregation_method=AggregationMethod.ADAPTIVE,
            min_match_ratio=0.8,
            sample_size_per_chunk=3,
            enable_early_termination=True,
            early_termination_chunks=3,
            collect_statistics=True,
        )


# =============================================================================
# Streaming Pattern Matcher
# =============================================================================


class StreamingPatternMatcher:
    """Chunk-aware pattern matcher for streaming data.

    This is the main interface for streaming pattern matching.
    It maintains state across chunks and provides aggregated
    results using configurable strategies.

    Example:
        # Basic usage
        matcher = StreamingPatternMatcher()

        for chunk in data_chunks:
            matcher.process_chunk(chunk, "column_name")

        result = matcher.finalize("column_name")
        for match in result.matches:
            print(f"{match.pattern.name}: {match.match_ratio:.2%}")

        # With configuration
        config = StreamingPatternConfig(
            aggregation_method=AggregationMethod.CONSENSUS,
            min_match_ratio=0.9,
        )
        matcher = StreamingPatternMatcher(config=config)

        # Process multiple columns
        for chunk in data_chunks:
            for col in ["email", "phone", "id"]:
                matcher.process_chunk(chunk, col)

        # Get all results
        results = matcher.finalize_all()
    """

    def __init__(
        self,
        config: StreamingPatternConfig | None = None,
        aggregation_strategy: AggregationStrategy | None = None,
        patterns: PatternRegistry | None = None,
        event_callback: PatternEventCallback | None = None,
    ):
        """Initialize the streaming pattern matcher.

        Args:
            config: Configuration for pattern matching
            aggregation_strategy: Override aggregation strategy
            patterns: Override pattern registry
            event_callback: Callback for pattern events
        """
        self.config = config or StreamingPatternConfig.balanced()

        # Allow overrides
        if aggregation_strategy:
            self._aggregation = aggregation_strategy
        else:
            self._aggregation = aggregation_strategy_registry.create_from_method(
                self.config.aggregation_method
            )

        self._patterns = patterns or self.config.patterns or BUILTIN_PATTERNS
        self._event_callback = event_callback

        # Internal matcher for per-chunk pattern detection
        self._chunk_matcher = NativePatternMatcher(
            patterns=self._patterns,
            min_match_ratio=0.0,  # We'll filter ourselves after aggregation
            sample_size=self.config.sample_size_per_chunk,
        )

        # State management
        self._column_states: dict[str, ColumnPatternState] = {}
        self._lock = threading.RLock()

    @property
    def patterns(self) -> PatternRegistry:
        """Get the pattern registry."""
        return self._patterns

    @property
    def aggregation_strategy(self) -> AggregationStrategy:
        """Get the current aggregation strategy."""
        return self._aggregation

    def process_chunk(
        self,
        chunk: pl.LazyFrame | pl.DataFrame,
        column: str,
        chunk_index: int | None = None,
    ) -> ChunkProcessingStatus:
        """Process a single chunk for pattern matching.

        This updates the internal state with pattern statistics
        from the chunk.

        Args:
            chunk: DataFrame or LazyFrame chunk to process
            column: Column name to analyze
            chunk_index: Optional chunk index (auto-incremented if not provided)

        Returns:
            Status of chunk processing
        """
        start_time = time.perf_counter()

        # Ensure LazyFrame
        if isinstance(chunk, pl.DataFrame):
            lf = chunk.lazy()
        else:
            lf = chunk

        with self._lock:
            # Get or create column state
            if column not in self._column_states:
                self._column_states[column] = ColumnPatternState(column_name=column)

            col_state = self._column_states[column]
            idx = chunk_index if chunk_index is not None else col_state.chunks_processed

            # Check early termination
            if self._should_terminate_early(col_state):
                self._emit_event("chunk_skipped", column, idx, {"reason": "early_termination"})
                return ChunkProcessingStatus.SKIPPED

        try:
            # Get chunk row count
            chunk_rows = lf.select(pl.len()).collect().item()
            if chunk_rows == 0:
                with self._lock:
                    col_state.add_chunk(0)
                return ChunkProcessingStatus.COMPLETED

            # Run pattern matching on chunk
            chunk_results = self._chunk_matcher.match_column(lf, column)

            # Get total non-null count
            total_count = (
                lf.select(pl.col(column).is_not_null().sum())
                .collect()
                .item()
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            with self._lock:
                # Update state for each pattern that was tested
                for pattern in self._patterns:
                    pattern_state = col_state.get_or_create_pattern_state(pattern)

                    # Find matching result
                    result = next(
                        (r for r in chunk_results if r.pattern.name == pattern.name),
                        None,
                    )

                    if result:
                        # Pattern was found in this chunk
                        stats = PatternChunkStats(
                            pattern_name=pattern.name,
                            match_count=result.match_count,
                            total_count=result.total_count,
                            chunk_index=idx,
                            processing_time_ms=elapsed_ms / len(list(self._patterns)),
                        )
                        pattern_state.add_chunk_stats(stats)
                        pattern_state.add_samples(result.sample_matches)
                    else:
                        # Pattern not found - record zero matches
                        stats = PatternChunkStats(
                            pattern_name=pattern.name,
                            match_count=0,
                            total_count=total_count,
                            chunk_index=idx,
                            processing_time_ms=elapsed_ms / len(list(self._patterns)),
                        )
                        pattern_state.add_chunk_stats(stats)

                col_state.add_chunk(chunk_rows)

            self._emit_event("chunk_processed", column, idx, {
                "rows": chunk_rows,
                "patterns_detected": len(chunk_results),
                "processing_time_ms": elapsed_ms,
            })

            return ChunkProcessingStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to process chunk {idx} for column '{column}': {e}")
            self._emit_event("chunk_failed", column, idx, {"error": str(e)})
            return ChunkProcessingStatus.FAILED

    def process_chunks(
        self,
        chunks: Iterator[pl.LazyFrame | pl.DataFrame],
        column: str,
    ) -> list[ChunkProcessingStatus]:
        """Process multiple chunks in sequence.

        Args:
            chunks: Iterator of chunks to process
            column: Column name to analyze

        Returns:
            List of processing statuses for each chunk
        """
        statuses = []
        for chunk in chunks:
            status = self.process_chunk(chunk, column)
            statuses.append(status)
            if status == ChunkProcessingStatus.SKIPPED:
                # Early termination triggered
                break
        return statuses

    def finalize(self, column: str) -> StreamingPatternResult:
        """Finalize pattern matching for a column.

        Aggregates all chunk statistics into final results.

        Args:
            column: Column name to finalize

        Returns:
            StreamingPatternResult with aggregated matches
        """
        with self._lock:
            if column not in self._column_states:
                return StreamingPatternResult(
                    column=column,
                    matches=[],
                    chunks_processed=0,
                    total_rows=0,
                    processing_time_ms=0.0,
                    aggregation_method=self._aggregation.name,
                )

            col_state = self._column_states[column]
            col_state.finalize()

        # Aggregate pattern statistics
        matches = []
        pattern_stats = {}

        for pattern_name, pattern_state in col_state.pattern_states.items():
            result = self._aggregation.aggregate(
                pattern_state,
                self.config.min_match_ratio,
            )

            if result is not None:
                matches.append(result)

            if self.config.collect_statistics:
                pattern_stats[pattern_name] = {
                    "total_matches": pattern_state.total_matches,
                    "total_rows": pattern_state.total_rows,
                    "overall_ratio": pattern_state.overall_match_ratio,
                    "chunks_with_matches": sum(
                        1 for s in pattern_state.chunk_stats if s.match_count > 0
                    ),
                    "variance": pattern_state.variance,
                    "is_consistent": pattern_state.is_consistent,
                }

        # Sort by match ratio
        matches.sort(key=lambda r: (-r.match_ratio, -r.pattern.priority))

        # Infer type from best match
        inferred_type = matches[0].pattern.data_type if matches else None

        self._emit_event("processing_complete", column, col_state.chunks_processed, {
            "matches": len(matches),
            "total_rows": col_state.total_rows,
        })

        return StreamingPatternResult(
            column=column,
            matches=matches,
            chunks_processed=col_state.chunks_processed,
            total_rows=col_state.total_rows,
            processing_time_ms=col_state.processing_duration_ms,
            aggregation_method=self._aggregation.name,
            inferred_type=inferred_type,
            pattern_stats=pattern_stats,
        )

    def finalize_all(self) -> dict[str, StreamingPatternResult]:
        """Finalize pattern matching for all processed columns.

        Returns:
            Dictionary mapping column names to their results
        """
        with self._lock:
            columns = list(self._column_states.keys())

        return {column: self.finalize(column) for column in columns}

    def reset(self, column: str | None = None) -> None:
        """Reset state for a column or all columns.

        Args:
            column: Column to reset, or None to reset all
        """
        with self._lock:
            if column is None:
                self._column_states.clear()
            elif column in self._column_states:
                del self._column_states[column]

    def get_current_state(self, column: str) -> ColumnPatternState | None:
        """Get current state for a column (for monitoring).

        Args:
            column: Column name

        Returns:
            Current column state or None
        """
        with self._lock:
            return self._column_states.get(column)

    def _should_terminate_early(self, state: ColumnPatternState) -> bool:
        """Check if early termination should be triggered."""
        if not self.config.enable_early_termination:
            return False

        if state.chunks_processed < self.config.early_termination_chunks:
            return False

        # Check if all patterns are clearly above or below threshold
        for pattern_state in state.pattern_states.values():
            if pattern_state.chunks_processed < 2:
                continue

            ratio = pattern_state.overall_match_ratio
            std = pattern_state.std_deviation

            # Pattern is clearly matching
            if ratio - 2 * std > self.config.min_match_ratio:
                continue

            # Pattern is clearly not matching
            if ratio + 2 * std < self.config.min_match_ratio:
                continue

            # Pattern is uncertain - continue processing
            return False

        # All patterns are determined
        return True

    def _emit_event(
        self,
        event_type: str,
        column: str,
        chunk_index: int,
        data: dict[str, Any],
    ) -> None:
        """Emit a pattern event."""
        if self._event_callback:
            event = PatternEvent(
                event_type=event_type,
                column=column,
                chunk_index=chunk_index,
                data=data,
            )
            try:
                self._event_callback(event)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")


# =============================================================================
# Integration with StreamingProfiler
# =============================================================================


class StreamingPatternIntegration:
    """Integration layer for StreamingProfiler.

    This class provides the interface for integrating streaming
    pattern matching with the existing StreamingProfiler.
    """

    def __init__(
        self,
        config: StreamingPatternConfig | None = None,
        patterns: PatternRegistry | None = None,
    ):
        """Initialize integration.

        Args:
            config: Pattern matching configuration
            patterns: Pattern registry to use
        """
        self.config = config or StreamingPatternConfig.balanced()
        self.matcher = StreamingPatternMatcher(
            config=self.config,
            patterns=patterns,
        )

    def process_column_chunk(
        self,
        chunk: pl.LazyFrame | pl.DataFrame,
        column: str,
        chunk_index: int,
    ) -> None:
        """Process a column in a chunk.

        Called by StreamingProfiler for each chunk.

        Args:
            chunk: Data chunk
            column: Column name
            chunk_index: Index of this chunk
        """
        self.matcher.process_chunk(chunk, column, chunk_index)

    def get_column_patterns(self, column: str) -> tuple[PatternMatch, ...]:
        """Get detected patterns for a column.

        Called by StreamingProfiler when building ColumnProfile.

        Args:
            column: Column name

        Returns:
            Tuple of PatternMatch objects
        """
        result = self.matcher.finalize(column)
        return tuple(r.to_pattern_match() for r in result.matches)

    def get_inferred_type(self, column: str) -> DataType | None:
        """Get inferred type for a column.

        Args:
            column: Column name

        Returns:
            Inferred DataType or None
        """
        result = self.matcher.finalize(column)
        return result.inferred_type

    def reset(self) -> None:
        """Reset all state."""
        self.matcher.reset()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_streaming_matcher(
    aggregation: str | AggregationMethod = "adaptive",
    min_match_ratio: float = 0.8,
    **kwargs: Any,
) -> StreamingPatternMatcher:
    """Create a streaming pattern matcher with common options.

    Args:
        aggregation: Aggregation method name or enum
        min_match_ratio: Minimum match ratio threshold
        **kwargs: Additional config options

    Returns:
        Configured StreamingPatternMatcher

    Example:
        matcher = create_streaming_matcher(
            aggregation="consensus",
            min_match_ratio=0.9,
        )
    """
    if isinstance(aggregation, str):
        aggregation = AggregationMethod(aggregation)

    config = StreamingPatternConfig(
        aggregation_method=aggregation,
        min_match_ratio=min_match_ratio,
        **kwargs,
    )

    return StreamingPatternMatcher(config=config)


def stream_match_patterns(
    chunks: Iterator[pl.LazyFrame | pl.DataFrame],
    column: str,
    *,
    aggregation: str = "adaptive",
    min_ratio: float = 0.8,
) -> StreamingPatternResult:
    """Convenience function for streaming pattern matching.

    Args:
        chunks: Iterator of data chunks
        column: Column to analyze
        aggregation: Aggregation method
        min_ratio: Minimum match ratio

    Returns:
        StreamingPatternResult

    Example:
        from truthound.profiler.streaming_patterns import stream_match_patterns

        # From file chunks
        result = stream_match_patterns(
            file_chunk_iterator("data.csv"),
            "email_column",
        )

        print(f"Best match: {result.best_match.pattern.name}")
        print(f"Chunks processed: {result.chunks_processed}")
    """
    matcher = create_streaming_matcher(
        aggregation=aggregation,
        min_match_ratio=min_ratio,
    )

    matcher.process_chunks(chunks, column)
    return matcher.finalize(column)


def get_available_aggregation_methods() -> list[str]:
    """Get list of available aggregation methods."""
    return aggregation_strategy_registry.list_strategies()
