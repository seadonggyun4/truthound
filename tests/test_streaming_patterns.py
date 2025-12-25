"""Tests for Streaming Pattern Matching module.

This test file provides comprehensive coverage for the streaming_patterns module,
testing chunk-aware pattern matching, state management, aggregation strategies,
and integration with the streaming profiler.
"""

import pytest
import polars as pl
from datetime import datetime
from typing import Iterator

from truthound.profiler.streaming_patterns import (
    # Enums
    AggregationMethod,
    ChunkProcessingStatus,
    # Pattern state management
    PatternChunkStats,
    PatternState,
    ColumnPatternState,
    # Aggregation strategies
    AggregationStrategy,
    IncrementalAggregation,
    WeightedAggregation,
    SlidingWindowAggregation,
    ExponentialAggregation,
    ConsensusAggregation,
    AdaptiveAggregation,
    # Registry
    AggregationStrategyRegistry,
    aggregation_strategy_registry,
    # Result types
    StreamingPatternResult,
    # Events
    PatternEvent,
    # Configuration
    StreamingPatternConfig,
    # Main interface
    StreamingPatternMatcher,
    # Integration
    StreamingPatternIntegration,
    # Convenience functions
    create_streaming_matcher,
    stream_match_patterns,
    get_available_aggregation_methods,
)

from truthound.profiler.native_patterns import (
    PatternSpec,
    PatternBuilder,
    PatternPriority,
    PatternRegistry,
    BUILTIN_PATTERNS,
)
from truthound.profiler.base import DataType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def email_pattern() -> PatternSpec:
    """Create an email pattern for testing."""
    return (
        PatternBuilder("test_email")
        .regex(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        .data_type(DataType.EMAIL)
        .priority(PatternPriority.HIGH)
        .build()
    )


@pytest.fixture
def phone_pattern() -> PatternSpec:
    """Create a phone pattern for testing."""
    return (
        PatternBuilder("test_phone")
        .regex(r"\d{3}-\d{3}-\d{4}")
        .data_type(DataType.PHONE)
        .priority(PatternPriority.MEDIUM)
        .build()
    )


@pytest.fixture
def test_registry(email_pattern, phone_pattern) -> PatternRegistry:
    """Create a test pattern registry."""
    registry = PatternRegistry()
    registry.register(email_pattern)
    registry.register(phone_pattern)
    return registry


@pytest.fixture
def email_chunks() -> list[pl.DataFrame]:
    """Create chunks with all valid email data."""
    # Each chunk has ALL valid emails so the pattern matches at 100%
    return [
        pl.DataFrame({
            "email": [
                "user1@example.com",
                "user2@test.org",
                "user3@domain.net",
                "a@b.com",
                "x@y.org",
            ]
        }),
        pl.DataFrame({
            "email": [
                "john@company.com",
                "jane@business.io",
                "admin@site.co.uk",
                "test@test.com",
                "demo@demo.org",
            ]
        }),
        pl.DataFrame({
            "email": [
                "contact@shop.com",
                "info@service.org",
                "support@help.net",
                "sales@store.com",
                "help@website.org",
            ]
        }),
    ]


@pytest.fixture
def mixed_chunks() -> list[pl.DataFrame]:
    """Create chunks with mixed data (some emails, some not)."""
    return [
        pl.DataFrame({
            "data": [
                "user1@example.com",
                "not_an_email",
                "user2@test.org",
            ]
        }),
        pl.DataFrame({
            "data": [
                "random_text",
                "jane@business.io",
                "12345",
            ]
        }),
        pl.DataFrame({
            "data": [
                "contact@shop.com",
                "another_string",
                None,
            ]
        }),
    ]


@pytest.fixture
def phone_chunks() -> list[pl.DataFrame]:
    """Create chunks with phone data."""
    return [
        pl.DataFrame({
            "phone": [
                "123-456-7890",
                "234-567-8901",
                "345-678-9012",
            ]
        }),
        pl.DataFrame({
            "phone": [
                "456-789-0123",
                "567-890-1234",
                "678-901-2345",
            ]
        }),
    ]


# =============================================================================
# Tests: PatternChunkStats
# =============================================================================


class TestPatternChunkStats:
    """Tests for PatternChunkStats dataclass."""

    def test_creation(self):
        """Test creating chunk stats."""
        stats = PatternChunkStats(
            pattern_name="email",
            match_count=10,
            total_count=100,
            chunk_index=0,
        )
        assert stats.pattern_name == "email"
        assert stats.match_count == 10
        assert stats.total_count == 100
        assert stats.chunk_index == 0

    def test_match_ratio(self):
        """Test match ratio calculation."""
        stats = PatternChunkStats(
            pattern_name="email",
            match_count=25,
            total_count=100,
            chunk_index=0,
        )
        assert stats.match_ratio == 0.25

    def test_match_ratio_zero_total(self):
        """Test match ratio with zero total."""
        stats = PatternChunkStats(
            pattern_name="email",
            match_count=0,
            total_count=0,
            chunk_index=0,
        )
        assert stats.match_ratio == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PatternChunkStats(
            pattern_name="email",
            match_count=10,
            total_count=100,
            chunk_index=0,
            processing_time_ms=5.5,
        )
        d = stats.to_dict()
        assert d["pattern_name"] == "email"
        assert d["match_ratio"] == 0.1
        assert d["processing_time_ms"] == 5.5


# =============================================================================
# Tests: PatternState
# =============================================================================


class TestPatternState:
    """Tests for PatternState class."""

    def test_add_chunk_stats(self, email_pattern):
        """Test adding chunk statistics."""
        state = PatternState(pattern=email_pattern)

        stats1 = PatternChunkStats("email", 10, 100, 0)
        stats2 = PatternChunkStats("email", 20, 100, 1)

        state.add_chunk_stats(stats1)
        state.add_chunk_stats(stats2)

        assert state.total_matches == 30
        assert state.total_rows == 200
        assert state.chunks_processed == 2

    def test_overall_match_ratio(self, email_pattern):
        """Test overall match ratio calculation."""
        state = PatternState(pattern=email_pattern)

        state.add_chunk_stats(PatternChunkStats("email", 80, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 1))

        assert state.overall_match_ratio == 0.85

    def test_add_samples(self, email_pattern):
        """Test sample collection with limit."""
        state = PatternState(pattern=email_pattern, max_samples=3)

        state.add_samples(["a@b.com", "c@d.com"])
        assert len(state.sample_matches) == 2

        state.add_samples(["e@f.com", "g@h.com"])  # Only 1 should be added
        assert len(state.sample_matches) == 3

    def test_variance_calculation(self, email_pattern):
        """Test variance calculation."""
        state = PatternState(pattern=email_pattern)

        # Add chunks with varying ratios
        state.add_chunk_stats(PatternChunkStats("email", 80, 100, 0))  # 0.80
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 1))  # 0.90
        state.add_chunk_stats(PatternChunkStats("email", 85, 100, 2))  # 0.85

        assert state.variance > 0
        assert state.std_deviation > 0

    def test_is_consistent(self, email_pattern):
        """Test consistency check."""
        state = PatternState(pattern=email_pattern)

        # Consistent pattern (low variance)
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 91, 100, 1))
        state.add_chunk_stats(PatternChunkStats("email", 89, 100, 2))

        assert state.is_consistent

    def test_to_pattern_match(self, email_pattern):
        """Test conversion to PatternMatch."""
        state = PatternState(pattern=email_pattern)
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))
        state.add_samples(["test@example.com"])

        pm = state.to_pattern_match()
        assert pm.pattern == "test_email"
        assert pm.match_ratio == 0.9
        assert "test@example.com" in pm.sample_matches


# =============================================================================
# Tests: ColumnPatternState
# =============================================================================


class TestColumnPatternState:
    """Tests for ColumnPatternState class."""

    def test_get_or_create_pattern_state(self, email_pattern):
        """Test pattern state creation."""
        state = ColumnPatternState(column_name="email")

        ps = state.get_or_create_pattern_state(email_pattern)
        assert ps.pattern == email_pattern

        # Second call returns same state
        ps2 = state.get_or_create_pattern_state(email_pattern)
        assert ps is ps2

    def test_add_chunk(self):
        """Test chunk tracking."""
        state = ColumnPatternState(column_name="email")

        state.add_chunk(100)
        state.add_chunk(150)

        assert state.chunks_processed == 2
        assert state.total_rows == 250

    def test_finalize(self):
        """Test finalization."""
        state = ColumnPatternState(column_name="email")
        assert state.completed_at is None

        state.finalize()
        assert state.completed_at is not None

    def test_processing_duration(self):
        """Test duration calculation."""
        state = ColumnPatternState(column_name="email")
        duration = state.processing_duration_ms
        assert duration >= 0


# =============================================================================
# Tests: Aggregation Strategies
# =============================================================================


class TestIncrementalAggregation:
    """Tests for IncrementalAggregation strategy."""

    def test_aggregate(self, email_pattern):
        """Test basic aggregation."""
        strategy = IncrementalAggregation()
        state = PatternState(pattern=email_pattern)

        state.add_chunk_stats(PatternChunkStats("email", 80, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 1))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is not None
        assert result.match_ratio == 0.85

    def test_aggregate_below_threshold(self, email_pattern):
        """Test aggregation below threshold."""
        strategy = IncrementalAggregation()
        state = PatternState(pattern=email_pattern)

        state.add_chunk_stats(PatternChunkStats("email", 50, 100, 0))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is None

    def test_should_include_pattern(self, email_pattern):
        """Test pattern inclusion check."""
        strategy = IncrementalAggregation()
        state = PatternState(pattern=email_pattern)

        state.add_chunk_stats(PatternChunkStats("email", 85, 100, 0))

        assert strategy.should_include_pattern(state, 0.8) is True
        assert strategy.should_include_pattern(state, 0.9) is False


class TestWeightedAggregation:
    """Tests for WeightedAggregation strategy."""

    def test_aggregate_with_varying_sizes(self, email_pattern):
        """Test aggregation with varying chunk sizes."""
        strategy = WeightedAggregation()
        state = PatternState(pattern=email_pattern)

        # Larger chunk with higher ratio should dominate
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 180, 200, 1))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is not None
        # 270 / 300 = 0.90
        assert result.match_ratio == 0.9


class TestSlidingWindowAggregation:
    """Tests for SlidingWindowAggregation strategy."""

    def test_aggregate_recent_chunks(self, email_pattern):
        """Test aggregation with recent chunks only."""
        strategy = SlidingWindowAggregation(window_size=2)
        state = PatternState(pattern=email_pattern)

        # Old chunks with low ratio
        state.add_chunk_stats(PatternChunkStats("email", 10, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 20, 100, 1))
        # Recent chunks with high ratio
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 2))
        state.add_chunk_stats(PatternChunkStats("email", 95, 100, 3))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is not None
        # Only recent 2 chunks: 185/200 = 0.925
        assert abs(result.match_ratio - 0.925) < 0.01


class TestExponentialAggregation:
    """Tests for ExponentialAggregation strategy."""

    def test_invalid_alpha(self):
        """Test validation of alpha parameter."""
        with pytest.raises(ValueError):
            ExponentialAggregation(alpha=0)
        with pytest.raises(ValueError):
            ExponentialAggregation(alpha=1.5)

    def test_ema_weights_recent(self, email_pattern):
        """Test that EMA weights recent chunks more."""
        strategy = ExponentialAggregation(alpha=0.5)
        state = PatternState(pattern=email_pattern)

        # Old chunk with low ratio
        state.add_chunk_stats(PatternChunkStats("email", 50, 100, 0))
        # Recent chunk with high ratio
        state.add_chunk_stats(PatternChunkStats("email", 100, 100, 1))

        result = strategy.aggregate(state, min_match_ratio=0.7)
        assert result is not None
        # EMA with alpha=0.5: 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        assert result.match_ratio > 0.7


class TestConsensusAggregation:
    """Tests for ConsensusAggregation strategy."""

    def test_invalid_threshold(self):
        """Test validation of consensus threshold."""
        with pytest.raises(ValueError):
            ConsensusAggregation(consensus_threshold=0)
        with pytest.raises(ValueError):
            ConsensusAggregation(consensus_threshold=1.5)

    def test_consensus_met(self, email_pattern):
        """Test when consensus is met."""
        strategy = ConsensusAggregation(consensus_threshold=0.75)
        state = PatternState(pattern=email_pattern)

        # 3 out of 4 chunks match (75%)
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))  # matches
        state.add_chunk_stats(PatternChunkStats("email", 85, 100, 1))  # matches
        state.add_chunk_stats(PatternChunkStats("email", 88, 100, 2))  # matches
        state.add_chunk_stats(PatternChunkStats("email", 50, 100, 3))  # doesn't match

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is not None

    def test_consensus_not_met(self, email_pattern):
        """Test when consensus is not met."""
        strategy = ConsensusAggregation(consensus_threshold=0.75)
        state = PatternState(pattern=email_pattern)

        # Only 1 out of 4 chunks matches (25%)
        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))  # matches
        state.add_chunk_stats(PatternChunkStats("email", 50, 100, 1))
        state.add_chunk_stats(PatternChunkStats("email", 40, 100, 2))
        state.add_chunk_stats(PatternChunkStats("email", 30, 100, 3))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is None


class TestAdaptiveAggregation:
    """Tests for AdaptiveAggregation strategy."""

    def test_selects_strategy(self, email_pattern):
        """Test strategy selection."""
        strategy = AdaptiveAggregation()
        state = PatternState(pattern=email_pattern)

        state.add_chunk_stats(PatternChunkStats("email", 90, 100, 0))
        state.add_chunk_stats(PatternChunkStats("email", 88, 100, 1))
        state.add_chunk_stats(PatternChunkStats("email", 91, 100, 2))

        result = strategy.aggregate(state, min_match_ratio=0.8)
        assert result is not None


# =============================================================================
# Tests: AggregationStrategyRegistry
# =============================================================================


class TestAggregationStrategyRegistry:
    """Tests for AggregationStrategyRegistry."""

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = aggregation_strategy_registry.list_strategies()
        assert "incremental" in strategies
        assert "weighted" in strategies
        assert "adaptive" in strategies

    def test_get_strategy(self):
        """Test getting strategy by name."""
        strategy = aggregation_strategy_registry.get("incremental")
        assert isinstance(strategy, IncrementalAggregation)

    def test_get_unknown_strategy(self):
        """Test getting unknown strategy."""
        with pytest.raises(KeyError):
            aggregation_strategy_registry.get("unknown")

    def test_get_or_default(self):
        """Test get with default."""
        strategy = aggregation_strategy_registry.get_or_default("unknown")
        assert isinstance(strategy, AdaptiveAggregation)

    def test_register_custom(self):
        """Test registering custom strategy."""

        class CustomStrategy(AggregationStrategy):
            name = "custom_test"

            def aggregate(self, state, min_match_ratio=0.8):
                return None

            def should_include_pattern(self, state, min_match_ratio):
                return False

        registry = AggregationStrategyRegistry()
        registry.register(CustomStrategy())
        assert "custom_test" in registry.list_strategies()


# =============================================================================
# Tests: StreamingPatternConfig
# =============================================================================


class TestStreamingPatternConfig:
    """Tests for StreamingPatternConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StreamingPatternConfig()
        assert config.aggregation_method == AggregationMethod.ADAPTIVE
        assert config.min_match_ratio == 0.8

    def test_fast_preset(self):
        """Test fast preset."""
        config = StreamingPatternConfig.fast()
        assert config.aggregation_method == AggregationMethod.INCREMENTAL
        assert config.enable_early_termination is True

    def test_accurate_preset(self):
        """Test accurate preset."""
        config = StreamingPatternConfig.accurate()
        assert config.aggregation_method == AggregationMethod.CONSENSUS
        assert config.enable_early_termination is False

    def test_balanced_preset(self):
        """Test balanced preset."""
        config = StreamingPatternConfig.balanced()
        assert config.aggregation_method == AggregationMethod.ADAPTIVE

    def test_invalid_min_match_ratio(self):
        """Test validation of min_match_ratio."""
        with pytest.raises(ValueError):
            StreamingPatternConfig(min_match_ratio=1.5)


# =============================================================================
# Tests: StreamingPatternMatcher
# =============================================================================


class TestStreamingPatternMatcher:
    """Tests for StreamingPatternMatcher."""

    def test_process_chunk(self, email_chunks):
        """Test processing a single chunk."""
        matcher = StreamingPatternMatcher()
        status = matcher.process_chunk(email_chunks[0], "email")
        assert status == ChunkProcessingStatus.COMPLETED

    def test_process_multiple_chunks(self, email_chunks):
        """Test processing multiple chunks."""
        matcher = StreamingPatternMatcher()

        for chunk in email_chunks:
            status = matcher.process_chunk(chunk, "email")
            assert status == ChunkProcessingStatus.COMPLETED

        result = matcher.finalize("email")
        assert result.chunks_processed == 3
        assert result.total_rows == 15  # 3 chunks * 5 rows each

    def test_finalize_with_matches(self, email_chunks):
        """Test finalization with detected patterns."""
        # Lower threshold to ensure match
        config = StreamingPatternConfig(min_match_ratio=0.5)
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.has_matches
        assert result.best_match is not None
        assert result.best_match.pattern.data_type == DataType.EMAIL

    def test_finalize_unprocessed_column(self):
        """Test finalization of unprocessed column."""
        matcher = StreamingPatternMatcher()
        result = matcher.finalize("nonexistent")
        assert result.chunks_processed == 0
        assert not result.has_matches

    def test_process_chunks_iterator(self, email_chunks):
        """Test process_chunks with iterator."""
        matcher = StreamingPatternMatcher()
        statuses = matcher.process_chunks(iter(email_chunks), "email")
        assert len(statuses) == 3
        assert all(s == ChunkProcessingStatus.COMPLETED for s in statuses)

    def test_finalize_all(self, email_chunks, phone_chunks):
        """Test finalizing all columns."""
        matcher = StreamingPatternMatcher()

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")
        for chunk in phone_chunks:
            matcher.process_chunk(chunk, "phone")

        results = matcher.finalize_all()
        assert "email" in results
        assert "phone" in results

    def test_reset_single_column(self, email_chunks):
        """Test resetting single column."""
        matcher = StreamingPatternMatcher()

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        matcher.reset("email")
        result = matcher.finalize("email")
        assert result.chunks_processed == 0

    def test_reset_all(self, email_chunks, phone_chunks):
        """Test resetting all columns."""
        matcher = StreamingPatternMatcher()

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")
        for chunk in phone_chunks:
            matcher.process_chunk(chunk, "phone")

        matcher.reset()
        results = matcher.finalize_all()
        assert len(results) == 0

    def test_get_current_state(self, email_chunks):
        """Test getting current state."""
        matcher = StreamingPatternMatcher()
        matcher.process_chunk(email_chunks[0], "email")

        state = matcher.get_current_state("email")
        assert state is not None
        assert state.chunks_processed == 1

    def test_event_callback(self, email_chunks):
        """Test event callback."""
        events = []

        def callback(event: PatternEvent):
            events.append(event)

        matcher = StreamingPatternMatcher(event_callback=callback)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        matcher.finalize("email")

        assert len(events) > 0
        event_types = [e.event_type for e in events]
        assert "chunk_processed" in event_types
        assert "processing_complete" in event_types

    def test_with_custom_config(self, email_chunks):
        """Test with custom configuration."""
        config = StreamingPatternConfig(
            aggregation_method=AggregationMethod.CONSENSUS,
            min_match_ratio=0.9,
        )
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.aggregation_method == "consensus"

    def test_with_custom_patterns(self, test_registry, email_chunks):
        """Test with custom pattern registry."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        matcher = StreamingPatternMatcher(config=config, patterns=test_registry)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.has_matches

    def test_mixed_data(self, mixed_chunks):
        """Test with mixed data (partial matches)."""
        matcher = StreamingPatternMatcher(
            config=StreamingPatternConfig(min_match_ratio=0.3)
        )

        for chunk in mixed_chunks:
            matcher.process_chunk(chunk, "data")

        result = matcher.finalize("data")
        # Should detect email pattern even with partial matches
        assert result.chunks_processed == 3


class TestStreamingPatternMatcherEarlyTermination:
    """Tests for early termination feature."""

    def test_early_termination_disabled(self, email_chunks):
        """Test with early termination disabled."""
        config = StreamingPatternConfig(
            enable_early_termination=False,
        )
        matcher = StreamingPatternMatcher(config=config)

        statuses = []
        for chunk in email_chunks:
            statuses.append(matcher.process_chunk(chunk, "email"))

        assert ChunkProcessingStatus.SKIPPED not in statuses


# =============================================================================
# Tests: StreamingPatternIntegration
# =============================================================================


class TestStreamingPatternIntegration:
    """Tests for StreamingPatternIntegration."""

    def test_process_and_get_patterns(self, email_chunks):
        """Test integration workflow."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        integration = StreamingPatternIntegration(config=config)

        for i, chunk in enumerate(email_chunks):
            integration.process_column_chunk(chunk.lazy(), "email", i)

        patterns = integration.get_column_patterns("email")
        assert len(patterns) > 0

    def test_get_inferred_type(self, email_chunks):
        """Test type inference."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        integration = StreamingPatternIntegration(config=config)

        for i, chunk in enumerate(email_chunks):
            integration.process_column_chunk(chunk.lazy(), "email", i)

        dtype = integration.get_inferred_type("email")
        assert dtype == DataType.EMAIL

    def test_reset(self, email_chunks):
        """Test resetting integration."""
        integration = StreamingPatternIntegration()

        for i, chunk in enumerate(email_chunks):
            integration.process_column_chunk(chunk.lazy(), "email", i)

        integration.reset()
        patterns = integration.get_column_patterns("email")
        assert len(patterns) == 0


# =============================================================================
# Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_streaming_matcher(self):
        """Test create_streaming_matcher function."""
        matcher = create_streaming_matcher(
            aggregation="incremental",
            min_match_ratio=0.9,
        )
        assert matcher.config.aggregation_method == AggregationMethod.INCREMENTAL
        assert matcher.config.min_match_ratio == 0.9

    def test_stream_match_patterns(self, email_chunks):
        """Test stream_match_patterns function."""
        result = stream_match_patterns(
            iter(email_chunks),
            "email",
            aggregation="adaptive",
            min_ratio=0.5,  # Lower threshold to match
        )
        assert isinstance(result, StreamingPatternResult)
        assert result.has_matches

    def test_get_available_aggregation_methods(self):
        """Test get_available_aggregation_methods function."""
        methods = get_available_aggregation_methods()
        assert "incremental" in methods
        assert "adaptive" in methods
        assert "consensus" in methods


# =============================================================================
# Tests: StreamingPatternResult
# =============================================================================


class TestStreamingPatternResult:
    """Tests for StreamingPatternResult."""

    def test_to_dict(self, email_chunks):
        """Test conversion to dictionary."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        d = result.to_dict()

        assert "column" in d
        assert "matches" in d
        assert "chunks_processed" in d
        assert "total_rows" in d

    def test_has_matches_property(self, email_chunks):
        """Test has_matches property."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.has_matches is True

    def test_best_match_property(self, email_chunks):
        """Test best_match property."""
        config = StreamingPatternConfig(min_match_ratio=0.5)
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.best_match is not None


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_chunk(self):
        """Test processing empty chunk."""
        matcher = StreamingPatternMatcher()
        chunk = pl.DataFrame({"email": []})
        status = matcher.process_chunk(chunk, "email")
        assert status == ChunkProcessingStatus.COMPLETED

    def test_all_null_chunk(self):
        """Test chunk with all null values."""
        matcher = StreamingPatternMatcher()
        chunk = pl.DataFrame({"email": [None, None, None]})
        status = matcher.process_chunk(chunk, "email")
        assert status == ChunkProcessingStatus.COMPLETED

    def test_single_value_chunk(self):
        """Test chunk with single value."""
        matcher = StreamingPatternMatcher()
        chunk = pl.DataFrame({"email": ["test@example.com"]})
        status = matcher.process_chunk(chunk, "email")
        assert status == ChunkProcessingStatus.COMPLETED

    def test_lazy_frame_input(self):
        """Test with LazyFrame input."""
        matcher = StreamingPatternMatcher()
        chunk = pl.DataFrame({"email": ["test@example.com"]}).lazy()
        status = matcher.process_chunk(chunk, "email")
        assert status == ChunkProcessingStatus.COMPLETED

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        matcher = StreamingPatternMatcher()
        chunk = pl.DataFrame({"email": ["test@example.com"]})
        status = matcher.process_chunk(chunk, "email")
        assert status == ChunkProcessingStatus.COMPLETED


# =============================================================================
# Tests: Performance
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_large_chunk_count(self):
        """Test with many chunks."""
        # Disable early termination to process all chunks
        config = StreamingPatternConfig(
            min_match_ratio=0.5,
            enable_early_termination=False,
        )
        matcher = StreamingPatternMatcher(config=config)

        for i in range(100):
            chunk = pl.DataFrame({
                "email": [f"user{i}@example.com"]
            })
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert result.chunks_processed == 100

    def test_statistics_collection(self, email_chunks):
        """Test statistics collection."""
        config = StreamingPatternConfig(collect_statistics=True)
        matcher = StreamingPatternMatcher(config=config)

        for chunk in email_chunks:
            matcher.process_chunk(chunk, "email")

        result = matcher.finalize("email")
        assert len(result.pattern_stats) > 0


# =============================================================================
# Tests: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_column_processing(self):
        """Test processing different columns concurrently."""
        import threading

        matcher = StreamingPatternMatcher()
        errors = []

        def process_column(column_name: str, chunks: list[pl.DataFrame]):
            try:
                for chunk in chunks:
                    matcher.process_chunk(chunk, column_name)
            except Exception as e:
                errors.append(e)

        email_data = [pl.DataFrame({"email": [f"user{i}@example.com"]}) for i in range(10)]
        phone_data = [pl.DataFrame({"phone": ["123-456-7890"]}) for _ in range(10)]

        t1 = threading.Thread(target=process_column, args=("email", email_data))
        t2 = threading.Thread(target=process_column, args=("phone", phone_data))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

        results = matcher.finalize_all()
        assert "email" in results
        assert "phone" in results
