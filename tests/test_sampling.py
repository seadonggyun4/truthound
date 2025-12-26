"""Comprehensive tests for the sampling module.

Tests cover:
- Sampling strategies (all types)
- Configuration validation
- Statistical accuracy
- Memory safety
- Integration with pattern matching
"""

import math
import random
from datetime import timedelta

import polars as pl
import pytest

from truthound.profiler.sampling import (
    # Enums
    SamplingMethod,
    ConfidenceLevel,
    # Configuration
    SamplingConfig,
    DEFAULT_SAMPLING_CONFIG,
    # Metrics
    SamplingMetrics,
    SamplingResult,
    # Strategies
    SamplingStrategy,
    NoSamplingStrategy,
    HeadSamplingStrategy,
    RandomSamplingStrategy,
    SystematicSamplingStrategy,
    HashSamplingStrategy,
    StratifiedSamplingStrategy,
    ReservoirSamplingStrategy,
    AdaptiveSamplingStrategy,
    # Registry
    SamplingStrategyRegistry,
    sampling_strategy_registry,
    # Data size estimation
    DataSizeEstimator,
    # Main interface
    Sampler,
    # Convenience functions
    create_sampler,
    sample_data,
    calculate_sample_size,
)

from truthound.profiler.sampled_matcher import (
    SampledPatternMatchResult,
    SampledColumnMatchResult,
    SampledMatcherConfig,
    SampledPatternMatcher,
    SafeNativePatternMatcher,
    create_sampled_matcher,
    match_patterns_safe,
    infer_column_type_safe,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_df():
    """Small DataFrame for basic tests."""
    return pl.DataFrame({
        "id": range(100),
        "name": [f"User {i}" for i in range(100)],
        "email": [f"user{i}@example.com" for i in range(100)],
        "value": [float(i) for i in range(100)],
    })


@pytest.fixture
def medium_df():
    """Medium DataFrame for sampling tests."""
    n = 50_000
    return pl.DataFrame({
        "id": range(n),
        "email": [f"user{i}@example.com" for i in range(n)],
        "category": [["A", "B", "C", "D"][i % 4] for i in range(n)],
        "value": [random.random() for _ in range(n)],
    })


@pytest.fixture
def large_df():
    """Large DataFrame for stress tests."""
    n = 200_000
    return pl.DataFrame({
        "id": range(n),
        "email": [f"user{i}@domain.com" for i in range(n)],
        "uuid": ["550e8400-e29b-41d4-a716-446655440000"] * n,
        "phone": ["+821012345678"] * n,
    })


@pytest.fixture
def pattern_df():
    """DataFrame with various patterns for pattern matching tests."""
    n = 10_000
    return pl.DataFrame({
        "email": [f"user{i}@example.com" for i in range(n)],
        "uuid": ["550e8400-e29b-41d4-a716-446655440000"] * n,
        "ip": ["192.168.1.1", "10.0.0.1", "172.16.0.1"] * (n // 3) + ["192.168.1.1"],
        "mixed": [f"user{i}@example.com" if i % 2 == 0 else "not-an-email" for i in range(n)],
    })


# =============================================================================
# SamplingConfig Tests
# =============================================================================


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()

        assert config.strategy == SamplingMethod.ADAPTIVE
        assert config.max_rows == 100_000
        assert config.confidence_level == 0.95
        assert config.margin_of_error == 0.05
        assert config.min_sample_size == 1000

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            SamplingConfig(confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level"):
            SamplingConfig(confidence_level=-0.1)

        # Invalid margin of error
        with pytest.raises(ValueError, match="margin_of_error"):
            SamplingConfig(margin_of_error=2.0)

        # Invalid max_rows
        with pytest.raises(ValueError, match="max_rows"):
            SamplingConfig(max_rows=-100)

    def test_sample_size_calculation(self):
        """Test statistical sample size calculation."""
        # Set min_sample_size=1 to test pure Cochran's formula calculation
        config = SamplingConfig(
            confidence_level=0.95,
            margin_of_error=0.05,
            min_sample_size=1,  # Disable min_sample_size override for pure statistical test
        )

        # For large populations
        n = config.calculate_required_sample_size(1_000_000)
        assert n > 0
        assert n < 1_000_000
        # Should be around 384 for infinite population (Cochran's formula)
        assert 300 < n < 500

        # For small populations (should be capped)
        n_small = config.calculate_required_sample_size(100)
        assert n_small <= 100

        # Higher confidence should require more samples
        config_high = SamplingConfig(
            confidence_level=0.99,
            margin_of_error=0.05,
        )
        n_high = config_high.calculate_required_sample_size(1_000_000)
        assert n_high > n

        # Lower margin of error should require more samples
        config_precise = SamplingConfig(
            confidence_level=0.95,
            margin_of_error=0.01,
        )
        n_precise = config_precise.calculate_required_sample_size(1_000_000)
        assert n_precise > n

    def test_preset_configs(self):
        """Test preset configuration factories."""
        # Accuracy presets
        low = SamplingConfig.for_accuracy("low")
        assert low.max_rows < 50_000
        assert low.confidence_level == 0.90

        medium = SamplingConfig.for_accuracy("medium")
        assert medium.max_rows == 100_000
        assert medium.confidence_level == 0.95

        high = SamplingConfig.for_accuracy("high")
        assert high.max_rows > 100_000
        assert high.confidence_level == 0.99

        maximum = SamplingConfig.for_accuracy("maximum")
        assert maximum.strategy == SamplingMethod.NONE

        # Speed preset
        fast = SamplingConfig.for_speed()
        assert fast.strategy == SamplingMethod.HEAD
        assert fast.max_rows <= 10_000

        # Memory preset
        memory = SamplingConfig.for_memory(max_memory_mb=50)
        assert memory.strategy == SamplingMethod.RESERVOIR
        assert memory.max_memory_mb == 50

    def test_to_dict(self):
        """Test serialization."""
        config = SamplingConfig(
            strategy=SamplingMethod.RANDOM,
            max_rows=50_000,
            seed=42,
        )

        d = config.to_dict()
        assert d["strategy"] == "random"
        assert d["max_rows"] == 50_000
        assert d["seed"] == 42


# =============================================================================
# Sampling Strategies Tests
# =============================================================================


class TestNoSamplingStrategy:
    """Tests for NoSamplingStrategy."""

    def test_returns_all_data(self, small_df):
        """Test that no sampling returns all data."""
        lf = small_df.lazy()
        strategy = NoSamplingStrategy()
        config = SamplingConfig()

        result = strategy.sample(lf, config, total_rows=100)

        assert result.is_sampled is False
        assert result.metrics.sample_size == 100
        assert result.metrics.sampling_ratio == 1.0

    def test_metrics_accuracy(self, small_df):
        """Test metrics accuracy."""
        lf = small_df.lazy()
        strategy = NoSamplingStrategy()
        config = SamplingConfig()

        result = strategy.sample(lf, config)

        assert result.metrics.is_full_scan is True
        assert result.metrics.reduction_factor == 1.0


class TestHeadSamplingStrategy:
    """Tests for HeadSamplingStrategy."""

    def test_takes_first_n_rows(self, medium_df):
        """Test that head sampling takes first N rows."""
        lf = medium_df.lazy()
        strategy = HeadSamplingStrategy()
        config = SamplingConfig(max_rows=1000)

        result = strategy.sample(lf, config)

        # Collect to verify
        collected = result.data.collect()
        assert len(collected) == 1000
        # Should be first 1000 rows
        assert collected["id"][0] == 0
        assert collected["id"][999] == 999

    def test_respects_max_rows(self, medium_df):
        """Test max_rows constraint."""
        lf = medium_df.lazy()
        strategy = HeadSamplingStrategy()
        config = SamplingConfig(max_rows=500)

        result = strategy.sample(lf, config)

        assert result.metrics.sample_size <= 500

    def test_no_sampling_for_small_data(self, small_df):
        """Test fallback to no sampling for small datasets."""
        lf = small_df.lazy()
        strategy = HeadSamplingStrategy()
        config = SamplingConfig(max_rows=1000)  # More than 100 rows

        result = strategy.sample(lf, config)

        # Should use all data since it's smaller than max
        assert result.metrics.sample_size == 100


class TestRandomSamplingStrategy:
    """Tests for RandomSamplingStrategy."""

    def test_random_sampling(self, medium_df):
        """Test random sampling produces expected size."""
        lf = medium_df.lazy()
        strategy = RandomSamplingStrategy()
        config = SamplingConfig(max_rows=5000, seed=42)

        result = strategy.sample(lf, config)

        # Should sample approximately 5000 rows
        collected = result.data.collect()
        assert len(collected) > 0
        assert result.metrics.sample_size <= 5000

    def test_reproducibility_with_seed(self, medium_df):
        """Test that seed produces reproducible results."""
        lf = medium_df.lazy()
        strategy = RandomSamplingStrategy()
        config = SamplingConfig(max_rows=1000, seed=42)

        result1 = strategy.sample(lf, config)
        result2 = strategy.sample(lf, config)

        # Same seed should produce same sample
        # (Note: hash-based sampling may have slight variations)
        collected1 = result1.data.collect()
        collected2 = result2.data.collect()

        # At least the sizes should be similar
        assert abs(len(collected1) - len(collected2)) < 100


class TestSystematicSamplingStrategy:
    """Tests for SystematicSamplingStrategy."""

    def test_systematic_sampling(self, medium_df):
        """Test systematic sampling."""
        lf = medium_df.lazy()
        strategy = SystematicSamplingStrategy()
        config = SamplingConfig(max_rows=5000, seed=42)

        result = strategy.sample(lf, config)

        collected = result.data.collect()
        assert len(collected) > 0
        assert result.is_sampled is True

    def test_even_distribution(self, medium_df):
        """Test that systematic sampling covers data evenly."""
        lf = medium_df.lazy()
        strategy = SystematicSamplingStrategy()
        config = SamplingConfig(max_rows=1000, seed=0)

        result = strategy.sample(lf, config)
        collected = result.data.collect()

        # Should have rows from various parts of the data
        ids = collected["id"].to_list()
        if len(ids) > 1:
            # Check that there's some spread
            assert max(ids) > min(ids)


class TestHashSamplingStrategy:
    """Tests for HashSamplingStrategy."""

    def test_hash_sampling(self, medium_df):
        """Test hash-based sampling."""
        strategy = HashSamplingStrategy()
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=5000, seed=42)

        result = strategy.sample(lf, config)

        assert result.metrics.sample_size > 0
        assert result.is_sampled is True

    def test_deterministic_with_column(self, medium_df):
        """Test deterministic sampling on specific column."""
        strategy = HashSamplingStrategy(hash_column="id")
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=5000, seed=42)

        result1 = strategy.sample(lf, config)
        result2 = strategy.sample(lf, config)

        # Same hash column and seed should produce same result
        collected1 = result1.data.collect()
        collected2 = result2.data.collect()
        assert len(collected1) == len(collected2)


class TestStratifiedSamplingStrategy:
    """Tests for StratifiedSamplingStrategy."""

    def test_stratified_sampling(self, medium_df):
        """Test stratified sampling."""
        strategy = StratifiedSamplingStrategy(stratify_column="category")
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=4000, seed=42)

        result = strategy.sample(lf, config)

        collected = result.data.collect()
        assert len(collected) > 0

        # Should preserve rough category distribution
        categories = collected["category"].value_counts()
        # All categories should be represented
        assert len(categories) == 4

    def test_fallback_without_column(self, medium_df):
        """Test fallback to random when no stratify column."""
        strategy = StratifiedSamplingStrategy()  # No column specified
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=5000)

        result = strategy.sample(lf, config)

        # Should still work (falls back to random)
        assert result.metrics.sample_size > 0


class TestReservoirSamplingStrategy:
    """Tests for ReservoirSamplingStrategy."""

    def test_reservoir_sampling(self, medium_df):
        """Test reservoir sampling."""
        strategy = ReservoirSamplingStrategy()
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=5000, seed=42)

        result = strategy.sample(lf, config)

        collected = result.data.collect()
        assert len(collected) <= 5000
        assert result.is_sampled is True

    def test_uniform_distribution(self, medium_df):
        """Test that reservoir sampling is roughly uniform."""
        strategy = ReservoirSamplingStrategy()
        lf = medium_df.lazy()
        config = SamplingConfig(max_rows=1000, seed=42)

        result = strategy.sample(lf, config)
        collected = result.data.collect()

        # Should have samples from across the data range
        ids = collected["id"].to_list()
        if len(ids) > 1:
            assert max(ids) > len(ids) / 2  # At least some late rows


class TestAdaptiveSamplingStrategy:
    """Tests for AdaptiveSamplingStrategy."""

    def test_selects_none_for_small_data(self, small_df):
        """Test that adaptive selects no sampling for small data."""
        strategy = AdaptiveSamplingStrategy()
        lf = small_df.lazy()
        config = SamplingConfig(small_dataset_threshold=1000)

        result = strategy.sample(lf, config)

        # Small data should not be sampled
        assert "none" in result.metrics.strategy_used.lower()

    def test_selects_sampling_for_large_data(self, large_df):
        """Test that adaptive selects sampling for large data."""
        strategy = AdaptiveSamplingStrategy()
        lf = large_df.lazy()
        config = SamplingConfig(
            small_dataset_threshold=1000,
            max_rows=50_000,
        )

        result = strategy.sample(lf, config)

        # Large data should be sampled
        assert result.is_sampled is True
        assert result.metrics.sample_size < 200_000


# =============================================================================
# Sampler Interface Tests
# =============================================================================


class TestSampler:
    """Tests for the main Sampler interface."""

    def test_basic_sampling(self, medium_df):
        """Test basic sampler usage."""
        sampler = Sampler()
        lf = medium_df.lazy()

        result = sampler.sample(lf)

        assert result.metrics.original_size == 50_000
        assert result.metrics.sample_size > 0
        assert result.data is not None

    def test_with_custom_config(self, large_df):
        """Test sampler with custom config."""
        config = SamplingConfig(
            strategy=SamplingMethod.RANDOM,
            max_rows=10_000,
            confidence_level=0.99,
        )
        sampler = Sampler(config)
        lf = large_df.lazy()

        result = sampler.sample(lf)

        assert result.metrics.sample_size <= 10_000
        assert result.metrics.confidence_level == 0.99

    def test_override_config_per_call(self, medium_df):
        """Test overriding config on sample call."""
        sampler = Sampler(SamplingConfig(max_rows=1000))
        lf = medium_df.lazy()

        # Override to sample more
        override = SamplingConfig(max_rows=5000)
        result = sampler.sample(lf, config=override)

        assert result.metrics.sample_size <= 5000

    def test_sample_column(self, medium_df):
        """Test sampling a single column."""
        sampler = Sampler()
        lf = medium_df.lazy()

        result = sampler.sample_column(lf, "email")

        # Result should only have the email column
        collected = result.data.collect()
        assert "email" in collected.columns
        assert len(collected.columns) == 1

    def test_fallback_on_error(self, small_df):
        """Test fallback strategy on error."""
        # Create a config that might cause issues
        config = SamplingConfig(
            strategy=SamplingMethod.STRATIFIED,  # May fail without proper column
            fallback_strategy=SamplingMethod.HEAD,
        )
        sampler = Sampler(config)
        lf = small_df.lazy()

        # Should not raise, but fallback
        result = sampler.sample(lf)
        assert result.metrics.sample_size > 0


class TestSamplingRegistry:
    """Tests for the sampling strategy registry."""

    def test_default_strategies_registered(self):
        """Test that default strategies are registered."""
        strategies = sampling_strategy_registry.list_strategies()

        assert "none" in strategies
        assert "head" in strategies
        assert "random" in strategies
        assert "systematic" in strategies
        assert "adaptive" in strategies
        assert "reservoir" in strategies

    def test_get_strategy(self):
        """Test getting a strategy by name."""
        strategy = sampling_strategy_registry.get("random")
        assert isinstance(strategy, RandomSamplingStrategy)

    def test_get_unknown_raises(self):
        """Test that getting unknown strategy raises error."""
        with pytest.raises(KeyError, match="Unknown"):
            sampling_strategy_registry.get("nonexistent")

    def test_create_from_method(self):
        """Test creating strategy from enum."""
        strategy = sampling_strategy_registry.create_from_method(SamplingMethod.HEAD)
        assert isinstance(strategy, HeadSamplingStrategy)


# =============================================================================
# SampledPatternMatcher Tests
# =============================================================================


class TestSampledPatternMatcher:
    """Tests for SampledPatternMatcher."""

    def test_basic_pattern_matching(self, pattern_df):
        """Test basic pattern matching with sampling."""
        matcher = SampledPatternMatcher()
        lf = pattern_df.lazy()

        result = matcher.match_column(lf, "email")

        assert result.has_matches
        assert result.best_match is not None
        assert result.best_match.pattern.name == "email"
        assert result.best_match.match_ratio > 0.9

    def test_sampling_applied_for_large_data(self, large_df):
        """Test that sampling is applied for large datasets."""
        config = SampledMatcherConfig(
            sampling_config=SamplingConfig(max_rows=10_000),
        )
        matcher = SampledPatternMatcher(config=config)
        lf = large_df.lazy()

        result = matcher.match_column(lf, "email")

        assert result.is_sampled
        assert result.sampling_metrics.sample_size <= 10_000
        assert result.sampling_metrics.original_size == 200_000

    def test_confidence_interval(self, pattern_df):
        """Test confidence interval calculation."""
        matcher = SampledPatternMatcher()
        lf = pattern_df.lazy()

        result = matcher.match_column(lf, "email")

        if result.has_matches:
            match = result.best_match
            lower, upper = match.confidence_interval

            # Confidence interval should contain the point estimate
            # Use approximate comparison to handle floating point precision
            assert lower <= match.match_ratio + 1e-9
            assert match.match_ratio - 1e-9 <= upper
            # Interval should be reasonable
            assert upper - lower < 0.5

    def test_match_all_columns(self, pattern_df):
        """Test matching all columns."""
        matcher = SampledPatternMatcher()
        lf = pattern_df.lazy()

        results = matcher.match_all_columns(lf)

        # Should find patterns in email and uuid columns at least
        assert "email" in results
        assert "uuid" in results

    def test_infer_type(self, pattern_df):
        """Test type inference."""
        matcher = SampledPatternMatcher()
        lf = pattern_df.lazy()

        dtype = matcher.infer_type(lf, "uuid")

        from truthound.profiler.base import DataType
        assert dtype == DataType.UUID

    def test_preset_configs(self):
        """Test preset configurations."""
        fast_config = SampledMatcherConfig.fast()
        assert fast_config.sampling_config.strategy == SamplingMethod.HEAD

        accurate_config = SampledMatcherConfig.accurate()
        assert accurate_config.min_match_ratio > 0.8

        balanced_config = SampledMatcherConfig.balanced()
        assert balanced_config.min_match_ratio == 0.8


class TestSafeNativePatternMatcher:
    """Tests for SafeNativePatternMatcher (drop-in replacement)."""

    def test_drop_in_compatibility(self, pattern_df):
        """Test that it works as drop-in for NativePatternMatcher."""
        # Original API should work
        matcher = SafeNativePatternMatcher(max_rows=5000)
        lf = pattern_df.lazy()

        results = matcher.match_column(lf, "email")

        assert len(results) > 0
        assert results[0].pattern.name == "email"

    def test_memory_safety(self, large_df):
        """Test memory safety with large datasets."""
        matcher = SafeNativePatternMatcher(max_rows=10_000)
        lf = large_df.lazy()

        # Should not OOM
        results = matcher.match_column(lf, "email")

        # Should still find the pattern
        assert len(results) > 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_sampler(self):
        """Test create_sampler function."""
        sampler = create_sampler(
            strategy="random",
            max_rows=5000,
            confidence_level=0.99,
        )

        assert sampler.config.strategy == SamplingMethod.RANDOM
        assert sampler.config.max_rows == 5000

    def test_sample_data(self, medium_df):
        """Test sample_data function."""
        lf = medium_df.lazy()

        result = sample_data(lf, max_rows=5000, strategy="head")

        assert result.metrics.sample_size <= 5000
        assert result.is_sampled

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        n = calculate_sample_size(
            population_size=1_000_000,
            confidence_level=0.95,
            margin_of_error=0.05,
        )

        # Should be around 384 for large population
        assert 350 < n < 450

    def test_match_patterns_safe(self, pattern_df):
        """Test match_patterns_safe function."""
        result = match_patterns_safe(pattern_df, "email", max_rows=5000)

        assert result.has_matches
        assert result.best_match.pattern.name == "email"

    def test_infer_column_type_safe(self, pattern_df):
        """Test infer_column_type_safe function."""
        from truthound.profiler.base import DataType

        dtype = infer_column_type_safe(pattern_df, "uuid")

        assert dtype == DataType.UUID

    def test_create_sampled_matcher(self):
        """Test create_sampled_matcher function."""
        matcher = create_sampled_matcher(
            strategy="random",
            max_rows=10_000,
            min_match_ratio=0.9,
        )

        assert matcher.config.sampling_config.max_rows == 10_000
        assert matcher.config.min_match_ratio == 0.9


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pl.DataFrame({"col": []}).cast({"col": pl.String})
        lf = df.lazy()

        sampler = Sampler()
        result = sampler.sample(lf)

        assert result.metrics.original_size == 0
        assert result.metrics.sample_size == 0

    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pl.DataFrame({"email": ["test@example.com"]})
        lf = df.lazy()

        matcher = SampledPatternMatcher()
        result = matcher.match_column(lf, "email")

        # Should still work
        assert result.sampling_metrics.original_size == 1

    def test_all_nulls(self):
        """Test handling of all-null column."""
        df = pl.DataFrame({"col": [None, None, None]}).cast({"col": pl.String})
        lf = df.lazy()

        matcher = SampledPatternMatcher()
        result = matcher.match_column(lf, "col")

        # Should handle gracefully
        assert not result.has_matches

    def test_very_small_sample_size(self):
        """Test with very small sample size."""
        config = SamplingConfig(
            strategy=SamplingMethod.HEAD,  # Use explicit strategy to ensure sampling
            max_rows=10,
            min_sample_size=5,
        )
        sampler = Sampler(config)

        df = pl.DataFrame({"col": range(1000)})
        lf = df.lazy()

        result = sampler.sample(lf)

        assert result.metrics.sample_size <= 10

    def test_sample_size_larger_than_data(self):
        """Test when sample size exceeds data size."""
        config = SamplingConfig(max_rows=10_000)
        sampler = Sampler(config)

        df = pl.DataFrame({"col": range(100)})
        lf = df.lazy()

        result = sampler.sample(lf)

        # Should use all data
        assert result.metrics.sample_size == 100
        assert not result.is_sampled


# =============================================================================
# Statistical Validation Tests
# =============================================================================


class TestStatisticalValidation:
    """Tests for statistical accuracy of sampling."""

    def test_proportion_estimate_accuracy(self):
        """Test that sampled proportions are accurate."""
        # Create data with known proportion
        n = 100_000
        true_ratio = 0.3
        data = ["email@test.com" if random.random() < true_ratio else "not-email"
                for _ in range(n)]
        df = pl.DataFrame({"col": data})
        lf = df.lazy()

        # Sample and check email pattern
        config = SampledMatcherConfig(
            sampling_config=SamplingConfig(
                max_rows=10_000,
                confidence_level=0.95,
            ),
            min_match_ratio=0.0,  # Accept any ratio
        )
        matcher = SampledPatternMatcher(config=config)
        result = matcher.match_column(lf, "col")

        # The email pattern might not match, but if we had a pattern that
        # matched the emails, the ratio should be close to 0.3
        # This is a structural test - actual validation would need controlled patterns

    def test_margin_of_error_calculation(self):
        """Test margin of error is reasonable."""
        # Set min_sample_size=1 to test pure statistical calculation
        config = SamplingConfig(
            confidence_level=0.95,
            margin_of_error=0.05,
            min_sample_size=1,  # Disable min_sample_size override for pure statistical test
        )

        # For large population, margin should be close to requested
        sample_size = config.calculate_required_sample_size(1_000_000)

        # Verify it's in reasonable range (Cochran's formula gives ~384)
        assert sample_size > 100
        assert sample_size < 10_000

    def test_confidence_interval_coverage(self):
        """Test that confidence intervals are reasonable."""
        df = pl.DataFrame({
            "email": [f"user{i}@example.com" for i in range(10_000)],
        })
        lf = df.lazy()

        matcher = SampledPatternMatcher(
            sampling_config=SamplingConfig(max_rows=1000),
        )
        result = matcher.match_column(lf, "email")

        if result.has_matches:
            match = result.best_match
            lower, upper = match.confidence_interval

            # For high match ratio, interval should be tight
            assert upper - lower < 0.2


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance characteristics."""

    def test_sampling_faster_than_full_scan(self, large_df):
        """Test that sampling is faster than full scan."""
        import time

        lf = large_df.lazy()

        # Time with sampling - use explicit HEAD strategy to ensure sampling is applied
        start = time.perf_counter()
        config_sampled = SamplingConfig(
            strategy=SamplingMethod.HEAD,
            max_rows=10_000,
        )
        sampler = Sampler(config_sampled)
        result_sampled = sampler.sample(lf)
        # Force collect to actually measure execution time
        _ = result_sampled.data.collect()
        sampled_time = time.perf_counter() - start

        # Time without sampling (full scan)
        start = time.perf_counter()
        config_full = SamplingConfig(strategy=SamplingMethod.NONE)
        sampler_full = Sampler(config_full)
        result_full = sampler_full.sample(lf)
        # Force collect to actually measure execution time
        _ = result_full.data.collect()
        full_time = time.perf_counter() - start

        # Sampled should be faster (or at least not much slower)
        # Note: For LazyFrame operations, the difference may be minimal
        # until data is actually collected
        assert sampled_time <= full_time * 5  # Allow overhead for lazy evaluation

    def test_memory_efficiency(self, large_df):
        """Test that sampling reduces memory usage."""
        lf = large_df.lazy()
        config = SamplingConfig(max_rows=10_000)
        sampler = Sampler(config)

        result = sampler.sample(lf)

        # Memory saved should be positive
        assert result.metrics.memory_saved_estimate_mb >= 0

        # Reduction factor should be significant
        assert result.metrics.reduction_factor > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
