"""Tests for enterprise-scale sampling strategies.

This module tests:
    - Block-based sampling for large datasets
    - Multi-stage hierarchical sampling
    - Memory-aware sampling with backpressure
    - Time-budget management
    - Statistical quality guarantees
"""

import time
import pytest
import polars as pl

from truthound.profiler.enterprise_sampling import (
    # Scale classification
    ScaleCategory,
    SamplingQuality,
    # Configuration
    MemoryBudgetConfig,
    EnterpriseScaleConfig,
    # Extended metrics
    BlockSamplingMetrics,
    # Monitoring
    MemoryMonitor,
    TimeBudgetManager,
    # Strategies
    BlockSamplingStrategy,
    MultiStageSamplingStrategy,
    ColumnAwareSamplingStrategy,
    ProgressiveSamplingStrategy,
    # Main interface
    EnterpriseScaleSampler,
    # Convenience functions
    sample_large_dataset,
    estimate_optimal_sample_size,
    classify_dataset_scale,
    # Constants
    LARGE_SCALE_THRESHOLD,
    XLARGE_SCALE_THRESHOLD,
)
from truthound.profiler.sampling import SamplingConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def small_lf() -> pl.LazyFrame:
    """Small dataset (1K rows)."""
    return pl.DataFrame({
        "id": range(1000),
        "value": [f"val_{i}" for i in range(1000)],
        "number": [i * 1.5 for i in range(1000)],
    }).lazy()


@pytest.fixture
def medium_lf() -> pl.LazyFrame:
    """Medium dataset (100K rows)."""
    n = 100_000
    return pl.DataFrame({
        "id": range(n),
        "value": [f"val_{i % 1000}" for i in range(n)],
        "number": [i * 0.5 for i in range(n)],
    }).lazy()


@pytest.fixture
def large_lf() -> pl.LazyFrame:
    """Large dataset (1M rows) - simulated with lazy ops."""
    n = 1_000_000
    return pl.DataFrame({
        "id": range(n),
        "category": [f"cat_{i % 100}" for i in range(n)],
    }).lazy()


# ============================================================================
# Scale Classification Tests
# ============================================================================

class TestScaleClassification:
    """Tests for dataset scale classification."""

    def test_small_scale(self):
        assert classify_dataset_scale(100) == ScaleCategory.SMALL
        assert classify_dataset_scale(999_999) == ScaleCategory.SMALL

    def test_medium_scale(self):
        assert classify_dataset_scale(1_000_000) == ScaleCategory.MEDIUM
        assert classify_dataset_scale(9_999_999) == ScaleCategory.MEDIUM

    def test_large_scale(self):
        assert classify_dataset_scale(10_000_000) == ScaleCategory.LARGE
        assert classify_dataset_scale(99_999_999) == ScaleCategory.LARGE

    def test_xlarge_scale(self):
        assert classify_dataset_scale(100_000_000) == ScaleCategory.XLARGE
        assert classify_dataset_scale(999_999_999) == ScaleCategory.XLARGE

    def test_xxlarge_scale(self):
        assert classify_dataset_scale(1_000_000_000) == ScaleCategory.XXLARGE
        assert classify_dataset_scale(10_000_000_000) == ScaleCategory.XXLARGE


# ============================================================================
# Configuration Tests
# ============================================================================

class TestMemoryBudgetConfig:
    """Tests for MemoryBudgetConfig."""

    def test_default_config(self):
        config = MemoryBudgetConfig()
        assert config.max_memory_mb == 1024
        assert config.reserved_memory_mb == 256
        assert config.available_memory_mb == 768

    def test_for_scale(self):
        small_config = MemoryBudgetConfig.for_scale(ScaleCategory.SMALL)
        assert small_config.max_memory_mb == 256

        large_config = MemoryBudgetConfig.for_scale(ScaleCategory.LARGE)
        assert large_config.max_memory_mb == 1024

        xlarge_config = MemoryBudgetConfig.for_scale(ScaleCategory.XLARGE)
        assert xlarge_config.max_memory_mb == 2048


class TestEnterpriseScaleConfig:
    """Tests for EnterpriseScaleConfig."""

    def test_default_config(self):
        config = EnterpriseScaleConfig()
        assert config.target_rows == 100_000
        assert config.quality == SamplingQuality.STANDARD
        assert config.confidence_level == 0.95

    def test_for_quality(self):
        sketch = EnterpriseScaleConfig.for_quality("sketch")
        assert sketch.quality == SamplingQuality.SKETCH
        assert sketch.target_rows == 10_000

        high = EnterpriseScaleConfig.for_quality("high")
        assert high.quality == SamplingQuality.HIGH
        assert high.confidence_level == 0.99

    def test_get_block_size(self):
        config = EnterpriseScaleConfig()

        # Small dataset
        assert config.get_block_size(100_000) == 100_000

        # Large dataset
        block_size = config.get_block_size(50_000_000)
        assert block_size >= 1_000_000

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            EnterpriseScaleConfig(target_rows=-1)

        with pytest.raises(ValueError):
            EnterpriseScaleConfig(time_budget_seconds=-1)


# ============================================================================
# Monitoring Tests
# ============================================================================

class TestTimeBudgetManager:
    """Tests for TimeBudgetManager."""

    def test_unlimited_budget(self):
        manager = TimeBudgetManager(0)
        assert not manager.is_expired
        assert manager.remaining_seconds == float("inf")

    def test_budget_tracking(self):
        manager = TimeBudgetManager(10.0)
        assert not manager.is_expired
        assert manager.remaining_seconds > 9.0
        assert manager.budget_ratio_used < 0.1

    def test_can_process_block(self):
        manager = TimeBudgetManager(10.0)
        assert manager.can_process_block(1.0)

        # With unlimited budget
        unlimited = TimeBudgetManager(0)
        assert unlimited.can_process_block(100.0)

    def test_checkpoint(self):
        manager = TimeBudgetManager(10.0)
        manager.checkpoint("start")
        time.sleep(0.01)
        manager.checkpoint("end")

        assert len(manager._checkpoints) == 2


class TestMemoryMonitor:
    """Tests for MemoryMonitor."""

    def test_basic_monitoring(self):
        config = MemoryBudgetConfig(max_memory_mb=1024)
        monitor = MemoryMonitor(config)

        # Should work without psutil
        current = monitor.get_current_mb()
        assert current >= 0

    def test_gc_threshold(self):
        config = MemoryBudgetConfig(
            max_memory_mb=1024,
            gc_threshold_mb=768,
        )
        monitor = MemoryMonitor(config)

        # Should not trigger GC for normal usage
        # (depends on actual memory, so just check it doesn't crash)
        should_gc = monitor.should_gc()
        assert isinstance(should_gc, bool)


# ============================================================================
# Block Sampling Strategy Tests
# ============================================================================

class TestBlockSamplingStrategy:
    """Tests for BlockSamplingStrategy."""

    def test_small_dataset_no_sampling(self, small_lf):
        config = EnterpriseScaleConfig(target_rows=10_000)
        strategy = BlockSamplingStrategy(config)

        base_config = SamplingConfig(max_rows=10_000)
        result = strategy.sample(small_lf, base_config)

        # Small dataset should not be sampled
        assert not result.is_sampled or result.metrics.sample_size == 1000

    def test_medium_dataset_sampling(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=10_000)
        strategy = BlockSamplingStrategy(config)

        base_config = SamplingConfig(max_rows=10_000)
        result = strategy.sample(medium_lf, base_config)

        assert result.is_sampled
        assert result.metrics.sample_size <= 10_000
        assert result.metrics.sampling_ratio < 1.0

    def test_block_metrics(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        strategy = BlockSamplingStrategy(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = strategy.sample(medium_lf, base_config)

        assert isinstance(result.metrics, BlockSamplingMetrics)
        assert result.metrics.blocks_processed >= 1

    def test_reproducibility_with_seed(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000, seed=42)
        strategy = BlockSamplingStrategy(config)
        base_config = SamplingConfig(max_rows=5_000, seed=42)

        result1 = strategy.sample(medium_lf, base_config)
        result2 = strategy.sample(medium_lf, base_config)

        # Same seed should produce same results
        assert result1.metrics.sample_size == result2.metrics.sample_size


# ============================================================================
# Multi-Stage Sampling Strategy Tests
# ============================================================================

class TestMultiStageSamplingStrategy:
    """Tests for MultiStageSamplingStrategy."""

    def test_multi_stage_sampling(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        strategy = MultiStageSamplingStrategy(config, num_stages=3)

        base_config = SamplingConfig(max_rows=5_000)
        result = strategy.sample(medium_lf, base_config)

        assert result.is_sampled
        assert "multi_stage" in result.metrics.strategy_used

    def test_stage_count_in_strategy_name(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        strategy = MultiStageSamplingStrategy(config, num_stages=5)

        base_config = SamplingConfig(max_rows=5_000)
        result = strategy.sample(medium_lf, base_config)

        assert "(5)" in result.metrics.strategy_used


# ============================================================================
# Column-Aware Sampling Strategy Tests
# ============================================================================

class TestColumnAwareSamplingStrategy:
    """Tests for ColumnAwareSamplingStrategy."""

    def test_column_aware_sampling(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=10_000)
        strategy = ColumnAwareSamplingStrategy(config)

        base_config = SamplingConfig(max_rows=10_000)
        result = strategy.sample(medium_lf, base_config)

        assert result.is_sampled
        assert "column_aware" in result.metrics.strategy_used

    def test_different_column_types(self):
        # DataFrame with various column types
        df = pl.DataFrame({
            "int_col": range(100_000),
            "str_col": [f"value_{i}" for i in range(100_000)],
            "float_col": [i * 0.5 for i in range(100_000)],
        })
        lf = df.lazy()

        config = EnterpriseScaleConfig(target_rows=5_000)
        strategy = ColumnAwareSamplingStrategy(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = strategy.sample(lf, base_config)

        assert result.is_sampled


# ============================================================================
# Progressive Sampling Strategy Tests
# ============================================================================

class TestProgressiveSamplingStrategy:
    """Tests for ProgressiveSamplingStrategy."""

    def test_progressive_sampling(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        strategy = ProgressiveSamplingStrategy(
            config,
            convergence_threshold=0.01,
            max_stages=3,
        )

        base_config = SamplingConfig(max_rows=5_000)
        result = strategy.sample(medium_lf, base_config)

        assert result.is_sampled
        assert "progressive" in result.metrics.strategy_used


# ============================================================================
# Enterprise Scale Sampler Tests
# ============================================================================

class TestEnterpriseScaleSampler:
    """Tests for EnterpriseScaleSampler."""

    def test_auto_strategy_selection(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(medium_lf)
        assert result.metrics.sample_size <= 5_000

    def test_explicit_strategy(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=5_000)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(medium_lf, strategy="block")
        assert "block" in result.metrics.strategy_used

    def test_list_strategies(self):
        sampler = EnterpriseScaleSampler()
        strategies = sampler.list_strategies()

        assert "block" in strategies
        assert "multi_stage" in strategies
        assert "column_aware" in strategies
        assert "progressive" in strategies

    def test_unknown_strategy_raises(self, medium_lf):
        sampler = EnterpriseScaleSampler()

        with pytest.raises(ValueError):
            sampler.sample(medium_lf, strategy="nonexistent")


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_sample_large_dataset(self, medium_lf):
        result = sample_large_dataset(
            medium_lf,
            target_rows=5_000,
            quality="standard",
        )

        assert result.is_sampled
        assert result.metrics.sample_size <= 5_000

    def test_estimate_optimal_sample_size(self):
        # For 1M population
        # Note: Cochran's formula gives ~384 for 95% confidence, 5% margin
        # But SamplingConfig has min_sample_size=1000 by default
        size = estimate_optimal_sample_size(
            1_000_000,
            confidence_level=0.95,
            margin_of_error=0.05,
        )
        assert 300 < size <= 1000  # Returns 1000 due to min_sample_size default

        # For 100M population
        size = estimate_optimal_sample_size(
            100_000_000,
            confidence_level=0.99,
            margin_of_error=0.01,
        )
        assert size > 10_000  # Higher confidence needs larger sample

    def test_sample_with_time_budget(self, medium_lf):
        result = sample_large_dataset(
            medium_lf,
            target_rows=5_000,
            time_budget_seconds=60.0,
        )

        assert result.metrics.sampling_time_ms < 60_000


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full sampling pipeline."""

    def test_full_pipeline(self, medium_lf):
        # Configure
        config = EnterpriseScaleConfig(
            target_rows=10_000,
            memory_budget=MemoryBudgetConfig(max_memory_mb=512),
            quality=SamplingQuality.STANDARD,
        )

        # Create sampler
        sampler = EnterpriseScaleSampler(config)

        # Sample
        result = sampler.sample(medium_lf)

        # Verify
        assert result.metrics.sample_size <= 10_000
        assert result.metrics.confidence_level == 0.95
        assert result.metrics.sampling_ratio < 1.0

    def test_sampling_preserves_schema(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=1_000)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(medium_lf)

        # Schema should be preserved
        original_schema = medium_lf.collect_schema()
        sampled_schema = result.data.collect_schema()

        assert set(original_schema.names()) == set(sampled_schema.names())

    def test_memory_efficiency(self, large_lf):
        """Test that sampling doesn't load full dataset into memory."""
        config = EnterpriseScaleConfig(target_rows=1_000)
        sampler = EnterpriseScaleSampler(config)

        # This should not OOM even with large dataset
        result = sampler.sample(large_lf)

        # Verify sampling occurred
        assert result.is_sampled
        assert result.metrics.sample_size <= 1_000


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        lf = pl.DataFrame({"col": []}).lazy()
        config = EnterpriseScaleConfig(target_rows=100)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(lf)
        assert result.metrics.original_size == 0

    def test_single_row(self):
        lf = pl.DataFrame({"col": [1]}).lazy()
        config = EnterpriseScaleConfig(target_rows=100)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(lf)
        assert result.metrics.sample_size == 1

    def test_target_larger_than_data(self, small_lf):
        config = EnterpriseScaleConfig(target_rows=1_000_000)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(small_lf)

        # Should return full data
        assert not result.is_sampled or result.metrics.sampling_ratio >= 1.0

    def test_very_small_target(self, medium_lf):
        config = EnterpriseScaleConfig(target_rows=10)
        sampler = EnterpriseScaleSampler(config)

        result = sampler.sample(medium_lf)

        assert result.is_sampled
        # Should still get some samples
        assert result.metrics.sample_size > 0
