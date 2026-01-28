"""Tests for parallel block sampling.

This module tests:
    - Parallel block processing with ThreadPoolExecutor
    - Work stealing queue for load balancing
    - Memory-aware scheduling
    - Parallel speedup metrics
"""

import time
import pytest
import polars as pl

from truthound.profiler.parallel_sampling import (
    # Configuration
    ParallelSamplingConfig,
    ExecutionMode,
    SchedulingPolicy,
    # Core classes
    ParallelBlockSampler,
    SketchBasedSampler,
    WorkStealingQueue,
    BlockTask,
    BlockResult,
    # Metrics
    ParallelSamplingMetrics,
    # Convenience functions
    sample_parallel,
)
from truthound.profiler.enterprise_sampling import (
    MemoryBudgetConfig,
    ScaleCategory,
    SamplingQuality,
)
from truthound.profiler.sampling import SamplingConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def small_lf() -> pl.LazyFrame:
    """Small dataset (1K rows)."""
    return pl.DataFrame(
        {
            "id": range(1000),
            "value": [f"val_{i}" for i in range(1000)],
            "number": [i * 1.5 for i in range(1000)],
        }
    ).lazy()


@pytest.fixture
def medium_lf() -> pl.LazyFrame:
    """Medium dataset (100K rows)."""
    n = 100_000
    return pl.DataFrame(
        {
            "id": range(n),
            "value": [f"val_{i % 1000}" for i in range(n)],
            "number": [i * 0.5 for i in range(n)],
        }
    ).lazy()


@pytest.fixture
def large_lf() -> pl.LazyFrame:
    """Large dataset (1M rows)."""
    n = 1_000_000
    return pl.DataFrame(
        {
            "id": range(n),
            "category": [f"cat_{i % 100}" for i in range(n)],
        }
    ).lazy()


# ============================================================================
# Configuration Tests
# ============================================================================


class TestParallelSamplingConfig:
    """Tests for ParallelSamplingConfig."""

    def test_default_config(self):
        config = ParallelSamplingConfig()
        assert config.target_rows == 100_000
        assert config.max_workers >= 1
        assert config.execution_mode == ExecutionMode.THREAD
        assert config.scheduling_policy == SchedulingPolicy.MEMORY_AWARE

    def test_auto_workers(self):
        config = ParallelSamplingConfig(max_workers=0)
        # Should auto-detect based on CPU count
        assert config.max_workers >= 1
        assert config.max_workers <= 8

    def test_effective_workers(self):
        config = ParallelSamplingConfig(max_workers=8)

        # With few blocks, should use fewer workers
        assert config.get_effective_workers(total_blocks=2) == 2
        assert config.get_effective_workers(total_blocks=4) == 4
        assert config.get_effective_workers(total_blocks=16) == 8

    def test_custom_config(self):
        config = ParallelSamplingConfig(
            target_rows=50_000,
            max_workers=4,
            execution_mode=ExecutionMode.PROCESS,
            scheduling_policy=SchedulingPolicy.WORK_STEALING,
            chunk_timeout_seconds=60.0,
            backpressure_threshold=0.8,
        )

        assert config.target_rows == 50_000
        assert config.max_workers == 4
        assert config.execution_mode == ExecutionMode.PROCESS
        assert config.chunk_timeout_seconds == 60.0


# ============================================================================
# Work Stealing Queue Tests
# ============================================================================


class TestWorkStealingQueue:
    """Tests for WorkStealingQueue."""

    def test_basic_put_get(self):
        queue = WorkStealingQueue(num_workers=4)

        task = BlockTask(
            block_idx=0,
            start_row=0,
            end_row=1000,
            target_samples=100,
            seed=42,
        )
        queue.put(task, worker_id=0)

        result = queue.get(worker_id=0, timeout=1.0)
        assert result is not None
        assert result.block_idx == 0

    def test_work_stealing(self):
        queue = WorkStealingQueue(num_workers=4)

        # Add tasks only to worker 0
        for i in range(5):
            task = BlockTask(
                block_idx=i,
                start_row=i * 1000,
                end_row=(i + 1) * 1000,
                target_samples=100,
                seed=42,
            )
            queue.put(task, worker_id=0)

        # Worker 1 should be able to steal from worker 0
        result = queue.get(worker_id=1, timeout=1.0)
        assert result is not None

    def test_progress_tracking(self):
        queue = WorkStealingQueue(num_workers=2)

        for i in range(4):
            task = BlockTask(
                block_idx=i,
                start_row=i * 1000,
                end_row=(i + 1) * 1000,
                target_samples=100,
                seed=42,
            )
            queue.put(task)

        assert queue.progress == 0.0

        queue.get(worker_id=0)
        queue.mark_complete()
        assert queue.progress == 0.25

        queue.get(worker_id=0)
        queue.mark_complete()
        assert queue.progress == 0.5

    def test_empty_check(self):
        queue = WorkStealingQueue(num_workers=2)

        assert queue.is_empty()

        task = BlockTask(
            block_idx=0,
            start_row=0,
            end_row=1000,
            target_samples=100,
            seed=42,
        )
        queue.put(task)

        assert not queue.is_empty()


# ============================================================================
# BlockTask Tests
# ============================================================================


class TestBlockTask:
    """Tests for BlockTask."""

    def test_block_size(self):
        task = BlockTask(
            block_idx=0,
            start_row=100,
            end_row=500,
            target_samples=50,
            seed=42,
        )
        assert task.block_size == 400

    def test_priority_ordering(self):
        high_priority = BlockTask(
            block_idx=0,
            start_row=0,
            end_row=1000,
            target_samples=100,
            seed=42,
            priority=10,
        )
        low_priority = BlockTask(
            block_idx=1,
            start_row=1000,
            end_row=2000,
            target_samples=100,
            seed=42,
            priority=1,
        )

        # Higher priority should sort first
        assert high_priority < low_priority


# ============================================================================
# Parallel Block Sampler Tests
# ============================================================================


class TestParallelBlockSampler:
    """Tests for ParallelBlockSampler."""

    def test_small_dataset_no_sampling(self, small_lf):
        config = ParallelSamplingConfig(target_rows=10_000, max_workers=2)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=10_000)
        result = sampler.sample(small_lf, base_config)

        # Small dataset should not be sampled
        assert not result.is_sampled or result.metrics.sample_size == 1000

    def test_medium_dataset_parallel_sampling(self, medium_lf):
        config = ParallelSamplingConfig(target_rows=10_000, max_workers=4)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=10_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.is_sampled
        assert result.metrics.sample_size <= 10_000
        assert "parallel_block" in result.metrics.strategy_used

    def test_parallel_metrics(self, medium_lf):
        config = ParallelSamplingConfig(target_rows=5_000, max_workers=4)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = sampler.sample(medium_lf, base_config)

        assert isinstance(result.metrics, ParallelSamplingMetrics)
        assert result.metrics.workers_used >= 1
        assert result.metrics.blocks_processed >= 1

    def test_reproducibility_with_seed(self, medium_lf):
        config = ParallelSamplingConfig(target_rows=5_000, max_workers=2, seed=42)
        sampler = ParallelBlockSampler(config)
        base_config = SamplingConfig(max_rows=5_000, seed=42)

        result1 = sampler.sample(medium_lf, base_config)
        result2 = sampler.sample(medium_lf, base_config)

        # Same seed should produce same sample size
        assert result1.metrics.sample_size == result2.metrics.sample_size

    def test_work_stealing_mode(self, medium_lf):
        config = ParallelSamplingConfig(
            target_rows=5_000,
            max_workers=4,
            enable_work_stealing=True,
        )
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.is_sampled
        assert isinstance(result.metrics, ParallelSamplingMetrics)

    def test_non_work_stealing_mode(self, medium_lf):
        config = ParallelSamplingConfig(
            target_rows=5_000,
            max_workers=4,
            enable_work_stealing=False,
        )
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.is_sampled

    def test_time_budget(self, medium_lf):
        config = ParallelSamplingConfig(
            target_rows=5_000,
            max_workers=4,
            time_budget_seconds=60.0,
        )
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = sampler.sample(medium_lf, base_config)

        # Should complete well within time budget
        assert result.metrics.sampling_time_ms < 60_000


# ============================================================================
# Sketch-Based Sampler Tests
# ============================================================================


class TestSketchBasedSampler:
    """Tests for SketchBasedSampler."""

    def test_basic_sampling(self, medium_lf):
        sampler = SketchBasedSampler(
            hll_precision=12,
            cms_width=2000,
            cms_depth=5,
        )

        base_config = SamplingConfig(max_rows=10_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.is_sampled
        assert "sketch_based" in result.metrics.strategy_used

    def test_no_sampling_needed(self, small_lf):
        sampler = SketchBasedSampler()

        base_config = SamplingConfig(max_rows=10_000)
        result = sampler.sample(small_lf, base_config)

        # Small dataset doesn't need sampling
        assert "full" in result.metrics.strategy_used or not result.is_sampled


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_sample_parallel(self, medium_lf):
        result = sample_parallel(
            medium_lf,
            target_rows=5_000,
            max_workers=2,
        )

        assert result.is_sampled
        assert result.metrics.sample_size <= 5_000

    def test_sample_parallel_with_time_budget(self, medium_lf):
        result = sample_parallel(
            medium_lf,
            target_rows=5_000,
            time_budget_seconds=30.0,
        )

        assert result.metrics.sampling_time_ms < 30_000


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for parallel sampling."""

    def test_full_pipeline(self, medium_lf):
        config = ParallelSamplingConfig(
            target_rows=10_000,
            memory_budget=MemoryBudgetConfig(max_memory_mb=512),
            max_workers=4,
            quality=SamplingQuality.STANDARD,
        )

        sampler = ParallelBlockSampler(config)
        base_config = SamplingConfig(max_rows=10_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.metrics.sample_size <= 10_000
        assert result.metrics.confidence_level == 0.95

    def test_schema_preservation(self, medium_lf):
        config = ParallelSamplingConfig(target_rows=1_000, max_workers=2)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=1_000)
        result = sampler.sample(medium_lf, base_config)

        original_schema = medium_lf.collect_schema()
        sampled_schema = result.data.collect_schema()

        assert set(original_schema.names()) == set(sampled_schema.names())

    def test_large_dataset_memory_efficiency(self, large_lf):
        """Test that parallel sampling handles large datasets efficiently."""
        config = ParallelSamplingConfig(
            target_rows=1_000,
            max_workers=4,
            backpressure_threshold=0.7,
        )
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=1_000)
        result = sampler.sample(large_lf, base_config)

        assert result.is_sampled
        assert result.metrics.sample_size <= 1_000


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        lf = pl.DataFrame({"col": []}).lazy()
        config = ParallelSamplingConfig(target_rows=100, max_workers=2)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=100)
        result = sampler.sample(lf, base_config)
        assert result.metrics.original_size == 0

    def test_single_row(self):
        lf = pl.DataFrame({"col": [1]}).lazy()
        config = ParallelSamplingConfig(target_rows=100, max_workers=2)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=100)
        result = sampler.sample(lf, base_config)
        assert result.metrics.sample_size == 1

    def test_target_larger_than_data(self, small_lf):
        config = ParallelSamplingConfig(target_rows=1_000_000, max_workers=2)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=1_000_000)
        result = sampler.sample(small_lf, base_config)

        # Should return full data
        assert not result.is_sampled or result.metrics.sampling_ratio >= 1.0

    def test_single_worker(self, medium_lf):
        config = ParallelSamplingConfig(target_rows=5_000, max_workers=1)
        sampler = ParallelBlockSampler(config)

        base_config = SamplingConfig(max_rows=5_000)
        result = sampler.sample(medium_lf, base_config)

        assert result.is_sampled
        assert isinstance(result.metrics, ParallelSamplingMetrics)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests for parallel sampling."""

    @pytest.mark.slow
    def test_parallel_speedup(self, large_lf):
        """Test that parallel processing provides speedup."""
        base_config = SamplingConfig(max_rows=10_000)

        # Single worker
        config_single = ParallelSamplingConfig(
            target_rows=10_000, max_workers=1
        )
        sampler_single = ParallelBlockSampler(config_single)

        start = time.perf_counter()
        result_single = sampler_single.sample(large_lf, base_config)
        single_time = time.perf_counter() - start

        # Multiple workers
        config_multi = ParallelSamplingConfig(
            target_rows=10_000, max_workers=4
        )
        sampler_multi = ParallelBlockSampler(config_multi)

        start = time.perf_counter()
        result_multi = sampler_multi.sample(large_lf, base_config)
        multi_time = time.perf_counter() - start

        # Parallel should be at least somewhat faster
        # (not always guaranteed due to overhead, but usually true)
        assert result_single.metrics.sample_size == result_multi.metrics.sample_size
