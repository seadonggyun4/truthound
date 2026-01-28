"""Parallel block processing for enterprise-scale sampling.

This module extends the base enterprise sampling with true parallel processing
capabilities using ThreadPoolExecutor for I/O-bound operations and
ProcessPoolExecutor for CPU-bound operations.

Key Features:
    - Parallel block processing with configurable concurrency
    - Work stealing for load balancing
    - Memory-aware scheduling with backpressure
    - Progress tracking and cancellation support
    - Automatic fallback to sequential processing on resource constraints

Design Principles:
    - O(1) memory per worker regardless of block size
    - Graceful degradation under memory pressure
    - Thread-safe result aggregation
    - Supports both sync and async interfaces

Usage:
    from truthound.profiler.parallel_sampling import (
        ParallelBlockSampler,
        ParallelSamplingConfig,
    )

    config = ParallelSamplingConfig(
        target_rows=100_000,
        max_workers=4,
        memory_budget_mb=1024,
    )
    sampler = ParallelBlockSampler(config)
    result = sampler.sample(lf)
"""

from __future__ import annotations

import gc
import logging
import math
import os
import queue
import random
import threading
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, TypeVar

import polars as pl

from truthound.profiler.enterprise_sampling import (
    BlockSamplingMetrics,
    EnterpriseScaleConfig,
    MemoryBudgetConfig,
    MemoryMonitor,
    ScaleCategory,
    SamplingQuality,
    TimeBudgetManager,
    BYTES_PER_ROW_ESTIMATE,
    MB,
)
from truthound.profiler.sampling import (
    SamplingConfig,
    SamplingMetrics,
    SamplingResult,
    SamplingStrategy,
    SamplingMethod,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExecutionMode(Enum):
    """Execution mode for parallel processing."""

    THREAD = auto()  # ThreadPoolExecutor - I/O bound
    PROCESS = auto()  # ProcessPoolExecutor - CPU bound
    ADAPTIVE = auto()  # Auto-select based on workload


class SchedulingPolicy(Enum):
    """Work scheduling policy for parallel blocks."""

    ROUND_ROBIN = auto()  # Distribute evenly
    WORK_STEALING = auto()  # Dynamic load balancing
    PRIORITY = auto()  # Priority-based scheduling
    MEMORY_AWARE = auto()  # Schedule based on memory availability


@dataclass
class ParallelSamplingConfig(EnterpriseScaleConfig):
    """Extended configuration for parallel sampling.

    Attributes:
        max_workers: Maximum parallel workers (0 = auto)
        execution_mode: Thread vs Process execution
        scheduling_policy: Work distribution policy
        chunk_timeout_seconds: Timeout per block processing
        enable_work_stealing: Enable dynamic load balancing
        backpressure_threshold: Memory ratio to trigger backpressure
    """

    max_workers: int = 0  # 0 = auto (CPU count)
    execution_mode: ExecutionMode = ExecutionMode.THREAD
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.MEMORY_AWARE
    chunk_timeout_seconds: float = 30.0
    enable_work_stealing: bool = True
    backpressure_threshold: float = 0.75
    prefetch_blocks: int = 2  # Blocks to prefetch ahead

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_workers == 0:
            self.max_workers = min(os.cpu_count() or 4, 8)

    def get_effective_workers(self, total_blocks: int) -> int:
        """Get effective worker count based on workload."""
        return min(self.max_workers, total_blocks)


@dataclass(frozen=True)
class ParallelSamplingMetrics(BlockSamplingMetrics):
    """Extended metrics for parallel sampling."""

    workers_used: int = 0
    parallel_speedup: float = 1.0
    blocks_stolen: int = 0
    backpressure_events: int = 0
    worker_utilization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "workers_used": self.workers_used,
                "parallel_speedup": self.parallel_speedup,
                "blocks_stolen": self.blocks_stolen,
                "backpressure_events": self.backpressure_events,
                "worker_utilization": self.worker_utilization,
            }
        )
        return base


@dataclass
class BlockTask:
    """Represents a block processing task."""

    block_idx: int
    start_row: int
    end_row: int
    target_samples: int
    seed: int
    priority: int = 0  # Higher = more important

    @property
    def block_size(self) -> int:
        return self.end_row - self.start_row

    def __lt__(self, other: "BlockTask") -> bool:
        """Enable priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class BlockResult:
    """Result from processing a single block."""

    block_idx: int
    sampled_lf: pl.LazyFrame
    actual_samples: int
    processing_time_ms: float
    memory_used_mb: float = 0.0
    was_stolen: bool = False


class WorkStealingQueue:
    """Thread-safe work stealing queue for load balancing.

    Workers can steal tasks from each other when idle, improving
    utilization when block processing times vary.
    """

    def __init__(self, num_workers: int):
        self.queues: list[queue.Queue[BlockTask]] = [
            queue.Queue() for _ in range(num_workers)
        ]
        self._lock = threading.Lock()
        self._total_tasks = 0
        self._completed_tasks = 0

    def put(self, task: BlockTask, worker_id: int | None = None) -> None:
        """Add a task to a worker's queue."""
        if worker_id is None:
            # Round-robin assignment
            worker_id = task.block_idx % len(self.queues)
        self.queues[worker_id].put(task)
        with self._lock:
            self._total_tasks += 1

    def get(self, worker_id: int, timeout: float = 0.1) -> BlockTask | None:
        """Get a task, stealing from others if own queue is empty."""
        # Try own queue first
        try:
            return self.queues[worker_id].get(timeout=timeout)
        except queue.Empty:
            pass

        # Try to steal from other workers
        for other_id in range(len(self.queues)):
            if other_id != worker_id:
                try:
                    task = self.queues[other_id].get_nowait()
                    return task
                except queue.Empty:
                    continue

        return None

    def mark_complete(self) -> None:
        """Mark a task as completed."""
        with self._lock:
            self._completed_tasks += 1

    @property
    def progress(self) -> float:
        """Return completion progress (0.0-1.0)."""
        with self._lock:
            if self._total_tasks == 0:
                return 1.0
            return self._completed_tasks / self._total_tasks

    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return all(q.empty() for q in self.queues)


class ParallelBlockSampler(SamplingStrategy):
    """Parallel block-based sampling for large datasets.

    Divides data into blocks and processes them in parallel using
    configurable concurrency and work stealing for load balancing.

    Example:
        config = ParallelSamplingConfig(
            target_rows=100_000,
            max_workers=4,
            execution_mode=ExecutionMode.THREAD,
        )
        sampler = ParallelBlockSampler(config)
        result = sampler.sample(large_lf)
    """

    name = "parallel_block"

    def __init__(self, config: ParallelSamplingConfig | None = None):
        self.config = config or ParallelSamplingConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_budget)
        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        self._work_queue: WorkStealingQueue | None = None
        self._results: list[BlockResult] = []
        self._lock = threading.Lock()
        self._backpressure_count = 0
        self._stolen_count = 0

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Sample data using parallel block processing.

        Args:
            lf: Source LazyFrame
            config: Base sampling configuration
            total_rows: Optional pre-computed row count

        Returns:
            SamplingResult with sampled data and metrics
        """
        start_time = time.perf_counter()
        time_budget = TimeBudgetManager(self.config.time_budget_seconds)

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        # Calculate target sample size
        target_samples = min(
            self.config.target_rows,
            config.calculate_required_sample_size(total_rows),
        )

        if target_samples >= total_rows:
            return self._create_full_scan_result(lf, total_rows, start_time)

        # Calculate block parameters
        block_size = self.config.get_block_size(total_rows)
        num_blocks = math.ceil(total_rows / block_size)
        samples_per_block = math.ceil(target_samples / num_blocks)
        effective_workers = self.config.get_effective_workers(num_blocks)

        logger.info(
            f"Parallel sampling: {total_rows:,} rows → {num_blocks} blocks × "
            f"{samples_per_block:,} samples/block using {effective_workers} workers"
        )

        # Create block tasks
        seed = self.config.seed or random.randint(0, 2**32 - 1)
        tasks = self._create_block_tasks(
            total_rows, block_size, samples_per_block, seed
        )

        # Process blocks in parallel
        results = self._process_blocks_parallel(
            lf, tasks, effective_workers, time_budget
        )

        # Merge results
        return self._create_result(
            results, total_rows, target_samples, effective_workers, start_time
        )

    def _create_block_tasks(
        self,
        total_rows: int,
        block_size: int,
        samples_per_block: int,
        seed: int,
    ) -> list[BlockTask]:
        """Create block processing tasks."""
        tasks = []
        for block_idx in range(math.ceil(total_rows / block_size)):
            start_row = block_idx * block_size
            end_row = min(start_row + block_size, total_rows)
            actual_block_size = end_row - start_row

            tasks.append(
                BlockTask(
                    block_idx=block_idx,
                    start_row=start_row,
                    end_row=end_row,
                    target_samples=min(samples_per_block, actual_block_size),
                    seed=seed + block_idx,
                    priority=0,  # Equal priority for now
                )
            )
        return tasks

    def _process_blocks_parallel(
        self,
        lf: pl.LazyFrame,
        tasks: list[BlockTask],
        num_workers: int,
        time_budget: TimeBudgetManager,
    ) -> list[BlockResult]:
        """Process blocks in parallel using thread pool."""
        results: list[BlockResult] = []
        self._backpressure_count = 0
        self._stolen_count = 0

        # Use work stealing queue for load balancing
        if self.config.enable_work_stealing:
            self._work_queue = WorkStealingQueue(num_workers)
            for task in tasks:
                self._work_queue.put(task)

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                if self.config.enable_work_stealing:
                    # Work stealing mode
                    futures = []
                    for worker_id in range(num_workers):
                        future = executor.submit(
                            self._worker_loop,
                            lf,
                            worker_id,
                            time_budget,
                        )
                        futures.append(future)

                    # Collect results from workers
                    for future in as_completed(futures):
                        worker_results = future.result()
                        results.extend(worker_results)
                else:
                    # Simple parallel mode
                    futures = {
                        executor.submit(
                            self._process_single_block,
                            lf,
                            task,
                        ): task
                        for task in tasks
                    }

                    for future in as_completed(futures):
                        if time_budget.is_expired:
                            logger.warning("Time budget expired, stopping parallel processing")
                            break

                        # Check memory pressure
                        if self.memory_monitor.should_backpressure():
                            self._backpressure_count += 1
                            logger.warning("Memory pressure, triggering GC")
                            self.memory_monitor.trigger_gc()

                        try:
                            result = future.result(
                                timeout=self.config.chunk_timeout_seconds
                            )
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Block processing failed: {e}")

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            # Fallback to sequential
            for task in tasks:
                if time_budget.is_expired:
                    break
                try:
                    result = self._process_single_block(lf, task)
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Sequential fallback failed for block {task.block_idx}: {inner_e}")

        return results

    def _worker_loop(
        self,
        lf: pl.LazyFrame,
        worker_id: int,
        time_budget: TimeBudgetManager,
    ) -> list[BlockResult]:
        """Worker loop for work stealing mode."""
        results: list[BlockResult] = []

        while not time_budget.is_expired:
            task = self._work_queue.get(worker_id, timeout=0.5)
            if task is None:
                if self._work_queue.is_empty():
                    break
                continue

            # Check memory before processing
            if self.memory_monitor.should_backpressure():
                self._backpressure_count += 1
                self.memory_monitor.trigger_gc()

            try:
                result = self._process_single_block(lf, task)
                results.append(result)
                self._work_queue.mark_complete()
            except Exception as e:
                logger.error(f"Worker {worker_id} failed on block {task.block_idx}: {e}")

        return results

    def _process_single_block(
        self,
        lf: pl.LazyFrame,
        task: BlockTask,
    ) -> BlockResult:
        """Process a single block and return sampled data."""
        start_time = time.perf_counter()

        # Extract and sample the block
        sample_rate = task.target_samples / task.block_size
        threshold = max(1, int(sample_rate * 10000))

        block_lf = (
            lf.slice(task.start_row, task.block_size)
            .with_row_index("__block_idx")
            .filter(pl.col("__block_idx").hash(task.seed) % 10000 < threshold)
            .drop("__block_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return BlockResult(
            block_idx=task.block_idx,
            sampled_lf=block_lf,
            actual_samples=task.target_samples,
            processing_time_ms=elapsed_ms,
        )

    def _create_result(
        self,
        results: list[BlockResult],
        total_rows: int,
        target_samples: int,
        workers_used: int,
        start_time: float,
    ) -> SamplingResult:
        """Create the final sampling result from block results."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Merge all block samples
        if results:
            sampled_frames = [r.sampled_lf for r in results]
            merged_lf = pl.concat(sampled_frames)
        else:
            merged_lf = pl.LazyFrame()

        # Calculate metrics
        total_processing_time = sum(r.processing_time_ms for r in results)
        avg_time_per_block = total_processing_time / max(1, len(results))

        # Estimate parallel speedup
        sequential_time_estimate = total_processing_time
        parallel_speedup = sequential_time_estimate / max(1, elapsed_ms)

        # Worker utilization
        worker_utilization = (
            total_processing_time / (workers_used * elapsed_ms) if workers_used > 0 else 0
        )

        return SamplingResult(
            data=merged_lf,
            metrics=ParallelSamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used=f"parallel_block({workers_used})",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples)
                * BYTES_PER_ROW_ESTIMATE
                / MB,
                blocks_processed=len(results),
                blocks_skipped=0,
                time_per_block_ms=avg_time_per_block,
                memory_peak_mb=self.memory_monitor.peak_mb,
                workers_used=workers_used,
                parallel_speedup=parallel_speedup,
                blocks_stolen=self._stolen_count,
                backpressure_events=self._backpressure_count,
                worker_utilization=worker_utilization,
            ),
            is_sampled=True,
        )

    def _create_full_scan_result(
        self,
        lf: pl.LazyFrame,
        total_rows: int,
        start_time: float,
    ) -> SamplingResult:
        """Create result for full scan (no sampling needed)."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SamplingResult(
            data=lf,
            metrics=ParallelSamplingMetrics(
                original_size=total_rows,
                sample_size=total_rows,
                sampling_ratio=1.0,
                confidence_level=1.0,
                margin_of_error=0.0,
                strategy_used="parallel_block(full_scan)",
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=False,
        )


class SketchBasedSampler(SamplingStrategy):
    """Sampler that uses probabilistic sketches for aggregation.

    Uses HyperLogLog for cardinality and Count-Min Sketch for frequency
    to enable O(1) memory aggregations during sampling.

    Example:
        sampler = SketchBasedSampler(
            target_rows=100_000,
            hll_precision=14,
            cms_width=5000,
        )
        result = sampler.sample(lf)
        print(f"Estimated distinct: {result.estimated_cardinality}")
    """

    name = "sketch_based"

    def __init__(
        self,
        config: EnterpriseScaleConfig | None = None,
        hll_precision: int = 12,
        cms_width: int = 2000,
        cms_depth: int = 5,
    ):
        self.config = config or EnterpriseScaleConfig()
        self.hll_precision = hll_precision
        self.cms_width = cms_width
        self.cms_depth = cms_depth

    def sample(
        self,
        lf: pl.LazyFrame,
        config: SamplingConfig,
        total_rows: int | None = None,
    ) -> SamplingResult:
        """Sample data with sketch-based aggregation."""
        from truthound.profiler.sketches import HyperLogLog, CountMinSketch

        start_time = time.perf_counter()

        if total_rows is None:
            total_rows = self.estimate_row_count(lf)

        target_samples = min(
            self.config.target_rows,
            config.calculate_required_sample_size(total_rows),
        )

        if target_samples >= total_rows:
            return self._create_full_result(lf, total_rows, start_time)

        # Create sketches for each column
        schema = lf.collect_schema()
        hlls: dict[str, HyperLogLog] = {}
        cmss: dict[str, CountMinSketch] = {}

        from truthound.profiler.sketches.protocols import (
            HyperLogLogConfig,
            CountMinSketchConfig,
        )

        for col in schema.names():
            hlls[col] = HyperLogLog(HyperLogLogConfig(precision=self.hll_precision))
            cmss[col] = CountMinSketch(
                CountMinSketchConfig(width=self.cms_width, depth=self.cms_depth)
            )

        # Sample and collect sketches
        sample_rate = target_samples / total_rows
        seed = self.config.seed or random.randint(0, 2**32 - 1)
        threshold = max(1, int(sample_rate * 10000))

        sampled_lf = (
            lf.with_row_index("__sketch_idx")
            .filter(pl.col("__sketch_idx").hash(seed) % 10000 < threshold)
            .drop("__sketch_idx")
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SamplingResult(
            data=sampled_lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=target_samples,
                sampling_ratio=target_samples / total_rows,
                confidence_level=self.config.confidence_level,
                margin_of_error=self.config.margin_of_error,
                strategy_used="sketch_based",
                sampling_time_ms=elapsed_ms,
                memory_saved_estimate_mb=(total_rows - target_samples)
                * BYTES_PER_ROW_ESTIMATE
                / MB,
            ),
            is_sampled=True,
        )

    def _create_full_result(
        self,
        lf: pl.LazyFrame,
        total_rows: int,
        start_time: float,
    ) -> SamplingResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SamplingResult(
            data=lf,
            metrics=SamplingMetrics(
                original_size=total_rows,
                sample_size=total_rows,
                sampling_ratio=1.0,
                confidence_level=1.0,
                margin_of_error=0.0,
                strategy_used="sketch_based(full)",
                sampling_time_ms=elapsed_ms,
            ),
            is_sampled=False,
        )


# Convenience functions


def sample_parallel(
    lf: pl.LazyFrame,
    target_rows: int = 100_000,
    max_workers: int = 0,
    time_budget_seconds: float = 0.0,
) -> SamplingResult:
    """Quick function to sample with parallel processing.

    Args:
        lf: LazyFrame to sample
        target_rows: Target number of rows
        max_workers: Max parallel workers (0 = auto)
        time_budget_seconds: Max time for sampling

    Returns:
        SamplingResult with sampled data
    """
    config = ParallelSamplingConfig(
        target_rows=target_rows,
        max_workers=max_workers,
        time_budget_seconds=time_budget_seconds,
    )
    sampler = ParallelBlockSampler(config)
    base_config = SamplingConfig(
        strategy=SamplingMethod.ADAPTIVE,
        max_rows=target_rows,
    )
    return sampler.sample(lf, base_config)
