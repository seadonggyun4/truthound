# Enterprise Sampling Guide

This guide covers Truthound's enterprise-scale sampling capabilities for datasets ranging from 100 million to billions of rows.

---

## Overview

Truthound provides specialized sampling strategies designed for enterprise-scale data processing:

- **O(1) Memory Footprint**: Process any dataset size with constant memory
- **Parallel Processing**: Multi-threaded block processing with work stealing
- **Probabilistic Data Structures**: HyperLogLog, Count-Min Sketch, Bloom Filter
- **Statistical Guarantees**: Configurable confidence levels and error margins
- **Time/Memory Budget Awareness**: Automatic adaptation to resource constraints

---

## Quick Start

```python
from truthound.profiler.enterprise_sampling import (
    EnterpriseScaleSampler,
    EnterpriseScaleConfig,
    sample_large_dataset,
)

# Quick sampling with defaults
result = sample_large_dataset(lf, target_rows=100_000)
print(f"Sampled {result.metrics.sample_size:,} rows")
print(f"Strategy: {result.metrics.strategy_used}")

# Advanced configuration
config = EnterpriseScaleConfig(
    target_rows=100_000,
    time_budget_seconds=60.0,
    quality=SamplingQuality.HIGH,
)
sampler = EnterpriseScaleSampler(config)
result = sampler.sample(lf)
```

---

## Scale Categories

| Category | Row Count | Default Strategy | Typical Use Case |
|----------|-----------|------------------|------------------|
| SMALL | < 1M | No sampling | Quick analysis |
| MEDIUM | 1M - 10M | Column-aware | Daily reports |
| LARGE | 10M - 100M | Block sampling | Data warehouse |
| XLARGE | 100M - 1B | Multi-stage | Enterprise analytics |
| XXLARGE | > 1B | Sketches + Multi-stage | Big data platforms |

```python
from truthound.profiler.enterprise_sampling import classify_dataset_scale

scale = classify_dataset_scale(500_000_000)  # Returns ScaleCategory.XLARGE
```

---

## Sampling Strategies

### 1. Block Sampling

Divides data into fixed-size blocks and samples from each block proportionally. Ensures even coverage across the dataset.

```python
from truthound.profiler.enterprise_sampling import BlockSamplingStrategy

config = EnterpriseScaleConfig(
    target_rows=100_000,
    block_size=1_000_000,  # 1M rows per block (0 = auto)
)
strategy = BlockSamplingStrategy(config)
result = strategy.sample(lf, base_config)

# Access block-specific metrics
print(f"Blocks processed: {result.metrics.blocks_processed}")
print(f"Time per block: {result.metrics.time_per_block_ms:.1f}ms")
```

**Best for**: Datasets 10M-100M rows, when coverage across data is important

### 2. Multi-Stage Sampling

Hierarchical sampling that progressively reduces data in multiple stages. Ideal for billion-row datasets.

```python
from truthound.profiler.enterprise_sampling import MultiStageSamplingStrategy

strategy = MultiStageSamplingStrategy(
    config,
    num_stages=3,  # 3 progressive reduction stages
)
result = strategy.sample(lf, base_config)
print(f"Strategy: {result.metrics.strategy_used}")  # "multi_stage(3)"
```

**Reduction formula**: Each stage reduces by factor `(total_rows / target)^(1/stages)`

**Best for**: Datasets > 100M rows, when quick estimates are acceptable

### 3. Column-Aware Sampling

Analyzes column types and adjusts sample size based on complexity:
- Strings: 2.0x sample multiplier (high cardinality)
- Categoricals: 0.5x multiplier (low cardinality)
- Complex types (List/Struct): 3.0x multiplier
- Numeric: 1.0x baseline

```python
from truthound.profiler.enterprise_sampling import ColumnAwareSamplingStrategy

strategy = ColumnAwareSamplingStrategy(config)
result = strategy.sample(lf, base_config)
```

**Best for**: Datasets with mixed column types, when accuracy varies by column

### 4. Progressive Sampling

Iteratively increases sample size until convergence. Supports early stopping.

```python
from truthound.profiler.enterprise_sampling import ProgressiveSamplingStrategy

strategy = ProgressiveSamplingStrategy(
    config,
    convergence_threshold=0.01,  # Stop when estimates stabilize within 1%
    max_stages=5,
)
result = strategy.sample(lf, base_config)
```

**Best for**: Exploratory analysis where quick estimates are needed first

---

## Parallel Block Processing

For maximum throughput, use the parallel block sampler with configurable concurrency.

```python
from truthound.profiler.parallel_sampling import (
    ParallelBlockSampler,
    ParallelSamplingConfig,
    sample_parallel,
)

# Quick parallel sampling
result = sample_parallel(lf, target_rows=100_000, max_workers=4)

# Advanced configuration
config = ParallelSamplingConfig(
    target_rows=100_000,
    max_workers=4,  # 0 = auto (CPU count)
    enable_work_stealing=True,  # Dynamic load balancing
    backpressure_threshold=0.75,  # GC trigger threshold
    chunk_timeout_seconds=30.0,  # Per-block timeout
)
sampler = ParallelBlockSampler(config)
result = sampler.sample(lf)

# Access parallel metrics
print(f"Workers used: {result.metrics.workers_used}")
print(f"Parallel speedup: {result.metrics.parallel_speedup:.2f}x")
print(f"Worker utilization: {result.metrics.worker_utilization:.1%}")
```

### Work Stealing

Enable work stealing for better load balancing when block processing times vary:

```python
config = ParallelSamplingConfig(
    target_rows=100_000,
    max_workers=8,
    enable_work_stealing=True,
    scheduling_policy=SchedulingPolicy.WORK_STEALING,
)
```

### Memory-Aware Scheduling

The parallel sampler automatically monitors memory and applies backpressure:

```python
config = ParallelSamplingConfig(
    target_rows=100_000,
    memory_budget=MemoryBudgetConfig(
        max_memory_mb=2048,
        gc_threshold_mb=1536,
        backpressure_enabled=True,
    ),
    backpressure_threshold=0.75,
)
```

---

## Probabilistic Data Structures

For 10B+ row datasets, use probabilistic sketches for O(1) memory aggregations.

### HyperLogLog (Cardinality Estimation)

Estimates distinct counts with configurable precision.

```python
from truthound.profiler.sketches import HyperLogLog, HyperLogLogConfig

# Create with specific precision
hll = HyperLogLog(HyperLogLogConfig(precision=14))  # ~16KB, ±0.41% error

# Or create with target error rate
from truthound.profiler.sketches import create_sketch
hll = create_sketch("hyperloglog", target_error=0.01)  # Auto-select precision

# Add values
for user_id in user_ids:
    hll.add(user_id)

# Or batch add (more efficient)
hll.add_batch(user_ids)

# Get estimate
distinct_count = hll.estimate()
error = hll.standard_error()
print(f"Distinct users: ~{distinct_count:,} (±{error:.2%})")
```

**Precision vs Memory vs Error**:

| Precision | Memory | Error |
|-----------|--------|-------|
| 10 | ~1KB | ±1.04% |
| 12 | ~4KB | ±0.65% |
| 14 | ~16KB | ±0.41% |
| 16 | ~64KB | ±0.26% |
| 18 | ~256KB | ±0.16% |

### Count-Min Sketch (Frequency Estimation)

Estimates element frequencies and finds heavy hitters.

```python
from truthound.profiler.sketches import CountMinSketch, CountMinSketchConfig

# Create with specific dimensions
cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

# Or create with target accuracy
cms = create_sketch(
    "countmin",
    epsilon=0.001,  # Error bound (0.1% of total count)
    delta=0.01,     # 99% confidence
)

# Add values
for item in stream:
    cms.add(item)

# Estimate frequency
freq = cms.estimate_frequency("popular_item")
print(f"Frequency: ~{freq:,}")

# Find heavy hitters (items appearing in >1% of stream)
heavy_hitters = cms.get_heavy_hitters(threshold=0.01)
for item, count in heavy_hitters:
    print(f"  {item}: ~{count:,}")
```

### Bloom Filter (Membership Testing)

Space-efficient set membership with no false negatives.

```python
from truthound.profiler.sketches import BloomFilter, BloomFilterConfig

# Create filter for expected capacity
bf = BloomFilter(BloomFilterConfig(
    capacity=10_000_000,  # Expected items
    error_rate=0.01,      # 1% false positive rate
))

# Add items
for item in items:
    bf.add(item)

# Test membership
if bf.contains(query_item):
    print("Item possibly in set")
else:
    print("Item definitely not in set")  # Guaranteed!

# Check current false positive rate
print(f"FP rate: {bf.false_positive_rate():.2%}")
```

### Distributed Processing with Sketches

All sketches support merging for distributed processing:

```python
# Process partitions in parallel
def process_partition(partition_data):
    hll = HyperLogLog(HyperLogLogConfig(precision=14))
    hll.add_batch(partition_data)
    return hll

# Merge results
with ThreadPoolExecutor(max_workers=8) as executor:
    partition_hlls = list(executor.map(process_partition, partitions))

merged_hll = partition_hlls[0]
for hll in partition_hlls[1:]:
    merged_hll = merged_hll.merge(hll)

print(f"Total distinct: ~{merged_hll.estimate():,}")
```

---

## Configuration Reference

### EnterpriseScaleConfig

```python
@dataclass
class EnterpriseScaleConfig:
    # Sampling targets
    target_rows: int = 100_000          # Target sample size
    quality: SamplingQuality = STANDARD  # Quality level

    # Resource budgets
    memory_budget: MemoryBudgetConfig   # Memory limits
    time_budget_seconds: float = 0.0    # 0 = unlimited

    # Block processing
    block_size: int = 0                  # 0 = auto
    max_parallel_blocks: int = 4

    # Statistical parameters
    confidence_level: float = 0.95
    margin_of_error: float = 0.05

    # Adaptive parameters
    min_sample_ratio: float = 0.001     # At least 0.1%
    max_sample_ratio: float = 0.10      # At most 10%

    # Reproducibility
    seed: int | None = None
    enable_progressive: bool = True
```

### Quality Presets

```python
from truthound.profiler.enterprise_sampling import SamplingQuality

# Available quality levels
SKETCH   # Fast approximation, 10K samples
QUICK    # 90% confidence, 50K samples
STANDARD # 95% confidence, 100K samples (default)
HIGH     # 99% confidence, 500K samples
EXACT    # Full scan, 100% accuracy

# Use preset
config = EnterpriseScaleConfig.for_quality("high")
```

### Memory Budget Configuration

```python
from truthound.profiler.enterprise_sampling import MemoryBudgetConfig

# Default configuration
config = MemoryBudgetConfig()  # 1GB max, 256MB reserved

# Auto-detect from system
config = MemoryBudgetConfig.auto_detect()  # Uses 25% of available RAM

# For specific scale
config = MemoryBudgetConfig.for_scale(ScaleCategory.XLARGE)  # 2GB

# Custom configuration
config = MemoryBudgetConfig(
    max_memory_mb=4096,
    reserved_memory_mb=512,
    gc_threshold_mb=3072,
    backpressure_enabled=True,
)
```

---

## Validators with Enterprise Sampling

Use `EnterpriseScaleSamplingMixin` in custom validators:

```python
from truthound.validators.base import Validator, EnterpriseScaleSamplingMixin

class MyLargeDataValidator(Validator, EnterpriseScaleSamplingMixin):
    """Validator optimized for large datasets."""

    sampling_threshold = 10_000_000   # Enable sampling above 10M rows
    sampling_target_rows = 100_000    # Target sample size
    sampling_quality = "standard"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Automatically samples if dataset is large
        sampled_lf, metrics = self._sample_for_validation(lf)

        # Validate on sampled data
        issues = self._do_validation(sampled_lf)

        # Extrapolate counts if sampled
        if metrics.is_sampled:
            issues = self._extrapolate_issues(issues, metrics)

        return issues
```

---

## Performance Guidelines

### Choosing the Right Strategy

| Dataset Size | Strategy | Workers | Expected Throughput |
|--------------|----------|---------|---------------------|
| < 1M | Column-aware | 1 | 10M+ rows/sec |
| 1M - 10M | Column-aware | 2-4 | 5M+ rows/sec |
| 10M - 100M | Block | 4-8 | 2M+ rows/sec |
| 100M - 1B | Multi-stage | 4-8 | 1M+ rows/sec |
| > 1B | Sketches | 8+ | 500K+ rows/sec |

### Memory Optimization

1. **Set appropriate memory budget**:
   ```python
   config = EnterpriseScaleConfig(
       memory_budget=MemoryBudgetConfig(max_memory_mb=2048),
   )
   ```

2. **Enable backpressure**:
   ```python
   config = ParallelSamplingConfig(
       backpressure_threshold=0.7,  # GC at 70% memory
   )
   ```

3. **Use streaming for sketches**:
   ```python
   hll = HyperLogLog()
   for chunk in lf.iter_slices(100_000):
       hll.add_batch(chunk["user_id"].to_list())
   ```

### Time Budget Management

```python
config = EnterpriseScaleConfig(
    target_rows=100_000,
    time_budget_seconds=30.0,  # Stop after 30 seconds
)
sampler = EnterpriseScaleSampler(config)
result = sampler.sample(lf)

if result.metrics.sampling_time_ms > 25_000:
    print("Warning: Approaching time budget")
```

---

## Best Practices

### 1. Start with Auto-Selection

Let the sampler choose the best strategy:

```python
sampler = EnterpriseScaleSampler(config)
result = sampler.sample(lf)  # Auto-selects based on data size
```

### 2. Use Reproducible Seeds for Production

```python
config = EnterpriseScaleConfig(
    seed=42,  # Reproducible results
)
```

### 3. Monitor Memory in Parallel Processing

```python
result = sampler.sample(lf)
if result.metrics.backpressure_events > 0:
    print(f"Memory pressure detected: {result.metrics.backpressure_events} events")
```

### 4. Validate Sample Representativeness

```python
# Compare sample statistics with known population
sample_df = result.data.collect()
if abs(sample_df["amount"].mean() - known_mean) > margin:
    print("Warning: Sample may not be representative")
```

### 5. Use Sketches for Exact Aggregations

```python
# For exact distinct count on 10B rows:
hll = HyperLogLog(HyperLogLogConfig(precision=16))  # ±0.26% error
# Process in chunks
for chunk in data_source.iter_chunks(1_000_000):
    hll.add_batch(chunk["user_id"])
print(f"Distinct users: ~{hll.estimate():,}")
```

---

## API Reference

### Core Classes

- `EnterpriseScaleSampler`: Main sampling interface with auto-strategy selection
- `BlockSamplingStrategy`: Block-based parallel sampling
- `MultiStageSamplingStrategy`: Hierarchical sampling for billion-row datasets
- `ColumnAwareSamplingStrategy`: Type-aware adaptive sampling
- `ProgressiveSamplingStrategy`: Iterative sampling with early stopping
- `ParallelBlockSampler`: Multi-threaded parallel processing

### Probabilistic Structures

- `HyperLogLog`: Cardinality estimation
- `CountMinSketch`: Frequency estimation
- `BloomFilter`: Membership testing
- `SketchFactory`: Factory for creating sketches

### Convenience Functions

- `sample_large_dataset()`: Quick sampling with quality presets
- `sample_parallel()`: Parallel sampling with configurable workers
- `estimate_optimal_sample_size()`: Calculate statistically optimal sample size
- `classify_dataset_scale()`: Classify dataset by scale category
- `create_sketch()`: Factory function for creating sketches

---

## See Also

- [Data Profiling Guide](index.md) - Core profiling documentation
- [Distributed Processing](distributed.md) - Multi-node processing
- [Statistical Methods](../statistical-methods.md) - Sampling theory background
