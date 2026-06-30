# Enterprise Sampling Guide

실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Memory, Footprint, Process을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Parallel, Processing, Multi-threaded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Probabilistic, Data, Structures, HyperLogLog, Count-Min, Sketch, Bloom, Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Statistical, Guarantees, Configurable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Time/Memory, Budget, Awareness, Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 빠른 시작

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Scale Categories

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Row, Count을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default, Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Typical, Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-----------|------------------|------------------|
| 실무 운영 가이드에서 SMALL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼-aware | Daily 리포트 |
| 실무 운영 가이드에서 LARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Block을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Data 웨어하우스 |
| 실무 운영 가이드에서 XLARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multi-stage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Enterprise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 XXLARGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sketches, Multi-stage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Big을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```python
from truthound.profiler.enterprise_sampling import classify_dataset_scale

scale = classify_dataset_scale(500_000_000)  # Returns ScaleCategory.XLARGE
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Sampling Strategies

### 1. Block Sampling

실무 운영 가이드에서 Divides, Ensures을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Best, Datasets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 2. Multi-Stage Sampling

실무 운영 가이드에서 Hierarchical, Ideal을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.enterprise_sampling import MultiStageSamplingStrategy

strategy = MultiStageSamplingStrategy(
    config,
    num_stages=3,  # 3 progressive reduction stages
)
result = strategy.sample(lf, base_config)
print(f"Strategy: {result.metrics.strategy_used}")  # "multi_stage(3)"
```

실무 운영 가이드에서 `(total_rows / target)^(1/stages)`, Reduction을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Best, Datasets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 3. 컬럼-Aware Sampling

실무 운영 가이드에서 Analyzes을(를) 다루는 항목입니다:
- 실무 운영 가이드에서 Strings을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Categoricals을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Complex, List/Struct을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.enterprise_sampling import ColumnAwareSamplingStrategy

strategy = ColumnAwareSamplingStrategy(config)
result = strategy.sample(lf, base_config)
```

실무 운영 가이드에서 Best, Datasets을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 4. Progressive Sampling

실무 운영 가이드에서 Iteratively, Supports을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.enterprise_sampling import ProgressiveSamplingStrategy

strategy = ProgressiveSamplingStrategy(
    config,
    convergence_threshold=0.01,  # Stop when estimates stabilize within 1%
    max_stages=5,
)
result = strategy.sample(lf, base_config)
```

실무 운영 가이드에서 Best, Exploratory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Parallel Block Processing

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Enable을(를) 다루는 항목입니다:

```python
config = ParallelSamplingConfig(
    target_rows=100_000,
    max_workers=8,
    enable_work_stealing=True,
    scheduling_policy=SchedulingPolicy.WORK_STEALING,
)
```

### Memory-Aware Scheduling

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Probabilistic Data Structures

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### HyperLogLog (Cardinality Estimation)

실무 운영 가이드에서 Estimates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Precision, Memory, Error을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Precision을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Memory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Error을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|--------|-------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Count-Min Sketch (Frequency Estimation)

실무 운영 가이드에서 Estimates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Space-efficient을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 레퍼런스

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

### Memory Budget 설정

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증기 with Enterprise Sampling

Use `EnterpriseScaleSamplingMixin` in custom 검증기:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 성능 Guidelines

### Choosing the Right Strategy

| 실무 운영 가이드에서 Dataset, Size을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Workers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Expected, Throughput을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------------|----------|---------|---------------------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼-aware | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼-aware | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Block을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multi-stage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Sketches을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Memory Optimization

1. 실무 운영 가이드에서 Set을(를) 다루는 항목입니다:
   ```python
   config = EnterpriseScaleConfig(
       memory_budget=MemoryBudgetConfig(max_memory_mb=2048),
   )
   ```

2. 실무 운영 가이드에서 Enable을(를) 다루는 항목입니다:
   ```python
   config = ParallelSamplingConfig(
       backpressure_threshold=0.7,  # GC at 70% memory
   )
   ```

3. 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:
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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 권장 방식

### 1. Start with Auto-Selection

실무 운영 가이드에서 Let을(를) 다루는 항목입니다:

```python
sampler = EnterpriseScaleSampler(config)
result = sampler.sample(lf)  # Auto-selects based on data size
```

### 실무 운영 가이드 개요

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

### 실무 운영 가이드 개요

```python
# For exact distinct count on 10B rows:
hll = HyperLogLog(HyperLogLogConfig(precision=16))  # ±0.26% error
# Process in chunks
for chunk in data_source.iter_chunks(1_000_000):
    hll.add_batch(chunk["user_id"])
print(f"Distinct users: ~{hll.estimate():,}")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API 레퍼런스

### Core Classes

- 실무 운영 가이드에서 `EnterpriseScaleSampler`, EnterpriseScaleSampler, Main을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `BlockSamplingStrategy`, BlockSamplingStrategy, Block-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `MultiStageSamplingStrategy`, MultiStageSamplingStrategy, Hierarchical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `ColumnAwareSamplingStrategy`, ColumnAwareSamplingStrategy, Type-aware을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `ProgressiveSamplingStrategy`, ProgressiveSamplingStrategy, Iterative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `ParallelBlockSampler`, ParallelBlockSampler, Multi-threaded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Probabilistic Structures

- 실무 운영 가이드에서 `HyperLogLog`, HyperLogLog, Cardinality을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `CountMinSketch`, CountMinSketch, Frequency을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `BloomFilter`, BloomFilter, Membership을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `SketchFactory`, SketchFactory, Factory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Convenience Functions

- 실무 운영 가이드에서 `sample_large_dataset()`, Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `sample_parallel()`, Parallel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `estimate_optimal_sample_size()`, Calculate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `classify_dataset_scale()`, Classify을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `create_sketch()`, Factory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 Data, Profiling, Guide, Core을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Distributed, Processing, Multi-node을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Statistical, Methods, Sampling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
