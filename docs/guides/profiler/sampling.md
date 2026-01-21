# Sampling Strategies

This document describes sampling strategies for processing large datasets.

## Overview

The sampling system implemented in `src/truthound/profiler/sampling.py` provides 8 different strategies.

## SamplingMethod Enum

```python
class SamplingMethod(str, Enum):
    """Sampling strategies"""

    NONE = "none"           # No sampling (full data)
    RANDOM = "random"       # Random sampling
    SYSTEMATIC = "systematic"  # Systematic sampling (every Nth row)
    STRATIFIED = "stratified"  # Stratified sampling
    RESERVOIR = "reservoir"    # Reservoir sampling (streaming)
    ADAPTIVE = "adaptive"      # Adaptive sampling (automatic selection)
    HEAD = "head"              # First N rows
    HASH = "hash"              # Hash-based (reproducible)
```

## SamplingConfig

```python
@dataclass
class SamplingConfig:
    """Sampling configuration"""

    strategy: SamplingMethod = SamplingMethod.ADAPTIVE
    max_rows: int = 100_000          # Maximum sample size
    confidence_level: float = 0.95   # Confidence level (0.0-1.0)
    random_seed: int | None = None   # Random seed (reproducibility)

    # Stratified sampling options
    stratify_column: str | None = None

    # Hash sampling options
    hash_column: str | None = None
```

## SamplingMetrics

```python
@dataclass
class SamplingMetrics:
    """Sampling result metrics"""

    original_row_count: int      # Original row count
    sampled_row_count: int       # Sampled row count
    sampling_ratio: float        # Sampling ratio
    confidence_level: float      # Confidence level
    margin_of_error: float       # Margin of error
    strategy_used: SamplingMethod
    execution_time_ms: float
```

## Strategy-Specific Usage

### NONE - No Sampling

```python
from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod

config = SamplingConfig(strategy=SamplingMethod.NONE)
sampler = Sampler(config)
result = sampler.sample(lf)
# Returns full data
```

### RANDOM - Random Sampling

```python
config = SamplingConfig(
    strategy=SamplingMethod.RANDOM,
    max_rows=10_000,
    random_seed=42,
)
sampler = Sampler(config)
result = sampler.sample(lf)

print(f"Sampled: {result.metrics.sampled_row_count}")
print(f"Margin of error: {result.metrics.margin_of_error:.2%}")
```

### SYSTEMATIC - Systematic Sampling

Selects every Nth row.

```python
config = SamplingConfig(
    strategy=SamplingMethod.SYSTEMATIC,
    max_rows=10_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# Evenly spaced sampling from sorted data
```

### STRATIFIED - Stratified Sampling

Maintains the distribution of a specific column while sampling.

```python
config = SamplingConfig(
    strategy=SamplingMethod.STRATIFIED,
    max_rows=10_000,
    stratify_column="category",  # Maintain this column's distribution
)
sampler = Sampler(config)
result = sampler.sample(lf)
# Category column proportions remain the same as original
```

### RESERVOIR - Reservoir Sampling

An algorithm suitable for streaming data.

```python
config = SamplingConfig(
    strategy=SamplingMethod.RESERVOIR,
    max_rows=10_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# Equal probability sampling with O(1) memory
```

### ADAPTIVE - Adaptive Sampling

Automatically selects the optimal strategy based on data size.

```python
config = SamplingConfig(
    strategy=SamplingMethod.ADAPTIVE,
    max_rows=50_000,
    confidence_level=0.95,
)
sampler = Sampler(config)
result = sampler.sample(lf)

# Automatic selection logic:
# - Small datasets: NONE
# - Medium datasets: RANDOM
# - Large datasets: RESERVOIR or HASH
```

### HEAD - First N Rows

The fastest sampling method.

```python
config = SamplingConfig(
    strategy=SamplingMethod.HEAD,
    max_rows=1_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# Returns only the first 1,000 rows
```

### HASH - Hash-Based Sampling

Reproducible deterministic sampling.

```python
config = SamplingConfig(
    strategy=SamplingMethod.HASH,
    max_rows=10_000,
    hash_column="id",  # Column for hash basis
)
sampler = Sampler(config)
result = sampler.sample(lf)
# Same ID always included in the same sample
```

## SamplingMethodRegistry

Thread-safe strategy registry.

```python
from truthound.profiler.sampling import SamplingMethodRegistry

# Retrieve strategy
strategy_class = SamplingMethodRegistry.get(SamplingMethod.RANDOM)

# Register custom strategy
@SamplingMethodRegistry.register("my_strategy")
class MyCustomStrategy:
    def sample(self, lf: pl.LazyFrame, config: SamplingConfig) -> SamplingResult:
        # Custom sampling logic
        pass
```

## Statistical Sample Size Calculation

```python
from truthound.profiler.sampling import calculate_sample_size

# 95% confidence level, 5% margin of error
sample_size = calculate_sample_size(
    population_size=1_000_000,
    confidence_level=0.95,
    margin_of_error=0.05,
)
print(f"Required sample size: {sample_size}")  # ~385
```

## Memory-Safe Sampling

The Sampler internally uses `.head(limit).collect()` to prevent OOM:

```python
# Safe implementation (internal)
def _safe_sample(self, lf: pl.LazyFrame) -> pl.DataFrame:
    # Apply limit without calling full collect()
    return lf.head(self.config.max_rows).collect()
```

## CLI Usage

```bash
# Random sampling
th profile data.csv --sample-size 10000 --sample-strategy random

# Hash-based sampling
th profile data.csv --sample-size 10000 --sample-strategy hash --hash-column id

# Adaptive sampling (default)
th profile data.csv --sample-size 50000
```

## Strategy Selection Guide

| Scenario | Recommended Strategy |
|----------|---------------------|
| Small data (<100K) | `NONE` |
| Quick preview | `HEAD` |
| General analysis | `RANDOM` or `ADAPTIVE` |
| Preserve distribution | `STRATIFIED` |
| Streaming data | `RESERVOIR` |
| Reproducibility needed | `HASH` |
| Sorted data | `SYSTEMATIC` |

## Next Steps

- [Pattern Matching](patterns.md) - Detect patterns in sampled data
- [Distributed Processing](distributed.md) - Parallel processing for large data
