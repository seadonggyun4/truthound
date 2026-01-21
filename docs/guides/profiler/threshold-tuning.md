# Automatic Threshold Tuning

This document describes the automatic threshold adjustment system based on data characteristics.

## Overview

The threshold tuning system implemented in `src/truthound/profiler/auto_threshold.py` automatically determines optimal validation thresholds by analyzing data distributions.

## TuningStrategy

```python
class TuningStrategy(str, Enum):
    """Threshold tuning strategies"""

    CONSERVATIVE = "conservative"    # Conservative (strict thresholds, fewer false positives)
    BALANCED = "balanced"            # Balanced (precision and recall balance)
    PERMISSIVE = "permissive"        # Permissive (loose thresholds, fewer false negatives)
    ADAPTIVE = "adaptive"            # Adaptive (learns from data)
    STATISTICAL = "statistical"      # Statistical (confidence interval based)
    DOMAIN_AWARE = "domain_aware"    # Domain-aware (applies domain knowledge)
```

## ColumnThresholds

Per-column threshold configuration.

```python
@dataclass
class ColumnThresholds:
    """Per-column thresholds"""

    column_name: str
    null_threshold: float = 0.0                  # Maximum allowed null ratio
    uniqueness_threshold: float | None = None    # Minimum unique ratio
    min_value: float | None = None               # Minimum value
    max_value: float | None = None               # Maximum value
    min_length: int | None = None                # Minimum length
    max_length: int | None = None                # Maximum length
    pattern_match_threshold: float = 0.8         # Minimum pattern match ratio
    allowed_values: set[Any] | None = None       # Allowed value set
    outlier_threshold: float = 0.01              # Outlier ratio
    confidence: float = 0.5                      # Threshold confidence
    reasoning: list[str] = field(default_factory=list)  # Tuning reasoning
```

## TableThresholds

Table-level threshold collection.

```python
@dataclass
class TableThresholds:
    """Table thresholds"""

    table_name: str
    columns: dict[str, ColumnThresholds] = field(default_factory=dict)
    duplicate_threshold: float = 0.0           # Allowed duplicate row ratio
    row_count_min: int | None = None           # Minimum row count
    row_count_max: int | None = None           # Maximum row count
    global_null_threshold: float = 0.1         # Global null ratio
    strategy_used: TuningStrategy = TuningStrategy.BALANCED
    tuned_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_column(self, name: str) -> ColumnThresholds | None:
        """Retrieve per-column thresholds"""
        return self.columns.get(name)
```

## ThresholdTuner

Automatic threshold tuning class. Strategy can be specified as a string or TuningStrategy enum.

```python
from truthound.profiler.auto_threshold import ThresholdTuner, TuningStrategy

# Specify strategy as string (recommended)
tuner = ThresholdTuner(strategy="balanced")

# Or use Enum
tuner = ThresholdTuner(strategy=TuningStrategy.BALANCED)

# Tune thresholds from profile
thresholds = tuner.tune(profile)

# Check per-column thresholds
for col_name, col_thresholds in thresholds.columns.items():
    print(f"{col_name}:")
    print(f"  null_threshold: {col_thresholds.null_threshold:.2%}")
    print(f"  min_value: {col_thresholds.min_value}")
    print(f"  max_value: {col_thresholds.max_value}")
    print(f"  confidence: {col_thresholds.confidence:.2%}")
```

## Strategy-Specific Behavior

### CONSERVATIVE - Conservative

```python
tuner = ThresholdTuner(strategy=TuningStrategy.CONSERVATIVE)
thresholds = tuner.tune(profile)

# Characteristics:
# - Strict ranges (mean ± 2σ)
# - Low null tolerance
# - High pattern matching requirements
```

### BALANCED - Balanced

```python
tuner = ThresholdTuner(strategy=TuningStrategy.BALANCED)
thresholds = tuner.tune(profile)

# Characteristics:
# - Moderate ranges (mean ± 3σ)
# - Reasonable null tolerance
# - Moderate pattern matching requirements
```

### PERMISSIVE - Permissive

```python
tuner = ThresholdTuner(strategy=TuningStrategy.PERMISSIVE)
thresholds = tuner.tune(profile)

# Characteristics:
# - Wide ranges (mean ± 4σ)
# - High null tolerance
# - Low pattern matching requirements
```

### ADAPTIVE - Adaptive

Automatically selects strategy by analyzing data distribution.

```python
tuner = ThresholdTuner(strategy=TuningStrategy.ADAPTIVE)
thresholds = tuner.tune(profile)

# Internal logic:
# - Analyze data variability
# - Analyze outlier ratio
# - Analyze distribution shape
# → Automatic optimal strategy selection
```

### STATISTICAL - Statistical

Threshold setting based on IQR, percentiles, and Wilson confidence intervals.

```python
# StatisticalStrategy sets parameters in constructor
from truthound.profiler.auto_threshold import StatisticalStrategy, ThresholdTuner

# Create statistical strategy directly
stat_strategy = StatisticalStrategy(
    percentile_low=0.01,    # 1st percentile
    percentile_high=0.99,   # 99th percentile
    iqr_multiplier=1.5,     # IQR multiplier
)

tuner = ThresholdTuner(strategy=stat_strategy)
thresholds = tuner.tune(profile)

# Sets range/null thresholds based on IQR and Wilson CI
```

### DOMAIN_AWARE - Domain-Aware

Automatically applies domain knowledge per DataType. Uses built-in DOMAIN_DEFAULTS.

```python
from truthound.profiler.auto_threshold import ThresholdTuner

# DomainAwareStrategy uses per-DataType defaults
# - EMAIL: null_threshold=0.1, pattern_threshold=0.95, min_length=5, max_length=254
# - PHONE: null_threshold=0.2, pattern_threshold=0.9, min_length=7, max_length=20
# - UUID: null_threshold=0.0, uniqueness_threshold=1.0, min_length=36, max_length=36
# - PERCENTAGE: min_value=0.0, max_value=100.0
# Also supports DataType.CURRENCY, BOOLEAN, KOREAN_PHONE, KOREAN_RRN, etc.

tuner = ThresholdTuner(strategy="domain_aware")
thresholds = tuner.tune(profile)
```

## IQR-Based Analysis

Outlier detection using Interquartile Range (IQR).

```python
def _compute_iqr_bounds(self, profile: ColumnProfile) -> tuple[float, float]:
    """Calculate IQR-based bounds"""
    q1 = profile.quantiles.get(0.25, profile.min_value)
    q3 = profile.quantiles.get(0.75, profile.max_value)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper
```

## Percentile-Based Analysis

```python
def _compute_percentile_bounds(
    self,
    profile: ColumnProfile,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> tuple[float, float]:
    """Calculate percentile-based bounds"""
    lower = profile.quantiles.get(lower_pct, profile.min_value)
    upper = profile.quantiles.get(upper_pct, profile.max_value)
    return lower, upper
```

## Quick Tuning

```python
from truthound.profiler.auto_threshold import tune_thresholds
from truthound.profiler.base import Strictness

# Convenience function - specify strategy and strictness
thresholds = tune_thresholds(
    profile,
    strategy="adaptive",
    strictness=Strictness.MEDIUM,  # LOOSE, MEDIUM, STRICT
)

# Table-level thresholds
print(f"Duplicate threshold: {thresholds.duplicate_threshold:.2%}")
print(f"Global null threshold: {thresholds.global_null_threshold:.2%}")

# Per-column thresholds
for col_name, col_thresh in thresholds.columns.items():
    print(f"{col_name}: null <= {col_thresh.null_threshold:.1%}")
```

## A/B Testing

```python
from truthound.profiler.auto_threshold import ThresholdTester, ThresholdTuner
import polars as pl

tester = ThresholdTester()

# Compare two threshold configurations
tuner_a = ThresholdTuner(strategy="conservative")
tuner_b = ThresholdTuner(strategy="permissive")

threshold_a = tuner_a.tune(profile)
threshold_b = tuner_b.tune(profile)

# Test with DataFrame
df = pl.read_csv("test_data.csv")
result = tester.compare(
    data=df,
    threshold_a=threshold_a,
    threshold_b=threshold_b,
)

print(f"Recommendation: {result.recommendation}")
print(f"Violations A: {result.violations_a}")
print(f"Violations B: {result.violations_b}")
```

## Threshold Export

```python
import json

# Save as JSON
with open("thresholds.json", "w") as f:
    json.dump(thresholds.to_dict(), f, indent=2)

# Save as YAML
import yaml
with open("thresholds.yaml", "w") as f:
    yaml.dump(thresholds.to_dict(), f)
```

## CLI Usage

```bash
# Automatic threshold tuning
th tune-thresholds profile.json -o thresholds.yaml

# Specify strategy
th tune-thresholds profile.json -o thresholds.yaml --strategy statistical

# Specify confidence level
th tune-thresholds profile.json -o thresholds.yaml --strategy statistical --confidence 0.99

# Generate rules with tuned thresholds
th generate-suite profile.json -o rules.yaml --thresholds thresholds.yaml
```

## Integration Example

```python
from truthound.profiler import TableProfiler, generate_suite
from truthound.profiler.auto_threshold import ThresholdTuner, TuningStrategy

# Profiling
profiler = TableProfiler()
profile = profiler.profile_file("data.csv")

# Threshold tuning
tuner = ThresholdTuner(strategy=TuningStrategy.ADAPTIVE)
thresholds = tuner.tune(profile)

# Generate rules with tuned thresholds
suite = generate_suite(
    profile,
    thresholds=thresholds,
)

# Save
save_suite(suite, "rules.yaml", format="yaml")
```

## Next Steps

- [Quality Scoring](quality-scoring.md) - Evaluate tuned threshold quality
- [Rule Generation](rule-generation.md) - Threshold-based rule generation
