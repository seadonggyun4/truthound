# Quality Scoring

This document describes the quality evaluation system for generated validation rules.

## Overview

The quality scoring system implemented in `src/truthound/profiler/quality.py` evaluates the accuracy and usefulness of rules.

## QualityLevel

```python
class QualityLevel(str, Enum):
    """Quality levels"""

    EXCELLENT = "excellent"      # 0.9 - 1.0
    GOOD = "good"                # 0.7 - 0.9
    ACCEPTABLE = "acceptable"    # 0.5 - 0.7
    POOR = "poor"                # 0.3 - 0.5
    UNACCEPTABLE = "unacceptable"  # 0.0 - 0.3
```

## ConfusionMatrix

Confusion matrix for rule evaluation.

```python
@dataclass
class ConfusionMatrix:
    """Confusion matrix"""

    true_positive: int = 0   # Correct violation detection
    true_negative: int = 0   # Correct pass
    false_positive: int = 0  # Incorrect violation detection (false alarm)
    false_negative: int = 0  # Missed violation (miss)

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        if self.true_positive + self.false_positive == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)"""
        if self.true_positive + self.false_negative == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def f1_score(self) -> float:
        """F1 score: 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / Total"""
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        if total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / total
```

## RuleQualityScorer

The main class for evaluating rule quality.

```python
from truthound.profiler.quality import RuleQualityScorer

scorer = RuleQualityScorer()

# Calculate rule quality score
score = scorer.score(rule, test_data)

print(f"Quality Level: {score.quality_level}")
print(f"Precision: {score.metrics.precision:.2%}")
print(f"Recall: {score.metrics.recall:.2%}")
print(f"F1 Score: {score.metrics.f1_score:.2%}")
print(f"Accuracy: {score.metrics.accuracy:.2%}")
```

## QualityScore

```python
@dataclass
class QualityScore:
    """Quality score result"""

    rule_name: str
    quality_level: QualityLevel
    overall_score: float          # 0.0 - 1.0
    metrics: ConfusionMatrix
    test_sample_size: int
    evaluation_time_ms: float

    # Detailed scores
    precision_score: float
    recall_score: float
    f1_score: float
```

## Basic Usage

```python
from truthound.profiler.quality import RuleQualityScorer, estimate_quality

# Quick quality estimation
quality = estimate_quality(rule, lf)
print(f"Estimated quality: {quality.level}")

# Detailed quality evaluation
scorer = RuleQualityScorer(
    sample_size=10_000,
    cross_validation_folds=5,
)
score = scorer.score(rule, test_data)
```

## Quality Estimation Strategies

### SamplingQualityEstimator

Sampling-based rapid estimation.

```python
from truthound.profiler.quality import SamplingQualityEstimator

estimator = SamplingQualityEstimator(sample_size=1_000)
result = estimator.estimate(rule, lf)

print(f"Level: {result.level}")
print(f"Confidence: {result.confidence:.2%}")
```

### HeuristicQualityEstimator

Heuristic-based fastest estimation.

```python
from truthound.profiler.quality import HeuristicQualityEstimator

estimator = HeuristicQualityEstimator()
result = estimator.estimate(rule, lf)

# Estimation based on rule characteristics (no actual execution)
print(f"Level: {result.level}")
```

### CrossValidationEstimator

Cross-validation-based accurate estimation.

```python
from truthound.profiler.quality import CrossValidationEstimator

estimator = CrossValidationEstimator(n_folds=5)
result = estimator.estimate(rule, lf)

print(f"Level: {result.level}")
print(f"Mean F1: {result.mean_f1:.2%}")
print(f"Std F1: {result.std_f1:.2%}")
```

## Rule Comparison

```python
from truthound.profiler.quality import compare_rules

# Compare multiple rules
comparison = compare_rules(
    rules=[rule1, rule2, rule3],
    test_data=lf,
)

# Best quality rule
best = comparison.best_rule
print(f"Best rule: {best.name}")
print(f"F1 Score: {best.score.f1_score:.2%}")

# Sorted by rank
for rank, (rule, score) in enumerate(comparison.ranked_rules, 1):
    print(f"{rank}. {rule.name}: {score.f1_score:.2%}")
```

## Trend Analysis

Track quality changes over time.

```python
from truthound.profiler.quality import QualityTrendAnalyzer

analyzer = QualityTrendAnalyzer()

# Add history
for score in historical_scores:
    analyzer.add_point(score)

# Analyze trend
trend = analyzer.analyze()

print(f"Direction: {trend.direction}")  # IMPROVING, STABLE, DECLINING
print(f"Slope: {trend.slope:.4f}")
print(f"Forecast: {trend.forecast_quality}")
```

## Quality Threshold Configuration

```python
from truthound.profiler.quality import QualityThresholds

thresholds = QualityThresholds(
    excellent_min=0.9,
    good_min=0.7,
    acceptable_min=0.5,
    poor_min=0.3,
)

# Determine quality level
level = thresholds.get_level(0.75)  # QualityLevel.GOOD
```

## Quality Report Generation

```python
from truthound.profiler.quality import QualityReporter

reporter = QualityReporter()

# Generate full suite quality report
report = reporter.generate(suite, test_data)

print(f"Suite Quality: {report.overall_level}")
print(f"Rules passed: {report.passed_count}/{report.total_count}")
print(f"Average F1: {report.average_f1:.2%}")

# Identify problematic rules
for issue in report.issues:
    print(f"  Rule {issue.rule_name}: {issue.problem}")
```

## CLI Usage

```bash
# Evaluate rule quality
th score-rules rules.yaml --test-data test.csv

# Compare rules
th compare-rules rules1.yaml rules2.yaml --test-data test.csv

# Generate quality report
th quality-report rules.yaml --output report.html
```

## Quality-Based Rule Filtering

```python
from truthound.profiler import generate_suite
from truthound.profiler.quality import filter_by_quality

# Generate rules
suite = generate_suite(profile)

# Filter by quality (keep only GOOD or better)
filtered_suite = filter_by_quality(
    suite,
    test_data=lf,
    min_level=QualityLevel.GOOD,
)

print(f"Original: {len(suite.rules)} rules")
print(f"Filtered: {len(filtered_suite.rules)} rules")
```

## Next Steps

- [Threshold Tuning](threshold-tuning.md) - Adjust thresholds for quality optimization
- [ML Inference](ml-inference.md) - ML-based quality prediction
