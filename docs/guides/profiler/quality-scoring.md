# Quality Scoring

이 문서는 생성된 검증 규칙의 품질 평가 시스템을 설명합니다.

## 개요

`src/truthound/profiler/quality.py`에 구현된 품질 스코어링 시스템은 규칙의 정확도와 유용성을 평가합니다.

## QualityLevel

```python
class QualityLevel(str, Enum):
    """품질 수준"""

    EXCELLENT = "excellent"      # 0.9 - 1.0
    GOOD = "good"                # 0.7 - 0.9
    ACCEPTABLE = "acceptable"    # 0.5 - 0.7
    POOR = "poor"                # 0.3 - 0.5
    UNACCEPTABLE = "unacceptable"  # 0.0 - 0.3
```

## ConfusionMatrix

규칙 평가를 위한 혼동 행렬입니다.

```python
@dataclass
class ConfusionMatrix:
    """혼동 행렬"""

    true_positive: int = 0   # 올바른 위반 감지
    true_negative: int = 0   # 올바른 통과
    false_positive: int = 0  # 잘못된 위반 감지 (오탐)
    false_negative: int = 0  # 놓친 위반 (미탐)

    @property
    def precision(self) -> float:
        """정밀도: TP / (TP + FP)"""
        if self.true_positive + self.false_positive == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        """재현율: TP / (TP + FN)"""
        if self.true_positive + self.false_negative == 0:
            return 0.0
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def f1_score(self) -> float:
        """F1 점수: 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """정확도: (TP + TN) / Total"""
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        if total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / total
```

## RuleQualityScorer

규칙 품질을 평가하는 메인 클래스입니다.

```python
from truthound.profiler.quality import RuleQualityScorer

scorer = RuleQualityScorer()

# 규칙 품질 점수 계산
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
    """품질 점수 결과"""

    rule_name: str
    quality_level: QualityLevel
    overall_score: float          # 0.0 - 1.0
    metrics: ConfusionMatrix
    test_sample_size: int
    evaluation_time_ms: float

    # 세부 점수
    precision_score: float
    recall_score: float
    f1_score: float
```

## 기본 사용법

```python
from truthound.profiler.quality import RuleQualityScorer, estimate_quality

# 빠른 품질 추정
quality = estimate_quality(rule, lf)
print(f"Estimated quality: {quality.level}")

# 상세 품질 평가
scorer = RuleQualityScorer(
    sample_size=10_000,
    cross_validation_folds=5,
)
score = scorer.score(rule, test_data)
```

## 품질 추정 전략

### SamplingQualityEstimator

샘플링 기반 빠른 추정입니다.

```python
from truthound.profiler.quality import SamplingQualityEstimator

estimator = SamplingQualityEstimator(sample_size=1_000)
result = estimator.estimate(rule, lf)

print(f"Level: {result.level}")
print(f"Confidence: {result.confidence:.2%}")
```

### HeuristicQualityEstimator

휴리스틱 기반 가장 빠른 추정입니다.

```python
from truthound.profiler.quality import HeuristicQualityEstimator

estimator = HeuristicQualityEstimator()
result = estimator.estimate(rule, lf)

# 규칙 특성 기반 추정 (실제 실행 없음)
print(f"Level: {result.level}")
```

### CrossValidationEstimator

교차 검증 기반 정확한 추정입니다.

```python
from truthound.profiler.quality import CrossValidationEstimator

estimator = CrossValidationEstimator(n_folds=5)
result = estimator.estimate(rule, lf)

print(f"Level: {result.level}")
print(f"Mean F1: {result.mean_f1:.2%}")
print(f"Std F1: {result.std_f1:.2%}")
```

## 규칙 비교

```python
from truthound.profiler.quality import compare_rules

# 여러 규칙 비교
comparison = compare_rules(
    rules=[rule1, rule2, rule3],
    test_data=lf,
)

# 최고 품질 규칙
best = comparison.best_rule
print(f"Best rule: {best.name}")
print(f"F1 Score: {best.score.f1_score:.2%}")

# 순위별 정렬
for rank, (rule, score) in enumerate(comparison.ranked_rules, 1):
    print(f"{rank}. {rule.name}: {score.f1_score:.2%}")
```

## 트렌드 분석

시간에 따른 품질 변화를 추적합니다.

```python
from truthound.profiler.quality import QualityTrendAnalyzer

analyzer = QualityTrendAnalyzer()

# 히스토리 추가
for score in historical_scores:
    analyzer.add_point(score)

# 트렌드 분석
trend = analyzer.analyze()

print(f"Direction: {trend.direction}")  # IMPROVING, STABLE, DECLINING
print(f"Slope: {trend.slope:.4f}")
print(f"Forecast: {trend.forecast_quality}")
```

## 품질 임계값 설정

```python
from truthound.profiler.quality import QualityThresholds

thresholds = QualityThresholds(
    excellent_min=0.9,
    good_min=0.7,
    acceptable_min=0.5,
    poor_min=0.3,
)

# 품질 수준 판정
level = thresholds.get_level(0.75)  # QualityLevel.GOOD
```

## 품질 리포트 생성

```python
from truthound.profiler.quality import QualityReporter

reporter = QualityReporter()

# 전체 스위트 품질 리포트
report = reporter.generate(suite, test_data)

print(f"Suite Quality: {report.overall_level}")
print(f"Rules passed: {report.passed_count}/{report.total_count}")
print(f"Average F1: {report.average_f1:.2%}")

# 문제 있는 규칙 식별
for issue in report.issues:
    print(f"  Rule {issue.rule_name}: {issue.problem}")
```

## CLI 사용법

```bash
# 규칙 품질 평가
th score-rules rules.yaml --test-data test.csv

# 규칙 비교
th compare-rules rules1.yaml rules2.yaml --test-data test.csv

# 품질 리포트 생성
th quality-report rules.yaml --output report.html
```

## 품질 기반 규칙 필터링

```python
from truthound.profiler import generate_suite
from truthound.profiler.quality import filter_by_quality

# 규칙 생성
suite = generate_suite(profile)

# 품질 기반 필터링 (GOOD 이상만 유지)
filtered_suite = filter_by_quality(
    suite,
    test_data=lf,
    min_level=QualityLevel.GOOD,
)

print(f"Original: {len(suite.rules)} rules")
print(f"Filtered: {len(filtered_suite.rules)} rules")
```

## 다음 단계

- [임계값 튜닝](threshold-tuning.md) - 품질 최적화를 위한 임계값 조정
- [ML 추론](ml-inference.md) - ML 기반 품질 예측
