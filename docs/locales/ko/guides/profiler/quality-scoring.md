# 품질 점수

`truthound.profiler.quality`는 Polars `DataFrame`에서 검증 규칙을 평가합니다.
양성 예측은 **값이 규칙을 통과함**을 뜻합니다. 정답도 같은 의미의 null 없는 Boolean
Series이며 입력의 모든 행과 정렬되어야 합니다. sampling은 데이터와 정답에 동일한
행 인덱스를 적용합니다.

## RuleQualityScorer

```python
import polars as pl
from truthound.profiler.quality import RuleQualityScorer, RuleType, ValidationRule

frame = pl.DataFrame({"amount": [10, -1, 20]})
rule = ValidationRule("nonnegative", RuleType.RANGE, column="amount",
                      min_value=0, nullable=False)
scorer = RuleQualityScorer(
    estimator="sampling",
    estimator_options={"sample_size": 1000, "random_seed": 7,
                       "confidence_level": 0.95},
    min_confidence=0.5, quality_threshold=0.70, max_cache_entries=128,
)
score = scorer.score(rule, frame, ground_truth=pl.Series([True, False, True]))
print(score.metrics.f1_score)
print(score.metrics.quality_level.value)
```

반환값은 규칙 이름·유형·열, `metrics`, `recommendation`, `should_use`, 대안을 담은
`RuleQualityScore`입니다. `QualityMetrics`는 precision·recall·F1·accuracy·specificity·MCC,
신뢰 구간, sample/population 크기와 평가 시간을 제공합니다. `ConfusionMatrix` 필드는
`true_positives`, `true_negatives`, `false_positives`, `false_negatives`이며 분모가 0이면
비율은 0입니다. `ValidationRule.validate_column()`은 null 허용, 사용자 callback,
패턴의 시작 위치 매칭을 포함해 단일 값 `validate()`와 동일하게 판정합니다.
callback은 Boolean을 반환해야 하며 신뢰된 애플리케이션 코드입니다.

## QualityLevel

F1 기준은 excellent ≥0.95, good ≥0.85, acceptable ≥0.70, poor ≥0.50,
unacceptable <0.50입니다. scorer의 수용 기준과 최소 confidence는 권고에 적용되며
이 등급 자체를 바꾸지 않습니다.

## 평가 전략

- `SamplingQualityEstimator(sample_size=1000, confidence_level=0.95,
  random_seed=None)`는 별도 RNG로 전역 random 상태를 보존합니다. 설정한 신뢰 수준은
  Wilson precision/recall 구간에 적용됩니다. 정답은 표본만이 아닌 전체 입력에 대응해야 합니다.
- `CrossValidationEstimator(n_folds=5, random_seed=None)`는 행 수 이하의 fold를 사용합니다.
  정답이 있으면 fold의 실제 confusion count를 합치고, 없으면 일관성으로 추정합니다.
  빈 입력은 빈 지표를 반환합니다.
- `HeuristicQualityEstimator(strictness=0.5)`는 데이터·규칙 특징으로 추정하며 실제 정답과
  비교한 정확도를 입증하지 않습니다.

정답이 없는 sampling·cross-validation 결과는 휴리스틱 추정입니다. `confidence`는 규칙이
옳다는 보증이 아닙니다. F1 신뢰 구간은 계산하지 않아 `f1_ci`는 빈 기본값입니다.
`estimate_quality(rule, frame, estimator="sampling", **options)`는 정답 없이 지표를
반환합니다. 명시적 정답은 estimator의 `estimate()` 또는 scorer의 `score()`에 전달합니다.

## 규칙 비교와 캐시

`scorer.score_all(rules, frame, ground_truth=None)`는 입력 순서대로 점수를 반환합니다.
`scorer.compare(rules, frame)`와 `compare_rules(rules, frame)`는 F1 내림차순 목록을
반환하며 보고서나 best-rule wrapper를 반환하지 않습니다.

크기가 제한된 프로세스 로컬 캐시는 전체 데이터·정답·규칙 매개변수·estimator 설정·권고
기준을 반영합니다. 사용자 estimator·규칙 callback·지원하지 않는 Object 값은 자동 캐시를
우회합니다. 캐시 결과는 복사하여 호출자의 변경이 이후 결과를 오염시키지 않게 합니다.
`use_cache=False`는 해당 호출을 우회하고 `max_cache_entries=0`은 캐시를 끄며
`clear_cache()`는 항목을 지웁니다. 영속 복구나 암묵적 TTL은 없습니다.

## 이력 분석

`QualityTrendAnalyzer(storage_path=None)`는
`record(rule_name, metrics, timestamp=None, data_size=0, notes="")`로 지표를 기록합니다.
`analyze_trend(rule_name, window_days=30)`는 dict를, `get_history`는 측정 dict 목록을
반환하며 규칙마다 최대 100개를 보존합니다. 알림 서비스나 예측 모델은 아닙니다.

## 승인된 MCP 품질 작업

비공개 `ProfilerRuntime` factory의 `quality_scorers`는 native `RuleQualityScorer`에,
`quality_rules`는 native `ValidationRule`에 별칭을 연결합니다. `inspect`로 목록을 봅니다.
`quality_score(scorer_id, rule_ids=[별칭하나], asset_id)`와
`quality_compare(scorer_id, rule_ids, asset_id)`는 고정 Arrow 입력과 선택적인
`label_column`, `use_cache`를 사용합니다. 결과는 규칙 별칭·지표·수용 판정을 제공하며
비공개 규칙 이름·callback·설정을 제외합니다. `quality_cache_clear(scorer_id)`는 선택한
scorer의 캐시를 지웁니다. 정확한 승인과 민감 결과 동의가 필요하며 규칙 목록은 중복 없이
최대 100개입니다. sampling·estimator 설정은 factory가 소유합니다.

## 다음 단계

- [임계값 조정](threshold-tuning.md)
- [ML 추론](ml-inference.md)

영속 이력은 협력하는 native 파일 잠금과 sibling 파일의 원자적 게시, 파일·디렉터리
flush를 사용합니다. writer는 추가 전에 최신 snapshot을 다시 읽습니다. 재진입은 enum,
timestamp, 구간, confusion count를 복구합니다. 손상되거나 너무 큰 snapshot은 빈 이력으로
취급하지 않고 실패합니다. 쓰기 실패 후 해당 owner의 추가 기록은 차단됩니다. 새 owner로
실제 파일을 확인한 뒤 추가 측정 여부를 결정해야 합니다. 게시 뒤 실패하면 새 측정이 파일에
남을 수 있으며 exactly-once receipt 프로토콜은 아닙니다. 규칙 4096개, 규칙당 100개 측정,
snapshot 16 MiB로 제한하고 상위 디렉터리는 미리 존재해야 합니다. 시간은 UTC 순으로
정렬합니다. `get_history(limit=0)`은 빈 목록이며 음수 한도는 거부합니다.

factory는 native `QualityTrendAnalyzer`를 `quality_trends`에 연결할 수 있습니다.
`quality_record`는 `quality_score`와 같은 scorer·규칙·입력에 `trend_id`를 추가하여
native 결과를 기록합니다. `quality_history(trend_id, rule_ids, limit=50)`와
`quality_trend(trend_id, rule_ids, window_days=30)`는 규칙 별칭 하나를 요구합니다.
이력에서 비공개 notes를 제외하며 저장 경로는 factory가 소유합니다.
