# ML 기반 타입 추론

`truthound.profiler.ml_inference`는 열 이름·표본 값·통계·테이블 문맥을 결합하여 native
`DataType`을 제안합니다. 외부 AI 서비스를 호출하지 않는 로컬 추론입니다. confidence는
점수이며 보정된 확률을 보증하지 않습니다.

## 추론과 설정

```python
import polars as pl
from truthound.profiler.ml_inference import MLTypeInferrer, InferrerConfig

inferrer = MLTypeInferrer(config=InferrerConfig(
    model="ensemble", confidence_threshold=0.5,
    use_caching=True, cache_size=1000, enable_learning=True,
    feedback_batch_size=100,
))
result = inferrer.infer(
    pl.Series("email", ["a@example.test", "b@example.test"]),
    {"table_name": "contacts"},
)
print(result.inferred_type.value, result.confidence)
```

명시적 `model=` 객체·이름이 `config.model`보다 우선하며 생략하면 설정 모델을 사용합니다.
`infer(column, context=None)`는 문맥을 복사한 뒤 열 정보를 추가합니다.
`infer_table(frame, table_name="")`은 인접 열 문맥을 제공하고 열별 dict를 반환합니다.
점수가 `confidence_threshold`보다 낮으면 `DataType.UNKNOWN`을 반환하고 최선 후보를
`alternatives`에 남깁니다. 유효하지 않거나 유한하지 않은 예측 점수는 실패합니다.

`InferenceResult`는 `column_name`, `inferred_type`, `confidence`, `alternatives`,
`reasoning`, `features_used`, `model_version`, `inference_time_ms`를 제공합니다.
`to_dict()`는 enum과 대안 예측을 변환합니다. 간편 함수는
`infer_column_type_ml(column, context=None, model="ensemble")` 및
`infer_table_types_ml(frame, table_name="", model="ensemble")`입니다.

## 특징 추출

`Feature`는 이름·숫자 값·`FeatureType`·importance·metadata를 담습니다. 유형은
`NAME_BASED`, `VALUE_BASED`, `STATISTICAL`, `CONTEXTUAL`, `PATTERN_BASED`입니다.
extractor의 `extract(series, context)`는 특징 목록을 반환하고
`FeatureExtractorRegistry.extract_all()`은 `to_array()`, `to_dict()`, `get_feature(name)`를
제공하는 `FeatureVector`를 반환합니다.

기본 registry에는 `NameFeatureExtractor`, `ValueFeatureExtractor`,
`StatisticalFeatureExtractor`, `ContextFeatureExtractor`가 있습니다. native 추출 오류나
유효하지 않은 특징은 부분 성공으로 숨기지 않고 작업을 실패시킵니다. 사용자 extractor와
모델은 신뢰된 애플리케이션 코드입니다.

## 모델과 캐시

`create_inference_model("rule_based" | "naive_bayes" | "ensemble")`는 native 모델을
반환합니다. `RuleBasedModel`은 가중 특징 규칙을 사용하며 feedback으로 학습하지 않습니다.
`NaiveBayesModel`은 Gaussian 특징 likelihood와 class prior를 학습하고 학습 전에는 규칙
기반 예측을 사용합니다. `EnsembleModel`은 가중 예측을 결합합니다. 가중치는 유한한 음이
아닌 수이고 합은 양수여야 합니다. ensemble은 자식 복사본을 학습한 뒤 목록을 교체합니다.

크기가 제한된 프로세스 로컬 캐시는 전체 열 내용·문맥·모델 상태·confidence 기준을 반영하고
복사본을 반환합니다. 사용자 모델·extractor·지원하지 않는 Object 값은 자동 캐시를 우회합니다.
`cache_size=0`, `use_caching=False`는 캐시를 끄고 `clear_cache()`는 항목을 지웁니다.
디스크 캐시나 TTL은 없으며 같은 inferrer의 추론·feedback은 잠금으로 직렬화합니다.

## Feedback과 영속화

```python
from truthound.profiler.base import DataType
inferrer.provide_feedback(pl.Series("count", [1, 2]), DataType.INTEGER)
```

학습이 켜져 있으면 `feedback_batch_size`(기본 100, 1–10000)까지 모은 뒤 모델을 학습하고
추론 캐시를 지웁니다. 꺼져 있으면 모델을 바꾸지 않습니다. `model_path`는 로컬 JSON
snapshot을 선택하며 생성 시 존재하면 읽고 feedback batch 완료 시 저장합니다. 아직
flush하지 않은 feedback은 프로세스 로컬이므로 재시작하면 사라집니다.

내장 모델의 `save(path)`·`load(path)`는 크기가 제한된 JSON을 사용하고 pickle·import
실행을 하지 않습니다. 규칙 매개변수·Naive Bayes 상태·중첩 ensemble 자식을 보존합니다.
전체 snapshot 검증 후 모델을 바꾸며 기존 Naive Bayes JSON도 읽습니다. snapshot은
16 MiB, 중첩은 8단계, ensemble마다 자식은 32개로 제한합니다. 저장은 비공개 sibling 파일,
원자 교체, 파일·디렉터리 flush를 사용하고 상위 디렉터리가 존재해야 합니다. 사용자 모델은
영속화 메서드를 명시적으로 구현해야 합니다.

협력하는 inferrer는 선택 경로를 잠그고 학습 전에 관측한 파일 digest를 검사합니다.
오래된 owner는 새 snapshot 덮어쓰기를 거부합니다. 학습·저장 실패는 대기 feedback을
보존하고 그 owner의 추론·학습을 차단합니다. 새 owner로 실제 저장 상태를 확인해야 하며
게시 후 실패라면 새 상태가 디스크에 남을 수 있습니다. exactly-once feedback receipt나
임의의 비협력 writer 조정을 제공하지 않습니다.

## 승인된 MCP 추론

비공개 `ProfilerRuntime.inferrers`는 native `MLTypeInferrer`를,
`inference_contexts`는 복사된 비공개 문맥을 별칭에 연결하며 `inspect`는 별칭을 제공합니다.
`inference_infer(inferrer_id, asset_id, column, inference_context_id?)`와
`inference_table(inferrer_id, asset_id)`는 고정 Arrow 입력을 사용합니다. 결과는 열·타입·점수·
대안을 반환하며 비공개 문맥·특징 진단·모델 경로·모델 버전을 제외합니다.

`inference_feedback(inferrer_id, asset_id, column, true_type, inference_context_id?)`는
native 학습을 호출하고 학습 활성 여부와 대기 feedback 수를 제공합니다. 완료 batch는
factory가 선택한 경로에 저장할 수 있습니다. `inference_cache_clear(inferrer_id)`는 선택한
캐시를 지웁니다. 정확한 승인과 민감 결과 동의가 필요하며 wire로 경로·모델 import·callback
코드를 지정하지 않습니다. 설치 worker 검증은 로컬 native 모델 학습과 새 owner 복구를
포함합니다.

## 다음 단계

- [샘플링](sampling.md)
- [품질 점수](quality-scoring.md)
- [시각화](visualization.md)

## 네이티브 sklearn 분류 모델

별도 `truthound.profiler.ml` API는 `TrainingData(features, labels, feature_names,
weights=...)`와 `train_type_classifier(data, model_type, config)`를 제공합니다.
random_forest, gradient_boosting, naive_bayes, logistic_regression, svm,
neural_network 및 forest·logistic regression·Gaussian Naive Bayes의 soft-voting
ensemble을 지원합니다. 가중 신경망 학습을 위해 `ml` extra는 scikit-learn 1.7
이상을 요구합니다. rule_based 예측에는 sklearn이 필요 없으며 train은 규칙 학습이나
고정 정확도 반환 대신 입력 예제를 평가합니다. 잘못된 모델 이름과 미학습 ML 예측은
다른 모델로 대체하지 않고 실패합니다.

특징은 유한 값·고유 이름을 가지며 라벨·선택적 sample ID·가중치와 정렬되어야 합니다.
가중치는 유한·음이 아닌 값이고 총합이 양수여야 합니다. 최소 두 클래스와 클래스별
cross_validation_folds개 이상의 예제가 필요하며 네이티브 stratified holdout 제약도
적용됩니다. 특징 선택은 각 평가 fold 안에서 SelectKBest를 독립적으로 학습해
max_features개까지 선택합니다. holdout 지표와 교차 검증 점수를 따로 측정한 뒤
게시할 모델은 전체 행으로 학습합니다. 평가는 실제 적용한 가중치를 사용합니다.
학습·교차 검증 실패는 기존 모델·라벨·지표를 변경하지 않습니다.

트리 설정은 forest/boosting, n_jobs는 forest/ensemble, early_stopping은
boosting/신경망에 적용합니다. 클래스 균형은 네이티브 class weight 또는 가중 학습을
사용합니다. 해당 기능이 없는 모델에 트리·스레드 설정을 적용했다고 주장하지 않습니다.
생성 시 설정을 복사하며 명시적 분류기 클래스가 backend를 선택합니다. 퇴화한 특징·수렴
경고를 포함한 sklearn 경고는 숨기지 않습니다.

규칙 모델은 특징 이름·측정 지표를 포함한 검증된 JSON으로 저장합니다. sklearn 모델은
backend 식별자·설정·지표·특징 스키마를 포함한 **신뢰하는 Python pickle**로 저장합니다.
역직렬화 시 Python 실행이 가능하므로 운영자 소유의 신뢰하는 파일만 로드해야 하며
버전 간 이식 가능한 형식은 아닙니다. 현재 format marker가 없는 이전 파일은 거부합니다.
원자적으로 교체하며 잘못된 로드는 현재 모델을 보존합니다. 위 MLTypeInferrer의 제한된
JSON snapshot과는 별도 형식입니다.

PretrainedModelManager.load_default()는 공유 규칙 객체를 변경하지 않고 추출한 특징
이름을 연결합니다. 정확도 0은 미평가를 뜻합니다. train_and_cache(name,
n_samples_per_type=500, cache=True)는 유형별 최소 250개의 seeded 합성 값으로
학습합니다. 합성 평가이며 실제 데이터셋 성능 주장이 아닙니다. UUID를 포함해 자체 RNG를
사용합니다. 이름은 길이가 제한된 단순 식별자입니다. cache=False는 캐시를 읽거나 쓰지
않고, 활성 캐시는 DEFAULT_CACHE_DIR의 운영자 소유 trusted pickle입니다. 손상되거나
설정이 다른 캐시는 명시적으로 실패하며 새 설정은 새 이름으로 학습합니다. 게시도 원자적입니다.

승인된 팩토리는 truthound.mcp.profiler_classifier의
NativeClassifierBinding(model, confidence_threshold=0.0, sample_size=1000)을
ProfilerRuntime.inferrers에 등록할 수 있습니다. 기존 inference_infer/inference_table은
이름이 있는 특징으로 네이티브 예측을 반환합니다. 모델 사본을 소유하고 암묵적 캐시는 없어
clear는 no-op입니다. sklearn 예측은 학습 특징 스키마와 일치해야 합니다. 학습·신뢰하는
모델 파일 로드는 팩토리가 담당하며 이 binding의 inference_feedback은
unsupported_operation으로 거부됩니다. wire 요청으로 pickle·학습 callback·경로를 받을
수 없습니다. 설치 MCP 검증은 네이티브 규칙 binding을, 네이티브 수명주기 테스트는 모델
7종을 검증합니다.
