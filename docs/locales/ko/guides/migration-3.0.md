# 마이그레이션 to Truthound 3.0

## Summary

실무 운영 가이드에서 Truthound을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 ValidationRunResult, `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 TruthoundContext, Truthound, `.truthound/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `compare`, `truthound.drift.compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `CheckpointResult.validation_result`, CheckpointResult.validation_result을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Fast Diagnostic

실무 운영 가이드에서 Run을(를) 다루는 항목입니다:

```bash
truthound doctor . --migrate-2to3
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## API Mapping

| 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Replacement을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| --- | --- |
| 실무 운영 가이드에서 `report = th.check(data)`, `report`, `Report`, Report을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ValidationRunResult, `run = th.check(data)`, `run`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `report.validation_run`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ValidationRunResult, `th.check()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `from truthound import compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `from truthound.drift import compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `CheckpointResult.validation_result`, CheckpointResult.validation_result을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `CheckpointResult.validation_run`, `CheckpointResult.validation_view`, CheckpointResult.validation_run, CheckpointResult.validation_view을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 검증기 subclass authoring | 실무 운영 가이드에서 `CheckSpecFactory`, CheckSpecFactory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Before and After

### 검증

```python
# 2.x
report = th.check(data, validators=["null", "unique"])
print(report.validation_run.execution_mode)
```

```python
# 3.0
run = th.check(data, validators=["null", "unique"])
print(run.planned_execution_mode)
print(run.execution_mode)
```

### 드리프트

```python
# 2.x
from truthound import compare
```

```python
# 3.0
from truthound.drift import compare
```

### 체크포인트

```python
# 2.x
stats = checkpoint_result.validation_result.statistics
```

```python
# 3.0
stats = checkpoint_result.validation_run
view = checkpoint_result.validation_view
print(view.statistics.total_issues)
```

## Zero-설정 Changes

실무 운영 가이드에서 Truthound, `.truthound/`을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `.truthound/config.yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `.truthound/catalog/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `.truthound/baselines/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `.truthound/runs/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `.truthound/docs/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `.truthound/plugins/`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `truthound.yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Reporter and Docs 마이그레이션

실무 운영 가이드에서 Reporter을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `render(run_result, *, context)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `RunPresentation`, RunPresentation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `execution_mode`, `planned_execution_mode`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `Report`, `truthound.stores.results.ValidationResult`, Report, ValidationResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Extension 마이그레이션

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 `CheckSpecFactory`, CheckSpecFactory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `CheckSpec`, CheckSpec을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Acceptance Checklist

- [ ] `truthound doctor . --migrate-2to3` 리포트 no blocking patterns
- 실무 운영 가이드에서 `Report`, Report을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `compare`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [ ] 체크포인트 integrations use `validation_run` or `validation_view`
- 실무 운영 가이드에서 ValidationRunResult을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `report.validation_run`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- [아키텍처](../concepts/architecture.md)
- 실무 운영 가이드에서 Zero-Config, Context을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [릴리스 노트: Truthound 3.0](../releases/truthound-3.0.md)

## Result persistence contracts

새 저장소 DTO는 내장 `_validation_run_contract` 버전 2를 사용합니다.
표시 이름으로 근거를 재배정하지 않고, 결과 형식에서 선택한 상세를 포함해
각 검사에 속한 이슈를 보존합니다. 같은 검증기 이름이나 표시 별칭을 사용해도
저장소 왕복 후 검사별 판정이 유지됩니다. 저장 결과는 canonical run 파일과
마찬가지로 민감한 산출물로 취급하세요. 요약 화면은 데이터 제거 정책이 아닙니다.

버전 없는 이전 DTO는 이슈의 귀속이 명확할 때 계속 읽을 수 있습니다.
이전 계약에 중복 검사 이름과 해당 이슈가 함께 있으면 `ValueError`로 거부합니다.
정확한 귀속을 복구하려면 검증을 다시 실행하세요. 내장 계약이 없는 DTO는 기존
행 단위 표현을 유지합니다. 알 수 없는 내장 계약 버전은 추측하지 않고 거부합니다.

`ValidationRunResult.from_artifact_dict(payload, expected_run_id=...)`는 저장된 실행을
엄격하게 읽습니다. 실행 식별자·차원·검사·이슈·실행 오류가 명시되어야 하며,
빈 실행도 빈 목록을 명시해야 합니다. AI 실행 파일 로딩은 요청한 실행 ID와의
일치를 확인합니다. 체크포인트의 null이 아닌 canonical `validation_run`도 같은
읽기 경계를 사용하며, null은 실행 결과 없음을 뜻합니다. 손상되거나 불완전한
산출물은 분석 전에 `ValueError`로 거부합니다. 필수 필드를 갖춘 기존 버전 없는
canonical 파일은 계속 지원합니다.

`ValidationRunResult.from_dict`는 의도된 빈 결과를 포함해 기존 호환성을 위한
관대한 생성 도우미로 유지합니다. 손상 검사 용도로 사용하지 마세요.
구조 검증은 진위 보장이 아니며 일관되게 조작된 모든 근거를 탐지하지는 못합니다.
severity 필터는 현재 뷰를 좁히며, 검사를 재실행하거나 버린 낮은 수준의 상세를
복구하지 않습니다.
