# 드리프트 and 이상치 설정

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Who This Is For

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- maintainers wiring platform-specific 드리프트 or 이상치 태스크
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## `DriftConfig`

오케스트레이션 실행에서 `DriftConfig`, DriftConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from common.base import DriftConfig

config = DriftConfig(
    method="ks",
    columns=("revenue", "count"),
    threshold=0.05,
    min_severity="medium",
    timeout_seconds=60,
    extra={"n_permutations": 1000},
)
```

### Builder Pattern

```python
config = DriftConfig()
config = config.with_method("psi")
config = config.with_columns(("revenue", "user_count"))
config = config.with_threshold(0.1)
```

## `AnomalyConfig`

오케스트레이션 실행에서 `AnomalyConfig`, AnomalyConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from common.base import AnomalyConfig

config = AnomalyConfig(
    detector="isolation_forest",
    columns=("amount",),
    contamination=0.05,
    threshold=None,
    timeout_seconds=120,
    extra={"n_estimators": 200},
)
```

### Builder Pattern

```python
config = AnomalyConfig()
config = config.with_detector("lof")
config = config.with_columns(("amount", "frequency"))
config = config.with_contamination(0.03)
```

## `StreamConfig`

오케스트레이션 실행에서 `StreamConfig`, StreamConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from common.base import StreamConfig

config = StreamConfig(
    batch_size=5000,
    max_batches=100,
    timeout_per_batch_seconds=30.0,
    fail_fast=True,
)
```

### Builder Pattern

```python
config = StreamConfig()
config = config.with_batch_size(10000)
config = config.with_fail_fast(True)
```

## 플랫폼-Specific 설정 Surfaces

오케스트레이션 실행에서 Individual을(를) 다루는 항목입니다:

| 플랫폼 | 드리프트 설정 | 이상치 설정 |
|----------|--------------|----------------|
| 오케스트레이션 실행에서 Dagster을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `DriftOpConfig`, DriftOpConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `AnomalyOpConfig`, AnomalyOpConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 Prefect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `DriftTaskConfig`, DriftTaskConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `AnomalyTaskConfig`, AnomalyTaskConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 Mage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `DriftBlockConfig`, DriftBlockConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `AnomalyBlockConfig`, AnomalyBlockConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 Kestra을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `DriftScriptConfig`, DriftScriptConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 `AnomalyScriptConfig`, AnomalyScriptConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Operational Guidance

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Pages

- [드리프트 Detection](../engines/drift-detection.md)
- [이상치 Detection](../engines/anomaly-detection.md)
