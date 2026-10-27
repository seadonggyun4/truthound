# 모델 모니터링

`truthound.ml.monitoring.ModelMonitor`는 성능, 예측·특성 드리프트와 정답이
있는 품질 지표를 수집합니다. 지표 저장소, 등록 모델, 알림 규칙과 선택한
전송기를 소유하며 `truthound.core` 검증 커널과는 별도인 고급 기능입니다.

## 네이티브 수명주기

```python
import asyncio

from truthound.ml.monitoring import ModelMonitor, MonitorConfig
from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule
from truthound.ml.monitoring.collectors.quality import QualityConfig


async def main():
    monitor = ModelMonitor()  # 네이티브 메모리 저장소이며 외부 전송기는 없음
    monitor.register_model("classifier", MonitorConfig(
        batch_size=10,
        collect_interval_seconds=60,
        alert_evaluation_interval_seconds=30,
        quality=QualityConfig(window_size=1000, task_type="binary"),
    ))
    monitor.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 100)))
    async with monitor:
        identity = await monitor.record_prediction(
            "classifier", {"age": 25}, prediction=1, latency_ms=5,
        )
        await monitor.record_actual("classifier", identity, actual=1)
        collected = await monitor.collect_now("classifier")
        metrics = await monitor.get_metrics("classifier")
        assert collected == 1 and metrics.accuracy == 1.0


asyncio.run(main())
```

`record_prediction`은 네이티브 예측 ID를 반환합니다. 정답은 해당 예측이 버퍼에
남아 있을 때만 입력할 수 있으며 없거나 이미 수집된 ID는 `KeyError`입니다.
`collect_now`는 수집이 확인된 버퍼 레코드 수를 반환합니다. 모델 ID는 유일해야
하며 재등록으로 대기 상태를 덮어쓰지 않습니다. `unregister_model`은 해당 모델의
대기 예측을 명시적으로 버리고 타이머를 중단합니다. 저장된 지표는 남습니다.

`start()`는 선택한 저장소와 모델별 수집·평가 타이머를 시작합니다. 실행 중 등록한
모델에도 두 타이머를 만듭니다. `batch_size`에 도달하면 시작 전에도 즉시 수집합니다.
모델별 예측·정답·기준 변경·수집을 직렬화하여 저장 대기 중 새 예측이 두 배치에
중복 수집되지 않도록 합니다.

| 설정 | 동작 |
| --- | --- |
| `batch_size` | 즉시 수집 기준인 양의 정수이며 기본값은 100입니다. |
| `collect_interval_seconds` | 시작 상태에서 미완성 배치를 수집하는 유한한 양의 주기이며 기본값은 60입니다. |
| `alert_evaluation_interval_seconds` | 최신 저장 지표 관찰과 모델별 보존 삭제를 실행하는 유한한 양의 주기이며 기본값은 30입니다. |
| `retention_hours` | 모델 이력 조회 범위와 삭제 기준인 유한한 양의 시간이며 기본값은 24입니다. 선택한 저장소의 별도 보존·개수 제한도 적용될 수 있습니다. |
| `enable_drift_detection`, `enable_quality_metrics` | 네이티브 수집기를 선택하는 실제 boolean이며 둘 다 기본값은 true입니다. |
| `performance`, `drift`, `quality` | 선택적 네이티브 `PerformanceConfig`, `DriftConfig`, `QualityConfig`입니다. 등록할 때 독립된 설정 복사본을 고정합니다. |

성능 지연은 `window_size`와 지정 백분위수를 사용합니다. 드리프트는 마지막
`reference_window_size`개 기준 레코드와 `detection_window_size`개 이동 탐지
창을 사용합니다. `set_reference_data`는 드리프트 수집이 켜진 등록 모델에 속하는
비어 있지 않은 네이티브 `PredictionRecord` 목록을 요구합니다. 품질은 배치를
가로지르는 정답 레코드 이동 창을 사용합니다. `QualityConfig.task_type`은
`binary`, `multiclass`, `regression` 또는 기본값 `auto` 휴리스틱입니다. 확률처럼
보이는 회귀 값이나 숫자 클래스가 모호하면 작업 종류를 명시하십시오.

## 알림·실패·종료

규칙은 실제 boolean을 반환해야 합니다. 새 저장 지표는 한 번 평가하며 같은
지표를 주기적으로 읽어도 이상·추세 관측치나 전송이 중복되지 않습니다. 지속시간
임계 규칙은 지정 시간이 지난 뒤에도 임계값을 넘는 후속 새 관측치를 요구합니다.
규칙·전송기 이름은 각각 유일해야 합니다. `remove_rule`, `remove_handler`는
지정한 구성요소를 제거합니다. 규칙 제거는 전송 없이 활성 상태를 로컬에서 해제합니다.

선택한 각 전송기의 `handle`·`resolve`는 `True`를 반환해야 합니다. `False`,
누락 응답과 예외는 실패입니다. 이력은 발생 시점을 기록하므로 모든 전송기가 수신했다는
증거가 아닙니다. 규칙 엔진은 기본적으로 발생 이력을 최대 10,000개까지 메모리에
유지하며 이후 해제 상태도 담습니다. 대시보드는 현재 지표·지표 이력·활성 알림과
실제 알림 이력을 제공합니다. 영구 저장은 선택한 저장소·전송기 구현의 계약입니다.

수집·평가 중 오류나 취소는 해당 모델을 불명확 상태로 고정합니다. 대기 예측의
개수·ID는 조회할 수 있지만 추가 수집·입력은 차단하고 타이머가 불명확한 효과를
재전송하지 않습니다. `get_status()`에서 상태·버퍼 수·저장소 수명주기를 확인합니다.
조사 후 `discard_uncertain(model_id)`를 명시적으로 호출하면 대기 레코드를 버리고
수집기·기준 상태를 초기화하며 현재 저장 지표를 전송 없이 관찰 완료로 처리합니다.
이미 저장한 지표나 알림을 되돌리지는 않습니다. 다음 드리프트 수집 전 기준을 다시 설정합니다.

`stop()`은 소유한 타이머를 취소·대기한 후 저장소를 중단합니다. 대기 레코드를
flush하거나 새 알림을 전송하지 않습니다. 종료 실패를 노출하며 저장소 정리가 확인되기
전에는 `start()`가 재시작을 거부합니다. 네이티브 `MonitoringPipeline`도 저장·전송
실패를 전파하고 명시적 `discard_uncertain(model_id)` 전까지 해당 모델의 후속
처리를 차단합니다. 불명확한 효과를 자동 재시도하지 않습니다.

기본 예측 버퍼·지표 저장소·규칙 이력·실패 상태는 메모리에 있습니다. 프로세스 손실
복구나 외부 알림의 정확히 한 번 전달을 보장하지 않습니다. 외부 저장소·전송기는
각 운영 계약에 따라 구성하고 검증해야 합니다.

## 승인된 MCP 바인딩

미배포 Core 프로파일은 `ml_services_open`, `ml_services_execute`를 제공합니다.
승인된 설치 팩토리는 `MLRuntime(monitor, models, rules, handlers)`를 반환합니다.
모델 별칭은 `MonitoringModel(native_id, config, features, prediction_column,
actual_column=None, latency_column=None)`을 선택합니다. `features`는 공개 특성
별칭을 고정 자산 열에 연결합니다. 규칙·전송기도 별칭으로 선택하는 팩토리 객체입니다.
호출자가 코드·실제 ID·연결 설정·전송 주소를 입력하지 않습니다. 초기 모니터는
중단 상태이며 등록 모델·규칙·전송기가 없어야 합니다. 비공개 `ml_services` 바인딩에
별칭과 설치 capability의 연결을 기록합니다.

명령은 `inspect`, `status`, `register`, `unregister`, `start`, `stop`, `shutdown`,
`record`, `feedback`, `reference`, `collect`, `evaluate`, `discard_uncertain`,
`metrics`, `dashboard`, `rule_add`, `rule_remove`, `handler_add`, `handler_remove`입니다.
각 명령은 정확한 승인과 민감정보 공개 허용을 요구합니다. 시작 승인은 중단·해제·만료까지
향후 주기 수집·평가·보존 삭제와 선택 전송기의 전달을 포함합니다. 설치 팩토리 진입·종료
동작은 운영자가 관리합니다.

`record`, `reference`는 고정한 자산 핸들에서 제한된 `offset`·`limit`만큼 읽습니다.
`feedback`은 `offset` 위치 한 행의 고정 정답 열과 `record`가 반환한 소유자 로컬
예측 참조를 사용합니다. 반환 참조는 배치 수집 이후에도 예측이 대기 중인지 표시합니다.
입력 하나라도 권한을 해제하면 모니터와 파생 산출물도 해제됩니다. 정리는 레코드를
flush하지 않고 타이머를 중단합니다. 출력은 유한한 표준 지표, 회귀 오차·설정 백분위수
등 내장 집계, 구성한 특성 별칭과 선택적 팩토리 고정 `MonitoringModel.metric_aliases`
매핑을 제공합니다. 임의 사용자 지표 이름·알림 메시지·실제 ID·전송기 메타데이터·주소는 제외합니다.

확인되지 않은 네이티브 동작은 현재 상태를 담은 실패 `native_monitor_incomplete`
산출물입니다. 같은 승인 키는 결과를 재생하며 수집·전송을 반복하지 않습니다. 대기 작업을
명시적으로 버리기 전 상태를 확인하십시오. `shutdown` 성공은 모니터와 팩토리 정리가
끝난 뒤에만 응답합니다. 같은 키로 결과를 재생할 수 있지만 다른 네이티브 명령은 거부합니다.
리소스 해제만으로 느린 팩토리 정리 완료가 증명되지는 않습니다. 이 소스 기능은 게시된
3.2.0 패키지에 포함되지 않습니다. [MCP 가이드](mcp.md#core-프로파일-아직-배포하지-않은-소스-기능)를 참조하십시오.


종료는 진행 중인 foreground 모델 작업도 기다린 뒤 저장소를 중단합니다. 정리 중에는
새 모델 작업을 차단하지만 이미 시작한 쓰기를 되돌리지는 않으며 중단 후 다음 전송기를
시작하지 않습니다. 자기 모델 작업 안에서 stop을 호출하면 자기 자신을 기다리지 않도록
거부합니다. `unregister_model`도 실행 중인 모델은 거부합니다. 전송기 편집은 다음
알림에 적용하고 현재 전송은 고정 목록을 사용하므로 대기 중 전송기 하나를 제거해도
다른 선택 전송기를 조용히 건너뛰지 않습니다. 상태에는 `stopping`을 포함합니다.
