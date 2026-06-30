---
title: Airflow SLA 모니터링
---

# Airflow SLA And Callbacks

오케스트레이션 실행에서 Airflow, SLA, DAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SLAConfig

오케스트레이션 실행에서 `SLAConfig`, SLAConfig, DAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import SLAConfig

config = SLAConfig(
    max_failure_rate=0.01,
    max_execution_time_seconds=300.0,
)
```

오케스트레이션 실행에서 Typical을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SLA메트릭

오케스트레이션 실행에서 Airflow, `SLAMetrics`, SLAMetrics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SLAViolation

오케스트레이션 실행에서 `SLAViolation`, `SLAViolationType`, SLAViolation, SLAViolationType을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SLAMonitor

오케스트레이션 실행에서 `SLAMonitor`, SLAMonitor, SLA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import SLAMonitor, SLAConfig

monitor = SLAMonitor(
    config=SLAConfig(
        max_failure_rate=0.05,
        max_execution_time_seconds=600.0,
    )
)
```

## Callback Types

오케스트레이션 실행에서 Airflow을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 `DataQualitySLACallback`, DataQualitySLACallback을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `QualityAlertCallback`, QualityAlertCallback을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `CallbackChain`, CallbackChain을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 Airflow, Airflow-native을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Registry Support

오케스트레이션 실행에서 `SLARegistry`, SLARegistry, DAGs, SLA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Airflow Pattern

```python
from truthound_airflow import DataQualityCheckOperator, DataQualitySLACallback, SLAConfig

sla_callback = DataQualitySLACallback(
    sla_config=SLAConfig(max_failure_rate=0.02, max_execution_time_seconds=300),
)

check = DataQualityCheckOperator(task_id="quality_check", data_path="users.parquet")
```

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- 오케스트레이션 실행에서 Operators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [레시피](recipes.md)
- [문제 해결](troubleshooting.md)
