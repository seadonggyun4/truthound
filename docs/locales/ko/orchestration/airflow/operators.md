---
title: Airflow Operators
---

# Airflow Operators

오케스트레이션 실행에서 Truthound, Airflow, DAGs, They, IDs, XCom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## DataQualityCheckOperator

오케스트레이션 실행에서 `DataQualityCheckOperator`, DataQualityCheckOperator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="quality_check",
    data_path="/opt/airflow/data/users.parquet",
    fail_on_error=True,
    rules=[
        {"column": "user_id", "type": "not_null"},
        {"column": "email", "type": "unique"},
    ],
)
```

오케스트레이션 실행에서 Typical을(를) 다루는 항목입니다:

- gate downstream 태스크
- publish structured 검증 결과 to XCom
- 오케스트레이션 실행에서 Airflow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## DataQualityProfileOperator

오케스트레이션 실행에서 `DataQualityProfileOperator`, DataQualityProfileOperator, DAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import DataQualityProfileOperator

profile = DataQualityProfileOperator(
    task_id="profile_users",
    data_path="/opt/airflow/data/users.parquet",
)
```

## DataQualityLearnOperator

오케스트레이션 실행에서 `DataQualityLearnOperator`, DataQualityLearnOperator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import DataQualityLearnOperator

learn = DataQualityLearnOperator(
    task_id="learn_users",
    data_path="/opt/airflow/data/baseline_users.parquet",
)
```

오케스트레이션 실행에서 Common을(를) 다루는 항목입니다:

- baseline learn 태스크
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `DataQualityCheckOperator`, DataQualityCheckOperator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## DataQualityStreamOperator

오케스트레이션 실행에서 `DataQualityStreamOperator`, DataQualityStreamOperator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import DataQualityStreamOperator

stream_check = DataQualityStreamOperator(
    task_id="stream_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

오케스트레이션 실행에서 Streaming을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Truthound-Specific Variants

오케스트레이션 실행에서 Truthound, `TruthoundCheckOperator`, `TruthoundProfileOperator`, `TruthoundLearnOperator`, TruthoundCheckOperator, TruthoundProfileOperator, TruthoundLearnOperator, Truthound-first을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## XCom Contract

오케스트레이션 실행에서 Airflow, Operators, Downstream을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 Guidance

오케스트레이션 실행에서 Make을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 Airflow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `warning_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Usage Pattern

1. 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 오케스트레이션 실행에서 Move, DAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 오케스트레이션 실행에서 Push, SLA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- 오케스트레이션 실행에서 Hooks을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Sensors, Triggers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 SLA, Callbacks을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [레시피](recipes.md)
