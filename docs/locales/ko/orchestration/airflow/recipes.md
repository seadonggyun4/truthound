---
title: Airflow 레시피
---

# Airflow 레시피

## Local 파일 Smoke Check

오케스트레이션 실행에서 Airflow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound_airflow import DataQualityCheckOperator

DataQualityCheckOperator(
    task_id="smoke_check_users",
    data_path="/opt/airflow/data/users.parquet",
    rules=[{"column": "user_id", "type": "not_null"}],
)
```

## Gate A Downstream 태스크 On Quality

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 오케스트레이션 실행 개요

오케스트레이션 실행에서 Airflow, SQL, Keep을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 프로파일 Before Enforcing Rules

오케스트레이션 실행에서 Truthound, `DataQualityProfileOperator`, Run, DataQualityProfileOperator, DAG을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Add SLA 알림 Without Rewriting 태스크

오케스트레이션 실행에서 Attach을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 오케스트레이션 실행 개요

1. 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. connection-backed 소스 check
3. XCom and 결과 consumer 검증
4. 센서 or SLA gating
5. 오케스트레이션 실행에서 DAG-wide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
