---
title: Prefect 플로우
---

# Prefect 플로우

오케스트레이션 실행에서 Truthound, Prefect, Flows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 플로우 설정

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 `FlowConfig`, FlowConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `QualityFlowConfig`, QualityFlowConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `PipelineFlowConfig`, PipelineFlowConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Quality 검증 플로우

```python
from prefect import flow, task
from truthound_prefect.tasks import data_quality_check_task

@task
def load_data():
    return ...

@flow(name="basic_quality_flow")
async def basic_quality_flow():
    data = load_data()
    result = await data_quality_check_task(
        data,
        rules=[{"column": "id", "type": "not_null"}],
    )
    return result
```

## 플로우 With 프로파일링

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task, data_quality_profile_task

@flow(name="full_quality_flow")
async def full_quality_flow():
    data = ...
    check_future = data_quality_check_task.submit(data, rules=[{"column": "id", "type": "not_null"}])
    profile_future = data_quality_profile_task.submit(data)
    return {"check": check_future.result(), "profile": profile_future.result()}
```

## 플로우 Decorators And Factories

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 `quality_checked_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `profiled_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `validated_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `create_quality_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `create_validation_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `create_pipeline_flow`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Recommended Usage Pattern

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Deployment Guidance

오케스트레이션 실행에서 Keep을(를) 다루는 항목입니다:

- 플로우 define 오케스트레이션
- blocks define reusable 설정
- 오케스트레이션 실행에서 Prefect을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 See, Deployment, Patterns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- [태스크](tasks.md)
- 오케스트레이션 실행에서 SLA, Hooks을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Deployment, Patterns을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
