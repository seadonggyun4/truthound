---
title: Mage 레시피
---

# Mage 레시피

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Validate A Dataset Before A Load

```python
from truthound_mage import CheckBlockConfig, CheckTransformer


def transform(df, *args, **kwargs):
    result = CheckTransformer(
        config=CheckBlockConfig(
            rules=[
                {"column": "id", "check": "not_null"},
                {"column": "email", "check": "email_format"},
            ]
        )
    ).execute(df)
    return result.result_dict
```

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Generate A 프로파일 For 오퍼레이터 Review

오케스트레이션 실행에서 `ProfileTransformer`, ProfileTransformer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Gate Downstream Work With A 센서

오케스트레이션 실행에서 `QualityGateSensor`, `DataQualitySensor`, QualityGateSensor, DataQualitySensor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Route Hard 실패 And Soft Warnings Differently

오케스트레이션 실행에서 Combine을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 `CheckTransformer`, CheckTransformer을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 SLA을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Keep Shared 설정 In One Place

오케스트레이션 실행에서 `CheckBlockConfig`, Create, CheckBlockConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 `ProfileBlockConfig`, `SensorBlockConfig`, ProfileBlockConfig, SensorBlockConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
드리프트 across 파이프라인.

## Related Pages

- 오케스트레이션 실행에서 Project, Layout을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `io_config.yaml`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [문제 해결](troubleshooting.md)
