---
title: dbt Generic Tests
---

# dbt Generic Tests

오케스트레이션 실행에서 Truthound, dbt, They을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
파일 or Python 오케스트레이션 code.

## Quick Model-Level Example

```yaml
version: 2

models:
  - name: test_model_valid
    tests:
      - truthound.truthound_check:
          arguments:
            rules:
              - column: id
                check: not_null
              - column: id
                check: unique
              - column: email
                check: email_format
              - column: user_uuid
                check: uuid_format
              - column: url
                check: url_format
```

## 컬럼-Level Convenience Tests

오케스트레이션 실행에서 Truthound을(를) 다루는 항목입니다:

```yaml
columns:
  - name: id
    tests:
      - truthound.truthound_not_null
      - truthound.truthound_unique

  - name: email
    tests:
      - truthound.truthound_email_format

  - name: age
    tests:
      - truthound.truthound_range:
          arguments:
            min: 0
            max: 150
```

## Supported Rule Families

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 `not_null`, `not_empty`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `unique`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `range`, `positive`, `non_negative`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `email_format`, `uuid_format`, `url_format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `in_set`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 `referential_integrity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Warning vs Error Behavior

오케스트레이션 실행에서 Truthound, dbt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

오케스트레이션 실행에서 `severity: warn`을(를) 다루는 항목입니다:

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```yaml
tests:
  - truthound.truthound_check:
      arguments:
        rules:
          - column: status
            check: in_set
            values: ["active", "inactive", "pending"]
      config:
        severity: warn
```

## Modern YAML Shape

오케스트레이션 실행에서 dbt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 `arguments`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 소스 Testing

오케스트레이션 실행에서 dbt을(를) 다루는 항목입니다:

```yaml
version: 2

sources:
  - name: raw
    tables:
      - name: users
        tests:
          - truthound.truthound_check:
              arguments:
                rules:
                  - column: user_id
                    check: not_null
                  - column: email
                    check: email_format
```

## Production Guidance

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 실패 Modes

- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
오케스트레이션 실행에서 YAML을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Pages

- 오케스트레이션 실행에서 Package, Setup을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 오케스트레이션 실행에서 Macros, Operations을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [문제 해결](troubleshooting.md)
