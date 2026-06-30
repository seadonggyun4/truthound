# 가이드

실무 운영 가이드에서 Truthound, Guides, Core을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `truthound`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 Truthound, Orchestration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
UI, jump to Truthound 워크플로우 documentation.

> 실무 운영 가이드에서 CLI, Looking, See, Reference을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 
> 실무 운영 가이드에서 API, Looking, Python, See, Reference을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## What Belongs Here

| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|--------------|
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | [검증기](validators/index.md) |
| 실무 운영 가이드에서 SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | [데이터 소스](datasources/index.md) |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| render output or persist 아티팩트 | 실무 운영 가이드에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | [프로파일러](profiler/index.md) |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Methodology, Workloads을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## 실무 운영 가이드 개요

- 실무 운영 가이드에서 Truthound, Host-native, Orchestration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Truthound, Repository을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Command, Reference을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Sequential, Tutorials을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Common Core Entry Points

```python
import truthound as th
from truthound.drift import compare

# Basic validation through the core kernel
run = th.check("data.csv")
print(f"Found {len(run.issues)} issues")

# Explicit validators when you need them
run = th.check("data.csv", validators=["null", "duplicate", "range"])

# Learn a baseline schema from trusted data
schema = th.learn("baseline.csv")

# Compare two datasets for drift
drift = compare("baseline.csv", "current.csv", method="auto")
```

## Guide Families

### 검증 and Data Access

- [검증기](validators/index.md)
- [데이터 소스](datasources/index.md)
- 실무 운영 가이드에서 Data, Masking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [개인정보](privacy.md)

### 실무 운영 가이드 개요

- [체크포인트 개요](checkpoints.md)
- [체크포인트 Family](checkpoint/index.md)
- [설정](configuration/index.md)
- 실무 운영 가이드에서 CI/CD을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [알림](notifications.md)

### Reporting, 아티팩트, and Persistence

- [리포터](reporters/index.md)
- 실무 운영 가이드에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Reporter, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [스토어](stores/index.md)

### 프로파일링 and Extended Core 워크플로우s

- [프로파일러](profiler/index.md)
- [성능](performance.md)
- [벤치마크 Methodology](benchmark-methodology.md)
- [벤치마크 Workloads](benchmark-workloads.md)
- 실무 운영 가이드에서 Great Expectations, Great, Expectations, Comparison을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [마이그레이션 to 3.0](migration-3.0.md)

## Suggested Reading Paths

### New adopter path

1. [검증기](validators/index.md)
2. [데이터 소스](datasources/index.md)
3. [리포터](reporters/index.md)
4. [체크포인트 개요](checkpoints.md)

### 플랫폼 team path

1. [설정](configuration/index.md)
2. [체크포인트 Family](checkpoint/index.md)
3. [스토어](stores/index.md)
4. [성능](performance.md)

### Extended 워크플로우 path

1. [프로파일러](profiler/index.md)
2. 실무 운영 가이드에서 Data Docs, Data, Docs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 Great Expectations, Great, Expectations, Comparison을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 Truthound, Orchestration을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Related Reading

- [시작하기](../getting-started/index.md)
- [튜토리얼](../tutorials/index.md)
- [레퍼런스](../reference/index.md)
- [개념 & 아키텍처](../concepts/index.md)
- 실무 운영 가이드에서 Truthound을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [Truthound 오케스트레이션](../orchestration/index.md)
- Truthound 워크플로우 documentation
