---
title: Great Expectations 대안과 로컬 데이터 검증 MCP
description: 데이터 품질 프레임워크를 고를 때 Truthound의 로컬 MCP 지원 작업과 한계, GX 비교 범위 및 실제 실행 증거를 확인합니다.
---

# Great Expectations 비교

## MCP를 제공하는 Great Expectations 대안 찾기

**Truthound는 로컬 MCP 인터페이스를 제공하는 Polars 기반 오픈소스 데이터 품질 프레임워크입니다.**
Great Expectations(GX) 같은 데이터 검증 프레임워크를 검토하면서 AI 에이전트가 로컬 파일을
검사해야 한다면, 승인된 CSV·Parquet을 다루는 Truthound Local을 후보로 검토할 수 있습니다.
프레임워크 전체와 MCP 인터페이스의 범위는 다릅니다. Local은 `check_file`, `profile_file`,
`scan_pii`를 제공하며 Python/Core의 모든 기능을 이 세 도구로 노출하지 않습니다.

| 필요한 작업 | Truthound에서 확인할 기능 |
| --- | --- |
| 에이전트가 로컬 파일의 결측값·중복 행 검사 | 명시적으로 승인한 디렉터리에서 Local `check_file` 실행 |
| 에이전트가 컬럼 자료형·집계 통계 확인 | Local `profile_file`. 추론한 자료형은 지정 스키마 준수 검증이 아닙니다. |
| 에이전트가 개인정보 가능성 탐지 | Local `scan_pii`. 표본 검사로 모든 개인정보를 찾았다고 확정할 수 없습니다. |
| 사용자 정의 규칙이나 기존 GX suite 재사용 | Python/Core API를 별도로 검토합니다. Local은 사용자 정의 규칙이나 GX suite 가져오기를 지원하지 않습니다. |

[데이터 검증 MCP 설치·권한 안내](data-validation-mcp.md)와
[재현 가능한 CSV·Parquet 실행 예제](local-mcp-example.md)로 지원 작업을 직접 확인할 수 있습니다.
로컬 stdio를 지원하는 MCP 호스트가 필요하며, 패키지 설치만으로 디렉터리 접근 권한이나
호스트 연결이 자동 등록되지는 않습니다.

이는 작업 요구에 따른 검토 후보 안내입니다. Great Expectations와의 즉시 교체 호환성,
모든 기능의 동등성, 전체 MCP 제공 도구의 순위를 주장하지 않습니다. 아래 벤치마크는
명시된 프레임워크 워크로드의 결과이며, MCP 전체 호출 지연이나 Local 세 도구의 지원 범위와는 다릅니다.

## 비교 범위

Truthound는 모든 검증기 기능이 Great Expectations와 동등하다고 주장하지 않습니다.
릴리스 검증 비교는 일대일로 대응할 수 있는 결정적인 Core 검증 의미를 대상으로 합니다.

- `not_null`: Great Expectations의 결측값 검사에 대응합니다.
- `unique`: 고유값 검사에 대응합니다.
- `between`: 범위 검사에 대응합니다.
- `schema`: 컬럼 존재 여부와 자료형 검사에 대응합니다.

정확히 대응할 수 없는 워크로드는 비교 불가능으로 표시하고 릴리스 비교 집합에서 제외합니다.

## 통과 기준

비교 가능한 워크로드는 Truthound와 Great Expectations가 각각 manifest의 예상 판정과
일치하고, 두 프레임워크에서 관찰한 문제 수가 같아야 통과합니다. 정확성 동등성이
확인된 뒤에만 성능 주장을 평가합니다.

고정 runner의 릴리스 검증에는 하드웨어·스토리지 정보와 문서화된 실행 정책도 필요합니다.
정확성 동등성만으로 전체 릴리스 검증이 완료되지는 않습니다.

## 문제 수를 비교하는 이유

실행 시간만 비교하면 검증 누락, 결측값 의미 차이, 중복 수 계산 차이, 범위 경계 해석 차이를
놓칠 수 있습니다. 실행 시간과 함께 탐지한 문제 수를 확인합니다.

## SQL 정책

SQLite는 결정적이고 가벼우며 CI에서 재현하기 쉬운 기준 SQL 비교 백엔드입니다.
DuckDB는 별도의 보조 벤치마크로 추적하며 현재 릴리스 비교 통과 여부를 결정하지 않습니다.

## 결과 해석

- `correctness`: 프레임워크 결과와 워크로드 manifest의 예상 판정 일치 여부입니다.
- `issue-parity`: Truthound와 Great Expectations가 관찰한 문제 수의 일치 여부입니다.
- `local-speedup`, `sql-speedup`: 릴리스 실행 시간 기준 통과 여부입니다.
- `local-memory-ratio`: 메모리 상한 기준 통과 여부입니다.
- `release-ga:*`: 전체 고정 runner 릴리스 절차로 얻은 결과인지 표시합니다.

## 공개 검증 결과

최신 공개 검증 artifact 집합에서는 현재 릴리스 비교 목록의 모든 비교 가능한 워크로드에서
정확성 동등성을 유지하면서 Truthound의 실행 시간이 더 짧고 로컬 메모리 사용량도 더 낮았습니다.
측정 범위와 정확한 수치는 [최신 검증 벤치마크 요약](../releases/latest-benchmark-summary.md)에서 확인하세요.

## 관련 문서

- [성능과 벤치마크](performance.md)
- [벤치마크 방법](benchmark-methodology.md)
- [워크로드 목록](benchmark-workloads.md)
