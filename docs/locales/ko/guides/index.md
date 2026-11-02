# 가이드

할 일이 정해졌다면 이 페이지에서 해당 작업의 문서로 이동하세요.
로컬 파일 검사부터 검증 규칙 설계, 결과 저장과 운영 자동화까지 공개 Core 기능을 안내합니다.

## 작업별 시작 지점

| 하고 싶은 일 | 먼저 볼 문서 |
| --- | --- |
| AI 에이전트로 CSV·Parquet의 결측·중복과 개인정보 표본 검사 | [데이터 검증 MCP](data-validation-mcp.md) · [실행 예제](local-mcp-example.md) |
| 검증기 선택과 검증 규칙 설계 | [검증기](validators/index.md) |
| 파일·SQL·웨어하우스 연결 | [데이터 소스](datasources/index.md) |
| 프로파일링과 규칙 학습 | [프로파일러](profiler/index.md) |
| 반복 실행과 정책 구성 | [체크포인트](checkpoint/index.md) · [체크포인트 개요](checkpoints.md) |
| 결과 표시·보고서·저장 | [리포터](reporters/index.md) · [Data Docs](datadocs/index.md) · [스토어](stores/index.md) |
| 성능 조정과 측정 근거 확인 | [성능](performance.md) · [벤치마크 방법론](benchmark-methodology.md) · [워크로드](benchmark-workloads.md) |

## 로컬 데이터 검증 MCP

Truthound Local MCP는 승인한 CSV·Parquet에 세 도구를 제공합니다.
`check_file`은 결측·중복 검사, `profile_file`은 자료형과 집계,
`scan_pii`는 개인정보 표본 탐지를 담당합니다. 설치는 데이터 접근 승인이 아닙니다.
사용자 지정 스키마 검증이나 LLM 답변 평가는 이 세 도구의 기능이 아닙니다.
[실제 입력과 결과를 재현](local-mcp-example.md)하거나
[상세 MCP 설정](mcp.md)을 확인하세요.

## 운영과 확장

- [설정](configuration/index.md), [CI/CD](ci-cd/index.md), [알림](notifications.md)
- [Data Docs](datadocs/index.md), [리포터 SDK](reporter-sdk.md), [개인정보](privacy.md), [데이터 마스킹](data-masking.md)
- [Great Expectations 비교](gx-parity.md), [3.0 마이그레이션](migration-3.0.md)

## 다른 진입점이 필요한 경우

- 처음부터 학습: [시작하기](../getting-started/index.md), [튜토리얼](../tutorials/index.md)
- 명령·API 조회: [CLI](../cli/index.md), [Python API](../python-api/index.md), [레퍼런스](../reference/index.md)
- 사람이 검토할 AI 제안·실행 분석: [Truthound AI](../ai/index.md)
- 스케줄러·워크플로우 연동: [Truthound Orchestration](../orchestration/index.md)
- 전체 구조: [개념과 아키텍처](../concepts/index.md)
