<div align="center">
  <img width="560" alt="Truthound Banner" src="assets/truthound_banner.png" />
</div>

# Truthound — Data Quality Workflow 3.x

Truthound는 데이터 품질 검증을 중심에 둔 워크플로우 프레임워크입니다. Core는 `TruthoundContext`, `ValidationRunResult`, 자동 suite 생성, 실행 계획 경계를 제공하고, Orchestration은 Airflow, Dagster, Prefect, dbt, Mage, Kestra 같은 실행 환경 안에서 같은 품질 검증 흐름을 반복 가능하게 만듭니다.

이 문서 포털은 상용 SaaS나 콘솔 제품을 소개하기보다, 오픈소스 Truthound가 데이터 품질 검증과 자동화된 워크플로우 안에서 어떤 역할을 하는지 설명합니다. 코드, CLI, API 이름은 정확성을 위해 원문 표기를 유지합니다.

## 계층별 Truthound

| 계층 | 담당 영역 | 먼저 볼 문서 |
|--------------|------------|-------------|
| **Truthound Core** | 검증 커널, 결과 모델, 리포터, Data Docs, 체크포인트, 프로파일링, 벤치마크 실행 경로 | [Core 시작하기](getting-started/quickstart.md) |
| **Truthound AI** | 선택형 제안 생성, 실행 분석, 승인 기록, 통제된 적용 흐름 | [Truthound AI](ai/index.md) |
| **Truthound Orchestration** | 스케줄러와 워크플로우 시스템 안에서 Truthound 실행 | [오케스트레이션 개요](orchestration/index.md) |
| **Truthound Workflow** | 데이터셋 저장소 운영 경계와 검토/증거 흐름 | 공개 MkDocs에서는 개념 수준만 다룹니다 |

## 시작 지점 선택

| 하고 싶은 일 | 시작 위치 |
|--------------|-----------|
| 거의 설정 없이 첫 검증을 실행 | [빠른 시작](getting-started/quickstart.md) |
| Core 흐름을 처음부터 따라가기 | [튜토리얼](tutorials/index.md) |
| Python 코드에서 Truthound 사용 | [Python API](python-api/index.md) |
| 터미널이나 CI에서 실행 | [CLI 레퍼런스](cli/index.md) |
| AI 제안과 분석 경계 이해 | [Truthound AI](ai/index.md) |
| Airflow, Dagster, Prefect, dbt, Mage, Kestra 연동 | [Truthound Orchestration](orchestration/index.md) |
| 전체 구조 이해 | [개념과 아키텍처](concepts/index.md) |

## Core가 먼저인 이유

Truthound Core는 런타임 계약, 벤치마크 근거, 결과 모델이 가장 엄격하게 검증되는 계층입니다.

- `ValidationRunResult`는 실행 결과의 표준 출력입니다.
- 자동 suite 선택은 불필요한 전체 실행을 줄입니다.
- planner/runtime 경계는 검증 실행을 예측 가능하게 만듭니다.
- `TruthoundContext`는 제로 설정 `.truthound/` 작업 영역을 관리합니다.
- 벤치마크 주장은 비교 가능한 Core 워크로드로 제한합니다.

## 검증된 Core 벤치마크 요약

<!--
FACT-CHECK LOCK, 2026-07-01:
이 숫자는 docs/releases/latest-benchmark-summary.md와 공개 release artifact set을
기준으로 한다. 로컬 .truthound/benchmarks/artifacts 디렉터리는 일부 원시
observation만 포함할 수 있으므로 전체 산출물 원천으로 쓰지 않는다.
-->

최근 고정 runner 벤치마크는 공개 release artifact set의 비교 가능한 8개 release-grade 워크로드에서 Truthound Core가 Great Expectations보다 빠르게 완료됐음을 보여줍니다. 로컬 속도 향상은 `1.51x`에서 `11.70x`, SQLite pushdown 속도 향상은 `3.69x`에서 `7.58x` 범위였습니다. 자세한 근거는 [최신 검증 벤치마크 요약](releases/latest-benchmark-summary.md)을 참고하세요.

## 계속 읽기

- [Core 시작하기](getting-started/index.md)
- [Truthound AI](ai/index.md)
- [Truthound Orchestration](orchestration/index.md)
- [릴리스 노트](releases/truthound-3.1.5.md)
- [3.0 마이그레이션](guides/migration-3.0.md)
