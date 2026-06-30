---
title: Truthound Orchestration
---

# Truthound Orchestration

Truthound Orchestration은 Airflow, Dagster, Prefect, dbt, Mage, Kestra 같은 실행 환경에서 Truthound 데이터 품질 검증을 자연스럽게 실행하기 위한 오픈소스 연동 계층입니다. 각 플랫폼의 작업 단위와 설정 방식을 존중하면서도, 품질 검증 결과와 실행 계약은 Truthound Core와 같은 기준으로 유지합니다.

## 이 문서가 필요한 사람

- 스케줄러나 워크플로우 시스템 안에서 Truthound 검증을 반복 실행하려는 데이터 엔지니어
- Airflow DAG, Dagster asset, Prefect flow, dbt test, Mage block, Kestra task에 품질 검증을 붙이려는 팀
- 실행 결과를 CI, 알림, 아티팩트, 관측성 시스템으로 연결해야 하는 운영 담당자
- 플랫폼별 코드는 다르지만 품질 검증 계약은 하나로 유지하고 싶은 팀

## 지원하는 플랫폼

| 플랫폼 | 주요 경계 | 적합한 사용 사례 |
|--------|-----------|------------------|
| Airflow | Operator, Sensor, Hook | DAG 기반 배치 검증과 SLA 운영 |
| Dagster | Resource, asset, op | asset 중심 데이터 품질 실행 |
| Prefect | Block, task, flow | Python 중심 flow와 배포 구성 |
| dbt | generic test, macro | warehouse-native 검증 |
| Mage | block, `io_config.yaml` | 노트북형 파이프라인 검증 |
| Kestra | YAML task, script | 선언형 워크플로우 실행 |

## 공유 런타임 경계

모든 플랫폼 연동은 같은 공유 런타임 개념을 사용합니다.

- `create_engine(...)`과 `EngineCreationRequest`로 실행 엔진을 선택합니다.
- `PlatformRuntimeContext`와 `AutoConfigPolicy`로 플랫폼별 기본값을 정리합니다.
- `resolve_data_source(...)`로 파일, DataFrame, SQL 소스를 일관되게 해석합니다.
- `run_preflight(...)`와 `build_compatibility_report(...)`로 실행 전 호환성을 확인합니다.
- 결과 payload는 XCom, Prefect artifact, Dagster metadata, Kestra output 같은 플랫폼별 표면으로 변환됩니다.

## 먼저 읽을 문서

1. [시작하기](getting-started.md)
2. [플랫폼 선택](choose-a-platform.md)
3. [아키텍처](architecture.md)
4. [제로 설정](zero-config.md)
5. [호환성](compatibility.md)
6. [운영 준비](production-readiness.md)

오케스트레이션 문서는 Truthound를 다른 제품처럼 포장하지 않습니다. 핵심은 데이터 품질 검증을 여러 실행 환경에서 일관되게 자동화하는 것입니다.
