# 저장소 CI와 full Nightly 검증

PR의 contract 4개 shard, fault/e2e, lint ratchet과 최종 gate를 유지합니다.
MCP distribution의 9개 OS/Python 조합, SDK 필수·skip 금지 검증,
실제 wheel/sdist/UV 설치 검증도 유지합니다. Core `src/**`, 공통 pytest 설정,
lockfile 변경을 MCP 영향에서 제외하지 않습니다. PR 또는 MCP 통과만으로
릴리스의 full 검증을 대체하지 않습니다.

## 원래 full 집합의 정확한 분할

예약 Nightly는 기존 integration-only 범위입니다. 릴리스 검증에는
`run_full_suite=true`로 수동 실행합니다. 원래 marker 식
`contract or fault or integration or soak or stress or scale_100m or e2e`와
`--run-integration --run-expensive --run-soak`를 그대로 사용합니다.
이는 설정된 marker 집합이며 저장소 모든 테스트를 선택한다는 뜻은 아닙니다.
수집·실행 job은 같은 frozen dev/MCP 의존성을 설치합니다.

node ID는 scale → soak/stress → integration → fault/e2e → contract 순서로
단 하나의 lane에 배정됩니다. stress 경로에는 root conftest가 이미 soak를
붙이므로 marker별 독립 명령을 단순 나열하지 않습니다. contract는 기존 PR의
파일 단위 균형 배정 알고리즘으로 4개 shard에 나누고 나머지 네 목적별 lane은
별도로 실행합니다. 수집 집합과 분할의 합집합 일치·교집합 0을 먼저 검증합니다.

동시에 최대 3개 lane만 실행하고 fail-fast는 끕니다. 각 job의 파일시스템,
의존성과 fixture는 독립적이며 session fixture·메모리 내 서비스는 공유되지
않습니다. setup 비용은 반복됩니다. 파일·테스트 수 균형이 실행시간 균형을
보장하지는 않습니다. 유료 runner나 서비스 계정을 새로 만들지 않습니다.

## 시간·자원과 실행 증거

| Lane | Job 분 | 테스트 강제 종료 초 | 진단 stack 초 |
| --- | ---: | ---: | ---: |
| 각 contract shard | 35 | 120 | 60 |
| Fault/e2e | 25 | 180 | 90 |
| Integration | 35 | 180 | 90 |
| Soak/stress | 35 | 300 | 120 |
| 100M scale | 120 | 600 | 120 |

scale의 테스트별 600초는 기존 suite 실행 예시를 따르며 10개 테스트와 setup을
job 예산 안에 둡니다. 다른 예산은 수십 초 stress generator와 기존 fault
deadline의 여유를 둔 공학적 설정이지 측정된 성능 약속이 아닙니다. 원래 데이터
규모·assertion·더 엄격한 throughput/runtime 기준은 바꾸지 않습니다.
scale은 최소 논리 CPU 4개, 전체 RAM 16GB, 가용 RAM 12GB, 여유 디스크 10GB를
실제로 검사합니다. 단위는 GiB가 아닌 십진 byte입니다. 공개 Ubuntu 표준 runner의
명목 자원과 별개로 실제 가용량이 부족하면 실패합니다. 이 검사가 OOM을 보장하지는 않습니다.

테스트 시작 시 node ID를 출력하고 부분 NDJSON에 단계별 결과·시간을 즉시
기록합니다. JUnit, 최종 실행 manifest, 의존성 버전, 자원 관측과 수집 계획은
7일 보관합니다. faulthandler의 soft timeout은 stack만 출력하며,
별도의 pytest-timeout thread deadline이 프로세스를 강제 종료합니다.
강제 종료로 최종 JSON/JUnit이 없을 수 있으므로 부분 증거는 보존하되 gate는 실패합니다.

항상 실행하는 최종 gate는 collector와 모든 matrix job 성공, 동일 checkout SHA·
run ID·plan digest·의존성, 정확한 node ID와 setup/call/teardown, JUnit 일치를
요구합니다. 취소·누락·추가 실행·실패·xfail·예상치 않은 skip은 성공이 아닙니다.
pytest가 수집 중 평가한 source skip/skipif만 optional 예상 skip으로 기록하고
사유·실행 일치를 확인하며 통과 수와 별도 집계합니다. 필수 MCP·scale에는 이
예외를 허용하지 않습니다. runtime skip을 자동 허용하지 않습니다.
수집 중 module skip도 별도 보존하며 실제 실행으로 세지 않습니다.

취소된 기존 단일 실행은 9,477개를 선택했고 19% 이후 진행 출력이 없었습니다.
이는 관측성 부족의 근거이지 특정 테스트 hang·OOM·제품 결함의 확정이 아닙니다.
새 lane 로그·stack으로 실제 장시간 구간을 확인해야 합니다.

## 설계 근거와 한계

[GitHub workflow 계약](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax),
[runner 자원 표](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[pytest fault-handler](https://docs.pytest.org/en/stable/how-to/failures.html),
[pytest-timeout 종료 한계](https://github.com/pytest-dev/pytest-timeout),
[SRE 22장](https://sre.google/sre-book/addressing-cascading-failures/)의 관련 부분을
확인했습니다. 기존 PR manifest runner와 Depot의 backend coverage·timeout 패턴을
참고했지만 그대로 복사하지 않았습니다. 제한된 공학적 조사이며 체계적 문헌고찰이나
속도 개선 실험 완료를 주장하지 않습니다.
