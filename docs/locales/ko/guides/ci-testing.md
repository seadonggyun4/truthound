# 저장소 CI와 full Nightly 검증

실계정 BigQuery·Snowflake·Redshift·Databricks fixture에는
`TRUTHOUND_CLOUD_DW_RUN_LIVE=1`, 기존 유료 OpenAI smoke에는
`TRUTHOUND_AI_RUN_LIVE_SMOKE=1`이 필요합니다. 별도 계정·비용 승인 없이 설정하지
않습니다. 이번 릴리스에서는 선택된 테스트를 제거하지 않고 `live_account`
SKIP/미검증으로 기록하여 필수 PASS와 분리합니다. 계정 없이 실행 가능한 생성기·설정
검증은 유지합니다. 기존 선택적 소비자 snapshot 검증도 없을 경우
`optional_consumer` 미검증으로 구분합니다. 로컬 라이브러리 누락이나 서비스
시작 실패, 예상하지 못한 runtime SKIP에는 이 예외를 적용하지 않습니다.
기존 수집 시점 선택적 SKIP(자격증명 기반 cloud store, Kafka/LocalStack
Testcontainers, 기본 비활성 Redis-stream 테스트)은 `other_optional_skipped`로
별도 보고하며 검증 완료로 표현하지 않습니다. 새 필수 실패나 runtime SKIP은 허용하지
않습니다.

integration VM의 Redis·Elasticsearch는 실제 로컬 서비스 컨테이너이며 mock으로
대체하지 않습니다. 기존 KMS/TMS mock fixture는 mock 검증임을 JUnit에 기록하며
실계정 검증으로 표현하지 않습니다. 시작·준비 실패는 FAIL입니다. SQLAlchemy
queue pool은 `pool_timeout`을 엔진에 전달하며 실제 SQLite 연결·쿼리로 검증합니다.

Elasticsearch 로그 fixture는 기존 100행을 유지하고 datetime 연산으로 초·분 경계를
넘어도 유효한 시각을 생성합니다. 정리는 literal 테스트 prefix로 이름을 먼저 조회한
뒤 일치하는 명시적 인덱스만 삭제합니다. 빈 값·wildcard prefix나 범위 밖 이름은
거부하며 Elasticsearch의 destructive-action 이름 보호 설정은 끄지 않습니다.
정리에 실패해도 fixture 종료는 실행하고, 정리 실패를 PASS로 숨기지 않습니다.

개발 검증은 PR·병합 큐·main push에서 실행하며 일반 작업 브랜치 push의 중복
실행은 제거합니다. 직접 push가 가능한 동안 main 검증은 유지합니다. 이전 PR
실행은 취소할 수 있지만 취소를 PASS로 기록하지 않습니다. contract 4개 분할과
fault/e2e는 유지하고 lint 3개 대상은 한 job에서 모두 검사합니다. 문서는
full/public strict 빌드·한국어·공개 제외 검사를 유지합니다.

MCP는 `verification/mcp_ci_gate.py`에 명시한 모든 필수 테스트 파일을
Linux Python 3.11에서 8개 비중복 분할로 실행합니다.
알고리즘 사례는 제거하지 않습니다. 소스·실행 증명 집계는 누락·중복·실패·skip을
거부합니다. 다른 환경에서는 프로세스 정리·권한·UTF-8·자원·영속성·승인 경계
검증을 수행하며 이를 전체 의미 동등성 검증으로 대체하지 않습니다. 기존 job의
15분 제한을 유지합니다.

wheel·sdist·MCPB 2종은 한 번 빌드합니다. 각 설치 job은 SHA256SUMS를 검사하고
동일 파일을 설치하며 MCPB 내장 wheel도 원본과 바이트 단위로 같아야 합니다.
기존 7개 새 환경 설치·실행기 검증은 모두 유지합니다. 개발 환경은 Linux
3.11/3.13과 Windows/macOS 3.11입니다. 주간·릴리스에서는
지원 OS/Python 9개 조합을 모두 검증합니다. OS 독립적
의미 검증 전체를 9번 반복하지 않습니다. 공통 소스·테스트·설정·의존성 변경은
보수적으로 MCP 영향에 포함합니다. 이 구성이 전체 Core 기능 동등성을 의미하지는
않습니다.

추가 OS·Python 환경의 경계 테스트는 각각 4개 비중복 분할로 실행합니다. 설치 전에 같은 환경의 분할을 집계하여 누락·중복·실패와
잘못된 검증 범위의 증거를 거부합니다. 테스트 범위와 15분 제한은 바뀌지 않으며
분할만으로 실패한 테스트를 통과한 것으로 처리하지 않습니다.

플랫폼 경계 검증은 전체 Linux 분할과 병렬로 실행하고 설치 검증과 job을 분리합니다.
각 job의 15분 한도는 유지합니다. 경계 테스트와 7개 설치가 하나의 한도를
소모하던 문제를 해결하며 테스트·환경을 제거하거나 반복하지 않습니다. 설치 job은
해당 환경의 경계 증거를 다운로드하고 최종 gate는 전체 테스트·빌드·플랫폼 계약·
설치를 모두 필수로 요구하여 실패·누락된 경계 검증을 우회하지 못하게 합니다.

`release-verify.yml`은 같은 SHA의 full Nightly·전체 MCP 호환성·strict 문서·기존
고정 runner 성능 gate·플러그인 설치·벤치마크 계약·lint ratchet을 모아 실행하며
게시하지 않습니다. `release-pypi.yml`은
정확히 같은 SHA에서 성공한 검증 run ID를 요구하고 해당 canonical 산출물을
다운로드하여 설치를 재확인한 뒤 게시합니다. 소스 재빌드와 전체 MCP 테스트의
중복 실행은 하지 않습니다. 게시 후 검사도 과거의 짧은 진단 제한 시간으로 전체
테스트를 재실행하지 않고 실제 릴리스 실행 증거를 확인합니다. 진단·실계정 검증은
수동 실행으로 유지하며 개발 필수 gate가 아닙니다.

## 원래 full 집합의 정확한 분할

Nightly는 기존 benchmarks/pdf extra와 `nightly-local` 개발 의존성 그룹의
OpenTelemetry SDK·Zstandard·pypdf를 설치합니다. Ubuntu는 Pango·Poppler·CJK
폰트를 설치하고 PDF smoke/render를 필수로 검증하며, 로컬 의존성 누락을 skip
면제로 바꾸지 않습니다.

dev 환경은 pytest 9를 사용합니다. native subtest는 부모의 setup/call/teardown과
분리하고 부모별 연속 순번을 기록하되 context·parameter 원문은 저장하지 않습니다.
부모 testcase 수와 부모+subtest JUnit 총계가 모두 일치해야 합니다. 실패·skip·xfail,
순번 누락·중복·다른 부모의 subtest는 거부하며, subtest를 무시하거나 별도의
최상위 선택 테스트로 중복 집계하지 않습니다. 수집 검증은 실행 전에 수행하여
finally hook이 pytest의 원래 missing-node 오류를 가리지 않게 합니다.

수집한 경로와 실제 import된 모듈 파일이 같은지 검사하고 소스 SHA-256을
기록합니다. 각 lane도 동일한 identity를 재검사하여 같은 파일명 때문에 다른
테스트가 대신 실행되지 않게 합니다. dev extra는 기본 압축 pipeline roundtrip에
필요한 LZ4 및 기본 키 저장소의 Argon2·cryptography를 명시적으로 설치합니다.
선택 의존성의 진단 정보 조회는 원래 테스트
결과를 지우지 않으며 full 수집·집계는 필수 MCP SDK 누락을 계속 거부합니다.

stress 지표는 중첩 property 조회가 가능한 재진입 잠금으로 일관된 summary를
만들고, 동시 기록 및 JSON/Markdown/HTML 보고서를 제한된 자식 프로세스에서
검증합니다. scale uniqueness fixture는 1억 행과 기존 300초 기준을 유지합니다.
stress 결과의 실패율은 early-stop 경로를 포함한 실제 완료·실패 작업 수로
계산하며 누락된 결과 속성 때문에 검사가 중단되지 않게 합니다.
Core HLL 검사가 아닌 정확한 Python set oracle이며, 기존 첫 1,000개 후보를
동일하게 사용하되 커지는 전체 set을 반복 복사하지 않습니다.
HEAD/NONE 메모리 sampling 시간 검사는 기존 5배 overhead 기준을 유지하면서
양쪽 동일 warmup 1회와 순서를 번갈아 측정하는 고정 7쌍의 중앙값을 사용합니다.
생성·sampling·collect 비용과 실제 행수를 확인하고 인위적으로 느린 경로가
거부되는 반례를 둡니다. cold-start가 아닌 warm 상대 overhead 검증이며,
통과할 때까지 반복하거나 별도 고정 runner 릴리스 성능 기준을 바꾸지 않습니다.

예약 Nightly는 주 1회 전체 설정 범위를 실행합니다. integration-only 수동 진단은
유지하되 릴리스 증거가 될 수 없습니다. 릴리스 검증 워크플로는
`run_full_suite=true`로 호출합니다. 원래 marker 식
`contract or fault or integration or soak or stress or scale_100m or e2e`와
`--run-integration --run-expensive --run-soak`를 그대로 사용합니다.
이는 설정된 marker 집합이며 저장소 모든 테스트를 선택한다는 뜻은 아닙니다.
수집·실행 job은 같은 frozen dev/MCP 의존성을 설치합니다.

node ID는 scale → soak/stress → integration → fault/e2e → contract 순서로
단 하나의 lane에 배정됩니다. stress 경로에는 root conftest가 이미 soak를
붙이므로 marker별 독립 명령을 단순 나열하지 않습니다. contract는 4개 shard를
사용하고 일반 파일은 기존 PR의 파일 단위 균형 배정을 유지합니다.
`test_mcp_algorithm_parity.py`의 독립 실행 가능한 파라미터 사례만 기존 MCP
매트릭스처럼 같은 네 shard에 균등 분산합니다. 큰 네이티브 동등성 모듈 하나가
한 lane에 몰리지 않게 하며 테스트·네이티브 계산·fixture 행·assertion·시간
제한은 제거하거나 축소하지 않습니다. 분할 허용 목록은 명시적이고 나머지 파일의
fixture 지역성은 유지하며 수집 순서가 계획을 바꾸지 않습니다. 나머지 네 목적별 lane은
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
7일 보관합니다. manifest runner는 checkout root를 Python import 경로에 넣어
별도 native worker도 부모와 동일한 테스트 함수를 import하게 합니다.
faulthandler의 soft timeout은 stack만 출력하며,
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


`dev` extra는 공유 설치형 서비스 fixture가 일반 quality 레인에서도 수집·실행되도록
MCP SDK를 포함합니다. `all` extra도 MCP를 활성화합니다. 새 기본 패키지 설치 검사는
SDK가 없는 환경에서의 동작도 계속 확인합니다.

벤치마크 CLI 작업은 소스와 게시된 wheel 측정에 `benchmarks` 런타임 extra를 사용합니다.
워크로드·판정·고정 runner 식별자·기준값은 유지합니다. 산출물을 기록한 뒤 릴리스
작업은 자신이 소유권을 기록한 임시 런타임만 제거합니다. run ID·시도·커밋·canonical
경로가 일치해야 하며 별칭이나 변경된 소유권 기록은 거부합니다. 다른 run의
디렉터리와 공유 캐시는 보존합니다.
