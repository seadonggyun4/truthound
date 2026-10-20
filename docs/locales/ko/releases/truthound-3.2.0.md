# Truthound 3.2.0

Nightly는 native PDF 의존성과 실제 로컬 Redis·Elasticsearch 서비스를 사용합니다.
SQLAlchemy queue pool 초기화의 `pool_timeout` 전달을 수정했고 native pytest
subtest도 원래 collection 오류를 가리지 않고 별도로 집계합니다.
OTEL SDK가 설치된 store tracing의 context manager 누락도 수정했으며 실제 SDK의
in-memory exporter로 예외 전달과 span 종료를 검증합니다. 실계정 warehouse·
유료 OpenAI 검증과 없는 선택적 소비자 snapshot은 SKIP/미검증으로 필수 PASS와
구분합니다. 실계정 검증을 완료했다고 주장하지 않으며 KMS/TMS의 기존 mock 결과도
mock 검증임을 명시합니다.

선택적 `truthound[mcp]` extra와 로컬 stdio 도구 `check_file`, `profile_file`,
`scan_pii`, Claude Desktop용 MCPB 생성 기능을 추가합니다. 기존 Core의 품질 계산과
공개 Python 시그니처를 재사용합니다. 제한과 데이터 처리 방식은
[MCP 가이드](../guides/mcp.md)를 참고하세요.

새로운 선택적 연동 표면을 minor 버전으로 구분합니다. `<3.2`에 고정한 소비자는
별도 호환성 검토 후 갱신해야 합니다. 이 문서는 제품 변경을 설명하며 Claude Desktop
마켓플레이스 승인과 번들 서명은 패키지 게시와 별개입니다. PyPI 버전 고정 MCPB와
wheel 포함 MCPB를 제공하며 최초 의존성 해석에는 네트워크가 필요합니다.
기본 패키지와 MCP extra는 기존 Core PII scanner API와의 호환성을 위해 Polars 1.x를
요구합니다. 이전의 상한 없는 기본 의존성은 Polars 2를 선택해 제거된 streaming 인자에서
실패할 수 있었습니다. 이제 wheel/sdist 기본 설치에서도 MCP SDK 없이 Core 검증·프로파일·
PII API를 실제 호출해 확인합니다.

요청 취소·stdin EOF·worker timeout 시 worker 종료·회수, 임시 입력 제거,
busy 잠금 해제를 연결합니다. 초기화 응답에 설정된 안전 지침을 포함합니다.
필수 연동 CI는 Core 소스만 바뀌어도 실행되며 SDK 부재·0개 수집·skip을
실패로 처리합니다. 새 wheel/sdist와 UV launcher의 실제 Python 3.11–3.13 및
의존성 버전을 기록합니다. 프로세스 메모리 상한·OS sandbox·병적인 커널이나
파일시스템 hang의 보편적인 종료 시한을 보장하는 것은 아닙니다.

MCP 프로파일의 `dtype`은 허용 목록의 타입 계열만 반환합니다. Enum 범주값·중첩
타입 상세·시간대·정밀도 매개변수는 제외하고 알 수 없는 문자열은 `Unknown`으로
바꿉니다. Core Python의 전체 자료형 정보는 유지합니다. 실제 stdio와 새 설치
런타임에서 Enum Parquet의 범주값 비노출을 회귀 검증합니다.

릴리스 검증은 기존 full Nightly marker 집합을 contract, fault/e2e,
integration, soak/stress, 실제 자원을 확인하는 scale lane으로 중복 없이 나눕니다.
최종 gate는 동일 SHA의 실제 실행 manifest와 JUnit을 대조합니다.
[저장소 CI 가이드](../guides/ci-testing.md)를 참고하세요.
README에는 버전이 일치하는 공식 Registry 메타데이터의 소유권 marker도 포함합니다.
유효한 메타데이터와 실제 Registry·marketplace 게시 완료는 구별합니다.

릴리스 CI는 수집 경로와 실제 import된 모듈의 identity·소스 해시를 대조하여
동일한 파일명 때문에 다른 테스트가 실행되지 않게 합니다. SDK 진단 정보 누락이
원래 결과를 지우지 않으며 full 검증은 SDK를 계속 필수로 요구합니다. Windows의
Registry 텍스트 읽기는 UTF-8을 명시합니다. LZ4 명시 설치, stress summary의
재진입 잠금과 exact-set scale fixture의 제한된 후보 추출은 기존 assertion과
행수·시간 기준을 유지하는 테스트 인프라 수정입니다.
