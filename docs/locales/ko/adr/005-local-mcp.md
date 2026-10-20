# ADR 005: 선택적 로컬 MCP 어댑터

상태: 3.2.0 구현 후보. 원격 서비스는 추가하지 않습니다.

## 결정과 근거

검증 커널은 MCP에 의존하지 않습니다. 외부 어댑터가 공식 Python SDK,
로컬 파일 정책, 별도 Core worker, 명시적 결과 투영을 조합합니다.
PyPI extra와 MCPB는 동일한 Truthound 버전을 사용합니다.

관련 7개 1차 자료를 범위가 정해진 엔지니어링 검토로 조사했습니다.
체계적 문헌고찰·유료 서적 전체 열람·논문의 성능 재현을 주장하지 않습니다.
자료 확인 날짜는 2026-10-20입니다.

| 자료와 열람 범위 | 적용 | 한계 |
| --- | --- | --- |
| [MCP tools 규격](https://modelcontextprotocol.io/specification/2025-11-25/server/tools), 구조화 응답·오류 절 | 품질 위반과 실행 오류 구분 | annotation은 권한 통제가 아님 |
| [공식 Python SDK](https://github.com/modelcontextprotocol/python-sdk), 서버·클라이언트 예제 | 프로토콜 재사용과 실제 stdio 테스트 | 모든 향후 규격 지원을 보장하지 않음 |
| [MCPB manifest](https://github.com/modelcontextprotocol/mcpb/blob/main/MANIFEST.md), UV·설정 절 | 폴더 선택과 버전 일치 배포 | 실제 Desktop GUI 검증은 별도 |
| [Parnas 1972](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html), 모듈 분해 논의 | 전송·정책·Core·패키징의 변경 경계 분리 | 성능 증거가 아닌 설계 원칙 |
| [Saltzer·Schroeder 1975](https://www.cs.virginia.edu/~evans/cs551/saltzer/), 보안 원칙 | 기본 거부와 요청별 접근 검사 | 애플리케이션 정책은 OS sandbox가 아님 |
| [Architecture Patterns with Python 2장](https://www.cosmicpython.com/book/chapter_02_repository), 의존성 역전 절 | 기존 Core 쪽으로 의존성 방향 유지 | 저장소 패턴을 MCP에 적용한 설계적 추론 |
| [MCP Safety Audit](https://arxiv.org/abs/2504.03767), 초록만 | 메타데이터를 명령으로 해석하지 않고 권한 축소 | 실험 재현·prompt injection 해결 주장 없음 |

## 대안과 검증

직접 JSON-RPC를 구현하면 규격 유지보수 책임이 중복됩니다. 원격 서비스는
인증·호스팅·데이터 전송 설계가 필요합니다. Core를 event loop에서 직접 실행하면
도구 검색과 취소가 막힙니다. 별도 worker와 임시 복사는 시작 비용을 늘리지만
worker timeout과 입력 프로젝트 상태 분리를 가능하게 합니다. 입력 파일 크기는
압축 해제 후 메모리 크기의 상한이 아닙니다. 악의적인 파일 경로 교체 경쟁을
완전히 방어한다고 주장하지 않습니다.

응답에서 표본값·경로는 제외하지만 컬럼명·집계도 민감할 수 있습니다. 실제 SDK
연결, Core 결과 일치, 경로 이탈·symlink·크기·timeout 오류, wheel/sdist 재설치,
MCPB manifest·launcher를 검증합니다. GUI 설치·디렉터리 승인 심사는 별개입니다.
