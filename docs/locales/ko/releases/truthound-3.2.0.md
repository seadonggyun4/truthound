# Truthound 3.2.0

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
