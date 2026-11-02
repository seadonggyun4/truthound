# Truthound 3.3.1

이 패치는 로컬 MCP tool 설명과 JSON Schema 필드 설명을 개선합니다.
`check_file`, `profile_file`, `scan_pii` 이름은 유지합니다. 검증 기능,
디렉터리 승인, 원문 공개 권한, 네트워크 전송을 추가하지 않습니다.
결측·중복 검사, 타입·집계 프로파일링, 표본 기반 개인정보 탐지 범위는 같습니다.

## 설치와 게시

릴리스 후보는 3.3.1입니다. 실제 게시는 같은 커밋의 Release verification과
Release PyPI 통과 및 새로운 PyPI 설치로 판정합니다. 문서나 커밋만으로
배포 완료를 뜻하지 않습니다. Registry 게시는 별도 단계입니다.

```bash
python -m pip install "truthound[mcp]==3.3.1"
```

후보 검증 동안 frozen Local plugin은 0.1.6/Core 3.3.0을 유지합니다.
3.3.1의 PyPI 게시를 확인한 뒤 의존성을 갱신합니다. 기존 Core packaging
검사는 게시 전 정확한 후보 wheel로 실행합니다. Styxis 공개 catalog는
Local `truthound` 하나이며 새 제품이나 공식 directory listing을 추가하지 않습니다.

## 검증과 한계

같은 커밋의 전체 테스트, 네이티브 호환성, 한영 문서, wheel/sdist 출처 검증을
유지합니다. metadata 개선만으로 Agent의 실제 선택 정확성이 증명되지는 않습니다.
client/model 관측을 따로 기록하며 모든 Agent의 발견을 보장하지 않습니다.
Local summary tool 설치는 데이터 접근을 승인하지 않고 원문을 반환하지 않습니다.
경로·열 이름·집계는 host 모델에 전달될 수 있습니다.
