# Truthound 3.3.2

Core 품질 계산과 기존 데이터 권한을 유지하면서 MCP 클라이언트의 도구 발견과 Claude Desktop 패키징을 수정합니다.

## 수정 사항

- 재귀 모델 도구 스키마에 MCP가 요구하는 최상위 객체 타입을 제공하고 중첩 검증 조건을 유지합니다.
- 빈 Desktop 작업공간·운영자 제어 설정의 미치환 문자열은 권한 없음으로 처리하며, 필수 데이터 경로의 미치환 문자열은 거부합니다.
- Local·Core Desktop 번들은 서로 다른 확장 ID와 작은 아이콘을 사용합니다.
- Local 플러그인 배포 ZIP에 Claude manifest를 포함합니다. 두 클라이언트 실행기를 각각 검증하며 manifest 버전이 다르면 패키징을 거부합니다.

## 설치

Codex·Claude Code와 별도 Claude Desktop 확장의 설치는 [Styxis Plugins](https://github.com/seadonggyun4/styxis-plugin#install)에서 시작합니다. Local 0.1.8과 Core 0.2.3은 공개 Core 3.3.2 런타임을 사용합니다. 현재 고정 소스 커밋은 카탈로그에 기록합니다. Code 플러그인 업데이트는 Desktop을 업데이트하지 않습니다. 설치만으로 데이터 접근을 허용하지 않습니다.

## 게시 및 검증

[PyPI 3.3.2](https://pypi.org/project/truthound/3.3.2/)는 커밋 `30a65a581b6dbd075f64e547f1a69c0957c24aab`의 [릴리스 검증](https://github.com/seadonggyun4/truthound/actions/runs/41954271544) 이후 게시되었습니다. [게시 워크플로](https://github.com/seadonggyun4/truthound/actions/runs/41961838401)는 다시 빌드하지 않고 검증된 파일을 사용했습니다. 공개 PyPI 해시는 다음과 일치합니다.

| 산출물 | SHA-256 |
| --- | --- |
| Wheel | `6ea961ec86d7bcc0a5baf8fa7a8521d3bd8638d627e10e1d1819f22fec7608cc` |
| sdist | `f51e7bd750ea28a469a2375e2cfa8f6d6b176361f9ad0319e8b22ead6505c763` |

회귀 검증은 재귀 스키마 동등성, 최상위 객체, 번들 ID, 선택 설정과 격리된 설치형 실행을 포함합니다. 플러그인 lock은 공개 PyPI 산출물을 사용하며 비교용 wheel을 대체 설치 소스로 사용하지 않습니다. 로컬 `3.3.1+desktop.3` 복구본은 미게시 빌드였습니다.

## 남은 제한

첫 Windows Python 3.12 Dask Arrow 스트림 검사는 네이티브 세션 초기화 중 시간 제한을 초과했습니다. 같은 커밋을 새 러너에서 재검증해 제한 변경 없이 214개 검사가 모두 통과했지만, 최초 초기화 지연의 원인은 아직 입증되지 않았습니다. 이번 패치가 이를 해결했다고 주장하지 않습니다.

MCP의 전체 Core 동등성은 여전히 미검증입니다. 도구 발견, 실행 성공, 데이터 품질 판정은 구분합니다. 실제 외부 환경 배포에는 별도의 설정·승인·검증이 필요합니다.
