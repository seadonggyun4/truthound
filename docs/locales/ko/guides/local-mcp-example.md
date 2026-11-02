---
title: 로컬 데이터 검증 MCP — CSV·Parquet 재현 예제
description: 공개된 Truthound Local MCP 3.3.1과 합성 데이터로 결측·중복 검사, 집계 프로파일링, 개인정보 표본 탐지를 재현합니다.
---

# 로컬 데이터 검증 MCP 실행 예제

AI 에이전트가 승인된 로컬 CSV·Parquet의 결측값과 중복 행을 검사하고, 열을
프로파일링하거나, 입력·공유 전에 개인정보 가능성을 살펴보려면 **Truthound Local MCP**를
사용할 수 있습니다. 이 예제는 공개 패키지 **Truthound 3.3.1**의 실제 실행 기록입니다.
AI 답변이나 검색 순위를 흉내 낸 결과가 아닙니다. 설치와 Local/Core 기능 구분은
[데이터 검증 MCP 안내](data-validation-mcp.md)에서 시작하세요.

## 입력: 합성 데이터 세 행

[재현 스크립트](https://github.com/seadonggyun4/styxis-plugin/blob/main/scripts/verify_local_example.py)는
임시 폴더에 다음 CSV와 같은 내용의 Parquet을 만듭니다.

```csv
email,score
geo-example-721@example.com,1
geo-example-721@example.com,1
,3
```

모두 가상의 테스트 값입니다. 스크립트는 자체 MCP 프로세스에 이 임시 폴더만
허용하며 고객 파일을 읽거나 데스크톱 플러그인의 기존 승인을 바꾸지 않습니다.
각 파일은 세 행·두 열이고, 중복 행 하나와 이메일 결측값 하나가 있습니다.

## 실제 도구와 결과

각 도구에는 절대 파일 경로인 `path`만 전달합니다. CSV와 Parquet 모두 MCP stdio로
다음 결과를 확인했습니다.

| 작업·도구 | 관측 결과 | 해석 |
| --- | --- | --- |
| 결측·중복 검사 — `check_file` | `ok: true`, 품질 `success: false`, 결측 검사 1개·중복 검사 1개의 문제 | 도구 실행은 완료됐지만 데이터 품질은 실패입니다. |
| 집계 프로파일 — `profile_file` | 세 행·두 열, 이메일 결측 비율 33.3%, String·Int64 자료형 | 관측 자료형과 집계이며 지정 스키마와의 일치 검증은 아닙니다. |
| 개인정보 탐지 — `scan_pii` | 이메일 소견 1개, 추정 건수 2 | 처음 1,000개 비결측값으로 추정하며 전체 개인정보 감사가 아닙니다. |

[기계가 읽을 수 있는 실행 기록](../assets/examples/local-mcp-3.3.1.json)에 여섯 번의
제한된 결과가 있습니다. 허용 폴더 밖의 호출은 차단됐고 입력 파일은 바뀌지 않았습니다.
결과에는 이메일 원문이 없었습니다. 경로·열 이름·집계는 선택한 AI 호스트에 전달될 수
있으므로 호스트의 데이터 처리 정책도 확인하세요.

## 공개 패키지로 재현하기

다음은 POSIX 셸과 Python 3.11–3.13 기준입니다. 실행 전에 스크립트를 검토하세요.
최초 준비에는 의존성 다운로드가 필요하며 MCP 호출은 로컬에서 실행합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "truthound[mcp]==3.3.1"
git clone https://github.com/seadonggyun4/styxis-plugin.git
.venv/bin/python styxis-plugin/scripts/verify_local_example.py --output local-example.json
```

스크립트는 패키지 식별, 도구 조회, 여섯 번의 정상 실행, 예상된 품질 실패,
폴더 밖 접근 차단, 원문 비노출과 입력 보존을 확인합니다. 끝나면 임시 파일을 지웁니다.
실행 예제이며 독립 성능 비교나 AI 검색 추천의 증거가 아닙니다.
개인정보 검사는 민감정보를 놓칠 수 있습니다.

## 어떤 공개 프로젝트를 설치하나요?

| 구성 | 역할 | 공식 출처 |
| --- | --- | --- |
| Truthound Python 패키지·Core 3.3.1 | 이 Local MCP 호출이 사용하는 실행 엔진 | [공개 패키지](https://pypi.org/project/truthound/3.3.1/) |
| Truthound Local MCP·플러그인 0.1.7 | 세 가지 로컬 파일 도구. 사용자 지정 스키마·규칙 인수는 받지 않습니다. | [공개 Local 소스](https://github.com/seadonggyun4/truthound/tree/main/plugins/truthound) |
| Styxis 플러그인 마켓플레이스 | 공개 Local 소스를 가리키는 설치 카탈로그 | [공개 카탈로그](https://github.com/seadonggyun4/styxis-plugin) |

[Registry 레코드](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)는
서버와 설치 정보를 식별하며 파일 접근을 승인하지 않습니다.
[클라이언트 설정](mcp.md#설치와-실행), [릴리스 검증](../releases/truthound-3.3.1.md),
[공개 제품 소개](https://styxis-io.com/ko/open-source/truthound)도 확인할 수 있습니다.
