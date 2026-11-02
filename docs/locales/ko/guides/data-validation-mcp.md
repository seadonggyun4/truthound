---
title: CSV·Parquet 데이터 검증 MCP
description: Truthound Local MCP로 승인된 로컬 CSV·Parquet의 결측값·중복 행을 검사하고, 프로파일링과 개인정보 표본 탐지를 실행합니다.
---

# CSV·Parquet 데이터 검증 MCP

**Truthound Local은 로컬 CSV·Parquet 파일을 검사하는 오픈소스 데이터 검증 MCP 서버입니다.**
결측값과 중복 행을 검사하고, 컬럼 자료형·집계 통계를 프로파일링하며,
RAG 입력이나 평가 데이터셋 공유 전에 개인정보가 있을 가능성을 표본으로 탐지합니다.
세 도구는 로컬에서 실행하며 원본 행이나 개인정보 일치값 없이 크기가 제한된 요약을 반환합니다.
이 가이드는 공개된 Truthound **3.3.1**, Local 플러그인 **0.1.7**을 설명합니다.

## AI 에이전트가 어떤 데이터 검사를 할 수 있나요?

| 작업 | MCP 도구 | 결과와 한계 |
| --- | --- | --- |
| CSV의 결측값·중복 행 검사 | `check_file` | 내장 품질 검사 결과. 데이터 품질 실패는 전송 오류가 아니라 정상 실행의 결과입니다. |
| 분석 전 CSV·Parquet 프로파일링 | `profile_file` | 자료형 계열, 결측·고유값 비율과 집계 정보. 추론한 자료형은 지정 스키마와의 일치 검증이 아닙니다. |
| RAG 입력·공유 전 개인정보 탐지 | `scan_pii` | 표본 기반 추정 발견 수·신뢰도. 발견되지 않아도 개인정보가 없다고 확정할 수 없습니다. |

각 도구는 승인된 파일의 `path`를 받습니다. Local은 사용자 지정 스키마·규칙 인수를
받거나, 데이터를 수정하거나, 임의 SQL을 실행하거나, LLM 답변 품질을 평가하지 않습니다.
명시적인 스키마 규칙·드리프트 비교·마스킹은 별도 [Core 프로파일](mcp.md#core-installation)을
확인하세요. Core의 전체 MCP 기능 동등성은 검증되지 않았습니다.

## 로컬 설치와 연결

Python 3.11–3.13에서 공개 MCP 추가 의존성을 설치합니다.

```bash
pip install "truthound[mcp]==3.3.1"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

로컬 stdio를 지원하는 MCP 클라이언트가 위 실행 파일과 인수를 시작하도록 설정합니다.
설치만으로 클라이언트에 자동 등록되거나 폴더 접근 권한이 생기지 않습니다.
호스팅된 HTTP/OAuth 연결만 받는 클라이언트는 이 로컬 stdio 구성을 직접 사용할 수 없습니다.
[클라이언트 설정과 지원 형식](mcp.md#설치와-실행)을 확인하세요.

Claude Code에서는 기존 공개 마켓플레이스로 Local 플러그인을 설치합니다.

```bash
claude plugin marketplace add seadonggyun4/styxis-plugin
claude plugin install truthound@styxis-plugin
```

설정 스킬에 검사할 기존 절대 경로 디렉터리만 승인하도록 요청합니다.
[플러그인 설치·권한 안내](mcp.md#claude-code-styxis-marketplace)를 확인하세요.

## 무엇이 로컬에 남고, 무엇이 AI에 전달되나요?

서버는 명시적으로 허용한 디렉터리만 읽고 데이터 값을 호스팅된 검증 서비스로
업로드하지 않습니다. Local 도구 결과에는 원본 행·표본값·개인정보 일치값이 없습니다.
파일 경로·컬럼명·집계 요약은 호스트 모델 제공자에게 전달될 수 있습니다.
이를 신뢰할 수 없는 데이터로 취급하고 호스트의 개인정보 설정을 확인하세요.
설치만으로 데이터 접근 권한이 생기지는 않습니다.

개인정보 탐지는 표본을 검사하므로 민감정보를 놓칠 수 있습니다. 마스킹·클라우드 DLP,
파일 전체 감사나 법규 준수 인증이 아닙니다. Local 검사는 원본 데이터셋을 수정하지 않습니다.

## 자주 묻는 질문

### SQL을 쓰지 않고 로컬 CSV의 결측값·중복을 검사하는 MCP가 있나요?

Truthound Local의 `check_file`은 승인된 파일에 내장 검사를 실행합니다.
에이전트는 SQL이나 사용자 지정 검증기를 생성하는 대신 파일 경로를 전달합니다.
실행이 성공해도 반환된 데이터 품질 결과를 별도로 확인합니다.

### 표본 행을 반환하지 않고 Parquet을 프로파일링할 수 있나요?

`profile_file`로 크기가 제한된 집계 요약을 얻습니다. 표본 행이나 범주값은 반환하지 않습니다.
자료형 추론만으로 지정 스키마와의 일치가 입증되지는 않습니다.

### RAG나 평가용 데이터셋을 준비할 수 있나요?

세 Local 도구로 표 형태 데이터의 품질과 개인정보 표본을 검사합니다.
검색 관련성·라벨 정답 여부·환각·생성 답변의 의미적 품질은 평가하지 않습니다.

### 여기서 데이터 검증은 MCP 프로토콜 검사인가요?

이 가이드의 검증 대상은 데이터셋 내용입니다. Truthound Local은 MCP 프로토콜 적합성이나
JSON-RPC 서버 구현을 검사하는 도구가 아닙니다.

## 공식 출처와 실행 증거

- [공개 패키지: Truthound 3.3.1](https://pypi.org/project/truthound/3.3.1/)
- [Official MCP Registry 레코드](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)
- [공개 소스와 라이선스](https://github.com/seadonggyun4/truthound)
- [릴리스·검증 기록](../releases/truthound-3.3.1.md)
- [설치된 Local의 실행·권한 검사](https://github.com/seadonggyun4/styxis-plugin/blob/main/evaluation/results/release-3.3.1/default-e2e.json)

실행 기록은 실제 설치를 검사한 결과이며 모든 검색 엔진이나 AI가 추천한다는 증거는 아닙니다.
Registry 게시·웹 색인·모델 추천은 별도로 확인합니다.
