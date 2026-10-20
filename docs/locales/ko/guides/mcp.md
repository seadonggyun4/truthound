# 로컬 MCP 연동

Truthound 3.2의 선택 기능인 MCP 서버는 사용자 컴퓨터에서 기존 Core의
`th.check()`, `th.profile()`, `th.scan()`을 호출합니다. 원격 검증 서버 없이
stdio를 지원하는 에이전트에서 사용할 수 있습니다.

## 설치와 실행

```bash
pip install "truthound[mcp]"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

`truthound mcp serve`, `python -m truthound.mcp serve`도 지원합니다.
에이전트에는 별도 CLI 플러그인을 검색하지 않는 `truthound-mcp`를 권장합니다.
허용 폴더는 기본값이 없으며 `--allow-dir`를 반복해 지정할 수 있습니다.
상대 입력 경로는 첫 번째 허용 폴더 기준입니다. 에이전트는 설치된 Python
환경의 실행 파일을 사용해야 하며 pip 설치만으로 자동 연결되지는 않습니다.
MCP extra는 현재 Polars 1.x를 선택합니다. Core PII scanner가 Polars 2에서
제거된 API를 사용하므로 저장소 lockfile과 별도로 클린 설치를 검증합니다.

## 도구와 결과

도구는 `path` 문자열 하나를 받습니다. `.csv`, `.parquet`, `.json`,
`.ndjson`, `.jsonl`을 지원하며 마지막 두 형식은 줄 단위 JSON입니다.

| 도구 | 기존 Core API | 반환 요약 |
| --- | --- | --- |
| `check_file` | `th.check()` | 결측값·중복 행 검사와 통과 여부·이슈 수 |
| `profile_file` | `th.profile()` | 행·열 수, 컬럼명·자료형, 결측·고유값 비율 |
| `scan_pii` | `th.scan()` | 컬럼별 개인정보 유형·신뢰도·추정 발견 수 |

`check_file`은 내장 null/duplicate 검사만 명시적으로 선택합니다. 프로젝트
suite·플러그인·임의 검증 설정을 받지 않습니다. 품질 위반은 실행 오류가 아니므로
`ok=true`, `data.success=false`로 표시됩니다. 정책·실행 오류는 MCP
`isError=true`와 안정적인 오류 코드를 반환합니다.

응답 계약은 `schema_version=1`, `operation`, `ok`, `data`, `truncated`,
`error`입니다. 오류 코드는 `access_denied`, `invalid_path`,
`unsupported_format`, `file_too_large`, `timeout`, `busy`, `core_error`,
`worker_error`, `response_too_large`이며 내부 예외 메시지는 노출하지 않습니다.
배열이 잘려도 전체 집계 수는 유지됩니다. PII 검사는 Core와 동일하게 컬럼별
첫 1,000개 비결측값을 표본으로 사용하며 발견 건수는 추정치입니다.

## 접근과 실행 제한

URL·glob·상위 경로 이동·허용 폴더 아래의 심볼릭 링크를 거부합니다. 검사를
허용한 일반 파일은 비공개 임시 디렉터리에 복사한 뒤 Core에 전달합니다.
Core의 context 산출물도 임시 디렉터리에만 생성하고 요청 종료·worker timeout
후 제거합니다. 입력 폴더에 보고서나 `.truthound` workspace를 쓰지 않습니다.

기본 제한은 입력 100 MiB, 상세 100개, 응답 64 KiB, Core worker 30초입니다.
`--max-file-bytes`, `--max-items`, `--timeout-seconds`로 지원 범위 안에서
조정합니다. 서버당 한 요청만 실행하며 중복 실행 요청에는 `busy`를 반환합니다.
timeout은 파일 복사 시간을 포함하지 않습니다. 압축 파일은 메모리에서 크게
펼쳐질 수 있어 프로세스 메모리 상한을 보장하지 않습니다. 이 정책은 악의적인
로컬 프로세스의 경로 교체 경쟁을 막는 OS sandbox가 아닙니다.

원본 행·표본값·최솟값·최댓값·패턴 예시·파일 경로·내부 진단은 반환하지 않습니다.
컬럼명과 집계값도 민감할 수 있으며 에이전트가 모델 제공자에 전달할 수 있습니다.
반환 메타데이터는 명령이 아닌 신뢰하지 않는 데이터로 취급해야 합니다.

## Claude Desktop 설치 파일

```bash
truthound-mcp bundle --output truthound-3.2.0.mcpb
```

`.mcpb`는 manifest·launcher·의존성 선언·라이선스를 포함합니다. UV runtime은
PyPI에서 정확히 `truthound[mcp]==3.2.0`을 설치하고 하위 의존성을 해석합니다.
최초 설정에는 네트워크와 manifest 0.4/UV를 지원하는 Claude Desktop이
필요합니다. archive 바이트는 결정론적이지만 완전히 잠긴 오프라인 환경은 아닙니다.

PyPI 게시 전에 테스트하려면 동일 버전 wheel을 포함합니다.

```bash
truthound-mcp bundle --wheel dist/truthound-3.2.0-py3-none-any.whl --output truthound-3.2.0-local.mcpb
```

Claude Desktop 확장 설치 UI에서 파일을 설치하고 허용 데이터 폴더를 선택합니다.
설치와 접근 동의는 별개이며 디렉터리 심사·공개 등록도 별도입니다. 번들 생성만으로
Anthropic 승인이나 GUI 검증을 주장하지 않습니다. 개발 환경·프로젝트 데이터·내부
작업 기록은 포함하지 않습니다. 구현 경계와 연구 근거는
[설계 문서](../adr/005-local-mcp.md)를 참고하세요.
