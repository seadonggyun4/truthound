# 로컬 MCP 연동

Truthound 3.2의 선택 기능인 MCP 서버는 사용자 컴퓨터에서 기존 Core의
`th.check()`, `th.profile()`, `th.scan()`을 호출합니다. 원격 검증 서버 없이
stdio를 지원하는 에이전트에서 사용할 수 있습니다.

## 설치와 실행

```bash
pip install "truthound[mcp]==3.2.0"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

`truthound mcp serve`, `python -m truthound.mcp serve`도 지원합니다.
에이전트에는 별도 CLI 플러그인을 검색하지 않는 `truthound-mcp`를 권장합니다.
허용 폴더는 기본값이 없으며 `--allow-dir`를 반복해 지정할 수 있습니다.
상대 입력 경로는 첫 번째 허용 폴더 기준입니다. 에이전트는 설치된 Python
환경의 실행 파일을 사용해야 하며 pip 설치만으로 자동 연결되지는 않습니다.
기본 패키지와 MCP extra는 Polars 1.x를 사용합니다. Core PII scanner가 Polars 2에서
제거된 API를 사용하므로 저장소 lockfile과 별도로 클린 설치를 검증합니다.

Linux/macOS/Windows CI는 Python 3.11–3.13을 검증합니다. Windows에서는
POSIX의 `bin/truthound-mcp` 대신 설치 환경의 `Scripts/truthound-mcp.exe`를
사용합니다. 아래는 클라이언트 형식에 맞춰 조정하는 수동 stdio 설정 예시입니다.

```json
{
  "mcpServers": {
    "truthound": {
      "command": "/absolute/path/to/venv/bin/truthound-mcp",
      "args": ["serve", "--allow-dir", "/absolute/path/to/data"]
    }
  }
}
```

로컬 stdio 서버이며 HTTP/OAuth 원격 endpoint를 제공하지 않습니다.
원격 connector가 필요한 클라이언트에는 이 설정을 직접 적용할 수 없습니다.
초기화 후 `profile_file`에 `{"path": "example.csv"}`를 전달하면 명시적으로
허용된 폴더의 파일을 검사하고 원본 행 대신 제한된 집계 메타데이터를 반환합니다.
`check_file`, `scan_pii`도 같은 입력 형식입니다.

## 도구와 결과

도구는 `path` 문자열 하나를 받습니다. `.csv`, `.parquet`, `.json`,
`.ndjson`, `.jsonl`을 지원하며 마지막 두 형식은 줄 단위 JSON입니다.

| 도구 | 기존 Core API | 반환 요약 |
| --- | --- | --- |
| `check_file` | `th.check()` | 결측값·중복 행 검사와 통과 여부·이슈 수 |
| `profile_file` | `th.profile()` | 행·열 수, 컬럼명·타입 계열, 결측·고유값 비율 |
| `scan_pii` | `th.scan()` | 컬럼별 개인정보 유형·신뢰도·추정 발견 수 |

`check_file`은 내장 null/duplicate 검사만 명시적으로 선택합니다. 프로젝트
suite·플러그인·임의 검증 설정을 받지 않습니다. 프로파일의 `dtype`은 허용 목록의
`Enum`, `List`, `Struct`, `Datetime` 같은 타입 계열만 반환하며 범주값·중첩 타입
상세·시간대·정밀도 매개변수는 제외합니다. 알 수 없는 자료형 문자열은 `Unknown`으로
표시합니다. Core Python API의 전체 자료형 메타데이터는 바꾸지 않습니다. 품질 위반은 실행 오류가 아니므로
`ok=true`, `data.success=false`로 표시됩니다. 정책·실행 오류는 MCP
`isError=true`와 안정적인 오류 코드를 반환합니다.

응답 계약은 `schema_version=1`, `operation`, `ok`, `data`, `truncated`,
`error`입니다. 오류 코드는 `access_denied`, `invalid_path`,
`unsupported_format`, `file_too_large`, `timeout`, `busy`, `core_error`,
`worker_error`, `response_too_large`이며 내부 예외 메시지는 노출하지 않습니다.
worker는 실행·임시 경로와 Windows CPU 아키텍처 식별자만 허용 목록으로 전달하며,
자격 증명이나 CPU 검사 우회 설정은 상속하지 않습니다.
배열이 잘려도 전체 집계 수는 유지됩니다. PII 검사는 Core와 동일하게 컬럼별
첫 1,000개 비결측값을 표본으로 사용하며 발견 건수는 추정치입니다.

## 접근과 실행 제한

URL·glob·상위 경로 이동·허용 폴더 아래의 심볼릭 링크를 거부합니다. 검사를
허용한 일반 파일은 비공개 임시 디렉터리에 복사한 뒤 Core에 전달합니다.
Core의 context 산출물도 임시 디렉터리에만 생성하고 요청 종료·worker timeout
시 회수합니다. MCP 취소와 stdin EOF도 worker에 전달하여 프로세스 종료·회수,
임시 복사본 삭제, busy 잠금 해제 순서로 정리합니다. 취소 후 후속 호출이
가능하지만 병적인 커널·파일시스템 hang까지 보편적인 종료 시한을 보장하지는 않습니다.
입력 폴더에 보고서나 `.truthound` workspace를 쓰지 않습니다.

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

게시된 버전은 [3.2.0 MCPB 다운로드](https://github.com/seadonggyun4/truthound/releases/download/v3.2.0/truthound-3.2.0.mcpb) → **Settings → Extensions → Advanced settings → Install Extension…** → 파일 선택·허용 폴더 설정 순서로 설치합니다. [공식 설치 안내](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)를 참고하세요. 이는 사용자 지정 확장 설치이며 Claude 공식 디렉터리 등재를 뜻하지 않습니다.

```bash
truthound-mcp bundle --output truthound-3.2.0.mcpb
```

`.mcpb`는 manifest·launcher·의존성 선언·라이선스를 포함합니다. UV runtime은
PyPI에서 정확히 `truthound[mcp]==3.2.0`을 설치하고 하위 의존성을 해석합니다.
최초 설정에는 네트워크와 manifest 0.4/UV를 지원하는 Claude Desktop이
필요합니다. archive 바이트는 결정론적이지만 완전히 잠긴 오프라인 환경은 아닙니다.
생성된 번들은 서명되지 않았으며 설치 시 host의 동의가 필요합니다.

PyPI 게시 전에 테스트하려면 동일 버전 wheel을 포함합니다.

```bash
truthound-mcp bundle --wheel dist/truthound-3.2.0-py3-none-any.whl --output truthound-3.2.0-local.mcpb
```

Claude Desktop 확장 설치 UI에서 파일을 설치하고 허용 데이터 폴더를 선택합니다.
설치와 접근 동의는 별개이며 디렉터리 심사·공개 등록도 별도입니다. 번들 생성만으로
Anthropic 승인이나 GUI 검증을 주장하지 않습니다. 개발 환경·프로젝트 데이터·내부
작업 기록은 포함하지 않습니다. 구현 경계와 연구 근거는
[설계 문서](../adr/005-local-mcp.md)를 참고하세요.

## OpenAI 로컬 플러그인과 저장소 마켓플레이스

OpenAI 플러그인은 Core와 별도인 버전 `0.1.1`이며 공개 PyPI의
`truthound[mcp]==3.2.0`과 고정된 UV dependency lock을 사용합니다.
로컬 데스크톱용이며 원격 서비스나 ChatGPT 웹 커넥터가 아닙니다.
UV를 설치해 데스크톱 앱의 PATH에서 사용할 수 있어야 하며 최초 설치에는
네트워크와 Python 3.11–3.13 런타임이 필요합니다.

마켓플레이스를 한 번 추가합니다.

```bash
codex plugin marketplace add seadonggyun4/truthound --ref main
```

필요하면 앱을 새로고침·재시작한 뒤 **Plugins → Truthound Plugins 소스 →
Truthound Local → Install → Set up**에서 읽기를 허용할 정확한 절대경로
폴더에 동의합니다. CLI 설치는 `codex plugin add truthound@truthound`입니다.
클라이언트 버전과 조직 정책에 따라 저장소 마켓플레이스 사용 가능 여부가 다릅니다.
패키지는 portable manifest와 구형 Codex용 호환 manifest를 함께 제공합니다.
**Set up**이 없는 클라이언트에서는 에이전트에게 패키지의 **Truthound setup**
스킬 실행을 요청하세요. 폴더 동의 절차를 생략하지 않습니다.
MCP 명령은 host별 인자 placeholder 치환에 의존하지 않고 플러그인 기준의 상대
`cwd`를 사용합니다. 0.1.1은 Codex 0.141.0에서 0.1.0의 시작 문제를 수정합니다.
OS별 CI는 실제 Codex 설치·정규화된 실행 설정과 frozen UV 실행을 함께 검사합니다.
Windows 검증기는 `.cmd` shim을 shell 평가하지 않고 npm의 Codex JS 진입점을
Node로 직접 실행합니다.

설치만으로 데이터 접근 권한은 생기지 않습니다. 세 도구는 정상 연결되지만
폴더 동의 전에는 `access_denied`를 반환합니다. 온보딩은 사용자 동의 후에만
패키지의 `permissions.py`를 실행하며 설정은 plugin·입력 폴더 외부의
`~/.truthound-openai/permissions.json`에 원자적으로 저장합니다.
다음 도구 호출부터 새 정책을 읽으므로 서버 재시작은 필요하지 않습니다.
재설정은 기존 목록을 교체하며 `permissions.py --revoke`로 철회합니다.
철회는 다음 호출에 적용되고 진행 중인 호출을 소급 취소하지는 않습니다.
누락·손상·안전하지 않은 설정·빈 목록은 접근을 거부합니다. POSIX에서는
파일 소유자·쓰기 모드를 검사하며 Windows는 사용자 폴더·NTFS 권한에
의존합니다. POSIX mode가 Windows ACL을 강제한다고 주장하지 않습니다.

Core의 MCP transport·service·파일 정책·worker를 재사용하고 로컬 동의 로더만
추가합니다. HTTP listener·업로드 도구·OAuth·자격증명·유료 모델 호출은 없습니다.
원시 값은 숨기지만 파일 경로·컬럼명은 host가 모델 제공자에게 보낼 수 있는
도구 메타데이터입니다. 선택적 `TRUTHOUND_OPENAI_CONFIG` 절대경로 override는
운영자가 지정한 별도 설정 파일이나 격리 테스트용이며 접근 권한 자체가 아닙니다.

### OpenAI 공식 디렉터리 승인 절차

저장소 배포는 **OpenAI 공식 공개 디렉터리 게시가 아닙니다.** 현재 플러그인은
OpenAI 승인이나 공개 listing ID가 없습니다. 공식 **Desktop only** 등재 절차는 다음과 같습니다.

1. 제출 조직 소유자가 OpenAI 담당자에게 로컬 MCP 파트너 자격·조직 allowlist를
   요청하고 승인 확인을 기다립니다.
2. 개발자·사업자 본인 확인을 완료합니다. 제출자는 조직 owner 또는
   **Apps Management Write** 권한이 필요합니다. 자격증명을 대화·ZIP에 넣지 않습니다.
3. 실제 게시자 소유의 개인정보·이용약관 URL, 심사 메타데이터, 양성 5개·음성 3개
   테스트와 접근 가능한 시연 영상 URL을 준비합니다. 현재 패키지는 법적 정책·영상
   심사 자료까지 완성된 제출물이 아닙니다.
4. 깨끗한 데스크톱 설치, 새 대화·기존 대화의 온보딩과 주요 workflow를 검증합니다.
5. [제출 포털](https://platform.openai.com/plugins)에서 승인된 로컬 MCP용
   **Skills only / Upload ZIP** 경로를 사용하되 실제 MCP 설정을 유지합니다.
   자동 검증 지적을 해결하고 게시자에게 정책 확인·동의를 받습니다.
6. 심사 제출 후 승인을 받은 버전에서만 **Publish plugin**을 선택합니다.
   실제 listing URL을 기록하고 공식 디렉터리의 Install을 별도 검증합니다.

[공식 패키징](https://developers.openai.com/plugins/build/plugins),
[데스크톱 제출 자격](https://developers.openai.com/plugins/deploy/desktop-only),
[심사·게시 요건](https://developers.openai.com/plugins/deploy/submission)을 참고하세요.
ZIP 생성·저장소 설치 성공을 allowlist·법적 동의·심사 승인·웹 지원으로 대신하지 않습니다.

## Registry 메타데이터와 설치

[공식 MCP Registry](https://registry.modelcontextprotocol.io/)에서 `io.github.seadonggyun4/truthound`를 검색할 수 있으며 [3.2.0이 게시되어 있습니다](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/3.2.0). 연동 클라이언트의 마켓플레이스에 표시되면 **Install**을 누르고 접근을 허용할 폴더를 지정합니다. Registry 등록만으로 모든 클라이언트의 마켓플레이스 노출·승인이 보장되지는 않습니다.

저장소의 `server.json`은 `io.github.seadonggyun4/truthound`, 기존 PyPI 패키지
`truthound` 버전 `3.2.0`과 로컬 stdio를 선언합니다. UV의 `--with`로 MCP
extra를 설치하고 기존 CLI 별칭을 실행합니다.

```bash
uvx --with "truthound[mcp]==3.2.0" truthound@3.2.0 mcp serve --allow-dir /absolute/path/to/data
```

GitHub namespace는 실제 저장소 소유자에 맞추며 README의 `mcp-name`도 정확히
일치해야 합니다. dev 환경의 `python verification/mcp_registry.py`가 버전 고정
공식 JSON Schema와 설치 계약을 검사합니다. 필수 디렉터리 입력에는 기본값이
없으며 검색 목록 등록이 파일 접근 권한을 부여하지 않습니다. UV 격리 환경은
무관한 CLI plugin 설치를 피하며, 수동 host 구성에서는 전용 `truthound-mcp`
실행 파일을 계속 권장합니다.

PyPI는 Python 산출물을 저장하고 pip/pipx/uvx는 이를 설치합니다. 공식 MCP
Registry는 검색용 메타데이터 목록이지 패키지 저장소나 원격 검증 서버가 아닙니다.
manifest 검증만으로 실제 등록·외부 marketplace 승인을 주장하지 않습니다.
Registry 인증은 GitHub CLI 로그인과 별개이며 새 OAuth grant는 사용자가 직접
승인합니다. 토큰·Trusted Publisher·계정을 자동 설정하지 않습니다.
[공식 PyPI 소유권 규칙](https://modelcontextprotocol.io/registry/package-types)과
[인증 가이드](https://modelcontextprotocol.io/registry/authentication)를 참고하세요.
