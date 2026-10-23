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

기본 도구 세 개는 `path` 문자열 하나를 받습니다. `.csv`, `.parquet`, `.json`,
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

## Core 프로파일: 아직 배포하지 않은 소스 기능

현재 소스에는 선택형 `core` 프로파일이 추가됩니다. **이미 게시된 3.2.0 wheel,
Registry 등록 정보, 버전이 고정된 마켓플레이스 플러그인에는 포함되지 않습니다.**
수정한 소스를 별도 환경에 빌드·설치한 뒤 사용해야 합니다. 기본 프로파일의 기존
도구 세 개와 버전 1 응답 계약은 바꾸지 않습니다.

```bash
truthound-mcp serve --allow-dir /absolute/path/to/data --profile core
```

다음 여덟 파일·조회 도구는 최초 호환 계약을 유지합니다. 보관 데이터용 추가
도구는 아래에서 설명하며 각 입력·출력은 명시적인 형식으로 검사합니다.
임의 Python 가져오기 경로, SQL, 셸 명령, 콜백이나 임의 키워드 인수는 받지 않습니다.

| 도구 | 기존 기능 연결과 동작 |
| --- | --- |
| `capabilities_list` | 현재 설치된 기능 목록·사용 가능 상태·제약의 페이지 조회 |
| `capability_describe` | 한 기능의 정확한 입력·출력 스키마와 효과 조회 |
| `validation_check` | 자동 검증 묶음 또는 명시적인 검증 규칙, 인라인 스키마와 실행 옵션 |
| `schema_learn` | 제약 추론 여부와 범주형 임계값을 지정하는 스키마 학습 |
| `data_read` | 기존 읽기 API의 표본 선택과 크기가 제한된 원본 행 페이지 |
| `data_mask` | 컬럼 선택, 가림·해시·가짜값 전략, 없는 컬럼의 엄격한 오류 처리 |
| `data_profile` | 기존 프로파일의 요약 또는 전체 통계 |
| `privacy_scan` | 기존 개인정보 검사의 요약 또는 전체 발견 정보, 추정 건수 유지 |

예를 들어 범위 검증 도구에는 다음 입력을 전달합니다.

```json
{
  "path": "example.csv",
  "checks": [{"validator": "between", "columns": ["age"], "min_value": 0, "max_value": 120}],
  "result_format": "summary",
  "evidence": "summary"
}
```

`checks`를 생략하면 Core 자동 검증 묶음을 사용합니다. 빈 배열은 검사 성공으로
취급하지 않고 거부합니다. 명시적인 식별자는 `null`, `not_null`, `unique`,
`duplicate`, `between`, `length`입니다. 인라인 `schema_spec`은 컬럼명을 키로
사용하며 각 항목의 `name`, `dtype`이 있어야 합니다. 키와 이름이 다른 경우나
알 수 없는 필드, 아직 검토하지 않은 정규식 패턴은 거부합니다. 정확한 입력 형식은
기능 설명 조회 도구로 확인합니다. 병렬 실행과 최대 worker 수, pushdown, 최소
심각도, 자동 스키마, 결과 형식, 예외 처리 여부, 재시도 수와 제외 컬럼을 전달합니다.
context는 명시적으로 일시적인 것이며 영구 프로젝트 context를 대신하지 않습니다.
자동 검증에 필요한 기준 스키마는 소유한 임시 작업공간에서 생성한 뒤 회수하며,
안전한 요약처럼 보이게 하려고 기준 학습 자체를 끄지 않습니다.
실제 계획·재시도·심각도 필터링은 Core가 수행합니다. 현재 병렬 실행 결과의 단계별
오류·건너뜀 집계가 충분하지 않아 품질을 통과로 추정하지 않고 `unknown`으로 표시합니다.
자동 검증 묶음도 대상별 실행 증거가 부족하므로 품질을 미확인으로 표시합니다.
명시적인 규칙이라도 컬럼이 없거나 제외됐거나 자료형이 맞지 않아 적용되지 않았다면
이슈가 없다는 이유로 통과시키지 않습니다. 적용 대상을 확인한 명시적 규칙은 실제
결과에 따라 통과 또는 실패를 반환합니다.

원본 행, 학습한 값 제약, `evidence="full"` 상세 결과는 폴더 권한 외에 운영자의
추가 허용이 필요합니다. 운영자가 `--profile core --allow-sensitive-results`로
재시작해야 하며 허용 전에는 파일을 읽기 전에 `approval_required`를 반환합니다.
모델이 도구 인수에 `approved: true`를 넣어도 승인되지 않습니다. 이 시작 옵션은
허용된 모든 파일의 민감한 값·상세 증거를 에이전트와 모델 문맥에 노출할 수 있도록
허용하므로 좁고 신뢰할 수 있는 폴더에만 사용해야 합니다. 일부 컬럼을 마스킹해도
나머지 컬럼이 안전해지는 것은 아닙니다. 원본 파일에 덮어쓰거나 외부로 내보내지 않습니다.

새 응답은 실행 상태와 데이터 품질 상태를 분리하고 형식이 지정된 데이터·오류·검사
범위·출처 정보를 제공합니다. 품질 위반 자체는 도구 실행 오류가 아닙니다. 실행 오류,
검사 없음, 불완전한 집계를 통과로 표시하지 않습니다. 실제 임시 복사본 바이트의
세션별 비밀키에 연결된 불투명 식별자를 기록하고, 콘텐츠 SHA-256은 운영자가 민감한
결과를 허용한 경우에만 공개합니다. 경로명이나 파일 상태만으로 식별하지 않습니다. 전체 상세
결과는 `tagged-v1` 표현입니다. 객체는 `{"$type":"object","entries":{...}}`로
감싸고 소수·날짜·시간·바이너리·유한하지 않은 실수에는 명시적인 태그를 붙입니다.
지원하지 않는 객체를 문자열로 바꾸지 않고 오류로 처리합니다.
JavaScript의 정확한 정수 범위를 넘는 값에는 십진 문자열 태그를 사용합니다. 원본 행으로
변환할 때 정밀도가 손실될 수 있는 나노초 날짜·기간과 시각 컬럼은 중첩 컬럼까지
검사해 오류로 처리하며 조용히 반올림하지 않습니다. 허용한 상세 결과에는
원본 값과 임시 파일·context 경로가 포함될 수 있습니다. 기존 요약 도구는 민감한
결과를 허용해도 계속 값을 가립니다.

원본 읽기와 마스킹은 `offset`, `limit`으로 페이지를 지정합니다. 기본 50행,
최대 100행이며 운영자의 상세 항목 제한보다 커질 수 없습니다. 요청마다 별도
복사본을 사용하므로 여러 페이지를 합칠 때는 출처 해시가 같은지 비교해야 합니다.
무작위 표본과 가짜값 마스킹은 결과가 달라질 수 있어 후속 페이지를 제공하지 않고
0이 아닌 시작 위치도 거부합니다. 안정적인 핸들은 아래의 보관 데이터 도구를 사용합니다.
전체 결과가 응답 크기 제한보다 크면 오류를 반환하며 일부를 자르고 전체 결과라고
표시하지 않습니다. 기존 파일·시간·동시 실행 제한과 메모리 한계는 그대로 적용됩니다.

### 승인된 네이티브 데이터소스 팩토리

`sources_open_binding`은 `binding_id`와 `idempotency_key`만 받습니다. Core 프로파일,
민감 결과 허용, 비공개 제어 디렉토리가 필요합니다. 운영자는 아래의 설치 확장 바인딩
계약으로 플러그인을 설치·고정하고, 인자 없는 팩토리를 capability로 노출한 뒤
비공개 `bindings.json`에 다음 매핑을 추가합니다. 저장소나 배포물에 넣지 않습니다.

```json
{"sources":{"orders":"orders-factory"}}
```

이는 기존 `plugins` 설정에 합칠 조각이며 완성된 바인딩 파일이 아닙니다.
고정된 플러그인 하나만 `orders-factory`를 노출해야 합니다. 팩토리는
`name="orders"`인 동기 Core `BaseDataSource`를 반환해야 합니다. 예를 들어 검토한
플러그인은 `SQLiteDataSource(table="orders", database=operator_selected_path,
config=SQLiteDataSourceConfig(name="orders"))`를 반환할 수 있습니다.
안정 포트인 `register_capability`로 팩토리를 등록합니다. 의존성과 자격증명 참조는
운영자가 구성합니다. MCP는 플러그인 설치, 연결 문자열·import 경로·설정 입력,
임의 환경변수 전달을 제공하지 않습니다. 자격증명을 플러그인 설정·이름·메타데이터·
데이터 컬럼·로그에 넣지 마세요.

```json
{"binding_id":"orders","idempotency_key":"orders-session-001"}
```

첫 호출은 플러그인을 import하지 않고 승인 계획을 반환합니다. 기존 운영자 CLI로
정확한 계획을 승인한 뒤 같은 요청을 반복합니다. 승인은 해당 세션 핸들의 신뢰된
확장 실행·네이티브 연결·읽기·종료에 적용됩니다. **OS 샌드박스나 읽기 전용 SQL
보증이 아닙니다.** 팩토리를 검토하고 최소 권한 DB 계정을 사용해야 합니다.
로컬 SQLite 검증으로 원격 공급자나 유료 서비스의 검증을 대신하지 않습니다.

반환된 소스는 `sources_inspect`, `sources_validate_connection`,
`sources_validate_columns`, `sources_check_size_limits`, `sources_sample`,
`sources_validate`에 사용할 수 있습니다. 검증은 실제 상주 소스를 Core의
`source=` 경로에 전달하며 DataFrame 복사본으로 대체하지 않습니다. 연결 설정 노출을
막기 위해 바인딩 소스의 조회 결과는 `config=null`입니다. 데이터 최신성·캐시·엔진은
네이티브 공급자의 의미를 따르며 불변 스냅샷이나 자동 재연결 서비스가 아닙니다.
라이브 바인딩 검증의 `input_sha256`·`input_identity`는 미확정으로 유지합니다.
소스 핸들의 리소스 해시는 바인딩 이름을 나타내며 DB 행의 해시가 아닙니다.
비동기 전용 공급자와 임의 프로토콜 객체는 받지 않습니다.

같은 키의 성공 재호출은 기존 유효 핸들을 반환하며 새 연결을 만들지 않습니다.
실행 시작 후 팩토리 실패·취소는 `outcome_unknown`으로 기록해 같은 키로 재실행하지
않습니다. 해제·만료된 핸들은 재호출로 복원되지 않습니다. 해제·세션 종료 시 소스를
먼저 닫고 플러그인을 종료합니다. 시간 초과나 제한된 정리 시간 초과 시 소유 worker를
강제 종료할 수 있으며 공급자 수준 정리는 보장하지 않습니다. 바인딩·패키지 identity·
정책 변경은 다음 확인에서 소스와 파생 산출물 권한을 회수하며 TTL이 유휴 worker의
수명을 제한합니다. 별도 감시 프로세스를 시작하지 않습니다.

### 보관 데이터·검증기 설정·산출물

같은 선택형 프로파일에서 다음 기능도 제공합니다.

| 도구 | Core 연결과 동작 |
| --- | --- |
| `validators_list`, `validators_describe` | 실제 등록 검증기 목록·상속 설정·값 표현 규칙·고정된 소스 구문 트리 식별자 |
| `assets_capture`, `assets_inline` | 파일 또는 형식을 지정한 메모리 테이블의 변경 불가 Arrow 스냅샷 |
| `assets_sqlite` | 일관된 비공개 백업을 통해 Core SQLite 데이터소스로 테이블 읽기 |
| `sources_open_sqlite` | 자원 한도에 포함되는 비공개 백업 위의 실제 상주 SQLite 데이터소스와 SQL 엔진 |
| `sources_open_duckdb` | WAL 없는 크기 제한 비공개 복사본 위의 네이티브 DuckDB SQL 소스·읽기 전용 물리 테이블·외부 접근과 확장 차단 |
| `streams_open`, `streams_next`, `streams_inspect` | 네이티브 CSV·JSONL·Parquet·Arrow 파일 스트림 커서·전체 배치·읽기 상태 |
| `assets_validate` | 검증기별 설정·참조 테이블·보관 스키마를 사용한 Core 검증과 정식 결과 산출물 |
| `assets_page`, `assets_sample`, `assets_mask` | 안정적인 무손실 행 페이지·시드 지정 Core 표본·마스킹 |
| `assets_profile`, `assets_scan` | 전체 Core 프로파일·개인정보 검사 결과의 비공개 보관 |
| `schemas_learn`, `schemas_create`, `schemas_inspect`, `schemas_export` | 학습·명시적 스키마와 승인된 조회·소유 JSON 및 원래 YAML 내보내기 |
| `profilers_catalog`, `profilers_profile`, `profilers_generate_rules` | 고급 프로파일 전체 설정·분석 구성 요소·프로파일 기반 규칙 생성 |
| `artifacts_export`, `artifacts_read` | 소유한 IPC·Parquet·CSV·JSON·NDJSON 산출물과 크기 제한 바이트 읽기 |
| `artifacts_render`, `artifacts_datadocs` | 정식 결과 재진입, JSON·Markdown·콘솔·HTML 리포터와 영·한 검증 Data Docs |
| `resources_describe`, `resources_release` | 세션 소유권·만료·무결성·현재 권한 확인과 파생 핸들 회수 |
| `contexts_open`, `contexts_inspect`, `contexts_initialize`, `plans_status` | 별도 작업공간 권한에 따른 설정 조회·승인된 초기화·실행 의도 상태 |
| `suites_create`, `suites_inspect`, `suites_validate` | 고유 검사 ID·스키마 핸들·태그·메타데이터·증거 및 심각도 정책을 가진 실제 커널 suite |
| `contexts_execute` | 승인된 소스 추적·기준 스키마·적용 suite·결과 및 문서 저장·메트릭·작업공간 목록 조회 |
| `sources_open_polars`, `sources_inspect`, `sources_sample` | 실제 상주 Polars 데이터소스·전체 네이티브 설정 및 메타데이터·Core 표본 추출 |
| `sources_validate_connection`, `sources_validate_columns`, `sources_check_size_limits`, `sources_validate` | 실제 연결·필수 열·크기 검사와 Core의 `source=` 실행 경로를 통한 검증 |
| `drift_compare` | 모든 공개 비교 메서드와 인수를 전달한 Core 데이터셋 비교 보고서 |
| `drift_catalog`, `drift_detect` | 구체 드리프트 탐지기 13종의 전체 생성자 설정과 두 Series 컬럼 비교 |
| `lineage_catalog`, `lineage_create`, `lineage_edit`, `lineage_query` | 네이티브 그래프 설정·불변 파생 편집·탐색 |
| `lineage_track` | 실제 Core의 소스·변환·검증·출력 수동 추적을 파생 그래프로 보관 |
| `lineage_load`, `lineage_export` | 승인된 로컬 네이티브 JSON 스냅샷 재진입과 Core JSON 산출물 |
| `lineage_impact`, `lineage_critical_paths`, `lineage_what_if_delete`, `lineage_schema_impact`, `lineage_dependency_chain` | 실제 Core 영향·의존성 분석 결과 보관 |
| `models_catalog`, `models_create`, `models_inspect` | 구체 ML 알고리즘 12종의 설정과 상주 수명주기 상태 |
| `models_fit`, `models_partial_fit`, `models_fit_predict`, `models_predict`, `models_score` | 기본 학습·지원되는 온라인 학습·학습과 예측·예측·이상 점수 |
| `models_detect`, `models_learn_rules`, `models_rules`, `models_diagnostics` | 두 입력의 ML 드리프트·규칙 학습·원래 모델 진단의 산출물 보관 |
| `models_rule_query` | 네이티브 학습 규칙 필터·suite/검증기 설정 변환, 자동 적용 없음 |
| `models_result_query` | 네이티브 예측 점수 순회·이상치 필터·임계값별 드리프트 컬럼 조회 |
| `schedules_preview` | 네이티브 cron 검증·일치·제한된 다음 시각 미리보기, 작업 등록 없음 |
| `models_ensemble_add`, `models_ensemble_remove` | revision을 확인하는 네이티브 편집과 제거한 모델의 복원 가능한 산출물 |
| `models_persist`, `models_resume` | 별도 승인된 비공개 모델 스냅샷과 새 세션 복원 |
| `jobs_submit`, `jobs_execute`, `jobs_status`, `jobs_result`, `jobs_cancel`, `jobs_recover` | 승인된 영구 검증 회차·결과 보존·불확정 lease 복구 |
| `schedules_register`, `schedules_status`, `schedules_disable` | 로컬 포그라운드 실행기를 통한 승인된 영구 cron 반복 예약·수명주기 |
| `storage_policy_register`, `storage_policy_status`, `storage_policy_disable` | 승인된 영구 시간·개수 자동 보존 정책 |
| `storage_retention_preview`, `storage_delete` | 네이티브 시간·개수 보존 정책 조회와 별도 승인된 정확한 대상 삭제 |
| `models_save`, `models_restore` | 네이티브 학습 상태 JSON 보관과 새 소유 모델 worker로 복원 |
| `schema_audit_contract`, `schema_audit_create`, `schema_audit_inspect` | 엄격한 네이티브 메타데이터 문서·전체 JSON Schema·정규화 해시 |
| `schema_audit_standards`, `schema_audit_implementation` | 표준·설계·카탈로그 감사의 전체 판정과 명시적인 미확인 범위 |

명시적 해제와 만료에서는 소유한 네이티브 reader를 먼저 종료·reap한 뒤
스냅샷을 삭제합니다. Windows의 SQLite 연결과 메모리 매핑된 Parquet·Arrow
스트림에도 같은 순서를 적용하며, 정리에는 새 읽기 권한이 아니라 세션 소유권을
사용합니다. 파일 삭제 실패는 성공으로 처리하지 않고 후속 정리를 위해 추적합니다.

파일 보관 후 받은 `data.resource_id`를 후속 도구의 `asset_id`로 전달합니다.
커널 suite는 같은 알고리즘·표시 이름을 반복해도 실제 검사 인스턴스를 구분합니다.
검사 메타데이터의 `_truthound_check_id`와 `_truthound_execution_incomplete`는
런타임 예약 필드이므로 사용자 메타데이터가 덮어쓸 수 없습니다. suite 증거 정책은
결과의 형식 표시에만 쓰지 않고 실제 검증기에 적용하며 factory의 보관 객체를
변경하지 않습니다. 심각도 필터는 실제 실행 실패·의존성 스킵을 성공으로 바꾸지 않습니다.
SQL fallback 실패나 쿼리 결과 누락·불일치도 빈 성공으로 처리하지 않습니다.
정식 결과 재진입은 구조화된 상세·실패 행 레코드·예외 및 재시도 증거를 보존합니다.
단, 네이티브 행 레코드 직렬화에는 원래 DataFrame의 물리적 dtype 스키마가 없으므로
그 정보는 보관된 타입 지정 자산을 사용해야 합니다.

`sources_open_polars`는 `asset_id`와 선택적 전체 `PolarsDataSourceConfig`인
`config`를 받아 실제 상주 데이터소스를 엽니다. 반환 ID를 `source_id`로 사용합니다.
`sources_validate`는 `asset_id`를 바꾸는 대신 실제 네이티브 객체를 Core의 `source=`로
전달합니다. 연결 검사 결과가 거짓이면 `connected=false`로 남고 크기 제한 예외는
실패입니다. 표본 추출은 새 상주 연결이 아닌 보관 Arrow 자산을 반환합니다.
입력·소스 회수 시 파생 자원과 소유 worker도 회수하고 정상 종료는 원래 context manager의
정리를 실행합니다. timeout·EOF·프로토콜 오류에서는 해당 소유 프로세스만 무효화합니다.
소스 핸들 해시는 설정의 식별자이지 별도 데이터셋 스냅샷 해시는 아닙니다.

`sources_open_sqlite`는 `path`, 리터럴 `table`명과 선택적 `config`를 받습니다.
DataFrame 대체물이 아니라 일관된 비공개 백업 위의 실제 `SQLiteDataSource`를 엽니다.
SQL·기본 설정을 전달하지만 `database`는 해당 백업에 고정하여 호출자가 URI를 바꿀 수
없습니다. 백업은 파일·세션 크기 한도에 포함되며 그 해시는 실제 DB 바이트를 식별합니다.
rollback journal DB와 실제 테이블만 받으며 WAL·뷰·모델이 작성한 SQL은 거부합니다.
소스를 회수하면 소유 백업도 회수합니다. `assets_sqlite`는 별도의 메모리 변환 도구입니다.
다른 SQL·클라우드·NoSQL·비동기 제공자의 동등성까지 증명하는 것은 아니며 네이티브
설정은 운영체제 sandbox가 아닙니다.

`sources_open_duckdb`는 `path`, 물리적 `table` 이름, 선택적 `config`를 받습니다.
네이티브 기본·SQL·DuckDB 설정을 전달하되 `database`는 비공개 복사본에,
`read_only`는 참으로 고정합니다. 이 미배포 체크아웃에서
`pip install '.[mcp,duckdb]' 'duckdb>=1.4,<2'`로 설치합니다. 안정 버전
DuckDB 1.4 이상·2 미만이 필요하며 구버전·사전 배포판은 `unsupported_dependency`입니다.
Core의 실제 SQL 엔진과 표본 추출을 사용하며 메모리 DataFrame 엔진으로 대체하지 않습니다.
WAL·뷰·임의 SQL은 거부하며 복사 도중 파일 상태나 WAL이 변하면 게시하지 않습니다.
[DuckDB 보안 지침](https://duckdb.org/docs/current/operations_manual/securing_duckdb/overview)에 따라
외부 파일 읽기, 확장 자동 설치·로딩, 커뮤니티·미서명 확장을 차단하고 설정을 잠급니다.
스레드는 기본 1·명시 시 최대 16이며 메모리는 `memory_limit`이 없으면
`max_memory_mb`를 MB 단위로 적용합니다. 연결 제어이지 OS sandbox나 전체 프로세스
메모리 제한을 뜻하지 않습니다. `unsupported_table`·잘못된 DB·한도·취소 실패에서
사용 가능한 소스를 게시하지 않습니다. 회수 시 소유 worker와 복사본만 정리하며
원본은 삭제하거나 수정하지 않습니다.

`streams_open`은 경로와 형식별 폐쇄형 `source`를 받습니다.

```json
{"path":"/approved/data.csv","source":{"format":"csv","config":{"chunk_size":1000,"columns":["age"],"skip_rows":0,"max_rows":10000,"separator":",","has_header":true}}}
```

형식은 `csv`, `jsonl`, `parquet`, `arrow`입니다. 공통 설정은 `chunk_size`, `columns`,
`skip_rows`, `max_rows`이며 Parquet·Arrow는 `use_mmap`, CSV는 `separator`,
`has_header`도 받습니다. CSV의 `skip_rows`는 네이티브 구현 그대로 헤더 앞의 물리적
행을 건너뜁니다. Arrow는 PyArrow가 필요합니다. 반환 자원 ID를 `streams_inspect`와
`streams_next`의 `stream_id`로 전달하며 진행마다 `idempotency_key`가 필요합니다.
배치는 `{artifact: <Arrow 자산 메타데이터>, exhausted: false}`, 종료는
`{artifact: null, exhausted: true}`를 반환합니다. 배치 ID를 `assets_validate` 등에
전달합니다. **배치 단위** 검사이지 전역 고유성·배치 간 window·Kafka offset·연속
실시간 검증이 아닙니다. 네이티브 배치 경계를 임의로 다시 나누지 않습니다.
같은 키의 재요청은 같은 배치를 재생하고 커서를 다시 진행하지 않습니다. 세션별 receipt는
`max_items`로 제한되며 서버 재시작 후 유지되는 영속 커서가 아닙니다. 진행 결과가
불확실하면 worker를 무효화하며 추정 위치에서 재시도하지 않습니다. 스트림 회수 시
입력 스냅샷과 배치도 회수합니다. 로컬 파일 스트림은 Kafka·Kinesis·PubSub 실계정이나
전체 realtime 동등성의 증거가 아닙니다.

예를 들어 보관 검증 입력은 다음과 같습니다. 예시 문자열 대신 실제 반환된 식별자를 사용합니다.

```json
{
  "asset_id": "asset_<returned-session-token>",
  "checks": [{"validator": "between", "settings": {"columns": ["age"], "min_value": 0, "max_value": 120}}]
}
```

검증기 설명은 실제 등록된 구체 검증기 260개의 생성자와 상속된 `ValidatorConfig`,
캐시·알고리즘별 설정을 보여줍니다. 알 수 없는 설정을 Core가 조용히 버리게 하지 않고
전송 경계에서 거부합니다. 수치 생성자 필드는 float 범위를 넘는 유한한 Python 정수도
강제 변환하지 않고 보존합니다. 네이티브 생성자·실행 오류는 그대로 실패이며,
NaN·무한대·불리언은 수치 설정으로 허용하지 않습니다. 자원·응답 예산은 늘리지 않습니다.
네이티브 스트리밍 검증기 6종(`streaming_null`,
`streaming_completeness`, `streaming_nan`, `streaming_range`,
`streaming_positive`, `streaming_outlier`)과 해당 `chunk_size` 설정도 포함합니다.
**이 목록은 260종 전체의 동등성 검증 통과 증거가 아닙니다.**
생성자 거부·네이티브 실행 오류·알고리즘 내부에서 잡힌 예외는 별도로 집계합니다.
잘못된 생성자 설정은 원문 없이 `validator_configuration_error`를 반환합니다.
승인된 신뢰 코드가 이미 시작된 뒤의 실패는 `outcome_unknown`을 유지하고 자동 재실행을
금지합니다. 선택 의존성 부재도 품질 성공이 아니라 실패입니다. Polygon 연산에는
`truthound[geospatial,mcp]`가 필요합니다. `column_consistency.rules`는 `when`,
`then`, 선택적 `description`만 받으며 두 조건은 아래의 제한된 Polars 식을 사용합니다.
Python 코드를 받지 않습니다.

개발 소스의 Core 수정은 MCP에서도 알고리즘 동작에 반영됩니다. 숫자형 PSI·CSI와
PSI 기반 feature drift는 기준 히스토그램 범위 밖의 관측치를 버리지 않고 집계합니다.
PSI의 캐시·비캐시 경로에 같은 꼬리 집계 규칙을 적용하며 임계값은 바꾸지 않습니다.
상수 기준분포도 빈 히스토그램으로 처리하지 않고 점질량을 보존합니다. 기본 참조 캐시는
검증기 인스턴스별로 구분하며 같은 기준분포를 인스턴스 간 재사용하려면 버전을 구분한
`cache_key`를 명시합니다.
`online_svm`이 사용하는 `validators.memory.SGDOneClassSVM`은 합성 음성 레이블을
넣는 이진 분류기 대신 scikit-learn의 실제 비지도 SGD one-class 추정기를 사용합니다.
scaler와 kernel projection은 첫 배치에서 학습한 뒤 고정하며 이후 배치는 추정기를
갱신합니다. 이전 이진 대체 모델의 예측과 동등하지 않습니다.
이 구성은 [scikit-learn의 one-class SGD 안내](https://scikit-learn.org/stable/modules/outlier_detection.html#scaling-up-the-one-class-svm)를 따릅니다.
미배포 소스 변경이며 이미 게시된 3.2.0 파일을 바꾼 것은 아닙니다.

Core의 JSON 스키마 검증기는 선택 라이브러리가 있으면 Draft 7을 사용합니다.
신뢰하지 않은 스키마 설정으로 외부 참조를 가져오지 않습니다. 전송 경계는 로컬
포인터·앵커를 따라가며, 원래 리터럴 데이터였던 위치도 참조 대상이 되면 실제 스키마로
검사합니다. 참조 분석에는 별도의 깊이·노드 예산이 있고, 초과하면 검증을 잘라내지 않고
실패합니다. 일반 속성명 `id` 등은 데이터로 유지하며, 실행 가능한 정규식·외부 참조
스키마는 검토된 운영자 바인딩이 필요합니다.
신뢰하지 않는 SQL·콜백·호출자 파일 경로·제한 없는 정규식은
`operator_configuration_required`를 반환합니다. 별도 운영자 승인 경계에서 정확한
검증기 템플릿과 설치된 확장의 네이티브 객체를 연결할 수 있지만 모델의 실행 입력을
신뢰하게 되는 것은 아닙니다. 인라인 JSON Schema의 `$ref`는
`#` 내부 참조만 허용하며 파일이나 네트워크를 조회하지 않습니다. 지원하지 않는
설정을 몰래 대체하지 않습니다.

참조 DataFrame·LazyFrame 인수는 `{"asset_id":"<returned-id>"}` 형식이고,
참조 테이블 사전은 이러한 값을 이름에 매핑합니다. 각각 권한을 재검사하고 결과의
부모 핸들로 연결합니다. 문자열 아닌 키를 가진 사전은
`{"entries":[{"key":1,"value":0.5}]}`로 키 자료형을 보존합니다.
날짜·날짜시각은 `{"value":"2026-01-01T00:00:00"}` 또는 ISO 날짜,
기간은 `{"value":1000000}`처럼 정수 마이크로초로 지정합니다. 자료형 설정은
메모리 테이블의 폐쇄형 `DType` 입력과 같습니다. Polars 식은 제한된 구문 트리로 전달합니다.
예를 들어 `{"op":"gt","left":{"op":"column","name":"age"},"right":{"op":"literal","value":120}}`입니다.
컬럼·리터럴, 산술·비교·불리언, null 검사·절댓값, 합계·평균·최솟값·최댓값·개수·고유 개수
노드를 지원하며 Python 표현식·SQL·직렬화된 실행 식은 받지 않습니다.

파일 보관의 `options`는 `FileDataSourceConfig`의 전체 설정을 전달합니다.
`options.sample_size`는 데이터소스 설정이고 최상위 `sample_size`는 별도의
`th.read()` 표본 크기입니다. `.pq`도 허용합니다. SQLite는 허용 폴더의
`.db`·`.sqlite`·`.sqlite3`와 실제 테이블명이 필요하며 SQL·뷰를 받지 않습니다.
백업 크기와 Core 메모리 변환 행 수를 제한하고 연결을 회수합니다. 읽기 전용
WAL 연결도 공유 메모리를 수정할 수 있으므로 WAL 데이터베이스는 거부합니다.
이 로컬 연결을 다른 SQL·클라우드 제공자나 자격 증명 지원으로 해석하면 안 됩니다.

핸들은 서버 세션에 귀속됩니다. 기본 제한은 128개·전체 1 GiB·수명 900초이며
`--max-handles`, `--max-session-bytes`, `--handle-ttl-seconds`로 지정합니다.
파생 핸들은 부모보다 오래 살 수 없고 부모를 회수하면 함께 무효화됩니다.
접근할 때마다 권한과 바이트 무결성을 확인하며 입력 종료 시 소유 파일을 삭제합니다.
영속 커서가 아닙니다. 보관 행 페이지는 물리값 태그로 중첩 나노초 시간 자료도 보존합니다.
신뢰하지 않는 인라인 자료형은 중첩 16단계·값당 고정 스칼라 슬롯 524,288개까지입니다.
네이티브 할당 전에 모든 행과 실제 중첩 리스트·구조체 원소의 스칼라 슬롯 합계에
16바이트를 곱해 남은 세션 바이트 예산 안인지 검사합니다. 작은 중첩 배열·null
선언이 거대한 버퍼를 요구하지 못하게 하는 보수적
허용 검사이며 운영체제 메모리 샌드박스나 Core 알고리즘 기준 변경이 아닙니다.
산출물 바이트 읽기는 최대 32 KiB의 base64 페이지·SHA-256·다음 위치를 반환합니다.
원본 행·스키마·바이트를 읽으려면 `--allow-sensitive-results`가 필요합니다.
산출물 생성만으로 그 내용이 공개되지는 않습니다. 렌더러는 출력·템플릿 경로를 받지
않으며 브라우저나 대시보드를 실행하지 않습니다. HTML은 여전히 신뢰하지 않는
콘텐츠이므로 호스트·브라우저에서 여는 행위는 별도로 판단해야 합니다.

스키마 생성은 `schema_spec`의 명시적인 컬럼 제약과 통계를 보관합니다. 내보내기는
`resource_id`, `format=json|yaml`을 받으며 기본은 YAML입니다. 에이전트가 지정한
경로에 쓰지 않고 소유 산출물을 만듭니다. YAML은 실제 Core의 `Schema.save()`를
사용하며 `Schema.load()`로 다시 읽을 수 있습니다. 인라인 정규식은 받지 않으며
실행 가능한 스키마 패턴은 검토된 네이티브 검증기 바인딩으로 연결합니다.

고급 프로파일 목록은 `ProfilerConfig`와 기본 구성 요소의 전체 생성자 설정을
보여줍니다. 프로파일 실행 입력은 `asset_id`, `config`, `options`, 선택적인
`column_analyzers`, `table_analyzers`, `type_inferrers`, 테이블의 `name`·`source` 표시명입니다.
구성 요소는 예를 들어 `{"component":"CorrelationAnalyzer","settings":{"threshold":0.8}}`
처럼 지정합니다. 컬럼 분석기 6종·테이블 분석기 3종·타입 추론기 3종을 연결합니다.
Core에서 `config`와 상속된 `options`는 별개이므로 합치지 않습니다. 예를 들어
DataProfiler의 실제 앞부분 표본은 `options.sample_size`가 지정하며
`config.sample_size`와 독립적입니다. Core에서 사용하지 않는 설정이 알고리즘을
바꾼다고 주장하지 않습니다. 알 수 없는 설정·임의 가져오기·사용자 정규식 정의는
거부합니다. 임의 확장 구성·스트리밍 프로파일·생성 규칙 적용 수명주기는 아직 별도 작업입니다.

`profilers_generate_rules`는 동일한 프로파일 입력에 필수 `generator`와
`strictness`(`loose`, `medium`, `strict`, 기본 `medium`)를 추가합니다.
예를 들어 `{"component":"SchemaRuleGenerator","settings":{}}`를 지정합니다.
목록에는 `StatsRuleGenerator`, `PatternRuleGenerator`, `MLRuleGenerator`도
있습니다. 선택한 네이티브 생성기 하나가 새 원본 프로파일을 받아 전체
`GeneratedRule.to_dict()` 결과를 소유권이 있는 산출물로 보관합니다.
`representation`을 `validator_configs`로 지정하면 대신
`{"validator_class": "...", "config": {...}}` 목록을 보관합니다. 각 설정은
요약에서 추론하지 않고 네이티브 `GeneratedRule.to_validator_config()`를 직접
호출하여 컬럼과 `mostly`까지 포함합니다. 기본값 `rules`의 결과는 그대로입니다.
내보내기는 검증기를 생성·승인·실행하지 않으며 이후 사용에도 기존 설정 검증과
접근 권한 검사가 필요합니다.
`column`을 생략하거나 `null`로 지정하면 테이블 전체 규칙을 생성합니다.
`"column":"amount"`처럼 실제 컬럼명을 지정하면 새 프로파일의 해당 컬럼에
네이티브 `generate_for_column()`을 호출합니다. 전체 규칙을 나중에 거르는
방식이 아니므로 테이블 수준 상관관계 규칙은 추가되지 않습니다. 없는 컬럼은
`core_error`로 실패하며 부분 산출물을 만들지 않습니다.
검증기를 실행하거나 suite를 적용하지 않으며, 예외를 숨겨 부분 규칙을 성공으로
반환하지 않습니다. 산출물 읽기에는 민감 결과 허용이 필요하고 원본 자산의 접근
권한을 유지하므로, 원본을 해제하면 규칙 산출물도 읽을 수 없습니다.

### 세션 소유 스키마 이력

`schema_history_create`는 소유한 상주 작업자에서 Core `InMemorySchemaHistory`를
생성합니다. `strategy`는 `semantic`(기본), `incremental`, `timestamp`, `git_like` 중
선택합니다. `max_versions` 기본값은 100이며 1부터 운영자의 `max_items`까지
(최대 1,000) 허용합니다. 결과에는 `history` 리소스 정보와 `revision=0`이 포함됩니다.

`schema_history_execute`는 `history_id`, `expected_revision`, 명시적인 `command`를
받습니다. 결과에는 이력 정보·현재 revision·`artifact`가 있으며, 완전한 네이티브
결과는 `artifacts_read`로 읽습니다.

| command의 action | 입력 필드 | 네이티브 결과 |
| --- | --- | --- |
| `save` | `asset_id`, 선택적 `version`, `metadata` | 자산의 행이 아닌 스키마로 생성한 새 버전 |
| `get` | `version_id` | 버전 또는 null |
| `latest`, `baseline` | 없음 | 보존된 최신·최초 버전 또는 null |
| `count` | 없음 | 네이티브 보존 버전 수 |
| `breaking_since` | `version_id`(ID 또는 버전 문자열) | 최신 버전과 비교한 네이티브 호환성 파괴 여부 |
| `by_version` | `version` | 버전 문자열에 해당하는 버전 또는 null |
| `list` | 선택적 `limit`, `since`, `until` | 최신순 목록, ISO 날짜 필터 경계는 제외 |
| `diff` | `source_id`, 선택적 `target_id`(ID 또는 버전 문자열) | 네이티브 변경·요약, 대상 기본값은 최신 버전이며 버전이 없으면 null |
| `rollback` | `version_id`(ID 또는 버전 문자열), 선택적 `reason` | 네이티브 롤백 메타데이터가 포함된 새 복사 버전 또는 null |
| `delete` | `version_id` | 삭제 여부 |

반환된 식별자를 사용하는 호출 예시입니다.

```json
{"history_id":"<history.resource_id>","expected_revision":0,"command":{"action":"save","asset_id":"<asset.resource_id>","metadata":{"author":"operator"}}}
```

```json
{"history_id":"<history.resource_id>","expected_revision":1,"command":{"action":"list","limit":10}}
```

`save`·`delete`·`rollback`은 없는 버전 때문에 false/null을 반환해도 revision을
증가시킵니다. 조회는 증가시키지 않으며 오래된 revision은 변경 전에 거부됩니다.
보존 수 초과 시 가장 오래된 네이티브 버전을 제거합니다. 롤백은 과거 이력을
덮어쓰지 않고 새 버전을 생성합니다. 네이티브 `limit=0`은 제한 없음을 의미합니다.
Core 생성 시각에 맞춰 시간대 없는 날짜 필터를 사용하세요. 시간대 유무가 다른
시각의 비교는 Core와 마찬가지로 실패합니다. UUID·생성/비교 시각과 timestamp/hash
버전 전략의 비결정적 값은 네이티브 동작을 유지합니다.

이 기능은 **세션 전용**이며 `FileSchemaHistory`, 외부 저장소, 세션 간 복구가 아닙니다.
리소스 해시는 생성 설정을 나타내며 변경되는 이력 내용의 해시가 아닙니다.
행을 저장하지 않지만 관찰한 모든 자산의 권한과 핸들은 버전 삭제·보존 만료 후에도
의존성으로 유지됩니다. 해제·만료·입력 권한 철회·작업자 실패는 이력과 파생 결과를
무효화합니다. 명령 시작 후 작업자나 출력이 실패하면 롤백을 주장하거나 불확실한
재시도를 허용하지 않고 핸들을 무효화합니다. 다음 세션 작업에서 정리하고, 수명
타이머가 유휴 작업자의 존속 시간을 제한합니다. 명시적 종료는 `resources_release`를
사용합니다. 출력과 보관 리소스에는 세션 용량 제한이 적용됩니다. 사용자 경로,
임의 코드·버전 전략, 원격·유료 서비스는 허용하지 않습니다.

### 드리프트와 상주 ML 모델

`profilers_schema_evolution`은 `baseline_id`와 `current_id`로 지정한 두 자산의
네이티브 스키마를 비교합니다. `detect_renames`(기본 `true`)와
`rename_similarity_threshold`(기본 `0.8`)를 Core의 `SchemaEvolutionDetector`에
전달합니다. 산출물에는 모든 네이티브 변경 레코드와 탐지 시각을 포함한 호환성
요약이 들어갑니다. 변경 순서는 Core의 정렬되지 않은 컬럼 집합 순회를 따릅니다.
컬럼 추가·삭제·이름 변경 추정·타입 변경을 탐지하며 제약·널 허용 여부의 변경은
추론하지 않습니다. 기준 스키마 저장·감시 실행·마이그레이션 적용도 하지 않습니다.
산출물을 읽으려면 민감 결과 허용과 두 원본 자산의 유효한 접근 권한이 필요하며,
어느 원본이든 해제하면 읽을 수 없습니다.

`profilers_column_renames`는 두 자산 사이에서 추가·삭제된 컬럼만 Core의
`ColumnRenameDetector`로 분석합니다. 확정·가능 후보, 유사도·신뢰도, 타입 호환성,
판정 이유·메타데이터·미매칭 이름을 모두 소유 산출물에 보존합니다. 실제 컬럼 이름을
바꾸거나 데이터를 마이그레이션하지 않습니다. 입력은 `baseline_id`, `current_id`,
`similarity_threshold=0.8`, `require_type_match=true`, `allow_compatible_types=true`,
`high_confidence_threshold=0.9`, `low_confidence_threshold=0.7`입니다.
`calculator`를 생략하면 Core의 기본 복합 계산기를 사용합니다. 명시적 종류는
`levenshtein`, `token`, `jaro_winkler`(`prefix_scale=0.1`), `ngram`(`n=2`),
`composite`(중첩 없는 가중 계산기 최대 16개)입니다. 복합 계산기의
`calculators=null`은 네이티브 기본값, `[]`는 네이티브 빈 가중치 동작을 유지합니다.
유한 가중치·임계값과 네이티브 값 보정 동작을 보존하며 임의 계산기 코드는 받지 않습니다.

```json
{"baseline_id":"<baseline-handle>","current_id":"<current-handle>","calculator":{"kind":"composite","calculators":[{"calculator":{"kind":"ngram","n":2},"weight":0.6},{"calculator":{"kind":"token"},"weight":0.4}]}}
```

스키마 변경 탐지와 마찬가지로 결과 읽기에는 민감 결과 허용과 두 부모 자산의 유효한
접근 권한이 필요합니다. 순서와 매칭은 Core 동작을 따릅니다. 이름 변경 추정이지
두 컬럼의 의미가 같다는 증명은 아닙니다.

`profilers_schema_compatibility`는 두 소유 자산 스키마에 Core의
`SchemaCompatibilityAnalyzer`를 실행합니다. 입력은 `baseline_id`, `current_id`,
`rules` (`types`, `nullability`, `column_presence` 이름의 순서 있는 목록), `strict`입니다.
`rules` 생략·`null`·빈 목록은 모두 Core의 기본 세 규칙을 유지합니다.
전체 산출물에는 `is_compatible`, `level`, `issues`, `rules_checked`가 포함됩니다.
읽기에는 민감 결과 허용과 두 부모 자산의 유효한 권한이 필요하며 임의 Python 규칙은
받지 않습니다. 현재 Core는 `strict` 값을 저장하지만 분석 동작에는 사용하지 않습니다.
또한 Polars 스키마에 null 허용 메타데이터가 없으므로 nullability 규칙은 문제를
반환하지 않습니다. 이를 실제 데이터의 null 허용 여부 검증으로 해석하면 안 됩니다.
Core는 규칙 예외를 기록하고 건너뛰기도 합니다. `rules_checked`는 평가된 규칙 목록이지
요청한 모든 규칙의 완료 보증이 아닙니다.

```json
{"baseline_id":"<baseline-handle>","current_id":"<current-handle>","rules":["types","column_presence"],"strict":false}
```

네이티브 panic은 일반 검증 실패와 다릅니다. 네이티브 확장이 canonical 결과를
반환하지 못하고 worker를 종료하면 MCP는 성공·부분 결과 산출물 없이
`worker_error`를 반환합니다. 예를 들어 Python 날짜 범위를 벗어나는 Polars
타임스탬프를 개인정보 검증기가 Python 값으로 변환할 때 발생할 수 있습니다.
이는 실패 격리이며 Core 결과를 복구하거나 개인정보 검증에 성공했다는 뜻이 아닙니다.

구체 탐지기의 설정은 `drift_catalog`에서 조회합니다. `drift_detect`는 `detector`,
`settings`, `baseline_id`, `current_id`, `baseline_column`, 선택적인 다른
`current_column`을 받아 해당 네이티브 탐지기의 `detect()`를 실행합니다. 비교
퍼사드의 기본값으로 생성자 설정을 대체하지 않습니다.

데이터셋 비교에는 `baseline_id`, `current_id`, 선택적인 `columns`, `threshold`,
`sample_size`, `method`를 전달합니다. 메서드는 `auto`, `ks`, `psi`, `chi2`, `js`,
`kl`, `wasserstein`, `cvm`, `anderson`, `hellinger`, `bhattacharyya`, `tv`
및 그 별칭 `total_variation`, `energy`, `mmd`입니다. 계산과 표본 선택은 Core가
수행하며 전체 보고서를 보관합니다. 두 스냅샷의 권한을 계속 확인합니다. Core가
어떤 컬럼에도 적용하지 않은 결과를 MCP 품질 통과로 해석하지 않습니다.

모델 목록은 최대 5종씩 실제 알고리즘명과 전체 설정을 보여줍니다. 모델 생성은
`algorithm`, `config`를 받으며 앙상블은 중첩 `detectors`도 지정할 수 있습니다.
ZScore·IQR·MAD·IsolationForest·Ensemble 이상 탐지, Distribution·Feature·Concept·
Multivariate 드리프트 탐지, DataProfileRuleLearner·ConstraintMiner·PatternRuleLearner를
연결합니다. 이름은 고정된 목록의 키이며 Python 가져오기 경로가 아닙니다. 모르는
설정을 ML의 임의 인수로 넘겨 조용히 버리지 않습니다.

생성 결과의 `model_id`와 `revision=0`을 받아 `asset_id`, `expected_revision=0`으로
학습하면 성공 시 revision 1을 반환합니다. 예측·점수·상태 조회는 현재 revision이
필요합니다. 규칙 학습·학습과 예측·두 입력 탐지도 revision을 증가시킵니다.
명시적인 ML 드리프트 탐지는 일반 모델·데이터 입력에 `reference_id`와 선택적인
`columns`를 추가합니다. 적용할 수 없는 메서드나 지원하지 않는 온라인 학습은
실패이며 건너뜀이나 몰래 수행한 재학습으로 대체하지 않습니다.
규칙과 진단 조회는 `model_id`, `expected_revision`이 필요합니다. 진단 산출물은
원래 설정·정보와 해당 모델이 제공하는 통계·특성 중요도·프로파일·기준 상관관계·컬럼
프로파일·학습 패턴·앙상블 이름과 가중치를 담으며 민감한 읽기 허용 후 확인합니다.

모델은 소유한 별도 프로세스에서 실제 학습 상태를 유지합니다. 세션당 최대 4개이며
핸들 제한도 적용합니다. 학습·기준 데이터 핸들은 계속 권한의 부모가 됩니다.
revision이 다르면 Core를 실행하지 않습니다. 시간 초과·취소·프로토콜 오류는 소유
프로세스를 종료·회수하고, 상태 변경 실패 시 불확실한 모델을 재사용하지 않습니다.
만료·회수·권한 철회·세션 종료도 모델과 종속 핸들을 무효화합니다. 여전히 호스트의
사용자 권한으로 실행되며 운영체제 수준 샌드박스는 아닙니다.
Windows 가상환경에서는 가상환경을 유지하면서 실제 인터프리터를 직접 실행합니다.
종료 시 중간 실행기가 아닌 해당 인터프리터의 종료를 기다린 후 작업 디렉터리를
제거합니다.

ML 산출물에는 `encoding`, `model_algorithm`, `model_revision`과 태그로 표현한
원래 `result`가 있습니다. Python 객체를 손실 직렬화하지 않고 사용한 모델 상태를
구분합니다. `models_save`는 `model_id`, `expected_revision`을 받아 네이티브 JSON을
보관합니다. `models_restore`에 해당 산출물의 `resource_id`를 주면 revision 1인
새 모델을 반환합니다. 목록의 12종은 학습 트리·앙상블 하위 모델·드리프트 기준
데이터·학습 규칙을 보존합니다. 버전이 있는 데이터 전용 형식이며 실행 가능한 형식,
중복 JSON 키, 모델 종류 불일치를 거부합니다. 불완전한 이전 저장 상태도 실패하며
준비 완료로 표시하지 않습니다. 복원 모델은 산출물의 권한 부모를 상속하므로 원래
학습 데이터를 해제하면 함께 무효화됩니다. 아래의 명시적 영구 저장 승인 전까지는
세션 소유 산출물입니다. 임의 플러그인 모델·
메서드별 전체 수명주기 동등성은 남은 작업입니다.

`models_result_query`는 `model_id`, `expected_revision`, `asset_id`, `action`을
받으며 `action`은 `all_scores`, `anomalies`, `drifted_columns` 중 하나입니다.
네이티브 모델로 예측하고 Core 결과의 순회·필터 메서드를 호출하여 revision을
증가시키지 않고 소유한 산출물에 보관합니다. `all_scores`는 일반 요약의 이상치뿐
아니라 정상 행과 점수 메타데이터도 포함합니다. `drifted_columns`에만 유한한
`threshold`를 지정할 수 있으며 기본값은 `0.5`입니다. Core의 경계값 포함 비교를
유지합니다. 모델 종류 불일치·미학습 모델·오래된 revision은 명시적으로 실패합니다.
산출물은 모델과 입력에 종속되며 읽기에는 공개 동의가 필요합니다. 매 호출은 새로
예측하며 이전 예측 산출물을 조회하는 기능은 아닙니다.

`models_rule_query`는 `model_id`, `expected_revision`, `asset_id`, `action`을
받습니다. `action`은 `suite`, `validator_specs`, `by_type`(`rule_type` 필요),
`by_column`(`column` 필요) 중 하나입니다. Core 규칙 학습기의 예측 결과와
원래 변환·필터 메서드를 사용하며 모델 revision을 변경하지 않고 전체 결과를
소유한 산출물에 보관합니다. 다른 action의 선택자를 함께 주면 거부합니다.
알 수 없는 유형·컬럼은 원래 빈 목록으로 반환하며 검증 PASS를 뜻하지 않습니다.
생성된 suite는 설정 데이터이며 자동 적용·실행하지 않습니다. 모델·입력 해제 시
파생 산출물도 무효화되고, 규칙 학습기가 아닌 모델에서는 명시적으로 실패합니다.

### 네이티브 스케줄 미리보기

`schedules_preview`는 cron `expression`(최대 256자), 명시적인 ISO 일시 `after`,
`count`(1–100, 기본 5)를 받습니다. 소유한 제한 시간 내 worker에서 Core의 파서,
`matches`, `next_n`을 그대로 호출합니다. 5·6·7 필드, 프리셋과 특수 필드는
네이티브 의미를 유지합니다. 시간대 없는 일시는 그대로 유지하고 명시적인 UTC
오프셋도 서버 시간대로 변환하지 않습니다. IANA 시간대·일광절약 예약 지원은 아닙니다.

잘못된 표현식은 `valid=false`와 원래 `errors`로, 유효하지만 Core의 4년 검색 범위
안에 요청한 개수만큼 결과가 없으면 `search_exhausted=true`로 구분합니다.
둘 다 데이터 품질 PASS가 아닙니다. 스케줄 등록·체크포인트 실행·외부 작업은
수행하지 않습니다. 영구 예약은 아래의 제한된 검증 회차에서 별도로 제공합니다.

### 앙상블 편집과 세션 간 모델 복원

`models_ensemble_add`는 `model_id`, `expected_revision`, 새 `detector` 명세와
유한한 `weight`(기본 1.0)를 받아 Core의 `add_detector`를 호출합니다. 하위 모델을
자동으로 학습하지 않습니다. `models_ensemble_remove`는 `index`(-8~7)로 제거하며,
제거한 모델의 설정과 학습 상태를 복원 가능한 모델 산출물로 반환합니다. 학습 모델은
revision 1, 미학습 모델은 revision 0으로 복원됩니다. 성공한 편집은 revision을
한 번 증가시키고, 실패한 변경은 상주 핸들을 무효화합니다. 하위 모델은 최대 8개,
중첩은 4단계입니다. `detector` 대신 `source_model_id`와 정확한 `source_revision`을
주면 기존 학습 모델을 네이티브 데이터 전용 저장·복원으로 복사해 편입합니다. 재학습이나
가변 Python 객체 공유는 하지 않으며 이후 원본 수정은 복사본에 반영되지 않습니다.
자기 편입과 의존 순환은 거부합니다. 원본 권한·수명은 유지하므로 원본 핸들 해제 시
파생 앙상블도 무효화됩니다. 오래된 원본 revision은 앙상블 변경 전에 거부합니다.

`models_persist`에는 모델 산출물의 `resource_id`와 8~128자 `idempotency_key`를
전달합니다. 운영자가 `approve-plan`으로 정확한 계획을 승인하기 전에는
`approval_required`입니다. 이 승인이 있어야 스냅샷 수명이 세션 밖으로 이전되며,
단순 저장만으로 영구 보존되지 않습니다. `models_resume`에는 `snapshot_id`를
전달합니다. 동일한 비공개 제어 디렉터리·정책의 새 세션에서 네이티브 JSON을
재학습 없이 복원하고 새 핸들을 발급하며 원본 디렉터리 권한을 다시 검사합니다.
정책이 바뀌면 복원과 파생 핸들 접근이 차단됩니다. provider 바인딩 모델은 제외됩니다.
성공 응답을 잃었다면 `models_resume`에 `snapshot_id` 대신 원래 `idempotency_key`를
전달할 수 있습니다. 둘 중 하나만 지정하며 저장을 반복하지 않고 커밋된 영수증을 조회합니다.
저장소는 클라우드나 작업공간 내보내기가 아닌 로컬 비공개 SQLite입니다. 최대 128개,
개별 16 MiB·합계 32 MiB이며 파일 예산도 적용됩니다. 복원 시 서명된 메타데이터와
바이트 해시를 검사하고 스냅샷과 성공 영수증은 한 트랜잭션으로 커밋합니다.
삭제는 아래의 별도 승인된 저장소 삭제 도구를 사용하며 제어 DB를 수동으로 수정하면 안 됩니다.

### 영구 검증 회차

`jobs_submit`은 `validation`(`assets_validate` 요청), `idempotency_key`, 선택형
`schedule`을 받습니다. 예약에는 `expression`, UTC offset이 있는 `after`,
`count`(1~100)를 지정하며 각 시각은 Core cron으로 계산합니다. 요청한 횟수 전체를
찾지 못하면 부분 등록하지 않고 거부합니다. 입력 보존과 지연 실행은
별도 운영자 계획 승인이 필요합니다. context/schema 핸들·참조 데이터 핸들·provider
바인딩·외부 액션은 허용하지 않습니다. 인라인 `schema_spec`과 제한된 검증기 설정은
사용할 수 있습니다.

모든 회차를 한 트랜잭션으로 등록하거나 전혀 등록하지 않으며 첫 `job_id`와 전체
`occurrence_ids`를 반환합니다. 회차마다 상태·실행 임대·결과가 독립적이며 한 회차를
취소해도 다른 회차는 취소되지 않습니다. 입력 복사본 전체가 기존 저장 한도에 포함됩니다.
무한 반복 규칙이 아닌 유한 반복입니다. 각 ID로 `jobs_status`, `jobs_execute`, `jobs_result`, `jobs_cancel`,
`jobs_recover`를 호출합니다. 클라이언트가 실행 시각이 지난 작업을 명시적으로 실행해야
하며 이 제출만으로 데몬 설치나 무한 반복 실행을 승인하지 않습니다. 유료 provider 호출·자동 재시도는 없습니다. 대기 입력과 완료된
네이티브 결과는 세션 종료 후에도 보존되고 결과는 새로 승인된 세션 산출물로 읽습니다.
실행 성공에도 품질 위반이 있을 수 있으며 Core 실행 오류는 `failed`로 남습니다.
취소한 대기 작업은 실행하지 않고 실행 중 취소는 다른 프로세스를 중단했다고 주장하는
대신 `busy`를 반환합니다. 실행 제한시간과 30초 정리 여유가 지난 미완료 lease는
명시적 복구로 `outcome_unknown`이 됩니다. 이는 PASS·자동 재시도가 아니며 30초는
네이티브 작업의 timeout 확대도 아닙니다. 성공 영수증은 계산·소유 자원 정리·결과의
트랜잭션 커밋이 끝나야 기록됩니다.

비공개 저장소는 작업 최대 128개, 입력/결과 개별 8 MiB, 작업 바이트 합계 16 MiB이며
기존 파일·세션 예산도 적용됩니다. 자동 보존 정리는 아래 별도 정책 승인이 필요합니다.
제출 성공 응답을 잃었다면 제출 이후 다섯 작업 도구에도 `job_id` 대신 원래
`idempotency_key`를 전달할 수 있습니다. 둘 중 하나만 지정하며 새 작업을 중복 제출하지
않고 기존 레코드를 조회한 뒤 권한을 다시 확인합니다.
반복 제출의 원래 키는 첫 회차와 전체 ID 목록을 조회합니다. provider 작업은
지원하지 않으며 전체 스케줄링 동등성 완료를 뜻하지 않습니다.

### 영구 반복 예약과 포그라운드 실행기

`schedules_register`는 `jobs_submit`과 같은 검증 요청을 받지만 `count: 1`인 예약이
필수입니다. 별도 승인 계획은 입력 보존과 횟수 제한 없는 반복 실행을 허용하며 정의와
첫 회차를 원자적으로 저장합니다. `schedules_status`·`schedules_disable`에는
`control_id` 또는 원래 `idempotency_key` 중 하나를 지정합니다. 해제하면 대기 중인
현재 회차를 취소하지만 이미 실행 중인 작업은 끝날 수 있습니다. 실패·취소·불확정 회차는
재시도 없이 예약을 일시 중지합니다. 성공한 회차만 순회당 한 번 다음 네이티브 cron
시각으로 이동하며 놓친 시각도 순서대로 처리합니다. 다음 시각을 찾지 못하면 오류로
중단하며 임의의 실행 시각을 만들지 않습니다.

운영자가 등록 서버와 동일한 권한·한도로 포그라운드 프로세스를 직접 시작해야 합니다.
아래 경로는 실제 존재하는 로컬 폴더로 바꿉니다.

```bash
truthound-mcp worker --allow-dir /data --control-dir /private/control \
  --profile core --allow-sensitive-results --poll-interval 5
```

`--once`는 한 번만 순회하며 `--poll-interval`은 1~3600초 정수입니다.
SIGINT/SIGTERM은 현재 제한된 작업을 마친 뒤 종료합니다. OS 서비스는 설치하지 않습니다.
복수 실행기는 트랜잭션 커서와 실행 lease를 공유합니다. 저장 한도는 그대로이며 한도가
차면 실행기를 중단합니다. 해제된 항목을 포함해 정의는 최대 64개입니다. 영구 보존 정책은
과거 완료 데이터를 정리하지만 활성·일시 중지 예약의 현재 회차는 보호합니다. 권한 변경·
변조·불확정 결과는 조용히 재시도하지 않습니다.

`storage_policy_register`는 미리보기 정책과 `idempotency_key`를 받으며 이후 자동
삭제에 대한 별도 승인이 필요합니다. 실행기는 매 순회에서 예약보다 먼저 승인된 정책을
평가합니다. `storage_policy_status`·`storage_policy_disable`에는 `control_id` 또는
원래 `idempotency_key`를 지정합니다. 해제는 이후 순회를 막지만 이미 커밋된 삭제를
되돌리지는 않습니다. 정책과 예약 커서는 세션·프로세스 재시작 후에도 남습니다.
항상 실행되는 OS 서비스 설치나 외부 provider 실행을 추가하는 기능은 아닙니다.

### 영구 저장 보존 정책과 삭제

`storage_retention_preview`에는 `kind`(`models` 또는 `jobs`)와 `max_age_minutes`
(1~5,256,000), `max_count`(1~128), `max_size_bytes`(1~33,554,432) 중 하나 이상을
지정합니다. Core의 `TimeBasedPolicy`·`CountBasedPolicy`·`SizeBasedPolicy`·
`CompositePolicy`를 재사용합니다. 기본 `mode="all"`은 모든 선택 정책이 보존해야
남기고, `mode="any"`는 하나라도 보존하면 남깁니다. 크기는 실제 작업 입력·결과
바이트 합계 또는 모델 payload 바이트이며 SQLite 파일 크기나 새 저장 한도가 아닙니다.
영구 자동 정책도 같은 필드를 받으며 재시작 후 승인된 조합을 유지합니다.
삭제 없이 `candidates`, `preserved`, `excluded` ID 목록을 반환합니다. 대기·실행 중·
불확정 작업과 생성 시각이 없는 과거 레코드는 제외합니다. 동일 생성 시각은 ID 순서로
정렬해 네이티브 개수 정책의 판정을 결정적으로 유지합니다.

`storage_delete`에는 중복 없는 정확한 `identities`(스냅샷·작업 ID 1~128개)와
`idempotency_key`를 지정합니다. 별도 운영자 승인이 필요하며 삭제 트랜잭션 안에서
소유권·서명·현재 권한·비활성 상태를 다시 검사합니다. 활성·불확정 작업은 삭제하지 않습니다.
대상 데이터 삭제와 성공 영수증을 원자적으로 커밋하고 재호출은 영수증만 반환합니다.
영구 원본 삭제 후 기존 복원 모델·결과 핸들의 접근도 거부합니다. 승인 영수증은 감사와
중복 실행 방지를 위해 남습니다. SQLite의 논리적 데이터 삭제이며 디스크 완전 소거를
보장하지 않습니다. 명시적 미리보기는 삭제하지 않으며 자동 정리는 별도 영구 정책 승인과
운영자가 시작한 실행기가 필요합니다. 아카이브·압축이나 전체 보존 하위 시스템 동등성을
주장하지 않습니다. 생성 시각 없는 과거 레코드도
정확한 ID를 지정하면 승인 후 삭제할 수 있습니다. 같은 비공개 제어 저장소·운영자 정책과
민감한 결과 허용이 필요합니다.

### 네이티브 스키마 감사

`schema_audit_contract`는 문서의 전체 네이티브 JSON Schema를 보관합니다.
`schema_audit_create`의 `document_json`에 엄격한 UTF-8 JSON을 전달하면 소유한
문서 산출물이 생깁니다. 전체 작업 입력 한도는 64 KiB입니다. 알 수 없는 필드,
중복 키, 잘못된 참조·문서 종류는 실패합니다. `schema_audit_inspect`는
`resource_id`를 받아 정규화 JSON과 원문·의미 해시를 제공합니다.
`schema_audit_standards`는 `standard_id`, `design_id`를,
`schema_audit_implementation`은 `design_id`, `catalog_id`를 받습니다.
전체 발견 사항·규칙별 검증 범위·원래 판정을 보관하며 UNKNOWN을 유지합니다.
비어 있거나 불완전한 검증을 PASS로 바꾸지 않으며 DB에 연결하지 않습니다.

선택적인 용어 매핑은 비공개 `bindings.json`의 `schema_audit`에 운영자가 설정하며
모델이 주장하는 승인 정보는 받지 않습니다. `binding_id` 사용에는
`idempotency_key`와 정확한 계획의 승인이 필요합니다. 입력 해시와 바인딩 파일
식별자를 확인하고 운영자 설정이 변경되면 결과 산출물 접근을 철회합니다.
문서·결과에도 세션 만료·용량 제한·부모 해제·민감한 결과 공개 정책을 적용합니다.

### 네이티브 계보 그래프

`lineage_track`는 `graph_id`와 `action`이 `source`, `transformation`, `validation`,
`output`인 `event`를 받습니다. 해당 Core `LineageTracker.track_*`를 호출하고 원본과
부모 소유권을 유지하는 새 그래프 핸들을 반환합니다. 모든 이벤트에 `name`, 선택형
`description`, `owner`, `tags`, JSON `properties`를 지정하며 속성으로 정식 인자를
덮어쓸 수 없습니다. 소스에는 `source_type`, `location`, `schema_spec`, 변환에는
`sources`, `operation`, `location`, `schema_spec`, `column_mapping`을 지정합니다.
열 매핑 값은 `[원본 노드, 원본 열]` 쌍의 목록입니다. 검증에는 `sources`, `validators`,
`result_summary`, 출력에는 `sources`, `output_type`, `location`을 지정합니다.
위치는 메타데이터일 뿐 파일·연결을 열거나 실행·게시하지 않습니다. 기존 노드 교체와
없는 원본 ID의 엣지 생략, 자동 생성 시각 등 Core 의미를 보존합니다. 수동 추적이며
자동 실행 이력·원격 전송이나 기록된 검증의 실제 실행을 증명하는 기능은 아닙니다.

추가 `dependency` 이벤트는 `source_id`, `target_id`, `edge_type`(기본값
`depends_on`)만 받아 Core `LineageTracker.add_dependency`를 호출합니다. 없는 노드나
순환 관계는 원본을 변경하지 않고 실패합니다. `lineage_query`는 `view: "tracked_lineage"`에
`node_id`, `direction`(`upstream`, `downstream`, 기본값 `both`)을 받거나,
`view: "tracked_path"`에 `source_id`, `target_id`를 받습니다. 네이티브 추적기의
엣지 방향 동작을 재해석하지 않습니다. 없는 끝점을 포함하여 경로가 없으면 빈 그래프가
아닌 JSON `null`을 반환합니다. 모든 산출물은 부모 소유권과 접근 철회를 유지합니다.

큰 관계 스키마는 승인된 운영자 검증기 설정 참조로 전달할 수 있으며 64 KB 도구 요청
한도는 그대로입니다. 깊은 관계 스키마의 재귀 탐색에서 Core `RecursionError`가 발생할
수 있습니다. MCP는 이를 검증 성공으로 바꾸지 않고 실행 실패와 정식 증거를 보존합니다.
동등성 테스트는 호출 스택 깊이에 따라 달라지는 반복 traceback 프레임 횟수만 정규화하며
예외 종류·메시지·프레임은 그대로 비교합니다.

동등성 증거는 Core가 보고 없이 삼키는 내부 예외도 별도로 구분합니다. 실패를 주입하지
않고 정확한 네이티브 메서드의 실제 예외와 정상 대조군을 관측한 뒤 Core/MCP 전체 결과를
비교합니다. 이를 실행 실패나 데이터 품질 통과로 바꾸지 않으며 집계 품질은 여전히
미확정입니다. critical 품질 이슈로 반환되는 예외와도 별도로 집계합니다.

`lineage_load`는 승인된 로컬 `.json`의 `path`와 선택형 Core `config`를 받으며
`persist_path`는 금지합니다. 크기 제한을 적용한 불변 파일 스냅샷을 만든 뒤
`LineageGraph.load`를 호출합니다. 입력 노드 레코드는 1,000개, 엣지는 2,000개로
제한하며 손상되거나 빈 네이티브 파일을 성공한 빈 그래프로 바꾸지 않습니다.
불러오기와 내보내기는 모두 민감 결과 권한이 필요합니다. `lineage_export`는
`graph_id`를 받아 Core `LineageTracker.export_to_json`의 정확한 UTF-8 바이트를
소유권이 있는 `application/json` 산출물로 반환합니다. `lineage_query`의 태그형 값이
아닌 네이티브 JSON입니다. `artifacts_read` 페이지를 Base64 디코딩해 이어 붙이고,
나중에 Core/MCP에서 다시 읽을 파일에는 해당 바이트를 보존합니다. 호출자가 출력 경로를
지정하거나 파일을 암묵적으로 게시하지 않습니다. 원본 권한 철회나 부모 해제는 파생
접근도 무효화합니다.

네이티브 전체·부분 그래프 렌더링은 `lineage_render`로 제공합니다.

계보 목록은 원래 설정·노드·엣지와 값 표현 규칙을 보여줍니다. 생성은 `config`,
`nodes`, `edges`를 받고 호출자의 `persist_path`는 거부합니다. 편집은 `graph_id`와
판별자 `edit`를 받습니다. 작업은 `add_node`, `remove_node`, `add_edge`,
`remove_edge`, `clear`이며 원본을 변경하지 않고 새 그래프 핸들을 만듭니다.
조회는 `graph_id`, `view`와 보기별 필드를 포함한 `query`를 받습니다. 보기는
`graph`, `nodes`, `edges`, `roots`, `leaves`, `topological`, `node`, `has_node`,
`node_edges`, `upstream`, `downstream`이며 탐색 깊이·엣지 방향을 명시합니다.

영향 분석은 `graph_id`, `node_id`, 선택적인 네이티브 `impact_rules`를 받습니다.
`lineage_impact`의 기본값은 `view="full"`입니다. `summary`는 네이티브
`ImpactResult.summary()`, `critical`은 `get_critical_nodes()`를 호출합니다.
`by_level`은 `level=none|low|medium|high|critical`, `by_type`은
`node_type`(예: `validation`)이 필수이며 다른 보기에는 해당 필드를 넣을 수
없습니다. 호출마다 캡처된 그래프와 깊이·검증 포함·영향 규칙으로 새 분석을
수행하며 이전 산출물을 필터링하지 않습니다. 텍스트·노드 목록도 민감 결과 권한이
필요한 소유 산출물이며 원본 그래프를 해제하면 접근이 취소됩니다.
예: `{"graph_id":"graph_<token>","node_id":"a","view":"by_level","level":"high"}`.

일반 영향 분석에는 `max_depth`, `include_validations`, 스키마 영향에는
`new_schema`를 추가합니다. 의존성 연결 조회는 대신 `source_id`, `target_id`를
받습니다. 전체 결과를 산출물로 보관하며 가상 삭제는 원본이나 외부 데이터를 지우지
않습니다.

`lineage_render`는 `graph_id`, `renderer=d3|cytoscape|graphviz|mermaid`,
`settings`의 전체 폐쇄형 네이티브 `RenderConfig` 설정과 선택적인 `root_node_id`,
`direction=upstream|downstream|both`를 받습니다. 부분 그래프의 탐색 깊이는
`settings.max_depth`로 지정하며 프로세스가 달라도 원래 그래프의 노드 순서를
유지합니다. 예시는 `{"graph_id":"graph_<token>","renderer":"mermaid","settings":{"orientation":"LR","theme":"dark"}}`입니다.
결과는 D3·Cytoscape JSON 또는 DOT·Mermaid 텍스트의 소유 산출물입니다. 민감한
결과 권한으로 `artifacts_read`에서 읽으며 원본 그래프 해제 시 파생 결과도
무효화됩니다. 알 수 없는 루트·설정은 실패합니다. 출력 경로·외부 실행 파일을
받거나 HTML·브라우저·이미지 변환을 실행하지 않습니다. 추적기·이력·대화형 HTML과
이미지 렌더링·OpenLineage 게시는 아직 별도 미완료 영역입니다.

### 영구 context와 운영자 승인

`sources_open_polars`에 소유한 `asset_id`와 선택적인 전체 네이티브
`PolarsDataSourceConfig` 설정을 `config`로 전달하면 `source` 리소스 ID가 반환됩니다.
이를 `source_id`로 사용합니다. `sources_validate`는 `assets_validate`와 같은 검증
설정을 받지만 호출자가 `asset_id`를 대체할 수 없습니다. 프레임으로 치환하지 않고
실제 상주 데이터소스를 Core에 전달합니다. 조회·검사 결과는 보관 산출물이며 읽기에는
민감정보 공개 허용이 필요합니다. 연결 실패 판정은 `connected=false`로 유지하고,
네이티브 크기 gate 예외는 실패로 반환합니다. 표본 추출 결과는 새 상주 연결이 아닌
Arrow 데이터 핸들입니다. 입력 데이터·소스를 해제하면 종속 핸들이 무효화되고 소유한
worker가 종료됩니다. 정상 종료 시 네이티브 context-manager 정리를 허용하며 시간
초과·EOF·프로토콜 실패 시 해당 소유 프로세스만 무효화·회수합니다. 소스 핸들의 해시는
설정의 식별자이지 별도 데이터 스냅샷 해시가 아닙니다. 이 수명주기 연결은 현재
**Polars만** 지원하며 SQL·클라우드·NoSQL·스트림·비동기 동등성은 아닙니다.
`assets_sqlite`는 여전히 프레임 적재이고 상주 SQL 소스·pushdown 검증이 아닙니다.
네이티브 설정은 운영체제 샌드박스가 아닙니다.

`suites_create`는 Python 코드가 아닌 버전 1 검증 의도를 보관합니다. 각 검사는
고유한 `check_id`, 구체 `validator`, 닫힌 `settings`와 선택적인 이름·분류·태그·
메타데이터를 가집니다. `schema_id`로 보관 스키마를 참조할 수 있으며 스키마만
있는 suite도 지원합니다. 이때 검사 ID `schema`는 예약됩니다. 스키마를 해제하면
종속 suite와 결과도 무효화됩니다. `evidence_policy`·`severity_policy`는 실제
커널 정책을 사용합니다. `suites_validate`에 `asset_id`·`suite_id`를 전달합니다.
이는 자동 기준 스키마 추론이 아닌 명시적인 커널 suite입니다. 영구 실행에는
`context_id`·멱등 키·정확한 운영자 승인이 추가로 필요합니다.

`contexts_execute`는 `context_id`, `idempotency_key`, `action`으로 구분하는
`operation` 객체를 받습니다. 지원 동작은 `resolve_source_key`,
`resolve_fingerprint`, `track_asset`, `get_or_create_schema_baseline`,
`read_applied_suite`, `write_applied_suite`, `persist_run`, `persist_docs`,
`metric_get`, `metric_put`, `metric_append_history`, `list_runs`, `list_docs`,
`list_baselines`, `list_applied_suites`입니다. 정확한 필드는 `capability_describe`로
확인합니다. 네이티브 context 생성 자체가 폴더 생성·손상 기록 격리를 수행할 수
있어 조회도 승인 대상입니다. 결과·문서 영구 저장은 소유한 정식 검증 산출물만
허용하며 적용 suite의 행위자 메타데이터는 승인 권한이 아닙니다. context 핸들을
해제해도 사용자 작업공간은 삭제하지 않습니다.

입력 폴더와 작업공간 권한은 분리됩니다. 서버에 `--allow-dir` 외에도
`--workspace-dir /absolute/project --control-dir /absolute/private-control`을 지정합니다.
제어 폴더는 데이터·작업공간과 겹치지 않아야 하고 POSIX에서는 사용자 소유·권한
`0700`이어야 합니다. `contexts_open`은 프로젝트 설정 또는 명시적인 `config`를
읽을 뿐 `.truthound`를 만들지 않습니다. 반환된 `context_id`와 8–128자의
`idempotency_key`로 초기화를 요청하거나, 두 값을 `assets_validate`에 전달해
실제 Core 프로젝트 기록을 남기는 검증을 요청합니다.

처음에는 승인 대기 계획이 반환됩니다. 운영자는 MCP 도구 밖에서 정확한 요청을
검토한 뒤 직접 승인합니다. 다음 예시도 실제 반환된 계획 식별자로 바꿔야 합니다.

```bash
truthound-mcp review-plan --control-dir /absolute/private-control --plan-id plan_<returned-token>
truthound-mcp approve-plan --control-dir /absolute/private-control --plan-id plan_<returned-token>
```

전용 CLI의 `revoke-plan`으로 취소할 수 있습니다. 승인은 기능·요청·세션·권한·작업공간
상태·효과·만료에 결합됩니다. 승인 후 동일한 도구 요청을 다시 전달합니다. 권한이나
작업공간이 바뀌거나 모델이 승인을 위조하면 거부합니다. 제어 폴더는 비공개 서명·승인
상태를 포함하므로 공유하거나 MCP로 노출하지 않습니다. 이 경계는 같은 사용자 권한의
악성 로컬 프로세스까지 막는 운영체제 샌드박스는 아닙니다.

영구 검증은 실제 Core context의 설정에 따라 카탈로그·기준 스키마·실행 결과·메트릭
이력·문서를 기록합니다. 완료된 실행 의도는 다시 기록하지 않고 영수증을 재생합니다.
이전 세션 산출물이 만료됐다면 `artifact_available=false`로 명시하며 검증을 다시
실행하지 않습니다. 효과 시작 후 시간 초과·취소는 `outcome_unknown`이며 성공한
롤백이나 재시도 허가로 표시하지 않습니다. 자동 충돌 복구는 아직 구현하지 않았습니다.
대상·건너뜀 집계의 완전성이 확인되기 전까지 보관 검증의 전체 품질은 `unknown`이며,
정식 Core 판정과 상세 결과는 산출물에서 확인합니다.

### 검토된 네이티브 검증기 바인딩

운영자는 보호된 제어 폴더에 `bindings.json`을 둘 수 있습니다. POSIX에서는 사용자
소유·권한 `0600`이어야 합니다. 데이터 폴더나 MCP 도구가 아닌 실행 설정입니다.

```json
{"validators":{"capital-a":{"validator":"regex","settings":{"pattern":"^A+$","columns":["code"]}}}}
```

`assets_validate`의 `checks`에 `{"validator":"regex","binding_id":"capital-a"}`와
`idempotency_key`를 전달한 뒤 위 CLI로 검토·승인합니다. 에이전트는 고정 설정을
덮어쓰거나 스스로 승인할 수 없습니다. 영속 요청·영수증은 원문 SQL·설정 대신
생성자 다이제스트를 남깁니다. 승인은 정확한 파일·권한·세션·효과에 결합되며 파일이
바뀌거나 철회되면 파생 산출물도 읽을 수 없습니다. 영수증 재생은 승인·설정 식별자를
보존하지만 새로운 민감한 읽기 권한을 부여하지 않습니다.

JSON으로 표현하지 못하는 값은 설치된 Core 플러그인으로 연결합니다. 같은 파일의
`plugins`에 설치 배포명·정확한 버전·`truthound.plugins` 진입점 이름·파일 `sha256`·
공개 capability 매핑과 선택적 플러그인 설정을 선언합니다. 코드를 가져오지 않고
설치 파일 다이제스트를 확인하는 명령은 다음과 같습니다.

```bash
truthound-mcp plugin-fingerprint --distribution your-reviewed-distribution
```

타입이 콜백인 필드는 `{"operator_capability":"approved-capability-id"}`로 연결합니다.
템플릿의 별도 `object_parameters`는 생성자 필드에 승인된 네이티브 capability 객체를
연결합니다. 완전한 Polars 식·자료형 매핑·참조 프레임 등을 전달할 수 있습니다.
고정 JSON·객체 매핑·호출자 설정은 중복할 수 없으며 일반 JSON을 실행 참조로
해석하지 않습니다. 승인 전에는 확장을 가져오지 않습니다. 소유 작업 프로세스가
배포·버전·파일을 다시 확인하고 실제 Core `PluginManager`의 로드·활성화·종료를
사용합니다. 자동 탐색·설치는 비활성화됩니다. 패키지 변경은 실행을 막고 불확실한
효과는 성공으로 재생하지 않습니다. 편집 가능 설치와 바이너리 전용 진입점은 지원하지
않습니다. 확장은 호스트 사용자 권한으로 실행되며 **운영체제 샌드박스나 전이 의존성
잠금이 아닙니다.** 승인 전에 코드와 데이터·네트워크 접근을 검토해야 합니다.

승인이 객체와 생성자 필드의 타입 적합성을 보장하지는 않습니다. 잘못된 일반 JSON
설정은 실행 전에 거부되지만, 승인된 네이티브 객체는 Core 생성자를 통과한 뒤 검증
도중 실패할 수 있습니다. Core가 기록한 실행 실패는 `core_execution_failed`로
유지하고 정식 결과를 보존합니다. 같은 멱등 키로 재요청해도 확장을 다시 실행하지
않습니다. 이는 생성자 거부나 효과 불확실 상태와 다르며, 승인 객체의 오류 검증을
일반 입력의 알고리즘 오류 검증으로 계산하지 않습니다.

이 구현은 **Core 전체 기능 동등성을 달성한 상태가 아닙니다.**
알고리즘별 전체 동등성·모든 데이터소스 수명주기·임의 프로파일 구성·전체 기계학습 영속화·
계보 추적기와 이력 및 대화형 렌더링·체크포인트·스트림·스토어·전체 플러그인 수명주기 도구·보안 관리·영속 작업은
아직 완성하지 않았습니다. 기능 조회도 전체 동등성이 거짓임을 표시하며 누락을 열거한
것이 구현이나 실계정 검증을 의미하지 않습니다. 원격 서버·자격 증명·유료 호출은
추가하지 않습니다.

동등성 증거는 생성자 거부·네이티브 실행 실패·알고리즘 내부에서 잡힌 예외를 구분합니다.
예를 들어 Core 쿼리·표현식 검증기는 예외를 `ExecutionIssue` 대신 심각한 품질 문제로
기록할 수 있으며 MCP는 이 구분과 전체 문제를 보존합니다. 모든 등록 검증기의 생성자
거부를 확인했다고 알고리즘별 전체 런타임 오류 분기까지 증명한 것은 아닙니다.

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

OpenAI 플러그인은 Core와 별도인 버전 `0.1.4`이며 공개 PyPI의
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
0.1.2는 표준 `./` 접두사를 사용하고, ZIP 생성 전에 두 portable manifest를
저장소에 고정한 공식 Agent Plugins 1.0.0 schema로 검증합니다.
0.1.4는 회색 외곽 여백이나 이미지 자체의 둥근 타일 없이 정사각형 아이콘·로고의
가장자리까지 주황 배경을 채웁니다. 모서리 클리핑과 UI 자체 여백은 host가
관리합니다. 기존 설치에서는 플러그인을
업데이트하고 클라이언트 캐시를 새로고침하기 전까지 이전 이미지가 보일 수 있습니다.
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

MCP CI는 OS·Python 조합마다 전체 필수 테스트를 결정적인 4개 분할로 나누되
15분 job 제한은 유지합니다. 모든 분할이 성공한 뒤에만 패키징을 시작합니다.
합산 gate는 전체 수집 목록·소스 해시·setup/call/teardown 단계와 JUnit을 대조하고,
누락·중복·SKIP·실패를 거부합니다. 일부 분할의 통과만으로 전체 검증 완료를
선언하지 않습니다.

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
