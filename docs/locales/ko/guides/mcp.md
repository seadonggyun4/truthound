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
`name="orders"`인 Core `BaseDataSource` 또는 `AsyncBaseDataSource`를 반환해야 합니다. 예를 들어 검토한
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
`sources_sample_source`,
`sources_validate`에 사용할 수 있습니다. 검증은 실제 상주 소스를 Core의
`source=` 경로에 전달하며 DataFrame 복사본으로 대체하지 않습니다. 연결 설정 노출을
막기 위해 바인딩 소스의 조회 결과는 `config=null`입니다. 데이터 최신성·캐시·엔진은
네이티브 공급자의 의미를 따르며 불변 스냅샷이나 자동 재연결 서비스가 아닙니다.
라이브 바인딩 검증의 `input_sha256`·`input_identity`는 미확정으로 유지합니다.
소스 핸들의 리소스 해시는 바인딩 이름을 나타내며 DB 행의 해시가 아닙니다.
비동기 소스는 상주 worker가 소유한 이벤트 루프에서 Core `AsyncToSyncAdapter`를
사용합니다. 네이티브 크기 제한·스키마 helper를 유지하고, 비동기 샘플도 같은
루프에서 연결한 뒤 각 연결을 닫고 루프를 종료합니다. 진입 실패 시 부분 연결을
정리하며 자동 재연결·재시도는 추가하지 않습니다. 로컬 비동기 소스의 수명주기와
표준 검증 결과 대조를 테스트했으며 외부 MongoDB·Elasticsearch 등 원격 제공자는
미검증입니다. 임의 프로토콜 객체는 바인딩 factory 계약에서 받지 않습니다.

동일한 root·표본 소스에 `sources_read`, `sources_profile`, `sources_scan`,
`sources_mask`, `sources_learn`도 사용할 수 있습니다. 각각 Core의 공개
`read(source, sample_size=...)`, `profile(source=...)`, `scan(source=...)`,
`mask(source=...)`, `learn(source=...)`를 직접 호출하며 provider의 materialization
한도와 표본 동작을 유지합니다. 읽기는 Core의 고정 seed 42를 사용하고 선택적인
`sample_size` 범위는 1–10,000,000입니다. profile·scan은 source 핸들만 받습니다.
mask는 선택적인 `columns`, 네이티브 `redact`·`hash`·`fake` 전략과 `strict`
(기본 false)를 받습니다. learn은 `infer_constraints`(기본 true)와
`categorical_threshold`(기본 20, 범위 0–10,000)를 받습니다.

읽기·마스킹은 페이지 조회와 후속 작업을 위한 불변 Arrow **asset**을 반환합니다.
fake 값은 호출당 한 번 생성하고 상주 소스는 변경하지 않습니다. profile·PII 보고서는
네이티브 JSON 표현과 공개 source 이름을 유지합니다. 학습된 **schema** 핸들은
`schemas_inspect`, `schemas_export`, `sources_validate(schema_id=...)`에 다시
사용할 수 있습니다. 보고서·스키마에 데이터 값이 포함될 수 있으므로 기존 민감 결과
정책을 적용합니다. 모든 결과는 소스의 권한·만료·수명에 종속됩니다. 출력 용량 초과나
네이티브 작업 실패는 부분 결과를 공개하지 않고 정상 소스 연결을 유지합니다.
PII 검사는 Core의 표본 탐지 동작을 유지하며 모든 민감 값을 찾았다는 보장이 아닙니다.

`sources_sample`은 기존 Arrow 스냅샷 계약을 유지합니다. `sources_sample_source`는
같은 `source_id`·`n`·선택적 `seed`로 기존 worker 안의 실제 네이티브 표본 datasource를
유지하고 **source** 핸들을 반환합니다. factory로 provider를 다시 생성하거나 Arrow
frame으로 대체하지 않습니다. 표본 소스도 inspect·연결·열·크기 검사·검증·Arrow 캡처와
추가 네이티브 표본을 지원합니다. worker당 자식 핸들은 최대 16개입니다. 네이티브 표본·
cache 의미가 유지되며 provider가 새 객체 대신 기존 객체를 반환할 수도 있습니다.

자식은 소유자의 연결·event loop 수명을 빌립니다. 자식 해제는 해당 객체의 마지막 별칭일
때 네이티브 context를 닫고 root·형제는 유지합니다. 부모 해제는 모든 후손을 무효화하고
자식 context를 먼저 닫은 다음 root·provider·plugin을 정리합니다. 제한 시간 내 정리가
실패하면 소유 worker를 종료하며 provider 측 cleanup은 보장하지 않습니다. 파생 live
표본의 입력 내용 hash는 미확정이고 매 접근에서 부모 권한을 재확인합니다. 별도 SQL·
Python·provider 경로·자격 증명·callback을 요청에 넣을 수 없습니다. 로컬 Polars와 실제
비동기 소스로 중첩 표본·검증·Arrow 캡처·공유 loop 종료 순서·한도 재사용을 검사합니다.


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
| `openlineage_services_open`, `openlineage_services_execute` | 고정 네이티브 emitter 승인·실행 전환·그래프 전송·결과 불명 판정·종료 |
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
`config.sample_size`는 앞부분 행 표본의 크기를 정합니다. 명시한
`options.sample_size`가 이를 재정의하며 원래 설정 객체는 변경하지 않습니다.
유효 설정은 기본 상관관계·패턴·분포 분석의 선택에도 적용됩니다. 이 경로의
표본 선택에는 `random_seed`가 사용되지 않습니다.
알 수 없는 설정·임의 가져오기·사용자 정규식 정의는
거부합니다. 사용자 확장과 suite 실행은 아래의 별도 승인된 `profiler_services_*` 수명주기를
사용합니다. 스트리밍·증분·분산 프로파일러는 별도 계약이 필요합니다.

고급 네이티브 프로파일은 구성 요소의 실패를 `execution_issues`에 보관합니다.
단계·구성 요소·예외 유형·선택적 컬럼명만 포함하고 예외 본문은 제외합니다.
컬럼과 테이블의 `complete` 속성은 선택한 분석의 완료 여부이며, 표본 설정이
있을 때 모든 입력 행을 분석했다는 뜻은 아닙니다. 불완전 결과의 JSON에는
`complete: false`와 진단이 포함되며 네이티브 저장·복원과 사전 변환도 이 상태를
유지합니다. MCP는 `execution_status: failed`, `native_profile_incomplete`와
소유권이 있는 부분 프로파일 산출물을 반환합니다. 읽기에는 민감 결과 허용과
원본 접근 권한이 필요합니다. 불완전 프로파일에서는 생성기를 호출하기 전에
규칙 생성을 중단합니다. 네이티브 suite 생성기도 선택한 생성기의 실패를
부분 성공으로 처리하지 않고 예외로 보고합니다.

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

### 승인된 사용자 프로파일러 세션

`profiler_services_open`·`profiler_services_execute`는 사용자 네이티브 구성을
승인된 단일 작업자에 유지합니다. 설치된 capability factory는 비동기 context
manager로 `truthound.mcp.profiler_services.ProfilerRuntime`을 반환합니다.
`profilers`에는 네이티브 `Profiler`, `generators`에는 사용자 규칙 생성기를 구성한
`ValidationSuiteGenerator`, `executors`에는 운영자의 registry/listener를 가진
`SuiteExecutor`를 지정합니다. 선택적 `profile_slots`·`suite_slots`는 고정 별칭을
절대 경로 `Path`에 연결합니다. 모든 설정·사용자 패턴·분석기·추론기·물리 경로와
종료는 factory가 소유합니다. 비공개 `plugins` 바인딩에 capability를 추가하고
`bindings.json`에 `"profiler_services":{"local":"profiler-factory"}`를 매핑합니다.
모델은 import 문자열·콜백·정규식 정의·검증기 클래스·경로를 전달하지 않습니다.
설치 코드는 신뢰된 코드이며 각 정확한 작업의 승인을 받습니다.

열기에서 받은 `resource_id`를 `service_id`로 사용하고 `binding_id`와 고유한
`idempotency_key`를 전달합니다.

| 명령 | 네이티브 효과 |
|---|---|
| `inspect` | 구성·슬롯·보관 결과의 제한된 별칭 목록 |
| `profile` (`profiler_id`, `asset_id`) | 선택한 구성을 실행하고 원래 `TableProfile` 보관 |
| `profile_get`, `profile_export` (`profile_id`) | 프로파일 조회 또는 재진입 가능한 advanced-profile 산출물 내보내기 |
| `profile_import` (`artifact_id`) | 소유한 advanced-profile 산출물을 디코딩하고 네이티브 `ProfileSerializer`로 복원 |
| `profile_save`, `profile_restore` (`slot_id`) | `profile_id` 저장 또는 운영자 슬롯에서 복원 |
| `generate_suite` (`profile_id`, `generator_id`) | `strictness`와 선택적 category/confidence 필터·`name`으로 네이티브 suite 생성 |
| `suite_get`, `suite_export` (`suite_id`) | 원래 생성 suite 조회·내보내기 |
| `suite_save`, `suite_restore` (`slot_id`) | `suite_id` 저장 또는 고정 슬롯의 네이티브 JSON 복원 |
| `suite_execute` (`suite_id`, `executor_id`, `asset_id`) | 선택적 `parallel`·`fail_fast`·`dry_run`으로 실행 |
| `profile_release`, `suite_release` | 작업자 내 결과 별칭 해제 |
| `slot_delete` (`slot_kind`, `slot_id`) | 선택한 영구 프로파일·suite를 명시적으로 삭제 |

소유자별 프로파일과 suite는 각각 최대 16개이며 세션 바이트 한도를 따릅니다.
Core 직렬화 후 고정 슬롯을 교체해 저장하며 트랜잭션 DB나 전원 중단 내구성을
보장하지 않습니다. 슬롯은 결과를 보관하고 구성 설정은 고정된 factory에 남습니다.
새 소유자의 슬롯 복원에는 새 승인이 필요하며 이전 세션 입력과 별도로 해당 저장
데이터 읽기를 명시적으로 승인합니다. 산출물 가져오기는 원본 산출물의 접근 권한
의존성을 유지합니다. 별도 소유자로 가져오며 자기 자신·하위 산출물의 의존성 순환은
승인 전에 거부합니다. 규칙을 검증기로 변환할 때 신뢰 코드가 실행될 수 있으므로
임의 suite 산출물 가져오기는 제공하지 않고 운영자 소유 슬롯에서만 복원합니다.

불완전 프로파일은 진단과 `native_profile_incomplete`를 반환하며 suite를 생성할 수
없습니다. 같은 실패 요청을 재전송해도 분석기를 다시 실행하지 않습니다.
검증기 변환 누락·실행 오류·건너뛴 규칙은 성공한 빈 실행 대신 실패한
`native_suite_incomplete` 산출물을 만듭니다. 실행이 완료되어도 데이터가 규칙을
위반하면 `success: false`일 수 있습니다. dry-run 성공은 생성 가능성만 확인하며
데이터 검증의 증거가 아닙니다. 네이티브 보고서에는 예외 본문 대신 유형을 보관합니다.
동기·비동기 실행은 전달한 네이티브 registry와 동일한 실패 집계를 사용합니다.
네이티브 구성 요소·생성기·검증기의 다음 콜백 전에 권한을 확인하며 소유자도 정책
변경을 감시합니다. 해제·만료·시간 초과·철회·세션 종료 시 소유자와 factory 자원을
회수합니다. 증분·청크 네이티브 프로파일러도 같은 프로파일/suite 보관 수명주기를 사용합니다.
factory는 `incrementals`(`IncrementalProfiler`), `streaming`(`StreamingProfiler`),
`chunk_factories`(네이티브 `ChunkIterator`를 반환하는 비동기 context manager),
`stream_progress` 콜백 별칭을 지정할 수 있습니다. `incremental_profile`은
`incremental_id`, `asset_id`, 선택적 `previous_profile_id`·`force_refresh`를 받아
프로파일과 함께 실제 재분석·재사용 컬럼과 변경 사유를 반환합니다.
`stream_profile`은 `streaming_id`, `stream_id`, 선택적 `progress_id`를 받습니다.
운영자 청크 소스를 직접 읽고 정상·실패 모두 iterator와 소스 context를 닫으며,
모델이 소스 경로나 콜백을 전달하지 않습니다.

증분 결과는 네이티브 modulo 표본 설정·hash를 담은 컬럼별
`content_fingerprint`를 저장합니다. 행·null 수만 같다는 이유로 재사용하지 않으며
지문이 없거나 불완전한 컬럼은 다시 분석합니다. 표본 밖 변경은 놓칠 수 있으므로
전체 데이터 무결성 검사는 아닙니다. 청크 오류는 현재 실행의 불완전 진단으로
남기며 청크 스키마가 달라지면 예외를 발생시키고 소스를 닫습니다. 청크 분석은
네이티브 스트리밍 통계이며 고급 DataProfiler의 모든 분석기를 실행하지 않습니다.
서로 다른 값의 추적 공간은 cardinality에 따라 커질 수 있어 청크 처리만으로
상수 메모리를 보장하지 않습니다. 분산 프로파일러 수명주기는 별도 계약입니다.

factory의 `caches`는 별칭을 `ProfileCache` 객체에, `cache_keys`는 네이티브
캐시 키 객체에, `cache_patterns`는 고정 논리 glob 패턴에 연결합니다.
`cache_get`·`cache_set`·`cache_get_or_compute`는 `cache_id`·`cache_key_id`를
받습니다. 저장에는 `profile_id`, 계산에는 `profiler_id`·`asset_id`가 추가됩니다.
선택적 `ttl_seconds=0`은 즉시 만료이며 생략하면 네이티브 기본 TTL을 따릅니다.
`cache_invalidate`는 키 하나, `cache_invalidate_pattern`은 `pattern_id`로 선택한
패턴을 무효화합니다. `cache_clear`는 선택한 캐시를 비우고 `cache_stats`는 일부
카운터를 반환합니다. 운영자는 데이터셋·설정·구현 버전을 구분해 키를 구성해야 하며
이 바인딩이 새로운 키 알고리즘을 만들지는 않습니다.

캐시 hit은 원래 네이티브 프로파일을 보관하며 불완전한 hit은 실패 산출물입니다.
파일 캐시는 비공개 임시 파일 직렬화 후 교체합니다. 손상 또는 접근 통계 저장 실패는
기존 바이트를 보존하고 `ProfileCacheReadError`·`ProfileCacheWriteError`를 발생시킵니다.
작업자 오류 후 MCP intent는 `outcome_unknown`이며 자동 재실행하지 않습니다.
새로 승인한 소유자의 명시적 무효화로 처리합니다. 파일 패턴 무효화는 삭제 전에
모든 논리 키/파일 매핑을 검증합니다. 키가 저장되지 않은 기존 항목은 알려진 키로
먼저 읽거나 명시적으로 비워야 합니다. 지원하지 않는 사용자 backend는 0개를
무효화한 척하지 않고 예외를 발생시킵니다. Redis 파싱·연결 오류도 손상 값을
보존하고 안전한 오류 유형을 보고하며 생성·종료 실패 시 소유 client/pool 정리를
모두 시도합니다. Redis namespace 스캔은 물리 prefix를 escape하고 각 논리 키를
다시 확인합니다. `cache_stats`는 물리 키·경로·endpoint·원본 metadata·예외 본문을
제외합니다. `connected`는 네이티브 관측 상태이며 새 health check가 아닙니다.
현재 키 개수를 얻지 못하면 `native_cache_incomplete`를 반환합니다. 실제 로컬 파일
영속성과 합성 Redis 프로토콜 장애를 검증했으며 실제 Redis 서비스는 미검증입니다.
트랜잭션 캐시나 전원 중단 복구를 보장하지 않습니다.

생성 suite 실행은 `timeout_seconds`(유한 양수, 최대 3600)와
`max_workers`(1–32)도 받습니다. 생략하면 선택한 executor의 설정을 유지하며 전체
MCP 작업에는 서버 정책의 시간 제한이 계속 적용됩니다. 검증기별 제한은 네이티브
`ProcessExecutionStrategy`를 사용합니다. 검증기 상태는 import/pickle 가능해야 하며
직렬화한 바인딩 guard가 검증 직전에 권한을 확인합니다. 직렬화 실패는 명확한
실패이며 중단할 수 없는 스레드로 우회하지 않습니다. 네이티브 deadline에는 자식
시작과 결과 대기가 포함되며 부모 쪽 직렬화와 마지막 정리에는 추가 시간이 들 수
있습니다. 자식은 별도의 상속 process group·비공개 결과 pipe·부모 수명 lease를
소유합니다. 시간 초과·정상 완료·부모 종료 모두 해당 실행 범위를 닫습니다.
오류는 원문·traceback 없이 유형으로 전달하며 결과 직렬화 실패를 시간 초과로
표시하지 않습니다.

일반 네이티브 스레드 전략은 시간 초과 후에도 daemon이 실행 중일 수 있으므로
`was_terminated`는 false, `outcome_unknown`은 true이며 자동 timeout 재시도를
중단합니다. 생성 suite의 검증기별 deadline에는 이 전략을 사용하지 않습니다.
병렬 fail-fast는 대기 작업을 취소하고 이미 실행 중인 작업을 회수하며 건너뛴
작업도 실패 집계에 남깁니다. 규칙 시작·완료 listener는 직렬화하여 호출하고
실패를 숨기지 않습니다. 설치 확장 코드의 OS sandbox를 제공하는 기능은 아닙니다.

### 승인된 상시 스키마 감시

`schema_monitor_open`은 세션 소유 프로세스에서 네이티브 `SchemaWatcher` 또는
`AsyncSchemaWatcher`를 실행합니다. 입력은 `binding_id`, `mode`(기본 `sync` 또는
`async`), `poll_interval`(0.1~3,600초, 기본 60), `max_events`(1~1,000, 기본 100,
운영자 `max_items` 이하), `idempotency_key`입니다. 변경·버전·호환성 파괴 알림은
Core가 계산하며 어댑터에서 별도 비교 알고리즘을 구현하지 않습니다.

운영자는 비공개 `bindings.json`에서 설치된 플러그인의 배포판·버전·진입점·SHA-256
고정 설정과 함께 다음 매핑을 구성합니다.

```json
{"schema_monitors":{"orders-watch":{"source":"orders-schema-factory","notifier":"orders-alert-factory"}}}
```

두 이름은 고정 플러그인의 `capabilities`에 등록되어야 합니다. 인자 없는 팩토리가
각각 Core `SchemaSource` 규약 객체(예: `FileSchemaSource`)와 선택적인
`AlertNotifier` 객체(`send(alert) -> bool`)를 반환합니다. 운영자가 구현한 HTTP·이메일
전송 함수는 네이티브 `CallbackNotifier`로 연결할 수 있습니다. 내장 원격 서버나
모델이 지정하는 URL·import 문자열·자격증명 입력은 없습니다. 대상·자격증명·연결
정책은 운영자가 관리합니다. 설치 플러그인은 신뢰된 코드이며 샌드박스가 아닙니다.
알림에는 스키마 상세와 로컬 경로를 포함할 수 있는 소스 식별자가 들어갑니다.
전송 승인은 고정된 대상에 이 정보를 공개하는 승인입니다.

```json
{"binding_id":"orders-watch","mode":"async","poll_interval":60,"max_events":100,"idempotency_key":"orders-watch-01"}
```

최초 호출은 설치 확장 실행·지속적인 스키마 읽기·선택적인 외부 알림 전송의
승인 계획을 반환합니다. 기존 운영자 승인 절차 후 동일 요청을 반복합니다.
승인 전에는 플러그인 로딩·읽기·전송이 없습니다. 반환된 `resource_id`를
`schema_monitor_status`·`schema_monitor_stop`의 `monitor_id`로 사용합니다.
두 호출은 산출물 리소스를 반환하며 `artifacts_read`로 상태·오류·검사/이벤트 수·
성공 전송 수·네이티브 버전과 알림을 읽습니다. 최초 등록은 기준 스키마이고,
호환성을 깨는 변경만 전송합니다. 동기 감시는 첫 주기 이후, 비동기 감시는
네이티브 동작대로 시작 직후 첫 검사를 수행합니다.

후속 도구 호출 없이 감시하다가 중지·해제·만료·세션 종료 시 멈춥니다. 중지 호출은
네이티브 감시 종료를 기다리고 플러그인을 종료합니다. 해제·만료·시간 초과는 해당
소유 작업자만 강제 회수하며 세션 종료가 정상 전송 완료를 보장하지는 않습니다.
검사·전송 전에 바인딩과 패키지 식별자를 다시 확인합니다. 부모 프로세스도 0.1초
간격으로 정책 변경을 확인하여 후속 도구 호출 없이 작업자를 회수합니다.
소스 실패·전송 거절/결과 불명확·승인 철회·이벤트/용량 한도
도달 시 추가 효과를 중지하고 `state="failed"`를 보고합니다. 상태 조회 자체의
성공이 감시 성공을 의미하지 않습니다. 네트워크 오류 전에 원격 효과가 발생했을
수 있어 자동 재전송하지 않습니다. 용량은 운영자 파일/세션 한도와 16 MiB 중
작은 값입니다. 승인 요청 재전송은 기존 핸들을 반환하며 새 작업자를 만들지 않습니다.

이 감시 이력은 `schema_history_*` 체크포인트와 별개인 전용 메모리 상태입니다.
세션 간 자동 재시작·영구 전송 대기열·정확히 한 번 전송 보장·감시 일시정지/재개·
소스 실행 권한 이전은 이 세션 소유 API에서 제공하지 않습니다. 로컬 실제 HTTP 전송과 동기·비동기
네이티브 동등성을 검증하며 클라우드·이메일·브로커 실서비스는 미검증입니다.

### 영구 스키마 감시 복구와 알림 전송 대기열

별도 승인되는 `schema_watch_register`에는 같은 바인딩·모드·주기·이벤트 설정,
`lifetime_seconds`(1~2,592,000초, 기본 86,400), 새 `idempotency_key`를 전달합니다.
운영자 CLI로 반환된 계획을 검토·승인한 후 동일 요청을 반복합니다. 승인에는 영구
플러그인 실행·재시작 후 자동 읽기·선택적인 외부 알림 전송이 명시됩니다.
등록 자체는 플러그인을 로딩하거나 검사하지 않습니다. `watch_id`는 세션 핸들이
아닌 영구 식별자이며 모델이 URL이나 자격증명을 지정하지 않습니다.

기존 `python -m truthound.mcp worker`를 등록 때와 **동일한** Core 프로필·민감 결과
동의·허용 디렉토리·비공개 제어 디렉토리·한도로 실행합니다. `--once`는 한 차례
검사하며, 기본 포그라운드 실행은 중지될 때까지 반복합니다. worker를 재시작하면
새 MCP 세션이나 재승인 없이 승인 만료·비활성화·정책/바인딩 변경·실패 전까지
감시를 복구합니다. OS 서비스 설치나 부팅 시 자동 실행은 하지 않습니다.
각 회차는 감시별로 대기 알림 한 건 전송 또는 예정된 네이티브 검사 한 번을
수행합니다. 전송 우선이므로 실제 검사 간격은 worker 주기와 대기량에도 좌우됩니다.

Core 동기·비동기 `check_now`, 버전 직렬화와 호환성 파괴 알림 생성을 재사용합니다.
서명된 SQLite 트랜잭션으로 마지막 스키마/체크섬·이력·카운터와 신규 알림을 함께
저장합니다. 내부 스키마 상태는 닫힌 값 규약과 빈 Arrow IPC의 Polars 타입 메타데이터로
손실 없이 복구합니다. 지원하지 않거나 손실이 생기는 값은 문자열로 바꾸지 않고 실패합니다.
이력·알림의 표현은 Core 직렬화를 유지합니다. 이전 개발 버전의 형식 번호 없는
체크포인트는 증거를 보존하고 실패 처리하며, 암묵적 변환 대신 새 등록 승인이 필요합니다.
중단 중 변경은 마지막
저장 기준과 재시작 시 비교하며, 검사 사이의 중간 변경을 재구성하지는 않습니다.
이 저장 상태는 `schema_history_*` 체크포인트와 별개입니다.

대기열 상태는 `pending → sending → sent | unknown`입니다. 고정된 네이티브
알림 객체를 호출하기 전에 전송 시도를 영구 기록합니다. 명시적인 `True` 응답만
성공이며 예외·거짓 응답·전송 중 프로세스 종료는 `unknown`으로 보존하고 자동
재전송하지 않습니다. 종료된 검사 소유자는 복구하되 살아 있는 소유자의 작업은
다른 worker가 가져가지 않습니다. 플러그인이 막힌 중에도 부모 파이프가 닫히면
해당 격리 자식만 종료합니다. 이는 **자동 시도 최대 한 번** 규약이며 전송
보장이나 원격 효과의 정확히 한 번 처리는 아닙니다. 운영자 알림 구현도 숨겨진
재시도를 해서는 안 됩니다. 불명확한 결과는 대상 측 확인이 필요하며 자동 대사는
제공하지 않습니다.

`schema_watch_status`·`schema_watch_disable`은 `watch_id`와 선택적인
`include_snapshot: true`를 받습니다. 상태·만료·검사/전송 수를 조회하며 선택적인
산출물은 `artifacts_read`로 민감한 이력·대기열 본문을 제공합니다. 비활성화는
새 작업을 막지만 진행 중인 전송을 취소하지 못하며 증거는 보존합니다. 정의는
32개, 정의당 최대 1 MiB이며 운영자 파일/세션 한도도 적용합니다. 이벤트 한도도
유지합니다. 용량 초과는 실패로 처리하고 알림을 몰래 삭제하지 않습니다.
로컬 실제 HTTP·별도 프로세스 재시작·전송 직후 충돌을 검증하며 클라우드·이메일·
브로커 실계정은 미검증입니다. 아직 게시된 패키지에 포함되지 않은 개발 기능입니다.

#### 승인 기반 일시중지·대사·대기열 보존

`schema_watch_update`에는 `watch_id`, 현재 상태의 `revision`을 넣은
`expected_revision`, 새 `idempotency_key`, `action`이 필요합니다. 운영자 CLI로
계획을 검토·승인한 뒤 같은 요청을 반복합니다. 검토 후 상태가 바뀌면 오래된
계획은 거부되므로 상태를 다시 조회하고 새 키로 요청합니다. 완료 요청 재전송은
저장된 결과를 반환하며 작업을 반복하지 않습니다.

- `pause`는 기준 스키마와 대기열을 유지하고 새 검사·전송을 막습니다. 진행 중
  작업은 끝날 수 있습니다. `resume`은 만료되지 않은 일시중지 감시만 재개하며
  승인 기간을 연장하지 않습니다. 중간 변경을 재생하지 않고 마지막 기준과
  비교합니다. disabled/failed/expired 상태는 재개하지 않습니다. Core 검사를
  감싸는 영구 스케줄 제어이며 네이티브 비동기 감시에 pause 메서드가 있다는
  뜻은 아닙니다.
- `renew`는 `lifetime_seconds`(1–2,592,000)를 요구하며 승인 실행 시각부터
  만료 기간을 설정합니다. active/paused/expired/failed에 적용하며 disabled는
  거부합니다. 만료된 active 감시는 paused가 되고 failed는 명시적 복구 전까지
  유지됩니다. 이력과 대기열은 보존합니다.
- `restart`는 만료되지 않은 failed 감시를 paused로 전환합니다. 기준과 대기열을
  보존하므로 재개 전에 원인인 provider 오류를 해결해야 합니다. 호환되지 않는
  체크포인트는 `reset_checkpoint: true`, `accept_history_loss: true`로 초기화합니다.
  다음 검사에서 새 기준을 만들고 이벤트 한도를 초기화하지만 대기열과 증가하는
  알림 번호는 보존합니다. 버린 기준 구간의 변경은 놓칠 수 있습니다.
- `delete`는 active가 아니고 진행 중 임대가 없는 감시와
  `accept_history_loss: true`를 요구합니다. pending/sending/unknown 알림이 있으면
  `discard_unresolved: true`도 필요합니다. 정의 삭제와 deleted 결과를 원자적으로
  저장하며 슬롯을 반환합니다. 이후 상태 조회는 invalid_handle이고 정확히 같은
  삭제 요청은 저장된 결과를 반환합니다. 승인 기록·내보낸 artifact·백업·SQLite
  여유 페이지의 물리적 보안 삭제를 제공하지 않습니다.
- `confirm_delivery`·`discard`·`retry`에는 unknown 상태의 `alert_id`와 대상 측
  확인을 나타내는 `evidence_reference`가 필요합니다. 비밀정보·원시 데이터가
  아닌 사례 식별자를 사용합니다. confirmed는 운영자 확인이지 관측된 전송
  성공이 아니며 discarded는 폐기이지 전달이 아닙니다. retry는 추가로
  `accept_duplicate_risk: true`와 중복 외부 전송 가능성에 대한 승인이 필요합니다.
  같은 네이티브 알림을 한 번 더 시도하도록 대기시키며 승인 호출에서 전송하지
  않습니다. 감시가 일시중지 상태면 그대로 대기합니다. 결정별 계획·참조·시각과
  전송 시도 횟수를 기록합니다.
- `retention`은 `retention: {"max_age_minutes": 1440, "max_count": 100,
  "mode": "all"}`을 받으며 나이·개수 중 하나 이상이 필요합니다. Core의
  TimeBasedPolicy·CountBasedPolicy·CompositePolicy를 재사용합니다. all은 모든
  규칙이 보존할 때, any는 하나라도 보존할 때 유지합니다. `retention: null`은
  이후 정리를 해제합니다. 승인 시점과 후속 worker 순회에서 해결 시각이 있는
  sent/confirmed/discarded 항목만 삭제합니다. pending/sending/unknown 및 해결
  시각이 없는 기존 항목은 보호하며 진행 중 대기열은 재정렬하지 않습니다.
  상태에 정책과 누적 `pruned_notifications` 수를 제공합니다.

- `history_retention`은 같은 `retention` 형식을 사용해 체크포인트 버전과 알림에
  각각 적용합니다. 최신 버전·현재 검사 기준·이벤트/검사 횟수·대기열이 참조하는
  알림은 보존하므로 지정 개수를 초과할 수 있습니다. 오래된 부모 참조는 삭제된
  버전을 가리킬 수 있습니다. `retention: null`로 이후 이력 정리를 해제합니다.
  상태는 history_retention과 누적 pruned_history를 제공합니다. 알림 번호는
  정리·프로세스 재시작·체크포인트 초기화 이후에도 증가합니다.

보존 정리는 동일한 권한 아래 worker 순회에서 일시중지·비활성 감시에도 적용됩니다.
대기열 보존 정책만으로는 체크포인트 이력을 지우지 않으며 별도 이력 정책이 필요합니다.
두 보존 정책 모두 승인 기록·외부 사본을 지우거나 정의 슬롯을 반환하지 않으며
물리적 보안 삭제도 아닙니다. 기존 이력·이벤트·용량 한도는 유지됩니다.
대상 서비스 자동 조회나 정확히 한 번 전달을 보장하지 않으며 실서비스는 미검증입니다.

### 세션 소유 스키마 이력

`schema_history_create`는 전용 작업자에서 Core `InMemorySchemaHistory`
(`backend="memory"`, 기본값) 또는 `FileSchemaHistory`(`backend="file"`)를
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
| `watch_add`, `watch_check` | `name`, `asset_id` | 최초 스키마 등록 또는 스냅샷 갱신과 네이티브 즉시 검사; 상태 또는 변경 이벤트 |
| `watch_remove`, `watch_status` | 삭제 시에만 `name` | 삭제 결과 또는 네이티브 소스 상태 |
| `alerts`, `alert_stats` | 없음 | 네이티브 로컬 알림 이력 또는 통계 |
| `alert_get`, `alert_acknowledge`, `alert_resolve` | `alert_id` | 네이티브 알림 또는 확인·해결 결과 |
| `snapshot` | 없음 | 파일 방식의 네이티브 JSON 체크포인트 산출물; 영구 저장 승인은 아님 |

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

감시 등록·검사·삭제와 알림 확인·해결도 revision을 증가시킵니다. 감시는 명시적으로
전달한 불변 자산의 스키마를 비교하며 외부 파일을 자동으로 읽거나 백그라운드
스레드를 시작하지 않습니다. 최초 등록은 기준 스키마이며 이력 버전을 저장하지
않습니다. 변경은 네이티브 `HistoryEventHandler`·`AlertingEventHandler`를 통해
버전으로 저장되고, 호환성을 깨는 변경은 로컬 알림이 됩니다. 이벤트·알림은
산출물로 읽으며 이메일·웹훅·외부 알림은 보내지 않습니다. 소스와 알림 수는
`max_versions`로 제한하며 알림 한도에 도달하면 조용히 삭제하지 않고 검사를
거부합니다. Core가 로그로만 보고하는 저장·감시 실패도 MCP에서는 실패입니다.

세션 간 복구에는 `backend="file"`로 생성한 뒤 `schema_history_persist`에
`history_id`·`expected_revision`·`idempotency_key`를 전달합니다. 기존 운영자
승인 절차와 제어 디렉터리가 필요합니다. 서명한 불변 체크포인트에는 네이티브
버전 파일·색인·로컬 알림과 확인·해결 상태가 포함되며 데이터 행은 저장하지
않습니다. 결과는 `snapshot_id`·`sha256`·`bytes`입니다. 같은 revision의 승인된
요청을 재전송하면 최초 저장 결과를 반환합니다.

```json
{"history_id":"<history.resource_id>","expected_revision":1,"idempotency_key":"history-checkpoint-01"}
```

새 세션에서 `schema_history_resume`에 `snapshot_id` 또는 최초 `idempotency_key`
중 하나만 전달합니다. 서명·파일 해시·운영자 정책·원본 소스와 작업공간 권한을
다시 확인하고 revision 0의 새 전용 이력 핸들을 생성합니다. 버전 ID·시각·스키마와
알림은 보존하지만 이전 핸들·감시 등록은 복구하지 않습니다. 복구본은 서로 독립된
작업 복사본이며 공유 쓰기 디렉터리가 아닙니다. 후속 변경을 보존하려면 새
체크포인트를 승인·저장해야 하며, 저장하지 않은 변경의 자동 장애 복구는 없습니다.
설치된 실행 코드 소스 바인딩은 이 저장 경로로 권한을 이전할 수 없습니다.

체크포인트는 건당 16 MiB(운영자 파일 제한이 더 작으면 그 값), 최대 128건·총
32 MiB입니다. `storage_retention_preview`와 승인된 자동 보존 정책의
`kind="histories"`, `storage_delete`의 체크포인트 ID 삭제를 지원합니다. 삭제하면
활성 복구 핸들도 무효화됩니다. 전용 작업 파일은 해제·만료 시 제거되며 승인한
체크포인트는 명시적 삭제 또는 승인한 보존 정책의 삭제 전까지 유지됩니다.
임의 디렉터리·아카이브·압축 이력 가져오기는 허용하지 않습니다.

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
받지 않습니다. 이 미출시 checkout의 `strict=true`는 보고된 문제가 하나라도 있으면
호환 판정을 거부하고 level `none`을 반환합니다. 실패하거나 잘못된 타입을 반환한 규칙은
`CompatibilityAnalysisError`로 실패하며 성공 artifact를 만들지 않습니다. 선택형
`baseline_nullability`·`current_nullability`는 컬럼별 선언 bool이며 두 map을 함께
제공하고 각각 최대 1,000개입니다. nullability 규칙을 선택해야 하며 공통 컬럼의 선언은
양쪽에 있어야 합니다. Core는 nullable→non-nullable 강화를 breaking으로 판단하고
없는 컬럼·타입이 잘못된 선언은 거부합니다. Polars dtype만으로 nullability를 선언하지는
않으므로 map 생략은 nullability 검증이나 표본 관측값으로의 제약 추론이 아닙니다.
규칙의 명시적 순서·중복 선택은 유지됩니다.

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
상주 native worker의 `HOME`·`USERPROFILE`은 Windows에서도 소유한 임시 작업
디렉터리로 지정합니다. 기본 native cache도 임시이며 영속 상태는 운영자가
고정한 slot이나 factory 설정이 필요합니다. 운영자의 실제 home이나 자격 증명은
상속하지 않습니다.
비공개 worker 출력 sink는 Windows SDK 진행 출력을 포함해 UTF-8을 명시하며,
이 stream은 MCP protocol 응답과 분리합니다.
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

### 승인된 네이티브 체크포인트 실행

개발 Core 프로파일의 `checkpoints_run`은 검증 작업과 달리
`truthound.checkpoint.Checkpoint.run`을 직접 호출합니다. 구성된 액션,
알림 조건·재시도·선택형 라우팅을 그대로 사용하며 게시된 3.2.0에는 없습니다.

운영자는 신뢰하는 플러그인을 설치하고 비공개 `bindings.json`에 배포판 버전,
해시·진입점을 고정합니다. 기존 `plugins` 바인딩에 팩토리 역량을 노출하고
`"checkpoints": {"daily": "checkpoint-factory"}`를 추가합니다. 인자 없는
팩토리는 네이티브 `Checkpoint`를 반환하는 동기 컨텍스트 관리자를 제공해야
하며, 종료 시 데이터소스와 액션 자원을 정리해야 합니다. 모델 인자나 공개
메타데이터에 자격증명을 넣지 마세요. 설치 코드는 호스트 사용자 권한으로
실행되며 샌드박스가 아닙니다.

```json
{"binding_id":"daily","use_router":true,"idempotency_key":"daily-reviewed-run-01"}
```

첫 요청은 플러그인을 가져오기 전에 승인 계획을 반환합니다. 기존 운영자
승인 CLI로 검토·승인한 뒤 동일 요청을 반복하세요. 승인에는 고정된 팩토리의
액션·외부 알림·네이티브 재시도가 포함됩니다. 바인딩·패키지·정책 변경은
권한을 무효화합니다. 모델이 임의 가져오기·연결·액션·컨텍스트 객체를
요청에 지정할 수 없습니다.

반환 핸들은 풍부한 값 인코딩을 사용한 네이티브 `CheckpointResult.to_dict()`
민감 산출물입니다. `artifacts_read`로 읽어 `validation_run`, `action_results`,
상태·오류를 확인하세요. 전송 성공은 결과 보관 성공이지 검증·알림 성공이
아닙니다. 민감 결과 허용이 필요하며 네이티브 진단에는 비공개 정보가 포함될
수 있습니다.

동일 요청 재생은 보관 결과를 반환하며 액션을 재실행하지 않습니다. 시간 초과,
종료·실행 중 권한 철회·확인 불가능한 실패는 `outcome_unknown`으로 남기고
자동 재시도하지 않습니다. 승인된 네이티브 액션 재시도는 별개이며 외부 효과를
반복할 수 있습니다. 프로세스 종료는 외부 효과를 취소하거나 서드파티의 정상
정리를 보장하지 않습니다. 트리거·체크포인트 예약·에스컬레이션 서비스는
시작하지 않습니다. 원격 실서비스는 미검증입니다.

`checkpoints_run_async`는 네이티브 `AsyncCheckpoint.run_async` 경로를 별도로
승인합니다. `binding_id`와 `idempotency_key`를 받으며 `use_router`는 받지
않습니다. 인자 없는 바인딩 팩토리는 `AsyncCheckpoint`를 제공하는 **비동기
컨텍스트 관리자**여야 하고 종료 시 실행기·데이터소스·액션을 정리해야 합니다.
결과 산출물·권한 철회·불명확한 결과 규칙은 동일합니다. 동기 Checkpoint를
단순 스레드 래퍼로 변환하는 기능이 아닙니다.

### 네이티브 체크포인트 서비스

개발 Core 프로파일은 `checkpoint_services_open`과
`checkpoint_services_execute`도 제공합니다. 팩토리는 비공개 `checkpoints`
바인딩에 등록합니다. `kind="runner"`의 팩토리는 중지된 네이티브
`CheckpointRunner`를 제공하는 동기 컨텍스트 관리자이며, 체크포인트·트리거·
콜백·대기열 한도는 운영자가 구성합니다. `kind="escalation"`은 중지된 네이티브
`EscalationEngine`을 제공하는 비동기 컨텍스트 관리자이며 정책·저장소·예약기·
알림 처리기는 운영자가 지정합니다. 소유자는 종료 시 서비스와 모든 자원을
정리해야 합니다. 모델은 팩토리·자격증명·알림 목적지·임의 컨텍스트·파이썬
객체를 지정할 수 없습니다.

```json
{"binding_id":"daily-runner","kind":"runner","idempotency_key":"runner-open-01"}
```

운영자 승인 CLI로 반환 계획을 승인한 뒤 같은 요청을 반복합니다. 반환된
`monitor_...` 핸들은 현재 MCP 세션 소유입니다. 열기만으로 예약을 시작하지
않습니다. 상태 조회·대기열 소비를 포함한 각 명령은 핸들에 묶인 별도 승인이
필요합니다.

```json
{"binding_id":"daily-runner","service_id":"monitor_<returned-id>","command":"start","idempotency_key":"runner-start-01"}
```

- 두 종류 모두 `start`, `stop`, `status`를 지원합니다.
- 실행기는 `results`와 `max_results`(1–1000, 기본 100, 정책 한도도 적용)를
  지원합니다. 네이티브 결과 대기열을 **소비**합니다. 승인 요청 재생은 보관한
  산출물을 반환하며 다시 소비하지 않습니다. 네이티브 대기열 용량·초과 결과
  폐기 동작은 바꾸지 않습니다.
- 에스컬레이션 엔진의 `trigger`는 `incident_id`와 선택적인 `policy_name`을
  받습니다. `record`, `acknowledge`, `resolve`, `cancel`, `escalate`에는
  `record_id`가 필요합니다. 확인·해결·취소 주체는 모델이 지정한 사람 이름이
  아니라 `mcp-approved-operator`로 기록합니다. 허용되지 않는 상태 전이를
  강제하지 않습니다. 네이티브 실패 결과를 전송 성공으로 바꾸지 않습니다.
  통계의 정수 키 `escalations_by_level`은 키 자료형 보존을 위해
  `[{"level":1,"count":2}]`로 인코딩합니다.

명령 결과는 `artifacts_read`로 읽으며 민감 자료이고 서비스 권한에 종속됩니다.
`stop`은 네이티브 실행을 중지하지만 소유자와 메모리 상태는 유지하여 나중에
승인된 `start`를 받을 수 있습니다. 핸들 해제·세션 종료·만료·권한 철회·응답
실패는 worker를 종료합니다. 불명확한 명령 효과는 자동 재시도하지 않습니다.
종료는 이미 전송한 알림을 취소하거나 서드파티의 정상 정리를 보장하지 않습니다.
이는 세션 소유 서비스입니다. 정책 설정은 아래 방식으로 영속화·복원할 수 있지만
영구 네이티브 실행은 아래의 별도 복구 설정을 운영자 팩토리에서 선택해야 합니다. 외부 알림·저장소
실서비스는 미검증입니다. 게시된 3.2.0이 아닌 개발 소스의 기능입니다.

추가 승인 기반 관리 명령은 네이티브 서비스 메서드를 재사용합니다.

- 실행기 `schedule_update`는 `checkpoint_name`, `trigger_name`과 비어 있지 않은
  `schedule_update` 변경 객체를 받아 내장 간격 또는 cron 트리거 하나를 편집합니다.
  `CheckpointRunner.edit_schedule`을 재사용하며 실행기와 실행 스레드가 모두
  종료되어야 합니다. 트리거 실행 횟수·마지막 실행 시각·연결을 유지하므로 새 최초
  실행으로 초기화하지 않습니다. 초·분·시간 간격, 활성 여부, 최대 실행 횟수,
  즉시 실행·catch-up·최대 동시 실행 설정, JSON 메타데이터, 시작·종료 시각,
  요일·시간대를 변경할 수 있습니다. 생략한 값은 유지하고 시작·종료 시각,
  요일·시간대만 명시적 null로 해제할 수 있습니다. 시각은 오프셋 없는 로컬
  datetime이며 빈 요일 목록과 0 이하의 총 간격은 거부합니다. 시간대·catch-up·
  동시성은 Core의 기존 의미를 유지하며 미구현 보장을 추가하지 않습니다.
  cron에는 `expression`과 공통 트리거 설정을 지정하며 간격 전용 필드는 거부합니다.
  이름은 바꾸지 않습니다. provider 트리거는 공통 필드(`enabled`, `max_runs`,
  `run_immediately`, `catch_up`, `max_concurrent`, JSON `metadata`)만 편집하며
  provider 전용 경로·연결·콜백은 팩토리 소유입니다. 편집 자체로 실행·재시작하지 않으며 정확한 요청별 승인이 필요합니다.
- 실행기의 `checkpoint_update`는 `checkpoint_name`과 비어 있지 않은
  `checkpoint_update`를 받으며 실행 종료 상태의 `CheckpointRunner.edit_checkpoint`를
  재사용합니다. `checks`는 기존 닫힌 registry 생성자·설정 계약을 재사용합니다
  (1–1000개, 콜백 binding 제외). 명시한 checks 목록은 검증 설정을 교체합니다.
  `min_severity`, `auto_schema`, `run_name_template`, `tags`, JSON `metadata`,
  `fail_on_critical`, `fail_on_high`, `timeout_seconds`(1–86400),
  `sample_size`(양수)도 지원합니다. `min_severity`·`sample_size`만 null로
  해제할 수 있고 생략한 값은 유지합니다. 소스·스키마 객체와 경로·액션·router·콜백은
  운영자 팩토리가 소유합니다.
- 실행기의 `run_once`는 `checkpoint_name`을 받고 `run_all`은 등록된 전체
  체크포인트를 실행합니다. 네이티브 액션과 결과 콜백도 실행하며 결과는 예약
  결과 대기열이 아닌 산출물로 반환합니다. `remove_checkpoint`는
  `checkpoint_name`을 받아 네이티브 삭제 여부를 반환합니다. 이미 실행 중인
  체크포인트를 취소하지는 않습니다.
- 에스컬레이션의 `policy`, `policy_remove`는 `policy_name`을 받으며 없는
  정책은 null/false를 반환합니다. 정책 제거가 기존 사건을 해결하지는 않습니다.
  `active_records`는 선택적인 `policy_name` 필터를, `incident_records`는
  필수 `incident_id`를 받습니다. 전체 네이티브 목록을 산출물 바이트 한도 내에서
  반환하며 결과를 조용히 잘라내지 않습니다.
- `policy_update`는 `policy_name`과 비어 있지 않은 `policy_update` 객체를
  받습니다. 기존 정책의 `enabled`, `severity_filter`, `max_escalations`,
  `cooldown_minutes`, `business_hours_only`, `business_hours_start`,
  `business_hours_end`, `business_days`, `timezone`을 수정합니다. 생략한
  설정은 유지하며 명시적 null과 알 수 없는 필드는 거부합니다. `description`,
  `triggers`, `levels`, JSON `metadata`도 편집합니다. `levels`를 지정하면 목록
  전체를 교체하며 수준 번호는 중복 없는 양수입니다. 수준별 지연·목적지·반복
  횟수와 간격·확인 요구·자동 해결·JSON 조건과 메타데이터를 지정합니다.
  목적지의 종류·식별자·이름·우선순위·JSON 메타데이터도 지원합니다. 네이티브
  트리거와 목적지 종류 각각 8종을 허용하지만 사용자 콜백·실제 전송기는 승인된
  운영자 팩토리가 제공합니다. 정확한 요청 승인은 정책 이름뿐 아니라 목적지도
  포함합니다. 요청·정책 산출물이 로컬에 보존되므로 설정에 비밀정보를 넣지 마세요.
  등록된 네이티브 정책을 교체하는 것이며 기존
  기록·예약 작업을 재작성하지 않습니다. 이후 정책 참조 시점도 Core 동작을 따릅니다.
  Core는 `policy_name` 명시 조회에서 활성화·심각도 선택 조건을 우회합니다.
  따라서 정책 비활성화는 자동 선택에 적용되며 명시 호출을 막지는 않습니다.
  업무 시간 강제 여부도 Core를 따르며 현재는 권고 동작입니다.
  시간대 문자열은 그대로 전달하며 Core는 조회 실패를 업무 시간으로 취급합니다.
  따라서 이 설정은 권한 통제 수단이 아닙니다.

```json
{"binding_id":"incident-engine","service_id":"monitor_<returned-id>","command":"policy_update","policy_name":"approved-policy","policy_update":{"enabled":false,"cooldown_minutes":30},"idempotency_key":"policy-disable-01"}
```

조회도 포함한 모든 명령은 정확한 요청별 승인이 필요합니다. 캐시 재생은 제거한
체크포인트를 재실행하거나 제거한 정책을 복원하지 않습니다. 잘못된 요청 형식은
실행 전에 차단하며 실행 시작 후 네이티브 예외·응답 유실은 안전한 재시도가 아닌
`outcome_unknown`으로 남습니다.

`policy_create`는 동일한 `policy_name`과 `policy_update` 형식으로 새 정책을
생성하며 생략한 필드는 네이티브 기본값을 사용합니다. 기존 정책 덮어쓰기는
거부하며 `policy_update`는 없는 정책 생성을 거부합니다. 빈 수준·트리거 목록도
네이티브 의미대로 허용하고 임의 수신자를 만들어 넣지 않습니다.
`trigger` 명령은 `trigger_type`(네이티브 기본값 `unacknowledged`)과 JSON
`context`도 받습니다. Core의 자동 정책 선택까지 그대로 전달하며 다른 명령에
이 필드를 지정하면 거부합니다.

`persist_configuration`은 중복 없는 `policy_names` 1–100개를 받아 정지 상태의
에스컬레이션 정책을 영속화하고 서명된 불변 `snapshot_id`·해시·바이트 수를
산출물에 반환합니다. 없는 정책은 삭제 표식으로 저장합니다. 비공개 승인 저장소에
스냅샷당 최대 1 MiB, 128개, 총 16 MiB로 제한하며 호출자가 경로를 지정하지
않습니다. JSON이 아닌 네이티브 메타데이터는 임의 변환하지 않습니다.
세션 종료 후 `checkpoint_services_open`에 `kind="escalation"`, 동일한
`binding_id`, `snapshot_id`, 새 멱등 키를 전달하면 별도 요청 승인 후 복원합니다.
변조된 내용·정책 변경·바인딩 변경은 팩토리 실행 전에 거부합니다. 고정된 운영자
팩토리를 다시 열고 선택한 정책·삭제 표식만 적용하며 서비스는 정지 상태입니다.
선택하지 않은 팩토리 정책은 유지합니다. 이는 **정책 설정만 복구**하며 사건 기록·
대기 작업·알림 전송 영수증·실행기 예약은 복원하지 않습니다. 사건 기록은 운영자의
네이티브 저장소가 소유하며 아래의 별도 영구 복구 계약을 사용할 수 있습니다.
세션을 열어도 자동 재시작 서비스를 설치하지 않습니다.

실행기의 `persist_configuration`에는 대신 중복 없는 `checkpoint_names` 1–100개를
지정하며 실행기·실행 스레드가 종료되어야 합니다. 서명된 스냅샷은 선택한 체크포인트의
편집 가능한 scalar 설정, 명시적으로 편집한 닫힌 검증 명세, 간격·cron·공통 트리거
설정을 저장합니다. 없는 체크포인트는 삭제 표식이며 트리거 이름은 명시적이고 중복이
없어야 합니다. 동일 binding의 새 승인으로
`checkpoint_services_open(kind="runner", snapshot_id=..., ...)`를 호출해 복원합니다.
소스·액션·provider 전용 필드·변경하지 않은 검증기는 팩토리가 제공하며 해당 객체를
직렬화하거나 스냅샷에서 import하지 않습니다. JSON이 아닌 metadata는 거부합니다.
복원은 일치하는 네이티브 객체를 편집하고 영구 트리거 이력을 보존하며 예약을 시작하거나
이전 액션을 재실행하지 않습니다. 실행기·에스컬레이션 스냅샷을 교차 사용할 수 없습니다.

### 영구 체크포인트 복구

운영자 팩토리에서 `RunnerConfig(journal_path=...)` 또는
`SchedulerConfig(journal_path=...)`를 선택합니다. MCP 요청은 journal 경로를
받지 않습니다. APScheduler 어댑터는 3.x API를 사용하며 선택형
개발 checkout의 `.[mcp,checkpoint-scheduler]`을 설치합니다. asyncio·실제 APScheduler에서 재시작과
알림 실행 종료를 검증합니다. 에스컬레이션의 사건 저장소는 파일 기반 SQLite이며 journal과
**동일한 파일**이어야 합니다. 메모리 저장소·서로 다른 파일은 거부합니다.
알림 완료 기록·사건 횟수·다음 예약을 한 트랜잭션으로 저장하기 위한 조건입니다.
실행기 트리거에는 중복 없는 명시적 이름이 필요하고 설정은 JSON 호환이어야 합니다.
추가 가변 상태를 가진 provider는 기본 실행 시각 외에 `export_runtime_state`와
`restore_runtime_state`를 구현해야 합니다. 사용자 provider 상태와 외부 브로커
전달은 해당 provider의 책임입니다.

영구 서비스를 열면 journal의 프로세스 단독 소유권을 획득합니다. 시간이 지났다는
이유로 살아 있는 소유권을 빼앗지 않습니다. `stop`은 소유권을 유지하며 `close`와
프로세스 종료는 해제합니다. 새 세션에서 동일한 승인 binding·팩토리를 새 요청으로
열고 `start`도 승인하면 아직 획득하지 않은 예약을 재개합니다. 정책·binding·제어
디렉터리·설치 plugin 식별자가 바뀌면 복구를 거부합니다. 예약 효과 직전에도 현재
binding과 설치 plugin 식별자를 재검사합니다. 에스컬레이션 작업은 네이티브 정책
정의도 고정하며 변경·비활성화 시 새 목적지에 알리지 않고 작업을 격리합니다.

네이티브 액션·알림 전송 전에 실행 의도를 영구 저장합니다. 획득 후 중단된 작업은
`unknown`이며 자동 재실행하지 않습니다. 불명확한 알림은 같은 사건의 후속·수동
에스컬레이션도 차단합니다. handler의 false·예외·취소·완료 저장 실패로 다음 알림을
등록하지 않습니다. 외부 서비스의 정확히 한 번 전달을 보장하지 않습니다. 실행기의
횟수·마지막 실행 시각은 복원되지만 메모리 결과 대기열은 복원하지 않으므로 필요한
결과는 별도로 보존해야 합니다. 액션·결과 콜백이 불명확하면 해당 트리거를 차단하며
수동 실행이 불명확했던 체크포인트의 새 수동 실행도 차단합니다.

두 서비스 모두 아래 명령에 별도 요청 승인이 필요합니다.

- `recovery_status`: `enabled`, `identity`·`state`만 담은 `attempts`,
  `next_offset`을 반환합니다. `recovery_limit`은 1–1000(기본 100, 정책 한도 적용),
  `recovery_offset`은 기본 0입니다. payload·원시 값·예외 원문·파일 경로는 반환하지
  않습니다. 살아 있는 실행은 `attempting`이며 새 단독 소유자만 이전 실행을 unknown으로
  분류합니다.
- `recovery_resolve`: 정지·실행 종료 상태에서 `attempt_id`, `resolution`
  (`completed` 또는 `cancelled`)을 받습니다. 운영자가 실제 결과를 독립적으로
  확인해야 하며 이 명령은 알림·체크포인트를 실행하지 않습니다. 알림 완료 확인은
  횟수와 동일 정책의 다음 수준 예약을 저장합니다. 폐기는 다음 알림을 만들지 않습니다.
  예약 실행기 결과의 완료 확인·폐기는 해당 트리거 회차를 소비하고 마지막 실행 시각을
  갱신합니다.
- `recovery_forget`: 정지·실행 종료 상태에서 중복 없는 `attempt_ids` 1–100개를
  받습니다. 완료·취소 기록만 원자적으로 삭제하며 pending·attempting·unknown은
  보호합니다. 논리적 삭제이며 디스크 완전 소거가 아닙니다.

```json
{"binding_id":"incident-engine","service_id":"monitor_<returned-id>","command":"recovery_status","recovery_limit":100,"idempotency_key":"recovery-status-01"}
```

네이티브 사용자는 종료 시 `await engine.close()` 또는 `runner.close()`를 호출합니다.
MCP worker도 종료 시 호출합니다. 다시 열어도 실행·재시도를 자동 승인하지 않습니다.
로컬 별도 프로세스 테스트는 실제 액션·알림 중단과 설치된 팩토리·세션 재열기를 다룹니다.
실제 외부 알림·클라우드·브로커·유료 AI 서비스는 미검증입니다.

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

`storage_retention_preview`에는 `kind`(`models`·`jobs`·`histories`)와 `max_age_minutes`
(1~5,256,000), `max_count`(1~128), `max_size_bytes`(1~33,554,432) 중 하나 이상을
지정합니다. Core의 `TimeBasedPolicy`·`CountBasedPolicy`·`SizeBasedPolicy`·
`CompositePolicy`를 재사용합니다. 기본 `mode="all"`은 모든 선택 정책이 보존해야
남기고, `mode="any"`는 하나라도 보존하면 남깁니다. 크기는 실제 작업 입력·결과
바이트 합계 또는 모델·이력 체크포인트의 payload 바이트이며 SQLite 파일 크기나 새 저장 한도가 아닙니다.
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

### 네이티브 loopback 대시보드 수명주기

`dashboards_open`의 비공개 `dashboards` binding은 고정된 비동기 팩토리에서 정지된
`truthound.profiler.dashboard.server.DashboardServer` 또는
`truthound.datadocs.dashboard.ReflexDashboardServer`를 제공합니다. 개발 checkout에서
Streamlit은 `.[mcp,dashboard-streamlit]`, Reflex는 `.[mcp,dashboard]`을 설치합니다.
Reflex는 운영자가 Node/npm으로 비공개 프로젝트를 명시적으로 준비한 뒤 binding을
엽니다. MCP start는 의존성 설치·빌드를 하지 않습니다.
[네이티브 준비 가이드](https://github.com/seadonggyun4/Truthound/blob/main/docs/locales/ko/guides/datadocs/dashboard.md)를 참고하세요.
host·port·페이지 설정·데이터 접근 설정은 팩토리가 소유합니다. 모델의
source 경로·Python 스크립트·port·host·설정은 받지 않습니다. MCP host는 `localhost`,
`127.0.0.1`, `::1`만 허용하며 원격 ChatGPT 서버를 만들지 않습니다.

`dashboards_execute`는 같은 binding·현재 `service_id`·정확한 멱등 키와 `command`
(`start`·`stop`·`status`)를 받고 각각 별도 승인이 필요합니다. start의
`timeout_seconds`는 (0, 30](기본 10)이며 프로세스 생존과 네이티브 health 준비를
구분하고 점유된 port를 거부합니다. 상태는 `running`, `ready`, `url`, `exit_code`입니다.
stop은 소유한 서버·프로세스 그룹만 종료합니다. 핸들 해제·만료·권한 철회는 worker의
자식을 정리하며 네이티브 자식도 부모 연결 종료·강제 종료를 감지합니다. 로컬 프로세스
수명주기이며 OS 서비스를 설치하지 않습니다.

네이티브 launcher는 모든 dataclass 설정을 비공개 JSON으로 정적 앱 진입점에 전달하며
제목을 실행 가능한 Python에 삽입하지 않습니다. 실제 Streamlit·준비된 Reflex의 시작·
HTTP 준비·정지·재시작·핸들 해제·부모 강제 종료와 Streamlit의 점유 port 실패·제목
렌더링을 로컬 테스트합니다. 해제는 제한된 네이티브·factory 정상 정리를 요청한 뒤
미리 식별한 소유 프로세스 트리를 회수하며 정리가 멈추면 강제 종료할 수 있습니다. 기존 UI의
`require_auth`는 인증을 강제하지 않고 파일 로딩 설정은 네이티브 의미를 유지합니다.
Reflex의 표시·내보내기 플래그는 UI 제어이며 비공개 빌드 프로젝트에서 프로필 값을
제거하거나 접근 권한을 강제하지 않습니다. 준비한 설정이 바뀌면 시작을 거부하며
새 빌드가 필요합니다. 원격 배포는 미검증입니다.

### 네이티브 브로커와 처리 수명주기

`brokers_open`은 비공개 `brokers` binding의 `binding_id`와 `idempotency_key`를
받습니다. 고정된 설치 capability의 비동기 소유자가
`truthound.mcp.brokers.BrokerRuntime`을 제공합니다. 연결 전 네이티브
`BaseStreamAdapter`, 선택형 `WindowProcessor`·`ExactlyOnceProcessor`, 운영자 지정
`output_topic`으로 구성하며 열기만으로 연결하지 않습니다. 모델은 endpoint·자격증명·
import 경로·topic·콜백을 지정하지 않습니다. 원시 메시지에는 민감 결과 허용이 필요하고
`monitor_...` 서비스 핸들의 해제·만료·권한 철회는 소유 프로세스를 종료합니다.

`brokers_execute`에는 같은 binding·`service_id`, 정확한 멱등 키와 명령을 지정합니다.

- `connect`·`disconnect`·`status`: 네이티브 연결·지표, 보관 batch 수, 구성된 처리기
  통계·watermark를 다룹니다. disconnect는 로컬 batch를 폐기하며 외부 효과 취소나
  서드파티 정상 정리를 보장하지 않습니다.
- `consume`: `limit` 1–1000(기본 100, 정책 한도), `timeout_ms` 1–30000(기본 1000).
  `batch_id`·rich-value 인코딩 메시지·`size`를 반환하며 cursor만 전진하고 자동 ack는
  하지 않습니다. batch는 최대 16개, 현재 산출물 바이트 예산 안에서 보관합니다.
- `ack`: 보관한 `batch_id`로 네이티브 `commit_batch`를 호출하고 해제합니다.
  어댑터는 partition별 마지막 offset만 커밋할 수 있습니다. 같은 승인 요청 재생은
  영수증만 반환하고 다시 커밋하지 않습니다.
- `window_add`: 보관 batch를 구성된 네이티브 window에 한 번 추가합니다.
  `window_flush`는 `force_emit`이며 둘 다 source를 ack하지 않습니다.
- `process`: 보관 batch를 네이티브 `ExactlyOnceProcessor.process_message`로 한 번
  처리합니다. 네이티브 offset 커밋·운영자 sink 전달이 발생할 수 있으며 실패·중복 제거를
  명시합니다. 예외 원문은 닫힌 실패 코드로 대체합니다.
- `recover`: 네이티브 transaction rollback 복구의 `rolled_back` 수를 반환합니다.
  외부의 정확히 한 번 전달 보장으로 해석하지 않습니다.
- `seek`: 보관 batch가 없을 때 0 이상의 `partition`·`offset`을 받습니다.
  `produce`는 JSON `message`의 `value`, 명시적 ISO `timestamp`, 선택 `key`·0 이상
  `partition`을 받으며 목적지는 팩토리가 소유합니다. `flush`는 네이티브 flush입니다.
  외부 효과와 네이티브 재시도가 발생할 수 있습니다.

조회도 각각 승인하며 응답 유실은 `outcome_unknown`으로 남아 자동 재시도하지 않습니다.
batch·window는 상주 소유자 상태이며 세션 간 상태·offset·복구 내구성은 네이티브
backend에 따릅니다. 테스트는 실제 Core 로컬 테스트 어댑터를 사용하며
Kafka·Kinesis·클라우드 실서비스는 미검증입니다.

### 네이티브 AI 제안 검토와 적용

`ai_services_open`의 비공개 `ai_services` binding은 비동기 소유자를 통해
`truthound.mcp.ai_services.AIRuntime`을 제공합니다. 네이티브 `TruthoundContext`,
명시적으로 지정한 provider, 운영자 지정 데이터·source·model과 선택형 run을 포함하며
workspace는 허용된 workspace root 안이어야 합니다. 모델은 자격증명·provider·model·
source 경로·주체를 지정하지 않습니다. 열기와 각 명령에는 정확한 요청 승인이 필요하고
provider 호출에는 네이티브 요약 공개와 비용이 발생할 수 있습니다. 로컬 테스트는
결정적 provider를 사용하며 유료 API를 호출하지 않습니다.

`ai_services_execute`는 binding·세션 `service_id`·멱등 키와 닫힌 명령을 받습니다.

- `suggest`: `prompt` 1–16000자, 선택형 `sample_size` 1–10000(기본 1000)으로
  네이티브 `suggest_suite`를 호출해 제안을 저장합니다.
- `explain`: 선택형 `run_id` 또는 운영자가 제공한 run으로 `explain_run`을 호출합니다.
  `include_history` 기본값은 true이며 분석을 저장합니다.
- `list_proposals`·`list_analyses`: `limit` 1–1000(기본 100, 정책 한도).
  후자는 선택형 `run_id` 필터를 받습니다. `proposal`·`history`에는 `proposal_id`,
  `analysis`에는 네이티브 `analysis_id`가 필요합니다.
- `approve`·`reject`·`apply`: `proposal_id`, 직전 결과의 `artifact_sha256`를 담은
  `expected_sha256`, 선택형 `comment`(최대 4096자)를 받습니다. 승인 요청이 제안
  revision을 고정하며 오래된 revision은 변경 전에 거부합니다. 각 결정은 새 digest를
  반환합니다. 승인과 적용은 별도 운영자 결정이며 주체는 `mcp-approved-operator`로
  고정합니다. 적용된 제안의 네이티브 종료 상태를 유지합니다.

redaction은 네이티브 `summary_only`로 고정하며 원시 샘플·실행 코드를 받지 않습니다.
응답은 `workspace_root`를 제외한 네이티브 필드를 rich-value 형식으로 보존합니다.
수명주기 상태는 네이티브 workspace에 남고 재열기는 동일 팩토리·새 승인이 필요합니다.
MCP 영수증이 다른 workspace writer와의 다중 파일 쓰기를 원자적으로 만들지는 않습니다.
불명확한 provider·적용 효과를 재시도하지 않으며 유료 provider·실제 외부 배포는 미검증입니다.

### 승인된 OpenLineage 전송

`openlineage_services_open`은 비공개 `openlineage_services` 바인딩에서 설치된
비동기 팩토리를 선택합니다. 팩토리는
`truthound.mcp.openlineage_services.OpenLineageRuntime(emitter, datasets={alias: dataset})`를
제공합니다. Emitter는 네이티브 `OpenLineageEmitter`이며 운영자가 전송 대상·인증·
데이터셋 메타데이터를 설정합니다. 열기만으로 이벤트를 보내지 않습니다.
이 기능은 미출시 소스 체크아웃 기능이며 공개 서비스 배포를 의미하지 않습니다.

각 `openlineage_services_execute` 요청은 `binding_id`, `service_id`, 정확한
`idempotency_key`를 포함하고 승인을 받아야 합니다. 명령은 `status`, `start`,
`running`, `complete`, `fail`, `abort`, `graph`, `resolve`, `shutdown`으로 제한됩니다.
`start`는 `job_name`, 선택적 활성 소유 `parent_run_id`, 입력용 `datasets` 별칭을
받습니다. `complete`의 별칭은 출력용입니다. 전환 명령은 소유한 `run_id`를
받고 `fail`은 고정된 운영자 보고 실패 메시지를 전송합니다. 요청으로 예외·인증·
주소·임포트 경로를 지정할 수 없습니다. `graph`는 소유 `graph_id`와 `job_name`을
받아 데이터셋 이름만 전송하고 해당 그래프의 권한·만료 의존성을 서비스에 유지합니다.

전송 실패는 `outcome_unknown` envelope와 `run_id`, `event_type`,
`outcome_unknown: true`를 가진 민감 아티팩트를 반환합니다. Owner는 `status`와
`resolve`를 위해 유지되며 같은 결과 불명 실행의 전환은 재전송하지 않습니다.
수신기를 별도로 확인한 뒤 `resolve`에 `run_id`와 명시적 boolean `delivered`를
전달하세요. 이 명령은 요청 없이 운영자 판정을 기록합니다. 같은 승인 요청의
재호출은 보존된 결과만 반환합니다. 상태 아티팩트에는 전송 설정·데이터셋 내용을
담지 않습니다. 새 실행은 보존된 활성·결과 불명 레코드 1000개로 제한됩니다.
그래프 네트워크 실패는 부분 전송을 남길 수 있으며 전체 그래프 트랜잭션이나
자동 재시도는 제공하지 않습니다.

`shutdown`은 네이티브 작업과 팩토리 종료를 회수한 후 응답합니다. 입력 그래프
해제는 종속 서비스를 철회합니다. 서비스 해제·만료·부모 소실도 owner를 종료합니다.
강제 종료는 원격 수신 여부를 판정하지 못하며 로컬 emitter 상태는 프로세스 소실
이후 영구 복구되지 않습니다. 소유한 loopback 수신기 검증은 실서비스 호환성
검증을 대신하지 않습니다.

### 네이티브 감사와 관측 관리

`audit_services_open`의 비공개 `audit_services` binding은 네이티브 `AuditStorage`를
담은 `truthound.mcp.administration.AuditRuntime`을 제공합니다.
`audit_services_execute`의 `query`·`read`·`count`·`record`·`flush`·`delete_before`는
각각 승인합니다. query 필드는 offset 있는 `start_time`·`end_time`, 네이티브
`event_types`, `actor_id`, `resource_id`, `outcome`, `limit` 1–1000(기본 100,
정책 한도), 0 이상 `offset`입니다. count는 시간·event type 필터만 지원하고 read는
`event_id`가 필요합니다. record는 `action`과 선택형 `outcome`·`message`·JSON
`data`로 고정 운영자 주체와 네이티브 checksum의 **custom** event를 생성하며 인증
이벤트를 사칭하지 않습니다. `delete_before`는 offset 있는 `before` 이전 기록을
별도 승인 후 영구 삭제합니다. 모델의 SQL·저장소 경로·주체 변경·자격증명은 받지 않습니다.

`telemetry_services_open`의 비공개 `telemetry_services` binding은 `TelemetryRuntime`의
네이티브 `MetricsRegistry`·`TracerProvider`와 명시적 metric·exporter alias를 제공합니다.
`telemetry_services_execute` 명령은 다음과 같습니다.

- `collect`: registry 값. `export`: 운영자 소유 `exporter_id`로 네이티브 외부 전달·저장.
- `metric`: 기존 `metric_id`, 유한 `value`, 선택형 문자열 `labels`, `update`.
  Counter·Gauge는 `inc`, Gauge는 `set`, Histogram·Summary는 `observe`이며
  네이티브 label·타입·값 검증을 유지합니다.
- `span`: `span_name`과 선택형 JSON `attributes`로 해당 소유자의 네이티브 span을
  생성·종료하고 trace·span ID를 반환합니다.
- `flush`: `timeout_ms` 1–30000(기본 1000). `shutdown`: tracer provider 종료.
  네이티브 false 결과를 그대로 반환하고 shutdown 후 다른 관측 명령은 종료합니다.

모든 관리 명령에 정확한 요청 승인과 민감 결과 허용이 필요합니다. 저장소·자격증명·
exporter endpoint·필터·provider 설정은 팩토리 소유입니다. 해당 상주 인스턴스를 다루며
다른 프로세스의 전역 관측이나 원격 보안 제어 시스템을 관리하는 계약은 아닙니다.
외부 감사 저장소·exporter는 미검증이며 테스트는 임시 SQLite, 네이티브 메모리 exporter와
의도적으로 실패하는 flush processor를 사용합니다.

### binding 기반 Core 보안·plugin 관리

`security_services_open`의 비공개 `security_services` binding은 고정된 비동기
팩토리에서 `truthound.mcp.platform_services.RBACRuntime`을 제공합니다. 네이티브
`RBACManager`, 선택형 고정 `tenant_id`, 명시적 읽기·쓰기 `attribute_keys`와 추가
`condition_fields`로 구성하며
모델은 tenant·저장소·자격증명·인증 콜백·policy evaluator를 지정하지 않습니다.
고정 tenant와 다른 대상은 CRUD·역할 할당 전에 거부합니다. 선택한 Core 관리자만
다루며 그 권한 판단으로 MCP 동의·폴더 권한·승인 요구를 변경하지 않습니다.

`security_services_execute`는 binding·현재 서비스 핸들·정확한 멱등 키와 명령을
받고 조회·변경 각각 승인합니다.

- `roles`·`principals`: 고정 tenant의 목록, `limit` 1–1000(기본 100, 정책 한도),
  0 이상 `offset`. `role`·`principal`에는 각각 `role_id`·`principal_id`가 필요합니다.
- `role_create`·`role_update`: `role_id`, 비어 있지 않은 `role_patch`의 `name`,
  `description`, `enabled`, `permissions`, `parent_roles`, `conditions`. 생성에는 이름이 필요합니다.
  권한은 아래의 문자열·구조화 객체이며 목록은 native set 전체를
  교체합니다. 부모 role은 해당 범위에 존재해야 합니다. `permission_add`·
  `permission_remove`는 `role_id`와 하나의 `permission`을 받습니다.
  `role_delete`는 네이티브 삭제 여부를 반환합니다.
- `principal_create`·`principal_update`: `principal_id`, 비어 있지 않은
  `principal_patch`의 `name`, `email`, `enabled`, `direct_permissions`, `attributes`.
  생성에는 이름이 필요하며 네이티브 기본 principal 종류를 사용합니다. attributes는
  팩토리가 선언한 키만 허용합니다. `principal_delete`는 네이티브 삭제 여부입니다.
- `assign_role`·`revoke_role`은 두 ID를 받습니다. `permissions`는 할당·상속 권한
  목록이며 접근 승인 판단이 아닙니다. `check`는 `principal_id`, `resource`, `action`,
  선택형 JSON `resource_attributes`로 allowed·effect·reason·matches를 반환합니다.
  `cache_clear`는 네이티브 캐시 무효화입니다.

권한 입력은 ALLOW를 뜻하는 `resource:action[:scope]` 문자열 또는 `resource`,
`action`, 선택형 `scope`(기본 빈 문자열), `effect`(`allow`·`deny`, 기본 `allow`)를
갖는 JSON 객체입니다. 각 구성요소 안에 `:`는 넣지 않습니다. role 권한·principal
직접 권한·추가·제거에 동일하게 적용합니다. ALLOW 문자열 제거는 별개 DENY 객체를
제거하지 않습니다.

role 조건은 `field`, `operator`, 필수 JSON `value`, 선택형 `description`을 받습니다.
연산자는 네이티브 `eq`·`neq`·`contains`·`not_contains`·`starts_with`·`ends_with`·
`matches`·`in`·`not_in`·`gt`·`lt`·`gte`·`lte`·`exists`·`not_exists`입니다. 편집은 호출자가
제공한 `resource.*`, `environment.timestamp`, 조회 가능한 principal의 id·tenant_id·
name·email·type·enabled·roles, 팩토리가 허용한 `principal.attributes.<key>`와
정확한 추가 `condition_fields`만 참조합니다. 값의 동적 `${...}` 참조에도 같은 허용
목록을 적용하며 다른 principal 속성을 조건으로 탐색하지 못합니다. 팩토리 소유 조건
중 범위 밖인 것이 있으면 role 조회에 `conditions_redacted=true`를 표시합니다.
조건의 리터럴·설명에 자격증명을 넣지 마세요.

결과는 권한 effect를 포함한 정의된 필드를 제공하고 임의 metadata·허용 목록 밖의
principal attributes를 제외합니다. 자격증명은 반환하지 않습니다. 저장·지속성과 hook은
팩토리 소유이며 서로 다른 프로세스의 RBAC 저장소 캐시 일관성을 자동 보장하지 않습니다.
Core 관리자는 판단과 변경을 직렬화하고 변경 후 캐시된 허용을 무효화하며 비활성
principal·role을 거부합니다. 명시적 DENY는 직접 ALLOW보다 우선하며 일반 resource
이름도 실제 종류로 처리합니다. 역할 diamond의 공통 조상은 허용하고 실제 순환은
거부하며 상속된 역할 조건을 적용합니다. 권한 목록·helper는 문맥에 따른 접근 판단이
아니므로 조건부 접근은 `check`를 사용합니다. 판단 캐시는 네이티브 무조건 role
evaluator에 한정합니다. 속성·소유권·시간·custom·조건부 role은 새로 평가하고 evaluator
구성 변경은 캐시를 무효화합니다. evaluator 실패를 다른 허용으로 대체하지 않고 접근을
거부하며 콜론을 포함한 주체·resource도 캐시 키에서 분리합니다.

네이티브 파일·SQLite 저장소는 재개방 후에도 DENY와 role 조건을 보존합니다. 일반
ALLOW는 기존 문자열을 유지하고 DENY는 구조화 권한 객체로 저장합니다. 이전 Core
reader는 객체를 읽지 못하므로 업그레이드가 필요합니다. 과거 기록에서 이미 사라진
DENY·조건은 이 변경으로 추정·복원할 수 없으며 운영자의 검토가 필요합니다. 파일 저장소는
손상된 기록을 거부하고 덮어쓰지 않으며 직렬화 후 원자적으로 교체합니다. 저장 실패 후
변경된 메모리 캐시는 무효화합니다. 새 네이티브 저장소와 실제 승인된 SQLite worker
종료·재개방을 로컬 검증하며 원격 저장소는 미검증입니다.

`plugin_services_open`의 비공개 `plugin_services` binding은 하나의 네이티브
`PluginManager`와 명시적 `classes`, 네이티브 `configs`, custom `hooks`, `providers`,
`reporters`, `capabilities`, `check_factories` alias를 담은 `PluginRuntime`을 제공합니다.
모델의 import 경로·distribution 이름·Python·임의 생성자·콜백 객체는 받지 않습니다.

`plugin_services_execute`는 다음을 지원합니다.

- `catalog`: 클래스 alias. `load`: `alias`로 네이티브 `load_from_class`를 호출하며
  자동 활성화하지 않습니다. `activate`, `deactivate`, `enable`, `disable`, `unload`,
  `status`, `health`는 로드된 alias를 다룹니다. 네이티브 상태·boolean health를
  유지하며 unload 뒤에는 새 load가 필요합니다.
- `reload`는 로드된 `alias`와 `reload_strategy="graceful"|"immediate"`를 받습니다.
  팩터리는 새 네이티브 클래스를 반환하는 `PluginRuntime.reload_factories[alias]`를
  제공해야 하며 wire에서 코드·경로·로더를 받지 않습니다. 진입 단계마다 binding
  권한을 확인하고 실제 teardown/setup/활성화를 수행합니다. 결과는 `reloaded`,
  `loaded`, `rolled_back`, `native_failed`와 본문 없는 `error_type`을 담습니다.
  setup 실패 시 정리가 확인된 경우에만 기존 클래스의 새 인스턴스를 복구하며
  외부 효과를 되돌리지 않습니다. 서명 바인딩은 별도 서명 갱신이 필요합니다.
  이 승인 명령은 자동 파일 감시를 시작하지 않습니다.
- `ports`: capability·provider·check factory 이름과 reporter 계약 버전.
  `provider_register`, `reporter_register`, `capability_register`,
  `check_factory_register`는 팩토리가 제공한 해당 alias 객체만 등록합니다.
  대응하는 `provider_unregister`·`reporter_unregister`·`capability_unregister`·
  `check_factory_unregister`는 manager 내부 port 조회를 제거합니다. provider·capability
  제거는 공유 자원을 닫지 않으며 factory가 cleanup을 계속 소유합니다. 전역 reporter
  생성자는 이 manager가 등록했고 다른 소유자로 교체되지 않은 경우에만 제거합니다.
- `hook_register`는 팩토리 콜백 `alias`, 선택형 `priority` 0–10000(기본 100)과
  `hook_type`(기본 `custom`)을 받습니다. 네이티브 수명주기 event와 추가 운영자 event
  별칭의 `PluginRuntime.hook_types`를 지원합니다. `hook_unregister`는 선택한
  handler/event를 제거하고 `hooks`는 callback repr 없이 metadata를 표시합니다.
  `hook_fire`의 기존 JSON `payload` 결과 목록에서 None은 성공과 실패를 구분하지
  못합니다. `hook_fire_detailed`는 handler별 success·result·오류 종류와 `native_failed`를
  반환하고 예외 본문은 숨깁니다. 외부 전달 증명이나 자동 재시도 권한은 아닙니다.
- `validate`는 같은 manager에 등록된 `check_aliases`와 보존된 `asset_id` 또는 등록된
  `provider_id` 중 하나를 사용합니다. factory는 네이티브
  `CheckSpecFactory.build_check_specs()`를 구현하며 기존 Validator 클래스·인스턴스는
  Core suite 컴파일을 사용합니다. provider는 네이티브 `DataAsset`을 반환합니다.
  suite 이름·결과 형식·parallel/max-workers·pushdown은 닫힌 명시적 필드입니다.
  실제 계산은 Core `ScanPlanner`·`ValidationRuntime`이 수행하고 before/after/issue hook
  결과를 데이터 품질 판정과 별도로 보존합니다.
- `render(alias,run_id)`는 등록된 v3 reporter로 실제 보존된 canonical run을 렌더링합니다.
  `run_id` 대신 표준 검증 `result_id`도 받을 수 있습니다. reporter 클래스·생성자·구성된
  객체는 factory가 소유하며 요청에 template·출력 경로·생성자 설정·실행 파일을 넣지
  않습니다. 텍스트·bytes와 report hook 결과는 크기가 제한된 JSON 산출물로 보존합니다.
- `run_get`·`run_export`·`run_release`는 네이티브 결과의 조회·내보내기·메모리 해제를
  제공합니다. canonical run ID와 다른 반환된 불투명 `plugin_run_...` 핸들을 사용합니다.
  남은 byte 예산 안에서 최대 16개 run을 유지합니다. `run_export`의 표준 canonical
  검증 산출물은 기존 reporter와 `store_services_execute(command="save",result_id=...)`에
  사용할 수 있습니다. 네이티브 run을 해제해도 이미 내보낸 불변 artifact는 지우지 않습니다.
  `provider_inspect(alias)`는 선택한 provider asset의 크기·backend를 조회합니다.

Core에서 직접 쓰는 새 `PluginManager.validate`·`validate_async`, `render`·`render_async`도
같은 manager 안의 이 흐름을 제공합니다. 비동기 버전은 데코레이터로 감싼 async handler까지
실제 결과를 기다리고 provider·kernel 동작은 호출한 스레드에서 수행합니다. 동기 버전은
실행 중 event loop에서 효과 전에 거부합니다. kernel port 검증은 context·workspace를
암묵적으로 저장하지 않으며 구성된 provider·reporter가 자원을 소유합니다. 검증 hook의
`checks`·`validators`는 불변 `CheckSpec` 의도입니다. issue hook도 네이티브 validator 이름과
alias가 달라도 해당 check 의도를 받습니다. hook 실패는 판정을 중단하지 않는 별도 결과로
보존하며 상세 API에서 성공한 None으로 바꾸지 않습니다. MCP를 통과하는 hook 반환 값은
기존 무손실 JSON 값 문법을 따라야 합니다.

사용한 보존 asset·result의 권한 수명은 소유자에 연결됩니다. 입력 해제·만료·철회 시 소유자와
종속 artifact를 정리합니다. 소유 artifact 읽기는 watchdog이 없어도 네이티브 정책·binding을 동기적으로 재확인하며,
background tick을 접근 권한 판단의 대체 수단으로 쓰지 않습니다. 네이티브 `set_execution_guard`는 provider·factory·report와
각 hook 직전에 binding을 재확인하여 철회 후 다음 handler의 실행을 차단합니다. unload
hook이 거부되어도 소유 plugin의 teardown·제거는 수행합니다. 설정된 reporter 인스턴스는
manager 내부에 유지하여 전역 생성자 registry를 손상시키지 않습니다. 정상 `unload`는
`{loaded:false}`를 유지하고 네이티브 정리 실패는 `native_failed=true`와 비활성화·teardown
완료·제거·hook 실패·예외 **종류**가 담긴 receipt를 추가하며 예외 본문은 숨깁니다. Core
`unload_plugin(name,detailed=True)`는 `PluginCleanupResult`를 반환하고 기존 호출은 None을
유지합니다. callback 완료가 원격 provider의 모든 자원 해제를 증명하는 것은 아닙니다. 신뢰된 확장 코드는
OS sandbox가 아니며 이미 시작한 외부 효과의 정확히 한 번 전달을 보장하지 않습니다.

모든 명령에 정확한 승인과 민감 결과 허용이 필요합니다. 등록은 해당 상주 관리자에
적용되며 영구 MCP binding 파일·다른 worker registry를 자동 변경하지 않습니다.
plugin 콜백이 unregister·teardown을 소유하고 해제 시 worker를 정리하지만 서드파티
정상 정리는 보장하지 않습니다. 임의 discovery·hot reload·자격증명 조회·원격 보안
제어는 이 계약에 포함되지 않으며 해당 네이티브 확장·관리 기능은 별도 작업이 남습니다.

### 네이티브 실행 엔진과 일반 저장소

직접 Python 실행 엔진은 설정과 캐시 결과를 복사합니다. 행 제한·offset은 음수·bool·
정수가 아닌 값을 거부합니다. 네이티브 limit=0은 빈 값/빈도 결과를 반환하고 offset과
limit=None 조합에 숨은 100만 행 제한을 두지 않습니다. MCP 결과 제한은 명시적인
1–1,000을 유지합니다. SQL 값 페이징은 provider의 LIMIT/TOP/ROWNUM 구문을 쓰며
SQLite의 제한 없는 offset과 다른 provider별 형식을 처리합니다. 명시적 소스 정렬이
없으면 SQL 행 순서는 안정적이지 않으며 SQL Server offset은 요청 열로 정렬합니다.
원격 provider의 실제 서비스 페이징은 미검증입니다. Polars/Pandas 필터는 네이티브
구문·소스 오류를 보존하되 MCP는 닫힌 오류 코드만 반환합니다. count나
__truthound_count라는 사용자 열도 빈도 집계에 사용할 수 있습니다. 설정 행 한도는
양의 정수이며 명시적 to_numpy_safe(max_rows=0)는 거부합니다.

`execution_services_open`은 비공개 `execution_services` binding의 승인된 비동기
factory에서 `truthound.mcp.execution_services.ExecutionRuntime`을 받습니다.
네이티브 `BaseExecutionEngine`, 완전한 운영자 SQL 문자열의 `predicates`·`queries`
별칭, 네이티브 Polars `expressions`를 지정합니다. 연결·backend·cluster 설정과
최종 provider 종료는 factory가 소유합니다. 요청에 SQL·Python·import·연결 URL·표현식
소스를 넣을 수 없습니다. 모든 명령은 `binding_id`, `service_id`, `command`, 정확히
승인된 `idempotency_key`를 사용하며 시작 엔진은 `engine_id="root"`입니다.

`execution_services_execute`는 행·null·고유값·완전성·통계·분위수·빈도·열 값·표본,
집계·정규식·범위·집합·중복 계산과 cache 정리를 Core에 위임합니다. 조건 계산과
`filter_by_condition`은 `predicate_id`, `execute_query`는 보간 없는 `query_id`를
사용합니다. 등록된 쿼리는 운영자가 정한 부수 효과가 있을 수 있으므로 승인 범위에
해당 전체 문장이 포함되어야 합니다. 네이티브 미지원 동작은 실패로 남습니다.
`inspect`는 기본 실행 한도와 공개 별칭만 표시하고 provider 설정은 숨깁니다.
빈도는 문자열이 아닌 키도 보존하도록 `{value,count}` 행으로 반환합니다.

`sample`, `filter_by_condition`, Polars의 `select(columns)`,
`filter(expression_ids)`, `with_columns(expression_ids)`는 실제 파생 엔진을
최대 16개 유지합니다. 반환된 `engine_id`를 사용하고 `release`로 해제합니다.
root 해제는 `resources_release(service_id)`를 사용합니다. SQL 필터는 부모 연결을
빌리는 지연 subquery이며 중첩 필터와 표본도 유지됩니다. 파생 객체를 먼저 정리한 뒤
부모와 factory를 종료합니다. 값·표본·테이블·NumPy 결과의 `limit`은 1–1,000이며
운영자의 `max_items`로 추가 제한됩니다. `to_numpy`는 행 수가 요청 한도를 넘으면
거부하고 `to_numpy_safe`는 네이티브 표본을 사용합니다. Core 자체 한도도 유지됩니다.
`to_polars`는 네이티브 lazy frame의 제한된 행을 반환합니다. NaN·시간 등은 기존
무손실 태그 인코딩을 사용합니다. 로컬 Polars·SQLite와 실제 로컬 Dask·Ray·Spark SDK
증거는 원격 SQL·원격 클러스터 배포 검증을 뜻하지 않습니다.

이 미출시 checkout에서는 선택한 runtime에 따라 `truthound[execution-dask,mcp]`,
`truthound[execution-ray,mcp]`, `truthound[execution-spark,mcp]`를 설치합니다.
Spark에는 운영자가 준비한 Java 17+와 올바른 Python worker 인터프리터가 추가로
필요하며 pip가 이 런타임을 설치하지는 않습니다. 분산 명령은 파티션 수,
1–1,000개 `repartition`·`coalesce`, metrics·정리와 제한 Arrow 변환을 제공합니다.
Dask는 `persist`, 선택 컬럼 엔진, 별칭 `native_filter`, head/tail·`compute`를 지원하고
요청에서 scheduler 주소·경로·조건식을 받지 않습니다. Ray 연결은 materialize·선택 컬럼·
take와 factory의 `row_filters`를 `filter_rows(row_filter_id)`로 선택하는 동작을 제공합니다.
Spark 연결은 persist/unpersist, factory 설정 checkpoint, 파생 엔진을 반환하는 별칭 SQL을
제공합니다. backend 미지원 명령은 실패합니다. 공유 client·cluster의 종료는 factory가
소유하며 파생 엔진 해제가 cluster를 종료하지는 않습니다. persist/checkpoint의 운영자
설정 저장 효과도 정확한 승인에 포함됩니다. 전체 compute·Arrow 변환은 먼저 행 수가
`limit` 이내인지 확인하므로 큰 입력은 select/sample을 먼저 사용합니다. Arrow 결과는
backend index 컬럼을 포함한 네이티브 schema를 보존합니다. metrics는 원문 오류·경고 없이
개수·실패 여부를 표시합니다.

네이티브 종료 완료 응답이 필요하면 승인된 root `shutdown` 명령을 사용합니다.
파생 엔진·root 엔진·factory의 종료가 끝난 뒤 반환하며 정리 오류나 제한 시간 초과를
성공으로 바꾸지 않습니다. 같은 shutdown 승인 키는 보존한 결과를 재생합니다.
이후 다른 엔진 호출은 `invalid_handle`로 실패하며 서비스 자원은 별도로 해제합니다.
`resources_release`는 제한된 강제 정리이므로 이것만으로 모든 factory finalizer의
완료를 입증하지 않습니다. 감독기는 driver가 살아 있을 때 소유 자식 PID를 기록해
driver 장애 후에도 정리 소유권을 유지합니다. 전역 Ray·Spark 종료 명령은 사용하지 않습니다.

Ray·Spark 표본은 네이티브 무작위 정렬로 `min(n, row_count)`개를 반환하며 shuffle이
필요할 수 있습니다. Ray `get_stats.count`는 null을 제외합니다. Spark distinct 집계는
`count_distinct`와 같은 null 제외 의미를 쓰고 Ray 네이티브 distinct 목록은 null을
포함합니다. Spark quantile은 상대 오차 0.01의 네이티브 근사 순위 의미를 유지하며 정수
확률 끝점도 받습니다. Ray의 정확한 median은 Core Polars 변환과 설정된 행 수 한도를
재사용하므로 무제한 분산 median이 아닙니다. Spark는 네이티브 median을 사용합니다.
PySpark의 기존 `MEMORY_ONLY_SER` 이름은 직렬화 `MEMORY_ONLY`로 연결하며 알 수
없는 네이티브 storage-level 이름은 거부합니다.

표준 NumPy 정수·bool·문자열·bytes·최대 64-bit 실수는 정확한 Python 값으로 정규화하여
기존 태그 인코딩을 사용합니다. 큰 unsigned 개수와 NaN/Inf도 보존합니다. 배열·복소수·
시간 스칼라·확장 정밀도 실수를 조용히 강제 변환하지 않습니다. 분산 FIRST/LAST는 row
count로 대체하지 않고 해당 연산을 선택합니다. 순서는 backend의 네이티브 실행 순서이며
순서 없는 분산 입력에 전역 정렬 계약을 추가하지 않습니다. 실제 Dask 검증은 파티션·집계·
null/빈 입력 first/last·선택 필터·persist·Arrow·파생 객체 종료를 포함합니다.

`store_services_open`은 비공개 `store_services` binding의 비동기 factory에서
`truthound.mcp.store_services.StoreRuntime`을 받습니다. 네이티브 `BaseStore`,
`CachedStore`, `BatchedStore`, `ReplicatedStore`, 선택적인 네이티브 영속화 DTO `ValidationResult`·
`ExpectationSuite`의 `records` 별칭, `data_assets` 별칭과 고정 버전 작업자를 지정합니다.
backend·namespace·직렬화·압축·자격 증명·정책·wrapper 설정과 저장소 종료는 factory가
소유합니다. `VersionedStore`도 네이티브 `BaseStore`이며 `config`는 실제 내부 backend
설정을 반영합니다. 요청에 연결·저장 경로를 넣을 수 없습니다.

`store_services_execute`는 초기화·저장·조회·존재·삭제·목록·query·개수·제한 삭제,
최신·이력·실패 조회·suite와 반복 조회를 Core에 위임합니다. 필터는 `data_asset_id`,
ISO 시간·상태·태그·정렬을 사용합니다. `limit`·`offset`과 반복 조회의 `batch_size`를
지원하고 Core 반복 조회는 전체 limit을 정확히 지킵니다. Core의 limit=0은 0개입니다.
record ID는 소유자 내의 불투명 참조이므로 영구 저장소를 다시 열면 목록·query로 새
참조를 받습니다. 내구성은 선택한 backend에 따르며 메모리 저장소·메모리 버전 정보가
자동 영속화되지는 않습니다.

`save`는 `record_id`, 보존된 canonical 검증 `result_id`, 선언형 `suite` 중 정확히
하나를 받습니다. 검증 결과는 Core `ValidationResult.from_validation_run_result`로
변환하며 전체 네이티브 영속화 payload를 보존합니다. 서비스는 해당 입력 권한을 계속
재확인하며 입력 해제·만료·권한 철회 시 소유자와 종속 artifact를 종료합니다. `suite`는
`data_asset_id`, 안전한 이름과 expectation·설정·태그를 받는 영속화 DTO입니다.
이는 불변 kernel `ValidationSuite`와 별개입니다. 조회는 판정·통계·시간·태그·expectation
설정을 표시하고 임의 metadata·실행 환경·issue 메시지·details·물리 asset 식별자는
숨깁니다. 버전 조회는 운영자 metadata·작성자·메시지를 숨기고 diff는 비공개 metadata
경로와 값 payload 없이 구조 경로·종류를 반환합니다.

버전 저장소는 `version_history`, `version_info`, `latest_version`, `count_versions`,
`diff`, `rollback`, 버전별 조회·존재·삭제, `expected_version`·`message` 저장을 지원합니다.
cache wrapper는 invalidate/all·refresh·warm·get-many·통계, batch wrapper는
add/start/stop/flush·통계를 제공합니다. `save`는 네이티브 즉시 저장입니다. 예약 batch
쓰기는 효과 직전에 binding을 재확인합니다. 대기 batch는 영구 journal이 아니며 중단·실패·
불명확한 flush를 저장 성공으로 간주하지 않습니다. 호환되지 않는 wrapper 명령은 거부하며
내부 저장소로 우회하지 않습니다. 파일 저장소는 `rebuild_index`를 제공합니다.
`RetentionStore`는 `retention_cleanup(dry_run=true)`, info·통계·이력·예측을 제공합니다.
`dry_run=false`는 네이티브 정책의 삭제·archive·압축·tier 효과에 대한 정확한 승인이
필요합니다. 결과는 개수·`dry_run`·`error_count`·`native_failed`를 보존하고 원문 오류는
숨깁니다. Core 정리 이력·통계는 preview도 포함하므로 그 누적 수치만으로 실제 삭제를
증명할 수 없습니다. 네이티브 보존 예약은 저장소 동작 중 실행하며 MCP daemon을
설치하지 않습니다. `RetentionStore.config`는 내부 backend 설정을 반영합니다.

`TieredStore`는 목록·info·통계, `tier_move(record_id,tier_id)`,
`tier_run(dry_run=true)`를 제공합니다. `StoreRuntime.tiers`에는 구성된 모든 네이티브
계층 이름의 공개 별칭을 지정합니다. preview·실제 이동은 불투명 record 참조·공개 계층
별칭·네이티브 처리·실패 개수를 반환하며 물리 metadata·오류 메시지는 숨깁니다.
배치 정보의 내구성은 선택한 metadata store에 따르며 기본값은 메모리입니다.

`ConcurrentFileSystemStore`는 `concurrent_apply(actions,dry_run=true)`,
`concurrent_stats`와 네이티브 `rebuild_index`를 제공합니다. 1–1000개 action은
`kind="save"`·`"delete"`와 불투명 `record_id`로 구성됩니다. save는 factory의 실제
record나 기존 네이티브 검증 DTO를 선택합니다. preview는 네이티브 buffer를 구성하고
rollback하며 직렬화 검증이나 대기 파일 변경은 수행하지 않습니다. `dry_run=false`는
하나의 정확히 승인된 호출 안에서 commit하고 MCP 호출 사이에 잠금을 유지하지 않습니다.
receipt는 네이티브 buffer·적용 작업 개수이며 완전한 crash 원자성의 별도 증거가 아닙니다.
통계는 backend 경로를 제외한 전체 lock 수치입니다.

네이티브 파일 잠금은 관련 index 갱신까지 유지합니다. index commit은 프로세스 간
잠금 안에서 최신 저장 상태를 읽고 서로 다른 소유자의 변경을 보존하며 충돌한 추가를
거부합니다. 손상된 index·WAL은 증거를 유지하고 실패합니다. 정상 WAL 복구는 기존
commit 항목을 먼저 보존합니다. 디스크 commit 실패 후 불확실한 index 소유자는
재사용하지 못합니다. 잠금 탐지는 호환되지 않는 실제 스레드 대기 순환을 따르며 일반
경합은 대기할 수 있습니다.

네이티브 batch context는 commit하지 않고 나가면 buffer를 rollback합니다. index
commit 이전 실패는 후속 작업을 중단하고 파일 잠금을 유지한 채 변경 파일을 복원합니다.
index commit이 불명확하거나 복원이 불완전하면 원본 hash·파일 대응 manifest를 포함한
비공개 `.batch-recovery-*` backup을 남기며 해당 저장소와 재열린 저장소를 명시적 운영자
조정 전까지 격리합니다. MCP는 `outcome_unknown`을 기록하고 소유자를 닫으며 재실행하지
않습니다. `concurrent_recovery_list(limit,offset)`는 네이티브 복구 조회를 호출하고 세션
소유의 불투명 `recovery_id`·단계·파일 개수를 반환합니다. 물리 경로·원본 hash·조회한
파일 상태 지문은 worker 내부에 보관합니다. `concurrent_reconcile(recovery_id,resolution)`는
새로운 정확한 승인이 필요합니다. `rollback`은 원본 바이트를 복원하고 원래 없던 파일을
제거합니다. `keep_current`는 실제 부분 효과를 포함한 현재 파일을 유지하고 관련 index만
조정하며 누락된 저장을 재실행하지 않습니다. 네이티브 `list_batch_recovery`와
`reconcile_batch(...,expected_fingerprint=...)`는 batch 소유권·변경 파일 잠금 안에서
manifest·backup 무결성, 안전한 경로와 현재 파일 지문을 재확인합니다. 상태가 바뀌면 새
조회·승인이 필요하고 실패 시 격리와 증거를 유지합니다.

MCP extra에 `filelock`을 포함하여 플랫폼 공통 저장소 복구의 기본값에 실제 프로세스
잠금 provider를 제공합니다. `filelock`·`portalocker`를 명시적으로 선택하려면 운영자
환경에 `truthound[stores-concurrency]`를 설치합니다. 네이티브 filelock의 0·비차단
제한 시간은 0으로 유지합니다. portalocker는 실제 lock API에 제한된 polling을 적용하고
공유 reader별 고유 descriptor와 open 실패 정리를 사용합니다. fcntl은 해제 후 잠금
파일 inode를 유지하여 대기 중·새 소유자가 서로 다른 파일로 소유권을 나누지 못하게 합니다.
실제 로컬 provider 경합과 세 프로세스 소유권 인계를 테스트하며 Windows 실행 증거는
최종 플랫폼 CI에서 확인해야 합니다.

하나의 안정된 네이티브 batch-owner 잠금이 게시·실행·복구의 소유권을 보장합니다. 실행
중인 소유자는 조정 실행을 막고 중단된 소유자의 OS 잠금은 해제됩니다. `preparing`
receipt는 backup 복사와 모든 파일 변경보다 앞서 저장하므로 해당 준비 상태를 명시적으로
해결할 때 현재 데이터를 변경하지 않습니다. 확정된 데이터·index 상태는 backup 삭제
전에 비공개 `.batch-resolved-*` 영역으로 원자적으로 이동합니다. 삭제 도중 중단되면
정상 데이터와 `cleanup_pending` 항목을 유지하고 `concurrent_recovery_cleanup(recovery_id)`가
네이티브 `cleanup_batch_recovery`의 나머지 정리를 별도 승인합니다. 정리 지문도 내부에
고정합니다. no-op 잠금은 프로세스 소유권을 보장하지 못하므로 복구에서 거부합니다.
manifest 읽기 한도는 16 MiB·100,000개 항목입니다. 대기 항목은 대상 데이터 파일 개수,
정리 항목은 남은 증거 파일 개수를 표시합니다. 명시적 조정·정리 기능이며 자동 multi-file
crash 원자성이나 외부 전달 보장은 아닙니다. index commit 전·후와 정리 중 실제 로컬
프로세스 중단을 테스트하며 전원 장애 내구성과 원격 backend는 미검증입니다.

`ObservableStore`는 일반 저장소 작업에서 factory가 소유한 같은 context·감사 backend·
metrics·tracer를 사용합니다. 네이티브 초기화는 `BaseStore`를 올바르게 통과하고 save·query
횟수를 한 번씩 집계합니다. 일부 close가 실패해도 audit·tracer·소유 metrics 서버를 모두
정리하며 내부 backend 종료 실패 시에도 관측 cleanup을 수행합니다.

`audit_query`는 선택적인 `record_id`, `audit.start_time`, `audit.end_time`,
`audit.event_type`, `limit`, `offset`을 받습니다. 네이티브 ISO timestamp 비교 규칙을
유지하고 `offset + limit <= 10,000`의 조회 한도를 적용합니다. 네이티브 시간·이벤트
유형·상태·기간·불투명 record 참조를 반환하며 actor·tenant·물리 ID·trace ID·요약·임의
metadata·오류 원문은 제외합니다. `metric_read(metric_id)`는 `StoreRuntime.metrics`의
별칭으로 신뢰된 `StoreMetric(name,labels,kind="counter")` query를 선택합니다. 네이티브
in-memory·Prometheus backend가 실제 counter·gauge 수치를 반환하고 이름·label은
비공개로 유지합니다. 임의 metrics나 Prometheus 원문 export는 공개하지 않습니다.
`observability_flush`는 네이티브 audit·tracer flush를 호출하며 원격 export 전달을
증명하지 않습니다. exporter 설정은 factory가 소유하고 외부 서비스는 미검증입니다.

`MigratableStore`는 `migration_status`, `migration_validate(record_id)`,
`migration_run(dry_run=true,batch_size=100)`을 제공합니다. batch size는 1–1000이고
callback·registry·목표/최소 버전·재시도 정책·필수 backup은 승인된 factory가 고정합니다.
상태는 네이티브 버전 분포·마이그레이션 필요 추정·표본 조회 실패·메모리 카운터를
유지합니다. validate는 네이티브 변환 경로를 확인하고 preview는 custom callback을
실행하거나 출력의 유효성을 증명하지 않습니다. 실제 실행은 네이티브 저장소 전체를
처리하며 실패·timeout 이전 일부 저장이 발생할 수 있습니다. receipt는 개수·dry-run·
success 판정과 `native_failed`·오류 개수를 유지하고 실제 record ID·오류 원문·callback
데이터는 숨깁니다. 자동 재실행은 추가하지 않습니다.

Core는 버전을 `ValidationResult.metadata`에 저장하여 저장소 재열기 후 완료한 변환을
반복하지 않습니다. 필수 backup 실패나 migration validator의 false는 원본 갱신을
차단하며 lazy 저장 실패와 불완전 query를 성공으로 숨기지 않습니다. 원본 backup은
버전을 바꾸지 않고 읽을 수 있고 후속 migration에서는 제외합니다. 기존 backup ID가
충돌하면 덮어쓰지 않고 실패합니다. 네이티브 `set_io_guard`는 callback·쓰기 직전
권한을 재확인합니다. lazy 읽기도 factory 정책에 따라 backup·갱신을 수행할 수 있으며
정확한 운영자 승인에 해당 효과를 포함합니다. backup은 네이티브 backend에 저장하지만
카운터는 프로세스 메모리이고 전체 저장소의 transaction rollback이나 영구 migration
scheduler는 아닙니다.

`ReplicatedStore`는 시작·정지·배출·통계와 선택적 `replica_id` 전체 동기화를 제공합니다.
`StoreRuntime.replicas`는 모든 구성 대상의 공개 별칭을 지정해야 합니다. 저장은 Core
`save_async`를 사용합니다. 동기·준동기 복제 실패는 primary 쓰기 후에도 예외를 발생시킬
수 있습니다. MCP는 이를 `outcome_unknown`으로 남기고 소유자를 종료하며 안전한 재실행으로
간주하지 않습니다. batch·전체 동기화 실패를 성공으로 덮어쓰지 않고, 전체 동기화 실패 시
대상은 읽기 라우팅을 위한 unhealthy 상태를 유지합니다. 네이티브 `stop`·`flush`는 해당
인스턴스의 비동기 작업·대기 쓰기를 배출합니다. `set_io_guard`는 background 복제·health
읽기를 포함한 primary·replica I/O 안에서 효과 직전 권한을 검사하여 재시도가 철회를
우회하지 못하게 합니다. 네이티브 재시도·best-effort 삭제 정책은 유지됩니다. 메모리
queue는 영구 receipt가 아니며 primary 저장만으로 replica 전달을 보장하지 않습니다.
통계의 lag·성공률 키에도 공개 replica 별칭을 사용합니다. 임시 로컬 저장소로 실제 효과와
부분 쓰기 quarantine을 검증하며 실제 cloud 복제·지역·자격 증명·프로세스 간 metadata
복구를 검증한 것은 아닙니다.

### binding 기반 tenant 관리

`tenant_services_open`은 비공개 `tenant_services` binding의 비동기 factory에서
`truthound.mcp.tenancy.TenantRuntime`을 받습니다. 하나의 네이티브 `TenantManager`,
공개 alias를 고정된 실제 ID로 연결하는 `tenants`, 선택형 네이티브 `TenantSettings`
객체의 `settings` alias를 제공합니다. 저장소·resolver·quota tracker·hook·실제 DB/schema
이름·알림 설정은 factory가 소유합니다. 선택한 Core 관리자만 다루며 MCP 데이터 접근·
동의를 변경하지 않습니다.

`tenant_services_execute`는 binding·서비스·키를 반복하고 정확한 승인을 받습니다.

- `catalog`: 컬렉션별 `limit` 1–1000(기본 100, 정책 한도), 0 이상 `offset`으로
  tenant/settings alias를 조회합니다. 다른 명령은 `tenant_id` alias를 선택하며 실제 ID나
  허용 목록 밖 tenant는 거부합니다.
- `get`·`delete`·`suspend`·`activate`는 네이티브 수명주기를 따릅니다. suspend의 선택형
  `reason`은 최대 4096자입니다. delete는 네이티브 usage 초기화와 factory의 hook도 호출합니다.
- `create`·`update`의 비어 있지 않은 `patch`는 `name`·`owner_id`·`owner_email`·
  `isolation_level`·quota 설정·운영자 `settings_id`를 받습니다. 생성에는 이름이 필요하고
  slug는 자동 생성하며 update로 `slug`를 편집합니다. 선택형 생성 `tier`는 네이티브
  기본값을 사용합니다. `set_tier`는 `tier`와 선택형 `update_quota`(기본 true)로 네이티브
  tier 기본 quota로 재설정할지를 선택합니다.
- `patch.quota`는 일/시간별 validations, 검증당 rows, storage bytes, 분/일별 API calls,
  일별 checkpoints, 최대 retention days/concurrent jobs/users와 `custom_limits`의 부분
  설정을 받습니다. 정수는 -1..2^63-1이며 강제 의미는 네이티브 tracker를 따릅니다.
  생성 시 tier 기본 quota를 확장하고 갱신 시 다른 필드를 유지합니다.
- `quota_check`·`usage_track`은 네이티브 `resource`와 양수 `amount`(기본 1, 최대 10^9)를
  받습니다. check는 네이티브 판정이며 track은 사용량 기록으로 quota가 검증된 작업을
  자동 실행하지 않습니다. `usage_summary`는 resource와 `window`(`hour`·`day`·`month`,
  기본 day)를 선택하고 네이티브 counter·limit·flag와 공개 tenant alias를 반환합니다.
- `context`는 해당 worker 안에서 네이티브 tenant 문맥에 진입·조회·종료합니다. 다른
  소유자에게 문맥을 유지하지 않으며 resolver 연결·데이터 행 격리는 별도의 명시적 작업을 사용합니다.

조회는 정의된 상태·tier·owner·quota·시간 필드만 제공하고 임의 metadata·실제 저장소 이름·
설정 endpoint·자격증명을 제외합니다. 모델의 콜백·import·경로·저장소 설정은 받지 않습니다.
지속성은 provider가 결정하며 실제 SQLite owner 종료·재개방과 상태·quota·문맥, 정지
metadata 복원을 로컬 테스트합니다. 사용량 tracker는 별개이며 기본 메모리 tracker는
프로세스와 함께 종료됩니다. 네이티브 0/무제한 tenant quota는 초과 summary를 보고하지
않습니다. 파일·원격 tenant 저장소나 분산 tracker 검증을 SQLite 증거로 대신하지 않습니다.

### 네이티브 rate budget과 비밀정보 메타데이터 관리

`rate_limit_services_open`은 비공개 `rate_limit_services` binding의 고정된 비동기
factory에서 `truthound.mcp.governance.RateLimitRuntime`을 받습니다. 네이티브
`RateLimiter`, 공개 alias를 실제 key로 연결하는 `subjects`, 선택형 불투명 `contexts`,
네이티브 `policies`·`quotas` alias, limiter에 연결한 것과 동일한 `PolicyRegistry`·
`QuotaManager`를 제공합니다. 저장소·extractor·설정은 운영자가 소유합니다. 이 budget은
MCP 자체 제한·동의를 변경하지 않습니다.

`rate_limit_services_execute`는 binding·현재 서비스 ID·정확한 멱등 키·명령마다
별도 승인을 받습니다.

- `catalog`는 alias를, `config`는 콜백을 제외한 네이티브 scalar 설정을 조회합니다.
- `acquire`·`peek`·`reset`은 `subject_id`가 필요하며 실제 key를 직접 받지 않습니다.
  acquire의 `tokens`는 1–1000000(기본 1)입니다. `wait=true`이면 (0, 10]의
  `timeout_seconds`가 필수이며 기다리지 않을 때는 생략합니다. 선택형 `context_id`는
  네이티브 정책 평가에 사용할 운영자 객체를 고릅니다. `peek`는 네이티브 의미를 따르며
  모든 정책을 통한 토큰 소비 시뮬레이션이 아닙니다.
- `quota_register`는 운영자 `quota_id`를 등록합니다. 이후 `quota_check`·`quota_consume`·
  `quota_usage`·`quota_reset`에 `subject_id`·`quota_id`를 사용하며 check·consume의
  사용량은 `tokens`입니다. 미등록 quota는 거부합니다. 소비는 네이티브 usage를 반환하며
  초과 소비의 네이티브 예외는 성공으로 처리하지 않고 핸들을 격리할 수 있습니다.
- `policy_register`·`policy_unregister`는 `policy_id`, `policy_resolve`는 `context_id`로
  선택하며 네이티브 match·설정과 공개 정책 alias를 반환합니다.

판정은 네이티브 allowed·remaining·limit·reset·retry·utilization과 HTTP 헤더를 제공합니다.
실제 bucket key와 임의 정책 metadata는 제외합니다. 성공 요청의 정확한 재호출은 보관한
산출물을 반환하고 다시 소비하지 않습니다. 지속성은 선택한 Core 저장소·manager 의미를
따릅니다. 메모리 budget은 소유자와 함께 종료되며 MCP 영수증으로 분산·영구 복구를
추가하지 않습니다. 로컬 메모리 저장소·quota를 테스트하며 원격 Redis·Memcached는 미검증입니다.

`secret_services_open`은 비공개 `secret_services` binding의 `SecretRuntime`을 받습니다.
네이티브 `SecretManager`, 공개 alias에서 `SecretReference`로의 고정 연결, provider
alias의 `SecretProviderSlot(provider, name, priority)`를 제공합니다. 실제 key·endpoint·
version·기본값·자격증명 해석은 factory가 소유하며 모델은 이 필드나 비밀 값을 보내지 않습니다.

`secret_services_execute`는 `catalog`, `resolve(reference_id)`, 전체 또는 선택한
`provider_id`의 `cache_clear`, 기존 `provider_id`의 `provider_add`·`provider_remove`를
지원합니다. resolve는 네이티브 조회를 수행하되 `resolved`와 존재 시 `expired`·
`created_at`·`expires_at`만 반환합니다. 값·해시·길이·실제 key/provider 이름·URI·임의
metadata는 반환하지 않습니다. `resolved`는 네이티브 조회·캐시 상태이며 외부 서비스가
자격증명을 받아들인다는 증명이 아닙니다. 실제 소비자는 운영자가 선택한 factory 내부에서
자격증명을 해석·사용하며 raw `get_value`·해석된 config를 전송하지 않습니다. 종료 시 선택한
manager의 캐시를 지우고 외부 client 정리는 factory가 맡습니다. 다른 소유자의 캐시는 독립적입니다.

두 catalog는 컬렉션별 `limit` 1–1000(기본 100, 정책 한도)과 0 이상 `offset`을 받습니다.
모든 호출에 민감한 결과 허용과 정확한 승인이 필요합니다. provider 실패에서 예외 원문을
반환하거나 효과를 자동 재실행하지 않습니다. Core도 공급자 예외의 내용을 포함하지 않는
일반 실패 경고만 기록하며 비엄격 모드의 없는 공급자는 `None`입니다. 합성 로컬 provider로
실제 캐시 hit·초기화·factory 종료를 검증하며 운영 환경 비밀이나 cloud secret 서비스는
읽거나 호출하지 않았습니다.

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
무효화됩니다. 알 수 없는 루트·설정은 실패합니다. `output_format="native"`가
기본값이며 전체 그래프의 HTML·Markdown·SVG·PNG 형식과 의존성은 아래의 확장
출력 계약을 따릅니다. 호출자가 출력 경로나 실행 파일을 지정하거나 브라우저를
자동 실행하지 않습니다. 트래커 문맥·이력과 OpenLineage 전송은 아래의 별도 승인 소유자를 사용합니다.

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

기능 조회는 `full_core_parity=false`를 반환합니다. 아래 구현된 작업과 승인 네이티브
서비스에는 범위를 명시한 회귀 검증이 있으며, 모든 알고리즘·설정·제공자의 동등성까지
증명하지는 않습니다. 네이티브 추적기 context·이력·렌더링, 체크포인트 서비스, 스트림,
스토어, 플러그인 수명주기와 영속 작업은 각각 문서화한 계약으로 제공됩니다. 서비스별
설정·영속화·복구·정리 제한을 확인하십시오. 작업 등록만으로 모든 제공자의 동작이
검증되지는 않습니다. 외부 실서비스와 소유자 강제 종료 후 복구에는 별도의 증거가
필요합니다. 원격 서버·자격 증명·유료 호출은 추가하지 않습니다.

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

MCP CI는 Linux Python 3.11에서 전체 필수 테스트를 결정적인 4개 분할로 나누되
15분 job 제한은 유지합니다. 모든 분할이 성공한 뒤에만 패키징을 시작합니다.
다른 선택된 OS·Python 조합은 플랫폼 경계 테스트를 실행하며 Windows는 2개
비중복 분할, 나머지는 단일 분할을 사용합니다. 설치 전에 해당 증거를 집계하며
경계 테스트를 전체 의미 동등성 검증의 대체물로 사용하지 않습니다.
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


### 승인된 네이티브 스트리밍 저장소

아직 배포하지 않은 Core 프로필의 `streaming_store_services_open`은 승인된 설치 capability가 반환하는 `StreamingStoreRuntime(store, runs={alias: StreamingRun(run_id, data_asset, metadata)})`을 사용합니다. store는 네이티브 `StreamingValidationStore`여야 하며 일반 `BaseStore` 바인딩과 별개입니다. 운영자 bindings의 `streaming_store_services`에 별칭→capability를 등록합니다. backend URL·자격증명·코덱·실제 run 식별자는 factory에서만 공급합니다.

`streaming_store_services_execute`는 `inspect`, `session_create/open/resume/status`, `writer_open/write/write_run/flush/close/abort/status`, `reader_open/read/reset/close/status`, `chunks_list`, `chunk_get`, `chunks_delete`, `finalize`, `stats`, 파일 저장소 전용 `storage_size` 명령을 명시적으로 제공합니다. 실제 명령은 `session_create`, `writer_write_run`처럼 밑줄로 연결합니다. 모든 동작은 승인이 필요하며 용량 제한된 비공개 artifact를 반환합니다. `run_id`는 고정 run 별칭이고 session·writer·reader·chunk ID는 해당 소유자 내부 참조입니다. `writer_write`는 타입이 지정된 영속화용 validator record를 최대 1,000개 받습니다. `writer_write_run`은 고정된 canonical 검증 artifact를 소비하며 verdict 스트림이 보존할 수 없는 실행 issue가 있는 run은 거부합니다. writer·reader 생성 시 `asynchronous=true`로 네이티브 async wrapper를 선택합니다. 읽기·청크 조회는 제한된 `limit`을 사용하며 목록에는 `offset`도 지원합니다.

명시적 `writer_close`와 `finalize`는 남은 버퍼를 flush합니다. `writer_abort`는 버퍼를 쓰지 않고 중단합니다. 서비스 해제·만료·입력 철회는 열린 writer를 먼저 abort하므로 정리 과정에서 새 레코드를 저장하지 않습니다. 이미 시작한 I/O의 취소나 디스크에 쓰지 않은 버퍼의 복구를 보장하지 않습니다. 새로 승인된 소유자는 같은 고정 run을 `session_open`하고 미완료 스트림을 명시적으로 `session_resume`할 수 있습니다. 불일치 복구 상태는 거부·보존하며 자동 채택·재실행하지 않습니다. 최종 집계는 네이티브 영속화 결과 projection을 반환하며 전체 결과를 메모리에 구성할 수 있습니다. 설치된 로컬 factory에서 파일·실제 SQLite 왕복을 검사하며 실제 S3·기타 DB 서비스는 미검증입니다.

파일 저장소·파일 SQLite의 `recovery_status(run_id=별칭)`는 민감한 경로·해시 대신 소유자 내부 `recovery_id`와 확정·미확정·누락 청크 수를 반환합니다. 별도 승인된 `recovery_reconcile(recovery_id=..., resolution="adopt"|"discard")`가 보존된 fingerprint를 네이티브 잠금 안에서 확인하여 명시적 선택만 수행합니다. 성공 후 이전 세션·청크 참조를 무효화하므로 `session_open`부터 다시 시작합니다. 불일치·손상된 확정 prefix는 복구 성공으로 처리하지 않습니다.

파일 SQLite의 복구는 실제 run 잠금과 `BEGIN IMMEDIATE` 아래에서 행 삭제·세션 갱신을 함께 commit합니다. fingerprint 변경·확정 청크 불일치 시 거부하며, 갱신 중 실패·프로세스 종료 시 트랜잭션을 롤백합니다. 미확정 행의 메모리 검사에는 세션의 `max_memory_mb` 한도를 적용합니다. `session_release`, `chunk_release`, `recovery_release`는 소유자 내부 참조만 해제하며 영구 데이터를 삭제하지 않습니다.

네이티브 S3 스트리밍은 opaque ETag 조건부 manifest·새 청크의 create-only 요청·쓰기 전 receipt 확인·소유 multipart abort를 사용합니다. 경로·ETag·upload ID는 모델에 반환하지 않습니다. 로컬 protocol 대역에서 불명확한 complete receipt와 정리 실패를 검사했으며 실제 S3·SDK의 배포 검증은 별도로 남아 있습니다.

`writer_status`는 flush하지 않고 버퍼·종료·지표 snapshot을 반환합니다. `reader_status`의 지표는 I/O로 읽은 레코드·bytes이며 prefetch가 포함될 수 있어 반환한 페이지 수와 다를 수 있습니다. 직렬화·압축된 단일 청크가 세션의 읽기 메모리 한도를 넘으면 backend 쓰기 전에 거부하고 버퍼를 보존합니다.

스트리밍 저장소의 동일한 opaque `recovery_status`·`recovery_reconcile` 계약은 승인된 S3 factory에도 연결됩니다. 모델은 객체 키·ETag·SHA 지문·multipart ID를 받지 않습니다. S3 폐기 장애는 manifest에 진행 상태를 남기며 새 recovery 참조와 명시적 폐기가 필요합니다. 전체 삭제 장애의 `pending_resolution=delete`는 새 승인된 `chunks_delete`로 완료합니다. 파일·SQLite의 실제 프로세스 손실 검사와 S3 합성 객체 프로토콜의 작업 프로세스 교체 검사는 실제 원격 S3 검증과 구분합니다.

일반 DB 저장소 binding은 `store_services_execute`의 `pool_status`·`pool_recycle`·root `shutdown`을 제공합니다. 풀 상태는 연결 URL·SQL·provider 예외 원문 없이 네이티브 상태와 숫자 지표를 표시하며 재활용은 선택한 네이티브 풀에 위임합니다. 승인된 `shutdown`은 소유 background 작업과 factory의 종료가 끝난 뒤 응답하고 실패·만료를 종료 성공으로 바꾸지 않습니다. 같은 승인 키는 결과를 재생할 수 있고 이후 다른 명령은 `invalid_handle`로 실패하며 서비스 자원은 별도로 해제합니다. 실제 SQLite pooled/legacy factory에서 재활용·종료·새 승인 소유자로의 재개방 후 데이터를 확인합니다. SQL 읽기 오류는 성공한 빈 artifact 대신 실패/불명확한 작업으로 남습니다.


`tenant_services_execute(command="isolation_apply")`는 고정 `tenant_id` 별칭과
고정된 `asset_id`, 선택형 `columns`, 제한된 `offset`·`limit`을 받습니다.
`TenantManager.apply_isolation`에 위임하여 비공개 행 페이지를 반환하며
`returned_rows`는 페이지 개수, `truncated`는 추가 행 존재 여부입니다. 임의 캡처 자원은
shared/row-level 전략만 적용하고 schema/database 출처를 추정하지 않으며 tenant의
활성 상태를 확인합니다. 민감 결과 허용과 정확한 승인이 필요하고 tenant 관리 권한이
입력 자원의 읽기 허용을 넓히지 않습니다. 입력 해제는 서비스·파생 artifact를 철회합니다.

factory의 `TenantRuntime.policies`는 별칭을 네이티브 `RowLevelPolicy` callback에
연결합니다. `policy_catalog`·`policy_add`·`policy_remove`·
`policy_update(policy_patch={enabled, priority})`는 선택한 tenant의 네이티브 row-level
전략을 사용하며 모델 코드는 받지 않습니다. 비어 있지 않은 patch는 두 필드 중 하나
이상을 바꾸며 priority 범위는 ±1,000,000입니다. metadata 결과에 실제 정책 이름·tenant
ID를 노출하지 않습니다. 정책은 소유자 내부 설정으로 SQLite tenant metadata에
영속화되지 않습니다. Core 기본 정책은 bool이 아닌 verdict·정책 오류를 거부하고 tenant
필터 뒤, 컬럼·페이지 제한 앞에서 실행합니다. 명시적인 네이티브 `fail_closed=False`
동작은 모듈 가이드를 따릅니다.


운영자 custom 규칙은 승인된 profiler factory의
`ProfilerRuntime.schema_analyzers`에 네이티브 `SchemaCompatibilityAnalyzer`를
별칭으로 연결합니다. `profiler_services_execute(command="schema_compare",
analyzer_id=..., baseline_id=..., current_id=...)`는 두 입력 자원을 고정해 선택한 analyzer에
위임하며 analyzer만 있는 factory도 지원합니다. callback·strict·선언 nullability 설정은
factory 소유이고 요청에서 규칙을 import하지 않습니다. 결과는 네이티브 호환성 네 필드를
보존하며 실패한 규칙을 호환 결과로 바꾸지 않습니다. 입력 해제는 소유자·파생 artifact를
철회하고 실행 guard는 정규화 전과 각 네이티브 규칙 앞뒤에서 실행합니다.

### 네이티브 ML 모니터 수명주기

`ml_services_open`은 비공개 `ml_services` 바인딩에서 승인된 `MLRuntime` 팩토리를
선택합니다. 모델 별칭은 네이티브 설정·자산 열 매핑을, 규칙·전송기 별칭은 고정 코드와
전송 설정을 선택합니다. `ml_services_execute`는 등록, 고정 자산의 예측·기준·정답,
주기 시작·중단, 수동 수집·평가, 지표·대시보드, 규칙·전송기 추가·제거, 불명확 버퍼의
명시적 폐기와 종료 완료 응답을 제공합니다. [모델 모니터링 가이드](ml-monitoring.md)에
전체 명령·수집 창·실패 계약을 설명합니다.

각 동작은 승인과 민감정보 공개 허용을 요구합니다. 시작은 중단·철회까지 향후 주기
수집·평가·보존 삭제를 포함합니다. 입력·파생 산출물은 소유자의 권한 기간에 묶입니다.
출력은 공개 모델·규칙·특성 별칭과 유한 지표를 사용하며 실제 ID·임의 알림 메시지·전송
주소는 제외합니다. 실패 `native_monitor_incomplete` 산출물은 외부 효과가 없다는
증거가 아니며 자동 재전송하지 않습니다. 기본 버퍼·이력은 메모리 상태이며 프로세스
손실 복구를 주장하지 않습니다. 설치형 로컬 검사는 네이티브 메모리 저장소와 로컬
전송기를 사용하며 실제 외부 서비스는 미검증입니다.


### 네이티브 저장 코덱·백프레셔 유틸리티

`store_utilities_open`은 비공개 `store_utilities` 바인딩에서 승인된
`StoreUtilityRuntime` 팩토리를 선택합니다. 네이티브 압축·암호화 파이프라인과
백프레셔는 팩토리 설정이며 키·provider·단계 코드·물리 슬롯을 호출자가 입력하지
않습니다. `store_utilities_execute`는 고정 자산·산출물 bytes 캡처, payload
보존·처리·역변환·export·import, 고정 비공개 슬롯 저장·복원·삭제, 네이티브 흐름
지표·허가·해제·reset과 종료 완료 응답을 제공합니다. [저장 유틸리티 가이드](stores/index.md#네이티브-코덱흐름-제어-유틸리티)에
명령·예산·binary 형식·영속화·실패 계약을 설명합니다.

각 동작은 승인과 민감정보 공개 허용을 요구합니다. 네이티브 단계 guard는 철회 후
다음 실행을 막고 입력 해제는 소유자·파생 산출물을 무효화합니다. 불완전 허가는 실패
산출물이며 명시적 native reset 전에는 재시도하지 않습니다. boolean 거부는 유효한
결정입니다. 암호화 복원은 실행 전에 선택 계획·헤더 불일치를 거부합니다. 출력은 키
자료와 provider 진단을 제외합니다. 실제 로컬 코덱·cryptography를 검사하며 외부
키 저장소·provider의 실서비스 검증을 주장하지 않습니다.


Ray 팩토리의 `concurrency`는 네이티브 task pool에, `use_actors`·`actor_pool_size`는
실제 callable-class actor pool에, `target_max_block_size`는 선택한 Dataset의 context에
적용합니다. Actor 모드와 task 동시성을 함께 지정하면 거부합니다. 배치·prefetch·pool·
블록 크기와 실제 boolean 모드의 범위를 검사합니다. 이미 초기화한 공유 Ray 클러스터의
CPU·GPU 용량을 다시 설정하지 않습니다. Lazy Ray 자식의 `num_partitions` 조회는 선택한
데이터를 materialize하여 실제 블록 수를 관찰하며 계획 추정치를 반환하지 않습니다.
따라서 분할 수 관찰도 팩토리 소스 계획을 실행할 수 있습니다. 실제 로컬 SDK actor와
작업 겹침을 검사하며 원격 클러스터 동작은 별도 미검증입니다.


저장 유틸리티의 키 바인딩은 네이티브 생성·get-or-create, 비공개 값을 제외한 상태·이력,
버전 회전·삭제, 선택 암호 적용과 manager 전체 만료 정리를 지원합니다. 키 자료·실제
ID는 응답에 넣지 않습니다. 코덱 실행 전 적용한 키를 다시 검증하고 불명확한 효과는
manager 별칭을 차단해 자동 재실행하지 않습니다. [키 영속화 계약](stores/index.md#키-영속화와-승인된-키-사용)을 참조하십시오.

### 분산 집계 계획과 검증기 실행

`ExecutionRuntime` 팩토리는 네이티브 `AggregationPlan`의 `plans` 별칭,
Core `Validator`의 `validators` 별칭과 고정 `AdapterConfig`인
`adapter_config`를 제공할 수 있습니다. `inspect`로 별칭을 조회합니다.
`aggregate_plan`은 `aggregation_plan_id`를 선택하며, `aggregate_stream`은
`batch_size`(1–100,000)와 응답 `limit`도 받습니다. `validate`는 분산 엔진에서
`validator_ids`(별칭 1–32개)와 선택적 `columns`를 받습니다. 호출자가 작성한
계획·SQL 필터·검증기 생성자·콜백은 wire 입력이 아닙니다. 기존 승인·민감 정보
공개 동의·소유자 수명주기가 적용됩니다.

계획은 로컬 Polars/Pandas/Arrow, Dask, Ray, Spark 데이터에서 실행됩니다.
Core에 등록된 집계기가 실제 worker 상태를 병합합니다. 평균·표준편차는 분할의
최종 요약값을 평균내지 않고 누적 상태를 병합합니다. 고정 SQL 필터는 각 worker
배치에서 Polars로 평가합니다. 전역 결과는 별칭별 값이며, 그룹 결과는 자료형을
유지한 모든 키를 보존합니다.

```json
{"groups": [{"keys": {"region": null}, "aggregations": {"amount_mean": 7.0}}]}
```

`aggregate_stream`은 독립적인 누적 `snapshots`와 `complete`를 반환합니다.
`complete: false`는 응답 한도로 스트림이 잘렸다는 뜻이며 마지막 snapshot을
최종 집계로 사용할 수 없습니다. Driver는 다음 배치 요청 전과 관측 후 권한을
검사합니다. 이미 제출한 네이티브 작업은 진행 중일 수 있습니다. 상태 메모리는
그룹 수에 따라 증가하며 `batch_size`가 전체 메모리를 격리하지는 않습니다.
distinct는 충돌하는 hash도 원래 값으로 구별하여 `max_sample`까지 정확히
집계하며, 한도를 넘으면 잘린 건수 대신 오류를 반환합니다. 열 distinct/count는
null을 제외하고 `count("*")`는 모든 행을 셉니다. 빈 mean/std는 등록된 Core
집계기의 0.0 계약을 따릅니다. 중복 결과 별칭은 오류입니다.

`DistributedValidatorAdapter`는 정확한 내장 `NullValidator`·
`NotNullValidator` 클래스의 네이티브 건수와 Core issue 생성기를 사용합니다.
실패 값 표본·빈도·전체 행이 필요하면 행 수 제한을 적용하는 Arrow-to-Polars
bridge를 사용합니다. 사용자 검증기는
`validate_distributed(engine, columns=None) -> list[ValidationIssue]`를
명시적으로 구현할 수 있습니다. 이름·분류만으로 검증 로직을 대체하지 않습니다.
`AUTO`는 다른 검증기에 대해 행 수가 `native_threshold` 이하면 전체 변환,
그보다 크면 표본 실행을 선택합니다. 강제 `NATIVE`는 네이티브 port가 없는
검증기를 거부합니다. `FULL_CONVERSION`·`SAMPLE_AND_LOCAL`은 실제 검증기를
호출하며 둘 다 요청 열을 적용합니다. 병렬 실행은 `max_parallel`과 입력 순서를
지키고 제출한 작업이 끝난 뒤 실패를 전달합니다. 순차 접근이 필요한 검증기는
`parallel_validators=False`를 선택합니다.

표본 issue의 건수·구조화된 비율은 관측한 행을 나타냅니다. 일부 행만 검사하면
각 issue에 `sampling: {total_rows, sample_rows, seed}`를 추가하며 건수를
전체로 외삽하지 않습니다. 검증 실패 시에도 파생 표본 엔진을 정리하고, root를
재사용한 표본은 root의 소유권을 가져가지 않습니다.

Dask 표본은 분산 무작위 순위로 정확히 `min(n, row_count)`개 행을 선택하며
seed는 분할 번호와 함께 적용합니다. 계산·persist에 설정한 scheduler를 적용하고,
결과 캐시 비활성화를 지키며 Arrow 변환을 한 번만 계산합니다. 엔진이 만든 분산
client는 context 종료 시 닫고 빌린 client는 호출자가 관리합니다. 자식은 부모의
client를 빌리므로 부모 종료 전에 작업을 끝내야 합니다.
`persist_intermediate`를 선택하면 생성 시 persist합니다. Ray의 null 포함 그룹
건수는 복사한 Dataset의 네이티브 hash shuffle을 사용하여 null 동등성과 호출자의
shuffle context를 보존합니다. 실제 로컬 Ray 2.48 및 현재 SDK로 호환 경로를
검증했으며 외부 cluster·서비스는 미검증입니다.

저장 유틸리티는 네이티브 `adaptive`·`chunked_compression`·
`chunked_encryption`도 선택합니다. 고정 adaptive 구성의 분석·추천은
`codec_analyze`·`codec_recommend`, 청크 평문 바이트 범위는 `codec_range`로
호출합니다. 헤더·index·크기·체크섬 오류를 보존하고 v2 암호화 index는 평문 반환
전에 인증합니다. [네이티브 코덱 설정·형식 계약](stores/index.md#스트리밍청크-암호화의-인증)을
참고하십시오.

증분 `stream_open/write/read/flush/finalize/abort/status/release` 명령은 승인된
`StoreCompressionStreamBinding`·`StoreEncryptionStreamBinding` 팩토리를
사용합니다. 실패·불명확한 native context 정리는 후속 작업을 차단하며 shutdown은
미완료 stream을 abort합니다. [증분 코덱 세션](stores/index.md#증분-코덱-세션)을 참고하십시오.

일반 파일 저장소 팩토리는 JSON/safe YAML·선택 네이티브 코덱·config 객체를
적용하며 결과·index를 각각 원자적으로 게시합니다. 손상은 오류이고
`rebuild_index`는 결과를 검증한 뒤 명시적으로 조정합니다. 메모리 팩토리는
용량·복사·기본 metadata를 적용하며 shutdown 후 이전 소유자의 상태를 보존하지
않습니다. 단일 writer 범위·ID 규칙·초기 검사·구형 이름은
[로컬 저장소 계약](stores/filesystem.md)을 참고하십시오.


Pandas 엔진은 전달한 변경 가능한 프레임을 참조합니다. 프레임을 변경한 뒤에는
`clear_cache()`로 계산 결과와 구체화된 Polars 변환 캐시를 함께 무효화합니다.
`cache_results=False`이면 호출마다 현재 프레임을 변환합니다.

`clear_cache`는 Ray·Spark 행 수 캐시도 무효화하며 `cache_results=False`는
Core 결과 캐시를 우회합니다. Ray는 연산별 명시적 매개변수가 없는 전역
COUNT/SUM/MEAN/MIN/MAX/STD/VAR/NULL_COUNT 집계에서 정확한 열 통계 캐시와 기본 결과를 재사용하며 MIN/MAX 쌍도 공유합니다.
`clear_cache()`는 두 캐시를 모두 무효화합니다. `ddof=0` 같은 명시적 매개변수,
그룹/필터 계획·FIRST/LAST·고유 값 수와 제한된 median 변환은 고유 실행을 유지합니다. `COUNT(*)`는
행 수를 유지하며 캐시된 null 결과도 null입니다. 네이티브 dataset persist는 이 캐시와 별개입니다.
`AggregationPlan.to_spec()`는 집계 매개변수·그룹 열·고정 SQL 필터를 독립적으로
복사합니다. 그룹/필터 spec도 계획과 같은 Core 상태 집계를 사용합니다. Dask
계획·spec은 선택 엔진의 scheduler를 유지하며, 독립 `DistributedAggregator`도
scheduler/client callable을 지정하는 `dask_scheduler` keyword를 받습니다.
로컬 Polars·Arrow table 계획에는 pandas가 필요하지 않습니다.


분산 엔진은 최초 생성 시 양수 `num_partitions`를 적용합니다. 명시적인 자식
`repartition`·`coalesce`는 요청한 파티션 수를 유지하며 sample·filter·projection은
루트 기본값을 다시 적용하지 않습니다. 파티션 수의 bool·0·음수는 오류입니다.
Spark는 입력 설정을 복사하고 선택 session의 Arrow·adaptive 설정에 `True`와
`False`를 모두 적용합니다. 추가 Spark 설정은 이 필드보다 우선합니다.

승인된 `AggregationPlan(scope=AggregationScope.PARTITION)`과 같은 scope의
네이티브 `AggregationSpec`는 각 네이티브 파티션을 따로 최종 집계합니다.
결과는 `{"partitions": [{"partition_id": 0, "result": {"x_sum": 3}}]}`이며 ID는
관측 순번으로, 재실행·재분할 후에도 유지되는 영구 ID가 아닙니다. 필터·선택 그룹은
파티션 안에서 적용되고 빈 파티션은 등록 집계기의 빈 결과를 반환합니다. Ray의
파티션은 네이티브 block이며 [Ray batch API](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)의
`batch_size=None`을 사용합니다. 내부 batch 상태는 block 안에서 병합합니다.
파티션 계획 stream은 완료한 파티션 결과의 독립적인 누적 목록을 반환하고 기존
`complete` 잘림 계약을 유지합니다. MCP에서는 기존 계획 별칭으로 선택합니다.

승인된 일반 S3 팩토리는 네이티브 형식·압축·metadata·object tag, 결과/index의
조건부 receipt, 오류를 보존하는 query와 명시적 `rebuild_index`를 적용합니다.
shutdown은 소유 client를 닫습니다. [S3 복구·provider 계약](stores/cloud-storage.md#조건부-게시와-복구)을
참고하십시오. 로컬 합성·botocore 검증은 실서비스 검증과 별개입니다.


`ExecutionMode.EAGER`는 생성 시 선택 네이티브 그래프를 계산하며 자식 엔진에도
적용합니다. Dask는 persist한 뒤 distributed client 사용 시 선택 future의 완료를
기다리고 전체 파티션을 driver로 가져오지 않은 채 작업 오류를 전파합니다.
`timeout_seconds`는 이 최초 대기를 제한하며, timeout이 원격 작업 중단을 증명하지는
않습니다. `persist_intermediate`만 선택하면 Dask의 일반적인 비동기 제출 동작을
유지합니다. Ray는 네이티브 `MaterializedDataset`을 보관하고 context 종료 시
엔진이 만든 참조를 해제합니다. 빌린 materialized dataset은 호출자가 관리합니다.
Spark는 즉시 persist·count하며 종료 시 엔진이 만든 캐시만 완료 대기와 함께
해제합니다. 이미 persist된 입력은 호출자 소유로 유지합니다. 명시적 `unpersist`는
선택 캐시를 해제합니다. 기본 `LAZY`는 지연 그래프를 유지합니다. MCP
`inspect.config`는 cluster 주소 없이 선택 실행 모드·초기 파티션 기본값·Arrow batch
크기·metrics 플래그를 제공합니다.


팩토리 소유 분산 설정의 `partition_strategy`는 최초 분할과 명시적
`repartition`에 적용합니다. `ROW_HASH`는 `partition_columns`를 사용하며 생략 시
모든 열을 사용합니다. `COLUMN`은 명시한 열의 같은 key를 함께 hash 분할합니다.
`ROW_RANGE`는 Spark·Ray의 정렬 범위를 사용합니다. Dask는 범위 열 하나를
지원하며 선택 scheduler로 파티션별 key 표본을 계획하고, 네이티브 Dask 작업에서
null 행을 보존한 채 분배·정렬합니다. driver 표본은 입력 파티션 수와 요청 파티션
수의 곱에 따라 증가합니다. Spark는 key 종류가 적으면 범위 파티션이 더 적을 수
있으며 `num_partitions`는 실제 개수를 반환합니다. `ROUND_ROBIN`은 각 입력
파티션/block의 0부터 행을 순환 분배하며 영구적인 전역 행 순서를 보장하지 않습니다.
`CUSTOM`은 같은 네이티브 자료형·schema와 요청 파티션 수를 반환하는
운영자 소유 `partitioner(data, count)`를 요구하고 잘못된 반환은 오류입니다.
MCP는 호출자의 partitioner 코드를 받지 않습니다. 일반 projection·filter 자식은
기존 분할을 유지하며 `coalesce`는 shuffle 전략과 별개로 각 backend의 축소
동작을 사용합니다. 범위 계획과 네이티브 schema·파티션 조회는 `LAZY`에서도
key를 읽거나 그래프를 실행할 수 있습니다. Ray 파티션 집계는 block별 Arrow 참조를
사용하여 빈 block의 빈 결과도 유지하고 원본 block 값은 Ray worker에 둡니다.


스트리밍 S3의 `recovery_status`는 영구 multipart intent·receipt 개수도 제공합니다.
`multipart_reconcile`은 보관한 `recovery_id`와 선택 `receipt_id`를 받습니다.
receipt 별칭은 팩토리 소유 `StreamingStoreRuntime.multipart_receipts`의 비공개
정확한 key→upload ID 매핑입니다. 생성 receipt가 없으면 운영자가 확인한 별칭이
필요하고, 기존 receipt가 있으면 별칭 없이 정리할 수 있습니다. inspect는 별칭만
보여 줍니다. 승인 명령은 해당 upload를 abort하고 불명확한 정리를 보존하며 새
복구 참조·개수를 반환합니다. provider key·upload ID·fingerprint는 숨깁니다.
완료 chunk의 채택·폐기는 별도입니다. [영구 multipart 소유 정보](stores/index.md#s3-multipart-소유-정보의-영구-복구)를
참고하십시오.

승인된 일반 GCS 팩토리는 공통 Core object 저장소 상태를 사용하고 네이티브
generation receipt·원본 bytes 읽기·명시적 index 재구축·실제 client 종료를
제공합니다. [GCS generation·복구 계약](stores/cloud-storage.md#generation-receipt와-복구)을 참고하십시오.

승인된 일반 Azure factory도 같은 Core inventory·복구 경로와 native 조건부
ETag·저장 바이트 복원을 사용합니다. blob 헤더 `metadata`와 영구 결과의
`record_metadata`는 구분합니다. `get_access_tier`는 선택한 결과의 관측 tier와
status를 읽고 `set_access_tier`는 `access_tier`(`Hot`, `Cool`, `Cold`, `Archive`)를
받아 `acknowledged`와 관측 상태를 반환합니다. 이는 rehydration 완료 응답이
아닙니다. 두 명령은 wrapper가 없는 선택된 Azure 저장소를 요구합니다. 클라이언트
입력으로 credential·endpoint·실행 callback을 전달하지 않습니다. shutdown은
소유한 SDK client와 기본 credential을 종료합니다.
[Azure 복구와 소유권](stores/cloud-storage.md#etag-응답-복구와-클라이언트-소유권)을 참고하세요.

### Native 실행 엔진 checkpoint와 Arrow cursor

Dask와 Ray의 `checkpoint_dir`는 운영자가 소유하는 로컬 디렉터리입니다.
`checkpoint()`는 고유 staging 디렉터리에 native Parquet를 쓰고 파일과 SHA-256
목록을 flush한 뒤 디렉터리 rename으로 완료 snapshot을 게시합니다. Python에서
반환된 child의 `checkpoint_path`를 조회할 수 있습니다.
Ray는 로컬 목적지 디렉터리를 생성하고 Windows drive 경로를 포함한 native
쓰기·읽기에 Arrow local filesystem을 명시적으로 선택합니다.
`execution-ray`·`all`·`dev` extra는 Windows 로컬 경로 처리 중 사용할 수 없는
선택 Azure filesystem을 조회하는 SDK 회귀 때문에 Ray 2.59.0을 제외합니다.
CI 밖에서도 같은 의존성 제한을 적용하며 native 계산·checkpoint assertion은 유지합니다.
`DaskExecutionEngine.from_checkpoint(path, config, client=...)`와
`RayExecutionEngine.from_checkpoint(path, config)`는 전체 파일 목록을 검증한
다음 native reader를 엽니다. 미완료 디렉터리, 손상 파일, 다른 backend의 응답은
거부합니다. 쓰기 실패의 pending 디렉터리는 조사용으로 남기고 engine 종료는
checkpoint 데이터를 삭제하지 않습니다. reader가 사용하는 완료 snapshot은
변경하지 않아야 합니다. manifest는 우발적인 손상을 탐지하며 인증 서명은
아닙니다. 이 로컬 프로토콜은 원격 파일시스템이나 객체 저장소의 트랜잭션을
보장하지 않습니다.

`checkpoint_enabled=True`는 비어 있지 않은 `checkpoint_dir`를 요구하며 root
생성 시 선택한 repartition 뒤에 실제 checkpoint를 실행합니다. 파생 projection·
filter engine은 자동으로 다시 checkpoint하지 않습니다. Dask/Ray는 저장한
native 데이터를 다시 열므로 원래 source graph를 재실행할 필요가 없습니다.
Spark는 native reliable DataFrame checkpoint와 선택한 SparkContext 디렉터리에
위임합니다. Spark checkpoint는 해당 native context 수명주기의 기능이며
Dask/Ray의 이식 가능한 manifest 형식과 구분합니다. MCP `checkpoint`는 세 엔진의
native child를 보존하며 실제 경로는 설치된 factory 안에 둡니다. 워커 재시작 후
factory가 선택한 Dask/Ray snapshot을 명시적으로 다시 열 수 있습니다.
[Dask native Parquet 게시](https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.to_parquet.html)를 참고하세요.

`iter_arrow_batches(batch_size=None)`는 타입이 있는 native batch를 반환하며
iterator `close()`를 제공합니다. Dask는 partition을 하나씩 계산하고 Ray는
설정한 prefetch로 native batch를 순회하며 Spark는 partition prefetch 없이
local row iterator를 사용합니다. Dask의 빈 stream도 schema를 유지하고 Ray는
null-only batch를 dataset schema에 맞춥니다. `STREAMING`은 Dask/Spark의 Arrow
변환에서 이 경로를 선택하며 Ray는 모든 mode에서 사용합니다. 완전한 Arrow table이나
Polars frame을 반환하는 API는 결과 전체를 여전히 보유합니다. native partition
할당은 batch 행 수보다 클 수 있으며 이 모드는 무제한 broker나 Spark Structured
Streaming query를 시작하지 않습니다.
[Ray iterator 수명주기](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_batches.html)와
[Spark local iterator](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toLocalIterator.html)를 참고하세요.

승인된 MCP `arrow_stream_open`은 `batch_size`를 받고 불투명한 `stream_id`를
반환합니다. `arrow_stream_read`는 같은 ID와 `engine_id`를 요구하고 `limit`를
행 수에 적용해 `rows`와 `complete`를 반환합니다. 페이지가 마지막 행에서 정확히
끝나면 소진 확인을 위한 읽기가 한 번 더 필요할 수 있습니다. `arrow_stream_close`는
cursor를 해제하며 child engine 해제나 shutdown도 종료를 시도합니다. 동시에 최대
16개 cursor를 열 수 있습니다. 읽기는 native 순회 전후 binding을 확인하고 private
artifact 바이트 예산을 적용합니다. cursor는 owner-local이며 소비한 읽기의 실패나
응답 손실을 새 iterator로 자동 재실행하지 않습니다.

native 계산 진입점은 중첩 호출에서도 하나의 협력적 `timeout_seconds` deadline을
공유합니다. 늦게 완료된 결과는 `DistributedOperationTimeout`을 발생시키고 결과
cache를 비웁니다. Arrow cursor는 batch 전후 deadline을 확인하고 실패·조기 종료 시
닫힙니다. blocking SDK 호출은 반환할 때까지 계속될 수 있으므로 borrowed worker
강제 종료나 엄격한 wall-clock 상한을 보장하지 않습니다. `retry_count`는 계산을
반복해도 안전하다고 명시한 `RetryableExecutionError`가 전파될 때만 추가 시도를
허용합니다. 일반 예외·timeout·결과가 불명확한 부작용을 자동 재시도하지 않습니다.
native framework는 worker 예외를 감싸거나 자체 task retry 정책을 적용할 수
있습니다. checkpoint·쓰기·cluster 수명주기는 native 실패 계약을 유지합니다.
`memory_fraction`은 iterator 시작 시 관측한 driver 가용 메모리의 해당 비율을
각 Arrow batch의 수락 한도로 사용합니다. 큰 batch는 반환 전에 `MemoryError`로
거부합니다. 이 협력적 전송 검사는 native 할당 뒤에 이루어지며 메모리를 예약하거나
worker heap·native partition buffer·prefetch·전체 수집 결과를 제한하지 않습니다.
그 자원은 선택한 backend에서 설정해야 합니다. execution extra는 이 관측을 위한
`psutil`을 포함합니다.

### 선택된 resolver 문맥과 검증된 SQL 소스

`TenantRuntime.contexts`는 비공개 문맥 dict를 별칭에 연결합니다. worker는 진입 시와
각 해석 시 복사합니다. `resolver_catalog`는 별칭을 나열하고 `resolve(context_id)`는
native resolver를 호출해 허용된 tenant 별칭 또는 `resolved: false`를 반환합니다.
허용 목록 밖의 tenant 해석은 실패합니다.
`resolve_or_create(context_id, tenant_id, default_name)`는 한 번 해석한 ID를
조회·생성 **이전**에 선택된 native ID와 비교합니다. Core의
`TenantManager.resolve_or_create(..., expected_tenant_id=...)`를 사용합니다.
Header resolver의 `required=True`는 누락된 header를 거부하고, 끝에 개행이 붙은
ID 등 잘못된 형식도 거부합니다. header·token·resolver 코드는 wire 입력이 아닙니다.

`TenantRuntime.sources`는 factory가 소유한
`truthound.multitenancy.sources.TenantSQLSource`에 별칭을 연결합니다.
`source_catalog`는 별칭을 나열하고
`source_read(tenant_id, source_id, columns?, offset?, limit?)`는 민감 결과 동의와
정확한 승인 아래 선택된 소스의 제한된 페이지를 반환합니다. native wrapper는
`SQLiteDataSource`를 빌려 고정 native tenant ID에 연결하고, SQLite savepoint 안에서
실제로 조회하는 동일 연결을 조회 전후 검증합니다. DATABASE 격리는 실제 DB 파일과
tenant의 절대 `database_name`을 비교합니다. SCHEMA 격리는 native schema 이름과
wrapper에 명시한 절대 `namespace_database` 모두 일치해야 합니다. 운영자가 attach한
SQLite DB도 지원합니다. 연결 정리는 factory 책임이며 경로와 namespace는 비공개입니다.
정지된 tenant, 다른 tenant ID, query mode, view, virtual table, 메모리 DB 및 검증되지
않은 원격 provider는 거부합니다. 일반 테이블 조회가 소스의 `materialization_row_limit`을
초과하면 행을 반환하기 전에 실패합니다. 페이지 크기는 이 한도를 대체하지 않습니다.
이미 반환한 데이터의 지속적인 회수가 아닌 구체화된 조회 snapshot입니다. 호출자가 제공한
Arrow 자산은 여전히 schema/DB 소속을 증명하지 못합니다. 설치 worker의 SQLite 재시작
검증은 원격 DB provider 지원의 증거가 아닙니다.

Profiler 서비스는 고정된 `quality_scorers`·`quality_rules`를 `quality_score`,
`quality_compare`, `quality_cache_clear`로 제공합니다. 입력은 고정 자산이며 정렬된
Boolean `label_column`을 지정할 수 있습니다. sampling·설정·사용자 규칙 코드는
factory가 소유합니다. [품질 점수](profiler/quality-scoring.md)를 참조하세요.

고정 profiler 서비스는 native 표본 snapshot, 로컬 타입 추론 feedback·모델 영속화,
완료 profile의 예산 제한 HTML artifact·고정 슬롯 내보내기도 제공합니다.
설정과 제약은 [샘플링](profiler/sampling.md), [ML 추론](profiler/ml-inference.md),
[시각화](profiler/visualization.md)를 참조하세요.

플러그인 소유자는 별칭 없이 `plugin_services_execute(command="shutdown")`도 받습니다. 승인된 명령은 플러그인·reload 정리와 factory 종료가 끝난 뒤 성공을 확인합니다. 정리 실패는 실패·불확실 결과로 남고 새 teardown으로 재실행되지 않습니다.


동시 저장소 팩토리는 복사된 JSON·안전한 YAML, 등록 압축 코덱, 메타데이터 기본값과 기대값 저장소의 선택 잠금 전략을 적용합니다. 기대값 저장소는 `query.order_by="name"`을 지원하고 태그 필터·페이지 처리를 네이티브로 적용합니다. 잘못된 레코드는 성공 결과에서 누락시키지 않고 조회·인덱스 재구축 오류로 보존합니다. 인덱스 작업은 WAL 게시 전에 검증하며 레코드 파일과 인덱스 파일 사이의 원자성을 보장하지는 않습니다.

### 승인된 벤치마크 워크로드

비공개 `benchmark_services` 바인딩은
`truthound.mcp.benchmark_services.BenchmarkRuntime`을 반환합니다.
`workloads` 맵은 `BenchmarkWorkload(runner, kind, target, parameters)`를
담으며 `kind`는 `single`, `suite`, `all`, `category`, `parity`입니다.
운영자가 네이티브 runner, suite, 프레임워크, 매개변수와 산출물 목적지를
팩터리에 고정합니다. 요청은 별칭만 받으며 임의 import, 경로, runner
키워드 인자를 받지 않습니다. 선택적 `comparators`, `thresholds` 맵은
네이티브 비교기와 임계값을 담습니다.

`benchmark_services_open`, `benchmark_services_execute`에는 승인과
멱등성 영수증이 적용됩니다. Execute는 `binding_id`, `service_id`,
`idempotency_key`를 반복합니다. 명령은 `inspect`, `run` (`workload_id`),
`result`, `release`, `report` (`result_id`, 선택 `format`: json/markdown/html),
`compare` (`result_id`, `baseline_id`, `comparator_id`), `threshold`
(`result_id`, `threshold_id`), `shutdown`입니다. 비교 대상의 네이티브
결과 종류가 일치해야 합니다. Parity 보고서는 현재 JSON만 지원합니다.

결과는 네이티브 측정값과 판정을 보존합니다. 실패한 측정과 빈 suite는
성공 영수증 대신 결과 산출물을 보존하는 `native_benchmark_incomplete`를
반환합니다. 임계값 검사는 실패하거나 빈 측정을 거부합니다. 반환 보고서는
매개변수, 태그, traceback, 상세 parity 관측값과 운영자 산출물 경로를
제외합니다. 일반 벤치마크 환경 정보는 민감 정보이며 공개 승인이 필요합니다.
소유자는 직렬화된 네이티브 결과 예산 안에서 최대 16개 결과를 유지하고 `release`로
제거합니다. 취소 시 소유한 네이티브 스레드를 회수한 뒤 정상 정리합니다.
강제 프로세스 손실은 결과 불명 상태이며 자동 재실행하지 않습니다.

네이티브 `BenchmarkConfig`는 잘못된 반복 횟수와 비유한 예산을 거부합니다.
시간과 프로세스 RSS 제한은 setup 및 반복 전후에 검사하며 cooldown도
남은 시간 예산을 따릅니다. 실행 중인 단일 반복을 강제로 중단하거나 모든
순간 메모리 증가를 감지하는 제한은 아닙니다. 실패해도 teardown을 실행합니다.
현재 증거는 로컬 합성 워크로드 검증이며 새 릴리스 parity 벤치마크가 아닙니다.
전체 parity 제공자 검증은 아직 남아 있습니다.

### 승인된 API 문서 생성

비공개 `documentation_services` 바인딩은
`truthound.mcp.documentation_services.DocumentationRuntime(jobs={alias: job})`를
반환합니다. `DocumentationJob(generator, source, kind="source")`는 설치된
팩터리에 네이티브 `DocumentationGenerator`, renderer·설정과 소스를 고정합니다.
`kind="module"`은 네이티브 모듈 추출·렌더링을, `kind="info"`는 운영자가
제공한 `ModuleInfo` 또는 `PackageInfo`를 사용합니다. 요청에서 소스 경로,
모듈 import, renderer나 출력 경로를 지정할 수 없습니다.

`documentation_services_open`, `documentation_services_execute`는 승인과
멱등성 영수증이 필요합니다. Execute는 `binding_id`, `service_id`,
`idempotency_key`를 반복하며 `inspect`, `generate` (`job_id`), `result`
(`result_id`), `read` (`result_id`, `file_id`), `release` (`result_id`),
`shutdown`을 선택합니다. 생성 결과는 파일 ID·상대 이름·크기·SHA-256의
목록을 보존합니다. `read`는 생성된 바이트 그대로 산출물로 반환합니다.
네이티브 JSON·MkDocs·Sphinx·HTML 형식은 팩터리에서 선택합니다. 모듈 단독
출력은 내용을 바꾸지 않고 `docs/module.txt`로 보존합니다.

생성은 MkDocs의 상위 설정 파일까지 포함하는 소유 임시 디렉터리를 사용합니다.
네이티브 부분 실패는 이미 생성된 파일과 `native_documentation_incomplete`를
보존하며 원본 오류 메시지를 가립니다. 빈 출력도 미완료입니다. 심볼릭 링크와
일반 파일이 아닌 항목은 거부합니다. 보존량은 서비스 바이트 예산, 생성당
256개 파일, 최대 16개 생성 결과로 제한합니다. `release` 또는 소유자 종료가
결과를 제거하고, 취소 시 소유 네이티브 작업을 회수한 뒤 정리합니다. 메모리 내
산출물 수명주기이며 소유자 간 복구나 자동 사이트 게시를 제공하지 않습니다.

문서에는 소스 코드·docstring·소스 링크가 포함될 수 있습니다. 운영자는 공개할
소스를 선택해야 하며 승인은 목록뿐 아니라 그 내용까지 포함합니다. 바이트
예산은 네이티브 생성 후 보존 산출물에 적용하며 renderer의 디스크·CPU·메모리
강제 샌드박스는 아닙니다. 팩터리는 신뢰된 설치 코드입니다. 생성 성공만으로
사이트 빌드·설치·배포가 검증된 것은 아닙니다.

벤치마크 팩터리는 별칭을 네이티브 `RegressionDetector`에 연결하는
`BenchmarkRuntime.histories`를 제공할 수 있으며 이력 루트는 운영자가 고정합니다.
`history_load`, `history_inspect`는 `history_id`가 필요합니다.
`history_baseline`은 기준을 결과 핸들로 복원하거나 `available: false`를
반환합니다. `history_save`, `history_check`, `history_report`,
`history_thresholds`에는 `history_id`와 suite `result_id`가 필요합니다.
`history_threshold`는 이미 바인딩된 `threshold_id`를 추가합니다.
`threshold`, `history_check`, `history_report`는 선택적 보존 `baseline_id`를
받습니다. 기준 부재·벤치마크 누락·측정 실패가 있는 이력 비교는 미완료 영수증을
반환합니다. 이력 보고서는 네이티브 텍스트 산출물입니다. 저장은 팩터리의 비공개
루트에 대한 승인된 쓰기이며 새 소유자가 루트 경로를 노출하지 않고 복원할 수
있습니다. 기준·이력의 두 쓰기는 각각 원자적이며 여러 파일의 단일 트랜잭션이
아니므로 불확실한 쓰기를 무조건 재실행하면 안 됩니다. 손상되거나 원본 측정이
없는 과거 요약 증거는 부재·복원 성공으로 처리하지 않고 거부합니다.

고정 벤치마크 워크로드는 `kind="generator"`, `"generator_lazy"`도 받습니다.
`runner`는 네이티브 `DataGenerator`, `target`은 캡처된 `GeneratorConfig`이며
`parameters`는 비어 있어야 합니다. `run`은 프레임을 보존하고 스키마·행 수
메타데이터를 반환합니다. `data_page`는 `result_id`, 음수가 아닌 `offset`,
0부터 1000까지의 `limit`을 받습니다. `report(format="ipc")`는 네이티브 Arrow
IPC 바이트를 그대로 내보냅니다. JSON 보고서는 프레임 메타데이터를, Markdown·
HTML은 벤치마크 결과를 처리합니다. 생성 프레임은 소유자의 결과 16개·메모리
추정 예산을 공유하며 `release`·종료로 정리합니다. 이 바이트 계산은 네이티브
생성 중의 일시적 메모리를 제한하지 않습니다. 시드 출력·IPC는 로컬 합성
데이터로 검증하며 이 검증은 실제 PII나 별도 기준 코호트를 사용하지 않습니다.

`kind="scaffold"`는 네이티브 `BaseScaffold`와 `ScaffoldConfig`를 고정하고
소유한 출력 루트에 파일을 생성합니다. 쓰기 전에 모든 경로를 검사하여 절대
경로·상위 경로 이탈·중복·심볼릭 링크를 거부합니다. 개별 파일 쓰기는 원자적이며
전체 파일 묶음은 단일 트랜잭션이 아닙니다. 매니페스트에는 실행 가능 여부도
포함됩니다. 사용자 로직을 구현해야 하며 미구현 데이터소스·액션은 오류를 발생시킵니다.

`kind="validate"`는 `DocumentationGenerator`와 소스를 고정하여 네이티브
`validate_docstrings`를 실행합니다. `strict=True`는 이 종류에서만 지원합니다.
진단은 `docs/docstrings.json`에 보존하며 검증 실패 시에도 산출물과
`native_documentation_incomplete`를 반환합니다. 파일에는 선택한 소스·docstring
내용이 포함될 수 있으므로 문서 생성과 동일한 공개 승인이 필요합니다.

벤치마크 `compare`와 이력 비교는 suite 항목이 누락되거나 중복되면 `native_benchmark_incomplete`를 반환합니다. 공통 항목만 비교하여 성공으로 처리하지 않습니다.

스캐폴딩 작업은 잘못된 설정의 네이티브 생성 실패를 보존합니다. 보존한 reporter 파일은 표준 Core 결과로 실행되며 통과 검사 변환을 Core와 공유합니다. 이슈 제한 0은 요약을 유지하면서 상세 행을 출력하지 않습니다.

계보 `lineage_render`의 `output_format="native"`는 기존 JSON·텍스트와 루트 기반
부분 그래프 출력을 유지합니다. 전체 그래프는 D3·Cytoscape·Mermaid의
`output_format="html"`, Mermaid의 `"markdown"`과 선택적 `title`, Graphviz의
`"svg"`·`"png"`를 지원합니다. 이미지에는 `.[mcp,lineage-visualization]`과 시스템
`dot` 실행 파일이 필요합니다. 지원하지 않는 렌더러·형식 조합 및 루트가 지정된
HTML·이미지 요청은 거부합니다. 문서·이미지의 너비와 높이는 1–4096이며 기존
산출물 바이트 예산을 적용합니다.

호출자가 출력 경로를 지정하거나 브라우저를 자동 실행하지 않습니다. 이미지는
로컬 Graphviz를 실행하며 서버는 원격 자원을 가져오지 않습니다. HTML에는
대화형 클라이언트 코드와 공개 라이브러리 URL이 있어 운영자가 브라우저로 열면
해당 라이브러리를 가져올 수 있습니다. 민감 결과 승인이 필요하며 원본 그래프를
해제하면 산출물 읽기도 철회됩니다. 노드 이름은 데이터로 처리하고 서로 다른
노드 ID를 합치지 않습니다. D3 고정 레이아웃과 Cytoscape 노드 크기를 적용하며,
부분 그래프 깊이는 명시적 인자가 없을 때 네이티브 설정을 따릅니다. 전체 그래프
문서는 부분 그래프를 선택하지 않습니다. 트래커 문맥·이력과 OpenLineage 전송은 아래의 별도 승인 소유자를 사용합니다.


### 승인된 네이티브 트래커 세션과 문맥 이력

비공개 `lineage_services` 팩터리는 `truthound.mcp.lineage_services`의
`LineageRuntime(tracker, jobs={별칭: LineageJob(callback, name, operation_type)},
files={별칭: 운영자_절대경로})`를 반환합니다. 팩터리는 네이티브 `LineageTracker`,
설정과 파일 별칭을 소유합니다. 콜백은 고정된 동기 `(tracker, context) -> None`
함수이며 요청은 코드·import·파일 경로를 전달하지 않습니다. 팩터리가 트래커를
넘길 때 활성 문맥이 없어야 합니다.

`lineage_services_open`과 `lineage_services_execute`는 운영자 승인, 제어 디렉터리,
민감 결과 동의가 필요합니다. 실행 요청은 `binding_id`, `service_id`,
`idempotency_key`와 다음 `command`를 사용합니다.

- `inspect`는 비공개 경로 없이 상태 개수와 작업·파일 별칭을 반환합니다.
- `track`은 `graph_id` 없이 기존 `lineage_track`과 같은 형식의 `event`를 받습니다.
- `run(job_id)`는 소유한 한 스레드에서 네이티브 `tracker.track(...)` 안에 콜백을
  실행합니다. 현재 트래커·중첩 문맥·자동 추적의 네이티브 동작을 적용합니다.
- `history(offset, limit)`는 완료 상태와 오류를 감춘 실패 이력을 조회합니다.
  제한 0은 빈 행을 반환하며 `snapshot`은 그래프를 반환합니다.
- `lineage(node_id, direction)`와 `path(source_id, target_id)`는 네이티브 질의입니다.
- `save(file_id)`·`load(file_id)`는 고정된 비공개 그래프 파일을 사용하고 `clear`는
  그래프와 문맥 이력을 지웁니다. 그래프 파일은 문맥 이력을 저장하지 않습니다.
- `export`는 네이티브 그래프 JSON이며 `render(rendering={...})`는 위의 닫힌
  렌더러·형식 설정을 재사용합니다. 렌더링 실패는 이미지로 잘못 표시하지 않고
  실패한 JSON 산출물로 보존하며 트래커를 유지합니다.
- `shutdown`은 소유자와 팩터리 정리가 끝난 뒤에만 완료를 응답합니다.

응답은 불변 세션 산출물이며 동일 승인 키 재호출은 콜백·파일 쓰기를 반복하지
않습니다. 완료 이력은 일반 구조화 메타데이터의 스냅샷이고, 불투명한 네이티브
객체는 전송하지 못할 수 있습니다. 콜백 실패는 부분 그래프 변경과 실패 문맥을
보존하며 롤백을 의미하지 않습니다. 네이티브 저장은 버전을 확인하고 게시한
바이트를 검증합니다. 명시적 주 저장은 다른 트래커의 최신 상태를 덮어쓰지
못합니다. 권위 있는 주 파일을 다시 읽어 버전을 확인하면 불명확한 게시를
해소할 수 있지만 다른 내보내기 파일을 읽는 것으로 주 파일의 불확실성을 해소하지
않습니다. 새 소유자는 설정한 그래프를 복원하되 네이티브 문맥 이력은 비어 있습니다.
취소는 소유한 스레드가 끝날 때까지 기다립니다. 강제 CPU 격리나 영구 콜백·이력
저널은 아니며 강제 프로세스 손실은 결과 불명으로 남고 작업을 자동 재실행하지
않습니다.

표현식 검증은 보고된 컬럼 자체의 실패 샘플·빈도·전체 행을 유지합니다. 앞 컬럼이
통과하거나 `mostly`로 생략돼도 뒤 실패의 증거가 다른 컬럼으로 이동하지 않습니다.
단독·배치 Core 실행은 같은 연결 규칙을 사용합니다.

### Core 3.3.0 설치 후보

공개 `truthound` 0.1.4 플러그인은 Core 3.2.0을 유지합니다. 별도 `truthound-core` 0.2.0 후보는 Core 3.3.0을 고정하며 일반 공개 인덱스 설치에는 해당 버전의 실제 게시가 필요합니다. 전이 의존성은 설치 시 허용 범위에서 해석하며 미게시 후보의 공개 인덱스 고정 lock은 포함하지 않습니다. 후보 CI는 소스 체크아웃 대신 새 wheel을 설치합니다.

```bash
codex plugin add truthound-core@truthound --json
truthound-mcp bundle --profile core --output truthound-3.3.0-core.mcpb
truthound-mcp bundle --profile core --wheel dist/truthound-3.3.0-py3-none-any.whl --output truthound-3.3.0-core-local.mcpb
```

설치만으로 데이터 접근을 허용하지 않습니다. 설치된 UV 프로젝트에서 `python -m truthound.mcp.consent configure --allow-dir <승인한-디렉터리> --confirm`을 실행합니다. `--workspace-dir`, `--control-dir`, `--allow-sensitive-results`는 각각 별도 운영자 승인 항목입니다. 전체 설정은 `~/.truthound-core/permissions.json`의 이전 권한을 대체하며 `python -m truthound.mcp.consent revoke`로 모두 철회합니다. 변경 시 세션을 종료하고 보유 핸들을 무효화하므로 재연결해야 합니다. 개별 실행 계획 승인은 여전히 MCP 외부의 신뢰된 운영자 명령으로 수행합니다.

Core MCPB는 현재 카탈로그와 데이터·작업공간·승인 저장소·민감 결과 설정을 제공합니다. 선택 디렉터리를 비우면 접근을 허용하지 않으며 민감 결과 공개 기본값은 false입니다. 데스크톱 설정 변경 후 재시작합니다. 기본 번들은 요약 도구 3개를 유지합니다. Core 지원은 부분적이며 공식 디렉터리 승인은 없습니다.
