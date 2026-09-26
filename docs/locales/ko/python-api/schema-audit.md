# 스키마 감사 계약

`truthound.schema_audit`는 별도로 임포트하는 개발 중 공통 모델과 감사 A/B 엔진입니다.
제한된 PostgreSQL14/17 카탈로그 수집기를 포함하지만 **ERD 편집·가져오기, 접속 인가·승인·Release는 구현하지 않습니다.**
기존 `th.check()`와 table-schema 검증 의미는 그대로입니다. 추가 extra는 필요 없으며
이 네임스페이스 자체는 표준 라이브러리만 사용합니다. 루트 패키지의 기존 의존성은 필요합니다.
미출시 소스 기능을 PyPI에 이미 배포된 기능으로 해석하지 마세요.

## 설계 → 카탈로그 감사 B

```python
from truthound.schema_audit import audit_implementation
from truthound.schema_audit.postgresql import collect_postgresql

# 호출자가 목적지 인가·TLS·인증을 완료한 전용 asyncpg.Connection입니다.
snapshot = await collect_postgresql(
    connection, schemas=("application",), binding_ref="approved-binding",
    snapshot_id="snapshot-1", timeout_ms=3000, attempts=2,
)
report = audit_implementation(design, snapshot)
result = report.as_dict()
```

호출자는 `asyncpg`를 설치하고 연결을 종료합니다. 수집기는 자격 증명을 해석하거나 SQL/URL을
받지 않습니다. 이미 트랜잭션이 열린 연결은 거부합니다. 읽기 전용 트랜잭션, repeatable-read
카탈로그 스냅샷, 조회 전 ACCESS SHARE 잠금, 명시한 스키마 범위를 사용합니다. 사용자 행은
조회하지 않고 FK를 따라 범위를 자동 확장하지 않습니다. 스키마 USAGE와 테이블 SELECT 권한이
필요하며 SELECT는 행 수집이 아닌 잠금 권한입니다. 시스템 스키마는 거부합니다. 접근 불가·부재
스키마와 숨겨진 테이블은 수집 완전성을 낮추며 거짓 누락 위반을 만들지 않습니다.

상한은 스키마 16개, 테이블 유사 객체 500개, 테이블·컬럼·제약조건 합계 10,000개, 메타데이터 값
4,096자입니다. `timeout_ms`는 100..10,000, `attempts`는 1..3입니다. 시도당 전체 시간은
statement timeout의 4배로 제한합니다. 카탈로그 변경·시간 초과만 제한적으로 재시도합니다.
`CollectionError.code`에 driver 오류 원문·접속 정보·SQL 값은 포함하지 않습니다. 취소를 전파하고
트랜잭션을 rollback합니다. 수집 이후 발생할 DDL까지 차단하거나 보증하지는 않습니다.

PG14/17 일반 테이블을 지원합니다. 15/16/18은 자동 허용하지 않으며 설계와 스냅샷의 major가
같아야 합니다. PG14는 PG15 이후 추가된 제약조건 컬럼을 참조하지 않는 별도 쿼리를 사용합니다.
PG14에서 음수 numeric scale 및 precision보다 큰 scale은 미지원입니다.
구현 감사 규칙팩은 v2이며 기존 결과를 새 규칙 결과로 재표시하지 않습니다.
파티션·상속·view·foreign table, 미검증 제약조건, initially
deferred, 기본값과 다른 FK match, 일부 컬럼만 SET NULL/DEFAULT, NULLS NOT DISTINCT UNIQUE는
완전성을 낮춥니다. 범위 밖 FK는 수집하지 않고 standalone unique index도 제약조건 coverage를
낮춥니다. 사용자 타입·도메인·enum, 배열 차원, timestamp precision 0은 v1 모델로 정확히 표현할
수 없는 부분을 UNKNOWN으로 보존합니다. 문자 길이는 바이트가 아니라 문자 수입니다.

방언별 정확한 namespace·테이블·컬럼·제약조건 이름으로 대응하며 키/참조 컬럼 순서, FK 동작,
타입 facet, nullable과 ordinal을 비교합니다. rename은 추정하지 않습니다. 논리명은 COMMENT
전체 값의 NFC를 비교하며 comment 부재나 외부 mapping 주장은 UNKNOWN입니다. 기본값·CHECK는
실행하지 않습니다. 양끝 공백만 제거한 동일 문자열은 일치, 다른 단순 리터럴은 차이,
그 밖의 표현식 차이는 UNKNOWN입니다. SQL 의미 동등성을 추정하지 않습니다. 기본값 부재와
존재는 차이입니다. 미지원 식별자 때문에 신뢰할 수 없는 누락·추가를 확정하지 않습니다.

결과는 입력 semantic digest, 규칙 버전·digest, 양쪽 출처와 fingerprint를 고정합니다.
`ImplementationReport.as_dict()`의 출력은 `truthound.implementation-audit/1`이며
`load_document()` 입력 계약은 아닙니다. FAIL이 있으면 NONCONFORMANT지만 UNKNOWN 집계도
보존합니다. 불완전 수집이나 빈 대상은 CONFORMANT가 될 수 없습니다. DB 구조는 수정하지 않습니다.

조사 근거: [PG17 catalog](https://www.postgresql.org/docs/17/catalog-pg-constraint.html)의
키 순서·FK 동작·검증 플래그, [가시성](https://www.postgresql.org/docs/17/infoschema-columns.html)의
권한 필터링, [트랜잭션 일관성](https://www.postgresql.org/docs/17/applevel-consistency.html)을
수집 경계에 적용했습니다. [Rahm·Bernstein 논문](https://www.microsoft.com/en-us/research/publication/on-matching-schemas-automatically/)
공개 초록을 근거로 추정 대응과 exact 비교를 분리했고,
[DDIA 저자 소개](https://dataintensive.net/)의 트랜잭션·재현성 관점을 검토했습니다.
유료 저서 전문 열람이나 표준 인증을 주장하지 않습니다.

## 읽기와 식별

```python
from truthound.schema_audit import (
    load_document, dump_document, raw_digest, semantic_digest, json_schema,
)

payload = b'{"contract_version":"truthound.schema-audit/1","kind":"DESIGN","id":"design1","revision":1,"dialect":{"name":"postgresql","major":17},"tables":[],"sources":[]}'
model = load_document(payload)
assert load_document(dump_document(model)) == model
original_sha256 = raw_digest(payload)
content_sha256 = semantic_digest(model)
structural_schema = json_schema()
```

빈 모델을 표현할 수 있지만 로딩 성공은 감사 PASS가 아닙니다.
`load_document(bytes) -> StandardRevision | DesignRevision | CatalogSnapshot`은
UTF-8 JSON만 받습니다. 모든 필드는 필수이며 미등록 필드·중복 키·지원하지 않는 버전·중복 ID·
깨진 그래프 참조를 거부합니다. 직접 생성자에서도 타입 강제 변환을 하지 않습니다.
frozen dataclass와 tuple을 사용하므로 입력 dict/list를 가변 상태로 보관하지 않습니다.

`ModelError.code`는 안정 오류 코드, `pointer`는 위치입니다. 알려진 필드는 JSON pointer,
중복 키·초기 자원 검사는 숫자 위치 또는 컨테이너 위치를 사용합니다. 미등록 키 이름·입력값을
오류에 포함하지 않습니다. 문법 오류는 루트 위치이며 parser 원문을 노출하지 않습니다.

상한은 입력·출력 20 MiB, 중첩 32, 문자열 4,096 Unicode 코드포인트, 배열당 50,000항목,
1,000테이블, 전체 50,000컬럼, 디코딩 1,000,000노드입니다. 정수는 ±(2^53−1)이며 bool과
구분합니다. NaN·Infinity·실수·NUL·단독 surrogate는 거부합니다. 이는 안전 경계이며
운영 처리량 보증이 아닙니다.

패키지의 `schemas/document-v1.json`은 `json_schema()`와 동일한 JSON Schema 2020-12입니다.
구조·기본 타입을 검증하며, 참조 무결성·자원 한도·관측 상태·제약 불변식은 반드시
`load_document`로 추가 검증해야 합니다. JSON Schema 검사만으로 전체 검증되지 않습니다.

## 모델

| 모델 | 책임 |
| --- | --- |
| `StandardRevision` | 단어·순서 있는 용어 구성·도메인·명명 정책·revision·출처 |
| `DesignRevision` | 방언·이름·테이블·컬럼·제약·revision·출처 |
| `CatalogSnapshot` | 객체 그래프와 opaque binding·adapter 버전·UTC 수집 시각·범위·속성별 coverage |
| `TypeDescriptor` | native token·type schema·길이/단위·precision/scale·timezone·배열 차원·도메인/enum 참조·collation |
| `Attribute` | VALUE/ABSENT/UNKNOWN/NOT_APPLICABLE·타입이 있는 값·고정 사유 코드 |
| `Identifier` | 디코딩된 원본 이름과 UNQUOTED/QUOTED/CATALOG 출처 |
| `LogicalName` | 논리명과 DESIGN/COMMENT/EXTERNAL_MAPPING/NONE 출처 |
| `SourceLocation` | 객체 ID·입력 SHA-256·JSON pointer. 파일시스템 경로가 아님 |
| `Finding`, `FindingSet` | 결과 ID·rule 버전·판정/사유·결정론적 정렬. 평가 엔진은 없음 |

`known(value)`, `absent()`, `unknown(reason=Reason.NOT_COLLECTED)`로 관측값을 구성합니다.
JSON null은 VALUE 상태에서 허용하지 않습니다. 실제 `DEFAULT NULL`은 관측된 SQL 표현식
문자열로 기록합니다. 없음과 권한 부족은 다릅니다. Catalog coverage는 TABLES, COLUMNS,
CONSTRAINTS, LOGICAL_NAMES, TYPES, DEFAULTS를 모두 명시합니다. 불완전 수집은 참조가
해결되지 않는 관계를 제외하고 partial coverage를 기록해야 하며, 완전한 그래프로 주장하지 않습니다.
타입의 도메인·enum 참조는 adapter의 opaque 참조이며 자동 해석되지 않습니다.
승인 증거와 tenant 인가는 소비하는 애플리케이션의 책임입니다.

## 정규화와 Digest

`normalize_identifier(identifier, dialect)`와 `normalize_type(type, dialect)`는 동등성
판정이 아닌 `Attribute`를 반환하며 미지원은 UNKNOWN입니다. 기본 지원은 PostgreSQL 14와 17입니다.
불변 `NormalizationRegistry`로 명시적 로컬 확장은 가능하지만 업로드 코드 실행·자동 fallback은
없습니다. 주입한 정규화기는 기본 semantic digest 정책을 변경하지 않습니다.

- quoted/catalog 이름은 그대로 보존하며 unquoted ASCII만 소문자화합니다. 비ASCII unquoted와
  기본 UTF-8 63바이트를 넘는 이름은 UNKNOWN으로 남기고 잘라내지 않습니다.
- 내장 타입은 `type_schema=known("pg_catalog")`이고 도메인·enum override가 없을 때만
  int4/integer, decimal/numeric 등의 명시 alias를 정규화합니다. `numeric(10,2)` SQL 선언,
  qualified token, serial pseudo-type, 사용자 타입을 추측해 파싱하지 않습니다.
- 모든 facet을 보존합니다. 길이 단위·배열·음수 scale·precision보다 큰 scale·timezone·collation을
  지우지 않습니다. 시간 타입과 timezone 관측값의 모순은 UNKNOWN입니다. SQL을 실행하지 않습니다.
- 논리명만 Python 공통 Unicode 3.2 데이터베이스의 NFC로 정규화합니다. 이후 추가 문자는
  보존하며 완전 정규화를 주장하지 않습니다. 물리명·약어·SQL 표현식에는 NFC를 적용하지 않습니다.

`raw_digest(bytes)`는 원본 바이트 SHA-256입니다. `semantic_bytes(model)`은
`truthound.schema-audit.semantic/1`과 `pg17-ascii-nfc32-logical-v1` 정책을 고정하며,
`semantic_digest(model)`은 그 결과의 SHA-256입니다. **RFC 8785/JCS 구현은 아닙니다.**
객체 키, ID 기반 컬렉션, coverage/scope는 정렬하지만 복합키·FK 대응 컬럼·용어 구성 순서와
컬럼 ordinal은 보존합니다. 루트 ID/revision·출처·수집 시각·adapter 버전·binding은 제외하고
객체 ID·방언·coverage는 유지합니다. 호출자는 envelope 식별·인가를 별도로 결합해야 합니다.
사람이 같은 설계로 보더라도 객체 ID가 다르면 digest가 같을 필요는 없습니다.

## 조사 근거

2026-09-26 확인. 아래는 설계 참고이며 인증·정확성 보증이 아닙니다. 유료 전문을 읽었다고
주장하지 않습니다.

| 참고자료 / 열람 범위 | 반영 |
| --- | --- |
| [ISO/IEC 11179-3:2023](https://www.iso.org/standard/78915.html), 공식 초록 | 식별·명칭·매핑 분리 |
| [PostgreSQL 17 식별자](https://www.postgresql.org/docs/17/sql-syntax-lexical.html), 식별자 절 | quoted/catalog 원형 보존 |
| [PostgreSQL 17 numeric](https://www.postgresql.org/docs/17/datatype-numeric.html), numeric 절 | 음수 scale·precision 독립 보존 |
| [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html), canonicalization 절 | 원본 바이트와 의미 정규화 구분 |
| [JSON Schema 2020-12](https://json-schema.org/draft/2020-12/json-schema-core), 개요 | 구조 계약과 의미 검증 분리 |
| [Unicode normalization](https://www.unicode.org/reports/tr15/), 정규화·안정성 절 | 명시적 버전 고정 |
| [Cupid 논문](https://www.microsoft.com/en-us/research/publication/generic-schema-matching-with-cupid/), 연구 초록 | 유사도와 확정 일치 구분 |
| [Barr 등의 Oracle Problem 논문](https://discovery.ucl.ac.uk/id/eprint/1471263/), 초록 | 독립 바이트 오라클·변형 관계 테스트 |
| [Specification by Example](https://www.manning.com/books/specification-by-example), 출판사 개요 | 정상·오류 구체 예제 |
| [Database System Concepts 7판](https://db-book.com/online-chapters-dir/index.html), 저자 장별 색인 | 구조 모델과 후속 의미 감사 구분; 전문 열람 아님 |

## 표준 → 설계 감사 A

```python
from truthound.schema_audit import audit_standards, load_document

def audit_bytes(standard_json: bytes, design_json: bytes) -> dict:
    return audit_standards(
        load_document(standard_json), load_document(design_json)
    ).as_dict()
```

`audit_standards(StandardRevision, DesignRevision, *, bindings=())`는 불변
`StandardsReport`를 반환합니다. `standards-v1`은 ASCII 물리명 문법·대소문자·바이트 길이,
NFC 논리명의 정확한 사전 대응, 단어 순서에 따른 약어 조합, 폐기 단어·용어,
도메인 자료형과 모든 명시 facet, 물리명 중복을 검사합니다. 테이블 논리명은 정확한
단어 또는 용어로, 컬럼은 용어로 대응합니다. 부분 문자열·번역·형태소·유사도는 통과 근거가
아닙니다. 접두·접미는 사전에 명시된 단어로 표현하며 임의 regex 정책은 지원하지 않습니다.

도메인 지정은 대응된 용어의 `domain_id`를 따릅니다. `TypeDescriptor.domain_ref`는 DB의
사용자 정의 타입 참조이며 표준 도메인 식별자가 아닙니다. 논리명이 명시적으로 없으면 FAIL,
관측되지 않았거나 모호하면 UNKNOWN입니다. 용어 대응이 불가능하면 도메인도 UNKNOWN입니다.
깨진 도메인·FK 참조는 공통 모델 단계에서 거부합니다. 미지원 방언·사용자 타입은 UNKNOWN으로
남기되, 확인 가능한 facet 차이는 독립 FAIL로 유지합니다. 문자 길이와 바이트 길이는 같다고
간주하지 않습니다. SQL 실행과 행 값 검사는 하지 않습니다.

각 검사에는 규칙·버전·대상·기대값·관측값·양쪽 입력 출처·표준 식별자·고정 fingerprint가
있습니다. 물리명 제안은 입력을 수정하거나 위반을 없애지 않습니다. 신뢰된 호출자만 승인 근거와
양쪽 입력 의미 digest에 고정된 `TermBinding`을 제공할 수 있습니다. 이 값 자체는 승인을
인증하지 않습니다. HTTP 서버가 승인 저장소를 통해 별도 확인해야 하며 Depot v1은 요청자의
binding 입력을 허용하지 않습니다.

결과는 입력 ID·revision·의미 digest, mapping digest, rule-pack digest를 고정합니다.
규칙 정책 변경 시 규칙 버전을 갱신해야 합니다. FAIL이 하나라도 있으면 NONCONFORMANT,
FAIL 없이 필수 UNKNOWN이 있거나 대상이 없으면 INDETERMINATE, 나머지는 CONFORMANT입니다.
규칙별 coverage에서 FAIL과 UNKNOWN을 함께 보존합니다. worker 성공은 감사 통과가 아닙니다.
결과 JSON 계약은 `truthound.standards-audit/1`이며 입력용 `load_document`와 `json_schema()`의
대상이 아닙니다.

실행 한도는 테이블+컬럼 10,000개, 객체당 출처 32개입니다. 결과는 최대 400,000개 검사이며
입력 배열의 50,000개 제한은 완화하지 않습니다. 이는 Core 계산 상한이며 Depot 전체 처리 용량을
보장하지 않습니다. Depot은 입력 20 MiB, 결과 원문 80 MiB, 저장 envelope 16 MiB의 별도
상한을 적용하고 초과 결과를 거절합니다. 결과 envelope는 상한이 있는 무손실 압축을 사용할 수 있습니다.
약어 조합이 4,096 code point를
초과하면 큰 문자열을 만들지 않고 UNKNOWN으로 판정합니다. 반복 용어 해석은 실행 내에서
캐시합니다. 이는 자원 보호 상한이며 처리량 보장이 아닙니다. 승인·Release 권한과 DB 수집·감사 B는
이 엔진의 범위가 아닙니다.

### 감사 A 조사 근거

- [ISO 11179-31 공개 초록](https://www.iso.org/standard/78925.html): 용어·도메인의 책임 구분.
- [SHACL 결과 보고서](https://www.w3.org/TR/shacl/#validation-report): 대상·경로·규칙별 근거. RDF/SHACL 호환성이나 판정 의미를 그대로 구현한 것은 아닙니다.
- [Rahm·Bernstein, On Matching Schemas Automatically](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-17.pdf): 대응 후보와 신뢰된 대응의 분리.
- [Helland, Immutability Changes Everything](https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper16.pdf): 고정된 입력과 재실행.
- [Kleppmann DDIA 저자 소개](https://dataintensive.net/): 아키텍처 선택의 trade-off. 학술저서 전문 완독이나 인증 근거로 사용하지 않습니다.

`tests/schema_audit/test_standards.py`의 정확한 기대값 및 변형 관계 테스트로 검증합니다.

## 공통 계약 검증 범위

`tests/schema_audit/test_contract.py`는 잘못된 입력, 불변성, unknown, 정규화, digest 고정
벡터를 검증합니다. wheel/sdist에서 패키지 리소스를 확인하고 기존 검증·공개 API 회귀는 별도로
실행합니다. 실제 eXERD 가져오기, DB 감사, Console 전체 E2E는 후속 PHASE 대상입니다.
