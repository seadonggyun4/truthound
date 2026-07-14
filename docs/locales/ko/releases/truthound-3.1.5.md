# Truthound 3.1.5 릴리스 노트

## 핵심 변경

Truthound 3.1.5는 Truthound 자체와 Truthound Depot 같은 소비자 제품이
공통으로 사용하는 SQL DataSource 계약을 복구합니다.

이 릴리스는 Core 결함과 소비자 adapter 결함을 분리합니다. Core 릴리스가
성공했다는 사실만으로 Depot connector가 인증되는 것은 아닙니다. Depot은
공개 배포된 artifact를 설치한 뒤 source, worker, persistence, UI matrix를
다시 검증해야 합니다.

## SQL DataSource 수정

- `row_count`, scalar query, 결과 변환이 tuple row, PyMySQL `DictCursor` 같은
  mapping row, `_mapping`을 제공하는 SQLAlchemy형 row를 모두 처리합니다.
- DB-API provider는 column name 정규화와 cursor cleanup을 공통 계약으로
  사용합니다.
- Polars fallback은 무제한 `fetchall()` 대신 provider별 제한 query와
  `fetchmany()` batch를 사용합니다.
- `materialization_row_limit + 1`행까지 확인하여 동시 데이터 증가가 조용한
  잘림으로 처리되지 않게 합니다.
- BigQuery, Snowflake, Redshift, Databricks, Oracle, SQL Server의 fallback은
  공통 bounded implementation을 사용합니다.

## 설치 extra

```bash
pip install truthound[postgresql]
pip install truthound[mysql]
pip install truthound[duckdb]
pip install truthound[snowflake]
pip install truthound[bigquery]
pip install truthound[redshift]
pip install truthound[databricks]
pip install truthound[oracle]
pip install truthound[sqlserver]
pip install truthound[sql-connectors]
```

## 공개 API 계약

SQL DataSource는 반드시 `source` keyword로 전달합니다.

```python
import truthound as th

validation = th.check(source=source)
profile = th.profile(source=source)
```

`materialization_row_limit`보다 큰 데이터는 pushdown 검증을 사용하거나
`source.sample(10_000)`처럼 명시적으로 제한해야 합니다. Truthound는 불완전한
데이터를 성공으로 반환하지 않고 `DataSourceSizeError`를 발생시킵니다.

## 검증 상태

| 증거 | 현재 상태 |
|------|----------|
| tuple, mapping, `_mapping` row 계약 | contract test 통과 |
| batch read와 조용한 잘림 금지 | contract test 통과 |
| SQLite 공개 `check`, `profile` 경로 | 3.1.5 wheel integration 통과 |
| DuckDB 공개 source 경로 | 3.1.5 wheel integration 통과 |
| PostgreSQL, MySQL 로컬 서비스 | 3.1.5 wheel integration 통과 |
| Snowflake, BigQuery, Redshift, Databricks, Oracle, SQL Server 실제 계정 | credential 기반 검증 전까지 미검증 |

credential 부재, skip, import 성공은 통과가 아닙니다.

## 알려진 제한

3.1.5 artifact의 Snowflake, BigQuery, Redshift, Databricks, Oracle, SQL Server는
provider-native schema 구현이 base class의 query 전용 abstract hook을 충족하지
못해 여전히 abstract class로 남습니다. 해당 provider를 사용하려면 3.1.6 이상으로
업그레이드해야 합니다. constructor 수정이 실제 credential 기반 지원 인증을
의미하지는 않습니다.

## 소비자 업그레이드 Gate

소비자 저장소는 Core 결함을 위한 로컬 우회 코드를 영구 유지하면 안 됩니다.

1. PyPI에서 공개 배포된 3.1.5 wheel을 설치합니다.
2. 설치 버전과 artifact hash를 확인합니다.
3. 대체된 workaround를 제거합니다.
4. 공개 API, worker, persistence, 재진입, provider QA를 다시 실행합니다.
5. 소비자 고유 결함은 Core 결함과 별도로 기록합니다.
