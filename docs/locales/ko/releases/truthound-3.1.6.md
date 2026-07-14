# Truthound 3.1.6 릴리스 노트

## 핵심 변경

Truthound 3.1.6은 3.1.5 소비자 통합 gate에서 발견된 SQL provider 생성 계약
결함을 복구합니다. Snowflake, BigQuery, Redshift, Databricks, Oracle, SQL Server는
provider-native schema 조회를 구현했지만 `BaseSQLDataSource`가 사용하지 않는
schema-query abstract hook을 강제하여 class를 생성할 수 없었습니다.

## Schema 조회 전략

SQL provider는 다음 두 전략 중 하나를 충족합니다.

- query 전략: `_get_table_schema_query()`를 구현하고 공통 DB-API schema runner 사용
- native 전략: provider metadata API 또는 dialect별 operation을 위해
  `_fetch_schema()` 구현

생성 시 두 전략 중 하나가 존재하는지 검증합니다. 공통 query runner는 tuple,
mapping, `_mapping` schema row를 정규화하고 성공과 실패 모두 cursor를 닫습니다.

## Provider class 계약

| Provider | Schema 전략 |
|----------|-------------|
| PostgreSQL | 공통 schema query |
| MySQL | 공통 schema query |
| SQLite | provider query/pragma 경로 |
| DuckDB | 공통 schema query |
| Snowflake | native information schema |
| BigQuery | native client metadata |
| Redshift | native information schema |
| Databricks | native `DESCRIBE` |
| Oracle | native catalog query |
| SQL Server | native information schema |

release workflow는 wheel에 `truthound[sql-connectors]`를 설치하고 모든 driver와
공개 provider class를 import하며, abstract class가 하나라도 있으면 실패합니다.

## 증거 경계

concrete class와 constructor/config test는 non-network package 증거입니다. 실제
provider 계정 인증을 대신하지 않습니다. Snowflake, BigQuery, Redshift,
Databricks, Oracle, SQL Server는 credential 기반 schema, row count, bounded read,
validation, profile, 재진입, cleanup이 끝날 때까지 실제 provider 수준에서
미검증입니다.

## 소비자 업그레이드 Gate

Truthound Depot 같은 소비자는 공개 배포된 3.1.6 artifact를 설치하고 runtime
version과 hash를 확인한 뒤 provider lifecycle을 처음부터 재실행해야 합니다.
local checkout이나 unpublished wheel은 소비자 인증 증거가 아닙니다.
