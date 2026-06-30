# CLI Data 소스 Guide

실무 운영 가이드에서 Truthound, CLI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Truthound, CLI을(를) 다루는 항목입니다:

1. 실무 운영 가이드에서 File, Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 `--connection`, `--table`, `--query`, Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 JSON, YAML, `--source-config`, Source을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Data 소스 Options

실무 운영 가이드에서 `check`, `scan`, `mask`, `profile`, `learn`, `compare`, `read`을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Option을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Short을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------|-------------|
| 실무 운영 가이드에서 `--connection`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `--conn`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Database을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `--table`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | 데이터베이스 테이블 name to read |
| 실무 운영 가이드에서 `--query`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | 실무 운영 가이드에서 SQL, `--table`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `--source-config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `--sc`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON, YAML, Path을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `--source-name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | | 실무 운영 가이드에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Connection String Formats

### PostgreSQL

```
postgresql://user:password@host:5432/dbname
```

실무 운영 가이드에서 PostgreSQL, SQL, Install을(를) 다루는 항목입니다:

```bash
pip install truthound[postgresql]
```

### MySQL

```
mysql://user:password@host:3306/dbname
```

실무 운영 가이드에서 MySQL, SQL, Install을(를) 다루는 항목입니다:

```bash
pip install truthound[mysql]
```

### SQLite

```
sqlite:///path/to/database.db
sqlite:///./relative/path.db
```

실무 운영 가이드에서 SQLite, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### DuckDB

```
duckdb:///path/to/database.duckdb
duckdb:///:memory:
```

실무 운영 가이드에서 Install, DuckDB을(를) 다루는 항목입니다:

```bash
pip install truthound[duckdb]
```

### Microsoft SQL Server

```
mssql://user:password@host:1433/dbname
```

실무 운영 가이드에서 SQL, Install, Server을(를) 다루는 항목입니다:

```bash
pip install truthound[mssql]
```

## 소스 설정 파일 Format

실무 운영 가이드에서 `--source-config`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### JSON Example

```json
{
  "type": "postgresql",
  "connection": "postgresql://user:password@host:5432/dbname",
  "table": "users",
  "source_name": "production-users"
}
```

### YAML Example

```yaml
type: postgresql
connection: postgresql://user:password@host:5432/dbname
table: users
source_name: production-users
```

### Using a SQL Query

```yaml
type: postgresql
connection: postgresql://user:password@host:5432/dbname
query: "SELECT id, name, email FROM users WHERE active = true"
source_name: active-users
```

## Dual-소스 설정 (for `compare`)

실무 운영 가이드에서 `compare`, `baseline`, `current`, You을(를) 다루는 항목입니다:

```yaml
baseline:
  type: postgresql
  connection: postgresql://user:pass@host/db
  table: users_baseline

current:
  type: postgresql
  connection: postgresql://user:pass@host/db
  table: users_current
```

실무 운영 가이드에서 Usage을(를) 다루는 항목입니다:

```bash
truthound compare --source-config compare_sources.yaml --method psi
```

실무 운영 가이드에서 Alternatively을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Per-Backend Install Hints

실무 운영 가이드에서 Truthound, Install을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Backend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Install, Command을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|----------------|
| 실무 운영 가이드에서 PostgreSQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[postgresql]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 MySQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[mysql]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 DuckDB을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[duckdb]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 SQL, Server을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[mssql]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 BigQuery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[bigquery]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Snowflake을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[snowflake]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `pip install truthound[databases]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 SQLite, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 보안 Considerations

실무 운영 가이드에서 CLI, Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 Recommended을(를) 다루는 항목입니다:

1. 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

    ```bash
    export DB_CONN="postgresql://user:password@host/db"
    truthound check --connection "$DB_CONN" --table users
    ```

2. 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

    ```bash
    chmod 600 db_config.yaml
    truthound check --source-config db_config.yaml
    ```

3. 실무 운영 가이드에서 `.pgpass`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

4. 실무 운영 가이드에서 Avoid, CI/CD, GitHub, Secrets, Vault을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 예시 for Each Command

### check

```bash
# Validate a PostgreSQL table
truthound check --connection "postgresql://user:pass@host/db" --table orders

# Validate with source config
truthound check --source-config prod_db.yaml --strict
```

### scan

```bash
# Scan a database table for PII
truthound scan --connection "postgresql://user:pass@host/db" --table customers
```

### mask

```bash
# Mask PII in a database table and write to a file
truthound mask --connection "sqlite:///data.db" --table users -o masked_users.csv
```

### 프로파일

```bash
# Profile a database table
truthound profile --connection "postgresql://user:pass@host/db" --table transactions
```

### learn

```bash
# Learn schema from a database table
truthound learn --connection "postgresql://user:pass@host/db" --table products -o schema.yaml
```

### compare

```bash
# Compare two database tables
truthound compare --source-config compare_sources.yaml --method psi --strict
```

### read

```bash
# Preview a database table
truthound read --connection "postgresql://user:pass@host/db" --table users --head 20

# Run a SQL query and export as CSV
truthound read --connection "sqlite:///data.db" --query "SELECT * FROM orders WHERE total > 100" --format csv -o high_orders.csv
```

## 함께 보기

- [Data Sources 개요](index.md)
- [데이터베이스 Connections](databases.md)
- 실무 운영 가이드에서 CLI, Core, Commands을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
