# 관계형 데이터베이스 DataSource

Truthound의 SQLite, DuckDB, PostgreSQL, MySQL, Oracle, SQL Server 연결과
검증 계약을 설명합니다.

## 개요

| 데이터베이스 | 드라이버 | 설치 | 기본 포함 |
|--------------|----------|------|-----------|
| SQLite | `sqlite3` | 별도 설치 없음 | 예 |
| DuckDB | `duckdb` | `pip install truthound[duckdb]` | 아니요 |
| PostgreSQL | `psycopg2` | `pip install truthound[postgresql]` | 아니요 |
| MySQL | `pymysql` | `pip install truthound[mysql]` | 아니요 |
| Oracle | `oracledb` | `pip install truthound[oracle]` | 아니요 |
| SQL Server | `pymssql` (`pyodbc`도 사용 가능) | `pip install truthound[sqlserver]` | 아니요 |

## 공통 계약

- thread-safe connection pool
- table mode와 지원되는 provider의 query mode
- SQL pushdown과 schema inference
- 공통 schema query 또는 provider-native metadata 조회 전략
- tuple, mapping, driver row를 column name 기준으로 정규화
- `fetch_size` batch와 `materialization_row_limit`를 적용한 bounded fallback

SQL DataSource는 위치 인자가 아니라 `source` keyword로 전달합니다.

```python
import truthound as th

validation = th.check(source=source)
profile = th.profile(source=source)
```

Polars fallback 기본 제한은 100,000행입니다. 제한을 넘으면 일부 데이터만
성공으로 반환하지 않고 `DataSourceSizeError`를 발생시킵니다. 대용량 profile이나
non-pushdown 작업은 `source.sample(10_000)`처럼 명시적인 sample을 사용합니다.

```python
from truthound.datasources.sql.base import SQLDataSourceConfig

config = SQLDataSourceConfig(
    fetch_size=10_000,
    materialization_row_limit=100_000,
)
```

### Provider 생성 계약

공개 지원 목록의 SQL provider는 credential 또는 network 검사 전에 concrete
class여야 합니다. provider는 공통 schema query 전략이나 native metadata 전략 중
하나를 사용합니다. 두 전략을 모두 구현하지 않은 사용자 정의 provider는 생성
시점에 schema strategy 오류로 실패합니다. release QA는 driver import뿐 아니라
built artifact에서 SQL provider class 10종을 직접 import하고 concrete 상태를
전수 검사합니다.

이 계약은 provider 생성 가능성을 증명할 뿐 외부 계정을 인증하지 않습니다.
운영 지원에는 실제 credential 기반 read, validation/profile 결과, 재진입과 cleanup
증거가 별도로 필요합니다.

### Capabilities

```python
from truthound.datasources import DataSourceCapability

# All SQL sources have these capabilities
source.capabilities
# {
#     DataSourceCapability.SQL_PUSHDOWN,
#     DataSourceCapability.SAMPLING,
#     DataSourceCapability.SCHEMA_INFERENCE,
#     DataSourceCapability.ROW_COUNT,
# }
```

## SQLite

실무 운영 가이드에서 SQLite, SQL, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Basic Usage

```python
from truthound.datasources.sql import SQLiteDataSource

# From database file
source = SQLiteDataSource(table="users", database="data.db")

# In-memory database
source = SQLiteDataSource(table="users", database=":memory:")
```

### Query Mode

실무 운영 가이드에서 SQL, Validate을(를) 다루는 항목입니다:

```python
# Table mode (validate existing table)
source = SQLiteDataSource(table="users", database="data.db")

# Query mode (validate query results)
source = SQLiteDataSource(
    database="data.db",
    query="SELECT id, name, email FROM users WHERE active = 1",
)

# Access query properties
print(source.is_query_mode)    # True
print(source.query_sql)        # "SELECT id, name FROM users WHERE active = 1"
print(source.table_name)       # None
```

### From DataFrame

실무 운영 가이드에서 SQLite, Polars, SQL, Create, Pandas, DataFrame을(를) 다루는 항목입니다:

```python
import pandas as pd
from truthound.datasources.sql import SQLiteDataSource

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
})

# Creates a temporary database file
source = SQLiteDataSource.from_dataframe(df, "test_table")

# Or specify database path
source = SQLiteDataSource.from_dataframe(df, "users", database="test.db")
```

### 설정

```python
from truthound.datasources.sql import SQLiteDataSource, SQLiteDataSourceConfig
import sqlite3

config = SQLiteDataSourceConfig(
    database="data.db",
    timeout=5.0,                                                # Connection timeout
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    isolation_level=None,                                        # Autocommit mode
)

source = SQLiteDataSource(table="users", config=config)
```

### SQLite-Specific Methods

```python
source = SQLiteDataSource(table="users", database="data.db")

# Get table info (PRAGMA table_info)
info = source.get_table_info()
# [{'cid': 0, 'name': 'id', 'type': 'INTEGER', 'notnull': 1, ...}, ...]

# Get index info (PRAGMA index_list)
indexes = source.get_index_info()

# Get foreign keys (PRAGMA foreign_key_list)
fks = source.get_foreign_keys()

# Optimize database
source.vacuum()     # Reclaim storage
source.analyze()    # Update statistics
```

## PostgreSQL

실무 운영 가이드에서 PostgreSQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설치

```bash
pip install psycopg2-binary
# or for production
pip install psycopg2
```

### Basic Usage

```python
from truthound.datasources.sql import PostgreSQLDataSource

source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    schema_name="public",  # Default schema
)
```

### Connection String

```python
source = PostgreSQLDataSource.from_connection_string(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table="users",
    schema_name="public",
)
```

### Query Mode

```python
# Validate results of a complex query
source = PostgreSQLDataSource(
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
    query="""
        SELECT u.id, u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at > '2024-01-01'
    """,
)
```

### 설정

```python
from truthound.datasources.sql import PostgreSQLDataSource, PostgreSQLDataSourceConfig

config = PostgreSQLDataSourceConfig(
    host="localhost",
    port=5432,
    database="mydb",
    user="postgres",
    password="secret",
    schema_name="public",
    sslmode="prefer",            # SSL mode: disable, require, verify-ca, verify-full
    application_name="truthound",

    # Connection pool settings
    pool_size=5,                  # Connections in pool
    pool_timeout=30.0,            # Timeout for acquiring connection
    query_timeout=300.0,          # Query execution timeout
    fetch_size=10000,             # Rows to fetch at a time
)

source = PostgreSQLDataSource(table="users", config=config)
```

### PostgreSQL-Specific Methods

```python
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
)

# Get table size information
size = source.get_table_size()
# {'total_size': '1024 MB', 'table_size': '512 MB', 'indexes_size': '512 MB'}

# Get table statistics from pg_stat_user_tables
stats = source.get_table_statistics()
# {'live_rows': 1000000, 'dead_rows': 500, 'last_vacuum': ..., ...}

# Get index information
indexes = source.get_index_info()
# [{'index_name': 'users_pkey', 'column_name': 'id', 'is_unique': True, ...}, ...]

# Get constraints
constraints = source.get_constraints()
# [{'constraint_name': 'users_pkey', 'constraint_type': 'p', ...}, ...]

# Maintenance operations
source.analyze()              # Update statistics
source.vacuum()               # Reclaim storage
source.vacuum(full=True)      # Full vacuum (locks table)
```

## MySQL

실무 운영 가이드에서 MySQL, SQL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설치

```bash
pip install pymysql
```

### Basic Usage

```python
from truthound.datasources.sql import MySQLDataSource

source = MySQLDataSource(
    table="users",
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
)
```

### Connection String

```python
source = MySQLDataSource.from_connection_string(
    connection_string="mysql://root:pass@localhost:3306/mydb",
    table="users",
)
```

### 설정

```python
from truthound.datasources.sql import MySQLDataSource, MySQLDataSourceConfig

config = MySQLDataSourceConfig(
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="secret",
    charset="utf8mb4",           # Character set
    autocommit=True,             # Auto-commit mode
    ssl={                        # SSL configuration
        "ca": "/path/to/ca.pem",
        "cert": "/path/to/client-cert.pem",
        "key": "/path/to/client-key.pem",
    },

    # Connection pool settings
    pool_size=5,
    pool_timeout=30.0,
    query_timeout=300.0,
    fetch_size=10000,
)

source = MySQLDataSource(table="users", config=config)
```

### MySQL-Specific Methods

```python
source = MySQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="root",
    password="secret",
)

# Get table status (SHOW TABLE STATUS)
status = source.get_table_status()
# {'Rows': 1000000, 'Avg_row_length': 128, 'Data_length': ..., ...}

# Get table size
size = source.get_table_size()
# {'data_size_mb': 128.5, 'index_size_mb': 32.1, 'total_size_mb': 160.6, 'approx_rows': 1000000}

# Get index information (SHOW INDEX)
indexes = source.get_index_info()

# Get CREATE TABLE statement
create_sql = source.get_create_table()
print(create_sql)
# CREATE TABLE `users` (
#   `id` int NOT NULL AUTO_INCREMENT,
#   ...
# )

# Maintenance operations
source.analyze()    # Update statistics
source.optimize()   # Defragment table
```

## Connection Pooling

실무 운영 가이드에서 SQL을(를) 다루는 항목입니다:

```python
from truthound.datasources.sql import SQLConnectionPool

# Pool is managed automatically, but you can access it
source = PostgreSQLDataSource(table="users", ...)

# Pool properties
print(source._pool.size)       # Pool size
print(source._pool.available)  # Available connections

# Context manager for connections
with source._get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    cursor.close()
```

### 설정

```python
from truthound.datasources.sql import SQLDataSourceConfig

config = SQLDataSourceConfig(
    pool_size=5,           # Maximum connections in pool
    pool_timeout=30.0,     # Timeout waiting for connection
    query_timeout=300.0,   # Query execution timeout
    fetch_size=10000,      # Batch size for fetching
    use_server_side_cursor=False,  # Server-side cursors for large results
    schema_name=None,      # Database schema
)
```

## Query Execution

실무 운영 가이드에서 SQL, Execute을(를) 다루는 항목입니다:

```python
source = PostgreSQLDataSource(table="users", ...)

# Execute query returning rows
results = source.execute_query("SELECT * FROM users WHERE age > %s", (30,))
# [{'id': 1, 'name': 'Alice', 'age': 35}, ...]

# Execute query returning single value
count = source.execute_scalar("SELECT COUNT(*) FROM users WHERE active = %s", (True,))
# 1000

# Built-in query builders
count_query = source.build_count_query("age > 30")
# "SELECT COUNT(*) FROM public.users WHERE age > 30"

distinct_query = source.build_distinct_count_query("email")
# "SELECT COUNT(DISTINCT email) FROM public.users"

null_query = source.build_null_count_query("phone")
# "SELECT COUNT(*) FROM public.users WHERE phone IS NULL"

stats_query = source.build_stats_query("age")
# "SELECT COUNT(age) as count, AVG(age) as mean, ..."
```

## 검증 Example

실무 운영 가이드에서 SQL, API을(를) 다루는 항목입니다:

```python
import truthound as th
from truthound.datasources.sql import PostgreSQLDataSource

# Create source
source = PostgreSQLDataSource(
    table="users",
    host="localhost",
    database="mydb",
    user="postgres",
    password="secret",
)

# Run validation - SQL pushdown executes queries on the database
report = th.check(
    source=source,
    validators=["null", "unique", "duplicate"],
    columns=["id", "email", "phone"],
)

# With rules
report = th.check(
    source=source,
    rules={
        "id": ["not_null", "unique"],
        "email": ["not_null", {"type": "regex", "pattern": r".*@.*"}],
        "status": [{"type": "allowed_values", "values": ["active", "inactive"]}],
    },
)

print(f"Found {len(report.issues)} issues")
```

## Sampling

실무 운영 가이드에서 SQL을(를) 다루는 항목입니다:

```python
source = PostgreSQLDataSource(table="users", ...)

# Create sampled source (uses LIMIT)
sampled = source.sample(n=10000)

# Sampled source wraps queries with LIMIT
print(sampled.full_table_name)
# "(SELECT * FROM public.users LIMIT 10000) AS sampled"
```

## Converting to Polars

실무 운영 가이드에서 Polars, Fetch, LazyFrame을(를) 다루는 항목입니다:

```python
source = PostgreSQLDataSource(table="users", ...)

# Convert to LazyFrame (fetches all data)
lf = source.to_polars_lazyframe()

# Apply Polars operations
result = (
    lf
    .filter(pl.col("age") > 25)
    .group_by("department")
    .agg(pl.col("salary").mean())
    .collect()
)
```

> 실무 운영 가이드에서 `to_polars_lazyframe()`, Warning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Factory Functions

실무 운영 가이드에서 SQL, Convenience을(를) 다루는 항목입니다:

```python
from truthound.datasources import get_sql_datasource

# SQLite
source = get_sql_datasource("data.db", table="users")

# PostgreSQL
source = get_sql_datasource(
    "postgresql://user:pass@localhost/mydb",
    table="users",
)

# MySQL
source = get_sql_datasource(
    "mysql://root:pass@localhost/mydb",
    table="users",
)
```

## Error Handling

```python
from truthound.datasources.sql import PostgreSQLDataSource
from truthound.datasources.base import (
    DataSourceError,
    DataSourceConnectionError,
)

try:
    source = PostgreSQLDataSource(
        table="users",
        host="nonexistent.host",
        database="mydb",
        user="postgres",
        password="wrong",
    )
    source.validate_connection()
except DataSourceConnectionError as e:
    print(f"Connection failed: {e}")

try:
    source = PostgreSQLDataSource(
        table="nonexistent_table",
        host="localhost",
        database="mydb",
        user="postgres",
        password="secret",
    )
    schema = source.schema  # Triggers schema fetch
except DataSourceError as e:
    print(f"Error: {e}")
```

## Checking Availability

실무 운영 가이드에서 Check을(를) 다루는 항목입니다:

```python
from truthound.datasources.sql import get_available_sources, check_source_available

# Get all SQL sources and their availability
sources = get_available_sources()
for name, cls in sources.items():
    status = "available" if cls is not None else "not installed"
    print(f"{name}: {status}")
# sqlite: available
# postgresql: available
# mysql: not installed
# ...

# Check specific source
if check_source_available("postgresql"):
    from truthound.datasources.sql import PostgreSQLDataSource
    # Use PostgreSQL...
else:
    print("Install psycopg2-binary: pip install psycopg2-binary")
```

## 권장 방식

1. 실무 운영 가이드에서 Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 `sample()`, Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. 실무 운영 가이드에서 Set, Prevent을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
5. 실무 운영 가이드에서 PostgreSQL, MySQL, SQL, `sslmode`, `ssl`, SSL, Configure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. 실무 운영 가이드에서 `check_source_available()`, Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
